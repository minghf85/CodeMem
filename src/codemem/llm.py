"""LLM 调用：重试、退避、截断检测。

add（抽取原子记忆）、answer（依证据作答）、eval（judge）、search（verifier 与 agent）
四处都要调 chat completion，且都要处理同一批坑：

- **429 / 503** —— 按 ``Retry-After`` 或 ``rate_limit_backoff`` 退避；其它错误普通指数退避；
  统一叠抖动，避免并发请求同时重试再次打爆限流。
- **空回复** —— 小模型关掉思考后偶发，补 ``reasoning`` 兜底，仍空则当瞬时错误重试。
- **被 max_tokens 截断**（``finish_reason == "length"``）—— 内容必然是不完整的 JSON；
  思考长度会波动，重试常能成功。所以单独成类 ``TruncatedCompletion``，让调用方能区分
  "截断了"和"真挂了"。
- **qwen3 关思考** —— 只有模型名含 ``qwen3`` 才发 ``chat_template_kwargs``；
  其它端点会因未知参数直接 400。

放在顶层而不是藏进 add/：这些都是**端点行为**，与"抽取记忆"这个业务无关。
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from typing import Any

import openai

# 谈判用的默认参数。调用方用 ``merge_call_config`` 把自己的模型配置叠上去。
DEFAULT_CALL_CONFIG: dict[str, Any] = {
    "base_url": "http://127.0.0.1:30000/v1",
    "api_key": "sglang",
    "model": "local",
    "temperature": 0.2,
    "max_tokens": 4096,
    "timeout": 300,
    "max_retries": 3,
    "backoff_base": 2.0,
    "rate_limit_backoff": 10.0,
    "max_backoff": 60.0,
    "backoff_jitter": 0.2,
    "enable_thinking": False,
}

# ``chat_completion`` 真正会读的键。配置里的其它键（业务参数）原样忽略。
CALL_KEYS = (
    "base_url", "api_key", "model", "temperature", "max_tokens", "enable_thinking",
    "timeout", "max_retries", "backoff_base", "rate_limit_backoff", "max_backoff",
    "backoff_jitter",
)


class TruncatedCompletion(RuntimeError):
    """模型输出因达到 ``max_tokens`` 而被截断（``finish_reason == 'length'``）。"""


def merge_call_config(config: dict[str, Any]) -> dict[str, Any]:
    """把调用方的配置叠到默认值上，只保留 ``CALL_KEYS``。

    为什么只挑键：配置文件通常把业务参数和模型参数混在一层（``search.yaml`` 里既有
    ``max_steps`` 也有 ``model``）。这里做一次收窄，让 ``chat_completion`` 拿到的永远
    是干净的调用参数 —— 否则少一个键就是运行时 ``KeyError``。
    """
    merged = dict(DEFAULT_CALL_CONFIG)
    for key in CALL_KEYS:
        if key in config and config[key] is not None:
            merged[key] = config[key]
    return merged


def make_client(config: dict[str, Any]) -> openai.AsyncOpenAI:
    """按配置建一个异步客户端。"""
    merged = merge_call_config(config)
    return openai.AsyncOpenAI(
        base_url=merged["base_url"],
        api_key=merged.get("api_key") or "not-needed",
        timeout=float(merged.get("timeout", 300)),
        max_retries=0,          # 重试由 chat_completion 自己管，避免两层退避相乘
    )


# ---------------------------------------------------------------------------
# 退避
# ---------------------------------------------------------------------------

def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _retry_after_seconds(exc: Exception) -> float | None:
    """解析 ``Retry-After`` 响应头（秒数或 HTTP-date 两种形式）。"""
    response = getattr(exc, "response", None)
    raw = response.headers.get("Retry-After") if response is not None else None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        target = parsedate_to_datetime(raw)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


def backoff_seconds(exc: Exception, attempt: int, config: dict[str, Any]) -> float:
    """本次失败后该等多久。429/503 用更长的 ``rate_limit_backoff``，其余用 ``backoff_base``。"""
    base = float(config.get("backoff_base", 2.0))
    cap = float(config.get("max_backoff", 120.0))
    jitter = float(config.get("backoff_jitter", 0.5))
    status = _status_code(exc)

    if status == 429:
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            delay = min(retry_after, cap)
        else:
            delay = min(float(config.get("rate_limit_backoff", 10.0)) * 2 ** (attempt - 1), cap)
    elif status == 503:
        delay = min(float(config.get("rate_limit_backoff", 10.0)) * 2 ** (attempt - 1), cap)
    else:
        delay = min(base * 2 ** (attempt - 1), cap)
    return delay + random.uniform(0, jitter * delay)


# ---------------------------------------------------------------------------
# 调用
# ---------------------------------------------------------------------------

async def chat_completion(
    client: openai.AsyncOpenAI,
    config: dict[str, Any],
    messages: list[dict[str, Any]],
) -> str:
    """调一次 chat completion 并返回正文。失败按 ``max_retries`` 重试。

    配置里若指定了与 ``client`` 不同的 ``base_url``，**显式切换** —— OpenAI SDK 的默认
    client 会忽略请求级 base_url，导致请求被静默发到旧地址（这个坑踩过）。
    """
    config = merge_call_config(config)
    base_url = config.get("base_url")
    if base_url and str(client.base_url).rstrip("/") != str(base_url).rstrip("/"):
        client = client.with_options(base_url=base_url)

    kwargs: dict[str, Any] = {
        "model": config["model"],
        "messages": messages,
        "temperature": config.get("temperature", 0.2),
        "max_tokens": config.get("max_tokens", 4096),
        "stream": False,
    }
    # 只有 qwen3 端点支持关闭思考；其它端点会因未知参数 400。
    if "qwen3" in str(config.get("model", "")).lower() and config.get("enable_thinking") is False:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    non_retryable = {400, 401, 403, 404, 422}
    last_error: Exception | None = None

    for attempt in range(1, int(config.get("max_retries", 3)) + 1):
        try:
            response = await client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            if not content.strip():
                # 关思考后偶发空回复：补 reasoning 兜底，仍空则当瞬时错误重试
                content = (getattr(message, "reasoning", None) or "").strip()
            if not content:
                raise RuntimeError("empty completion content")
            if getattr(choice, "finish_reason", None) == "length":
                raise TruncatedCompletion(
                    f"completion truncated by max_tokens (tail: {content[-80:]!r})"
                )
            return content
        except openai.APIStatusError as exc:
            last_error = exc
            if exc.status_code in non_retryable:
                raise
        except Exception as exc:  # noqa: BLE001 - 连接/超时/解析等瞬时错误
            last_error = exc
        if attempt < int(config.get("max_retries", 3)):
            await asyncio.sleep(backoff_seconds(last_error, attempt, config))

    if isinstance(last_error, TruncatedCompletion):
        raise TruncatedCompletion(f"chat completion failed after retries: {last_error}")
    raise RuntimeError(f"chat completion failed after retries: {last_error}")


async def complete(
    config: dict[str, Any],
    messages: list[dict[str, Any]],
    client: openai.AsyncOpenAI | None = None,
) -> str:
    """一次性调用（自建 client 时用）。有现成 client 就传 ``client`` 复用连接。"""
    if client is not None:
        return await chat_completion(client, config, messages)
    owned = make_client(config)
    try:
        return await chat_completion(owned, config, messages)
    finally:
        await owned.close()


def complete_sync(config: dict[str, Any], messages: list[dict[str, Any]]) -> str:
    """同步入口，给脚本/自测用。"""
    return asyncio.run(complete(config, messages))
