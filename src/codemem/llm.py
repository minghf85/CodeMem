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

async def _complete_choice(
    client: openai.AsyncOpenAI,
    config: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    require_text: bool = True,
) -> Any:
    """调一次 chat completion，返回 ``choices[0]``。失败按 ``max_retries`` 重试。

    ``tools`` 非空时走 **OpenAI 原生函数调用**（``tools=`` 参数）。这一模式下结果是
    ``message.tool_calls``，``content`` 可以是空的 —— 所以此时传 ``require_text=False``，
    把"空正文"从错误降级为合法状态（"既没有正文也没有工具调用"仍然算失败、要重试）。

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
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
        # agent loop 一步只执行一个调用，所以要求端点一次只给一个（否则并行调用会被丢弃）。
        # 这是可选参数：不支持的端点 400，下面的循环会把它去掉再试。
        kwargs["parallel_tool_calls"] = False
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
            tool_calls = getattr(message, "tool_calls", None)
            content = message.content or ""
            if require_text:
                if not content.strip():
                    # 关思考后偶发空回复：补 reasoning 兜底，仍空则当瞬时错误重试
                    content = (getattr(message, "reasoning", None) or "").strip()
                if not content:
                    raise RuntimeError("empty completion content")
                if getattr(choice, "finish_reason", None) == "length":
                    raise TruncatedCompletion(
                        f"completion truncated by max_tokens (tail: {content[-80:]!r})"
                    )
            else:
                # 原生工具调用：有 tool_calls 就成功；都没有才算失败。
                if not tool_calls and not content.strip():
                    raise RuntimeError("empty completion content (no tool_calls)")
                if not tool_calls and getattr(choice, "finish_reason", None) == "length":
                    raise TruncatedCompletion(
                        f"completion truncated by max_tokens (tail: {content[-80:]!r})"
                    )
            return choice
        except openai.APIStatusError as exc:
            last_error = exc
            # 端点不认识 ``parallel_tool_calls``（可选参数）就 400 —— 去掉它再试一次，
            # 而不是把整次调用判死。工具本身仍能正常工作，只是可能一次给多个调用。
            if (exc.status_code == 400 and "parallel_tool_calls" in kwargs
                    and "parallel_tool_calls" in str(exc)):
                kwargs.pop("parallel_tool_calls", None)
                continue
            if exc.status_code in non_retryable:
                raise
        except Exception as exc:  # noqa: BLE001 - 连接/超时/解析等瞬时错误
            last_error = exc
        if attempt < int(config.get("max_retries", 3)):
            await asyncio.sleep(backoff_seconds(last_error, attempt, config))

    if isinstance(last_error, TruncatedCompletion):
        raise TruncatedCompletion(f"chat completion failed after retries: {last_error}")
    raise RuntimeError(f"chat completion failed after retries: {last_error}")


async def chat_completion(
    client: openai.AsyncOpenAI,
    config: dict[str, Any],
    messages: list[dict[str, Any]],
) -> str:
    """调一次 chat completion 并返回正文（文本协议：正文就是全部信息）。"""
    choice = await _complete_choice(client, config, messages, require_text=True)
    message = choice.message
    content = message.content or ""
    if not content.strip():
        # 关思考后偶发空回复：补 reasoning 兜底（返回值不再校验，故这里再兜一次）
        content = (getattr(message, "reasoning", None) or "").strip()
    return content


async def chat_completion_message(
    client: openai.AsyncOpenAI,
    config: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> Any:
    """调一次并返回 assistant **message 对象**（含原生 ``tool_calls``）。

    给"用原生函数调用而不是文本协议"的模型（如 gpt-4o-mini）用：``tools`` 非空时走
    ``tools=``，``message.tool_calls`` 是要执行的动作，``message.content`` 是纯文本
    （在 agent 协议里就是"做完了"信号）。调用方负责把两者折回自己的协议。
    """
    choice = await _complete_choice(
        client, config, messages, tools=tools or None, require_text=not tools
    )
    return choice.message


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
