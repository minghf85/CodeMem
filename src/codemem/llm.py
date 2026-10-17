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
- **连接建立超时单独设短** —— 远程端点在高并发下会丢一部分**新建**连接；``connect_timeout``
  比读超时短得多，好让失败快速发生、把重试预算花在重试而不是等内核超时（见 ``make_client``）。

放在顶层而不是藏进 add/：这些都是**端点行为**，与"抽取记忆"这个业务无关。
"""

from __future__ import annotations

import asyncio
import random
import re
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
    "connect_timeout": 10.0,
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
    "timeout", "connect_timeout", "max_retries", "backoff_base", "rate_limit_backoff",
    "max_backoff", "backoff_jitter",
)


class TruncatedCompletion(RuntimeError):
    """模型输出因达到 ``max_tokens`` 而被截断（``finish_reason == 'length'``）。

    ``content`` 是**被截断的那段正文** —— 调用方可以按对象边界抢救出前面完整的记录
    （见 ``io.salvage_json_records``），而不是把整轮产出当作没发生。
    """

    def __init__(self, message: str, content: str = "") -> None:
        super().__init__(message)
        self.content = content


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
    """按配置建一个异步客户端。

    ``timeout`` 是**读超时**（一次生成可能很慢，要留足）；``connect_timeout`` 单独设、
    且**明显更短**。原因：远程端点在高并发下会丢掉一部分**新建**连接（SYN 被丢，实测
    16 并发时约一半失败，而串行复用连接几乎不失败）。若不单独限时，一次连接失败要等内核
    SYN 重传超时（~60s）才轮到重试 —— 既慢，又把重试预算耗在"等超时"上。短 connect
    超时让失败**快速**发生，退避后重试才有意义。

    **连接池按并发配**（``max_connections`` / ``max_keepalive_connections``）：默认池只留
    少量 keep-alive 连接，32 路并发时每批请求都要**新建**连接 —— 而那正是远程端点会丢的
    东西（同上）。把 keep-alive 上限设成并发量，池里的连接就能被下一批直接复用，
    新建连接数降到接近 0（``llm.max_connections`` / ``llm.max_keepalive_connections`` 可覆盖，
    ``max_keepalive_connections`` 缺省取 ``concurrency``）。

    连接池这一段是**加速件**，不是依赖：拿不到 ``httpx`` 时退回 SDK 自建客户端（超时照旧），
    功能一模一样，只是池子不能用并发量去调。**绝不能因为一个可选优化让整条链路起不来。**
    """
    merged = merge_call_config(config)
    read_timeout = float(merged.get("timeout", 300))
    connect_timeout = float(merged.get("connect_timeout", 10.0))

    try:
        import httpx
    except ImportError:                     # pragma: no cover - 取决于运行环境
        return openai.AsyncOpenAI(
            base_url=merged["base_url"],
            api_key=merged.get("api_key") or "not-needed",
            timeout=openai.Timeout(
                connect=connect_timeout, read=read_timeout,
                write=read_timeout, pool=read_timeout,
            ),
            max_retries=0,
        )

    concurrency = max(1, int(config.get("concurrency", 8) or 8))
    max_connections = int(config.get("max_connections", max(32, concurrency * 2)) or 0)
    keepalive = int(config.get("max_keepalive_connections", concurrency) or 0)
    # 超时**建在 http_client 上**：传了 http_client 时，SDK 直接用它（不再另建一个默认客户端），
    # 所以这里的 Timeout 就是生效的那一份 —— 不要再给 AsyncOpenAI 传 timeout，免得两处各说各话。
    timeout = httpx.Timeout(
        connect=connect_timeout, read=read_timeout, write=read_timeout, pool=read_timeout,
    )
    return openai.AsyncOpenAI(
        base_url=merged["base_url"],
        api_key=merged.get("api_key") or "not-needed",
        max_retries=0,          # 重试由 chat_completion 自己管，避免两层退避相乘
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=max_connections or None,
                max_keepalive_connections=keepalive or None,
            ),
            timeout=timeout,
        ),
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
                        f"completion truncated by max_tokens (tail: {content[-80:]!r})",
                        content=content,
                    )
            else:
                # 原生工具调用：有 tool_calls 就成功；都没有才算失败。
                if not tool_calls and not content.strip():
                    raise RuntimeError("empty completion content (no tool_calls)")
                if not tool_calls and getattr(choice, "finish_reason", None) == "length":
                    raise TruncatedCompletion(
                        f"completion truncated by max_tokens (tail: {content[-80:]!r})",
                        content=content,
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
    raise RuntimeError(f"chat completion failed after retries: {describe_error(last_error)}")


#: 宽字符（CJK 等）范围：这些字一个≈一个 token，不能按"4 字符≈1 token"折算。
_WIDE_RE = re.compile(r"[⺀-鿿豈-﫿가-힯＀-￯]")


def estimate_tokens(text: str) -> int:
    """粗估 token 数：宽字符一字算一个，其余按 4 字符一个。

    **宁高勿低**：估多了只是把输入切小一点（多一次调用），估少了就是 400 报错、整条 message
    一条 atom 都拿不到。所以 ASCII 也按 4 字符/token 而不是更乐观的 5~6。

    放在 ``llm`` 而不是某个步骤里：add 的两条路（atommem 的抽取、summem 的总结）都要按它
    给 prompt 定预算，两边必须用同一把尺子。
    """
    if not text:
        return 0
    wide = len(_WIDE_RE.findall(text))
    return wide + (len(text) - wide + 3) // 4


#: 端点抱怨"输入太长"时正文里的典型措辞。各家措辞不一，故按关键词匹配而不是精确串：
#: vLLM/sglang 是 ``maximum context length``、OpenAI 是 ``context_length_exceeded``、
#: 另一些网关写 ``too long`` / ``reduce the length``。
_CONTEXT_LENGTH_HINTS = (
    "context length", "context_length", "maximum context", "max context",
    "reduce the length", "too many tokens", "token limit", "too long",
)

#: 端点**内容策略**拦截时正文里的典型措辞（实测：OpenAI 系网关的
#: ``The response was filtered due to the prompt triggering content management policy``，
#: Azure 用 ``content_filter``，另一些写 ``safety system`` / ``flagged``）。
_CONTENT_FILTER_HINTS = (
    "content management policy", "content_filter", "content filter", "content policy",
    "safety system", "moderation", "was filtered", "flagged",
)

#: 这些状态码下**重发同一个 prompt 也是同样结果**：请求非法 / 权限 / 找不到 / 太大 / 内容被拦。
_PERMANENT_STATUS = frozenset({400, 401, 403, 404, 413, 422})


def _walk_causes(exc: BaseException | None):
    """沿 ``__cause__`` / ``__context__`` 走一遍异常链（SDK 常把细节埋在底层）。"""
    seen: set[int] = set()
    node = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        yield node
        node = node.__cause__ or node.__context__


def _matches(exc: BaseException | None, hints: tuple[str, ...]) -> bool:
    """异常链里有没有节点是 4xx（或没有状态码）且正文里带某个关键词。"""
    for node in _walk_causes(exc):
        status = getattr(node, "status_code", None)
        if status is not None and status not in _PERMANENT_STATUS:
            continue
        text = f"{node}".lower()
        if any(hint in text for hint in hints):
            return True
    return False


def is_context_length_error(exc: BaseException | None) -> bool:
    """判断这次失败是不是"输入超出模型上下文"。

    为什么要单独认出来：**它和网络抖动不是一类问题** —— 重试解决不了，缩输入才能解决。
    调用方（atommem 的抽取）据此把 prompt 预算砍半再试，而不是把一个超大 prompt 原样重发
    ``max_retries`` 次（那只会把 4xx 烧成几分钟的空转，最后整条 message 一条 atom 都没有）。
    """
    return _matches(exc, _CONTEXT_LENGTH_HINTS)


def is_content_filter_error(exc: BaseException | None) -> bool:
    """判断这次失败是不是被端点的**内容策略**拦下的（``code: content_filter``）。

    它对我们的意义和"超上下文"正好相反：**缩输入救不回来**（拦的不是长度，是内容），
    所以重试是纯浪费 —— 但值得试一次"把上下文去掉、只抽这一条"（触发词常常在**前文**里）。
    """
    return _matches(exc, _CONTENT_FILTER_HINTS)


def is_permanent_prompt_error(exc: BaseException | None) -> bool:
    """这次失败是不是"重发同一个 prompt 也不会有不同结果"。

    内容策略、非法请求、模型不存在、认证失败……都属于这一类。**别重试**：原地重试、补漏重试
    都只会把同一份 prompt 再烧一遍（实测：一次内容策略拦截会在补漏里被反复重试到天荒地老）。
    调用方据此把它记成 ``blocked``（不自动重试、但留痕可见），而**不是** ``failed``。

    "输入超上下文"**不算**这一类 —— 它虽然也是 400，但那是"缩一缩就能过"的。
    """
    if is_context_length_error(exc):
        return False
    for node in _walk_causes(exc):
        status = getattr(node, "status_code", None)
        if status in _PERMANENT_STATUS:
            return True
    return False


def describe_error(exc: Exception | None) -> str:
    """把异常写成带**底层原因**的一行。

    OpenAI SDK 的连接失败只把顶层异常写成 ``APIConnectionError: Connection error.``
    —— 真正有用的信息（``ConnectError`` / ``ConnectTimeout``）在 ``__cause__`` 里，不打出来
    就完全没法判断"是超时、被重置、还是连不上"。
    """
    if exc is None:
        return "unknown error"
    text = f"{type(exc).__name__}: {exc}"
    cause: BaseException | None = exc
    seen = set()
    while getattr(cause, "__cause__", None) is not None and id(cause) not in seen:
        seen.add(id(cause))
        cause = cause.__cause__
    if cause is not None and cause is not exc:
        text += f" (cause: {type(cause).__name__}: {cause!r})"
    return text


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
