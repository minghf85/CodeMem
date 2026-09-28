"""agent loop：让模型用 read/write/edit/bash 在一个目录里自主探索、写出 evidence.jsonl。

设计见 ``docs/evomem_plan_v3.md`` §5。这个 loop 的目标是**对齐成熟 code agent 的标准做法**，
同时针对实测暴露的四个失败点做了具体修正（每条都有日志里的证据）：

======================  =====================================================  ==========
实测问题                 做法                                                    借鉴来源
======================  =====================================================  ==========
93% 的观测是空（`jq >> `  每次变更型调用后**附加证据摘要**（行数/唯一 id/末尾行）     code agent 的
重定向 stdout 为空），模                                                       post-write 反馈
型看不见自己刚写了什么    → ``_evidence_digest``
60/60 步没有一次完成信号  **重复检测**：同样的工具+参数再来一次就直接拒绝并警告     重复工具调用检测
→ 12~24 步全烧在重复追加  → ``_RepeatTracker``
空转被算作"有进展"       **进展改成"唯一 id 数"变化**，不是行数变化                —
（往文件里追加重复行）    → ``EvidenceState.fingerprint``
prompt 随历史无限涨       **上下文压缩**：超出预算即把中段坍缩成摘要，保留           auto-compaction
（3.3k→4.5k tok / 12 步） system + 最近 N 条 + 工具用过的 id 集合
======================  =====================================================  ==========

外加两条稳健性措施：

1. **错误重试**：模型调用失败 / 输出解析不出工具调用时，**注入一条纠正消息再重试一次**
   （``max_parse_retries``），而不是立刻当作"完成信号"。实测 8B 偶尔会只吐一个 ``{``
   或加一段解释文字；这类瞬时畸形重试一次通常就好。
2. **空回复才终止**：只有模型**明确回复纯文本**才视为"我做完了"。解析失败不再是完成信号 ——
   它和"完成"语义完全不同，混在一起会让 verifier 对着一份半成品去判。

本模块不知道 verifier 的存在：验证由 ``evomem_v3.run_qa`` 编排（它在 agent 停下后调用
verifier，并把 ``missing[]`` 作为 user 消息再喂回来继续跑）。这样 agent loop 可以单独测试。

自测：``python scripts/test_evomem_v3.py``（假模型 + 真工具，纯 CPU）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import toolcfg
from ..log import Logger, block
from .evidence import EvidenceState, read_evidence_state
from .toolcfg import ToolCall
from .tools import ToolError, ToolResult

# 模型调用签名：(messages, meta) -> 模型原始输出。
# ``meta`` 里带 round/step/attempt 等上下文，供调用方记日志；返回空串表示调用失败。
ModelCall = Callable[[list[dict[str, str]], dict[str, Any]], Awaitable[str]]


# ---------------------------------------------------------------------------
# 证据状态：行数 + **唯一 id 数**（进展判定的依据）
# ---------------------------------------------------------------------------

def _evidence_digest(path: Path, state: EvidenceState, limit: int = 3) -> str:
    """变更型调用之后附在观测末尾的证据摘要。

    **这是修 P0 的核心**：``jq ... >> evidence.jsonl`` 的 stdout 是空的（重定向走了），
    所以模型收到的观测是 ``(no output)`` —— 它看不见自己刚写了什么，于是盲目重试。
    这里由 harness 主动把"文件现在长什么样"告诉它。
    """
    if not path.exists():
        return f"\n\n[{path.name}: does not exist yet]"
    if state.lines == 0:
        return f"\n\n[{path.name}: still empty]"
    lines: list[str] = []
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        content = []
    for line in content[-limit:]:
        line = line.strip()
        if line:
            lines.append(f"  {line[:160]}" + ("…" if len(line) > 160 else ""))
    tail = "\n".join(lines)
    return (
        f"\n\n[{path.name} now: {state.summary()}]\n"
        f"[last {len(lines)} line(s)]\n{tail}"
    )


def repeat_notice(call: ToolCall, count: int) -> str:
    """同一个工具+参数被重复调用时的拒绝消息（重复检测）。"""
    return (
        f"ERROR: you already ran this exact {call.tool} call {count} times "
        f"({call.describe()}). Running it again cannot change the result.\n"
        f"Do something DIFFERENT. Common reasons you are stuck:\n"
        f"- The id is NOT in that file. ids differ by file: records in inputs/atommem.jsonl end "
        f"in a sequence number (session_1_3_1); MESSAGES in inputs/msgmem.jsonl do not "
        f"(session_1_3). A jq select that matches nothing prints NOTHING and still exits 0.\n"
        f"- To resolve a relative time, read the session's timestamp from inputs/msgmem.jsonl:\n"
        f"    jq -c 'select(.metadata.id==\"session_1_3\") | {{id:.metadata.id,time:.metadata.time}}' "
        f"inputs/msgmem.jsonl\n"

        f"  then run timecalc on it. Do not guess the date.\n"
        f"- To see what you already have: jq -r '.metadata.id' evidence.jsonl\n"
        f"When the evidence already answers the question, reply with plain text (no JSON) to finish."
    )


def oversize_notice(records: int, limit: int) -> str:
    """evidence 明显过大时的提醒。

    实测：把"集合型问题要收全成员"写进 prompt 后，模型在一条**单事实**问题上也照做，
    一次 write 了 **658 条**记录，直接顶爆模型上下文（请求 44900 tokens > 40960 上限），
    之后 verifier 连调用都失败。prompt 的软约束又一次不可靠 —— 所以这里在观测里
    明确把"太多了"这个事实回给模型。

    注意**不是硬截断**：截断会静默丢证据（正是我们一直在避免的事）。只提醒，让模型自己精简。
    """
    return (
        f"\n\n[evidence.jsonl has {records} records -- that is far too many (aim for 5-15). "
        f"You are supposed to extract a SMALL set that answers the question, not dump every "
        f"candidate you saw. Keep only the records the answer needs, merge related ones into a "
        f"single record, then `write` the trimmed list. If the question asks for a SET of things "
        f"(activities, hobbies), list them in ONE record rather than one record per member.]"
    )


def write_nudge_notice(step_count: int, have: int) -> str:
    """"读了很多步却一个字没写"时的强制提醒。

    实测（qa15）：12 步里读了 12 次、evidence 始终为空。prompt 里明明写着
    "By turn 4 you MUST have written something"，但 8B 会无视它 —— 它找到一条候选后
    就一头扎进去反复核对来源，再也不出来。同一条经验在这个项目里已经反复出现：
    **prompt 的软约束在 8B 上不可靠，管用的是一期一会的机制**。

    所以由 harness 主动在观测里插话：把你已经查到的东西**现在**写下来。注意措辞是
    "把已有的写下来"而不是"赶紧写" —— 空 write 或编造记录都是更坏的结局。
    """
    if have:
        return (
            f"\n\n[You have gone {step_count} steps and evidence.jsonl still has only "
            f"{have} record(s). Write down what you have ALREADY found before searching more.]"
        )
    return (
        f"\n\n[You have gone {step_count} steps and evidence.jsonl is still EMPTY. "
        f"Stop reading and WRITE what you have already found, now. Use `write` with the full "
        f"list of records, or `jq ... > evidence.jsonl`. You can always improve it afterwards -- "
        f"an empty file scores zero no matter how well you searched.\n"
        f"If you are unsure whether a candidate belongs, include it with the wording you can "
        f"support and move on. Do not spend more turns re-reading the same record.]"
    )


def no_output_notice(call: ToolCall) -> str:
    """命令**成功但没有任何输出**时的提示（`jq` 选不到任何东西就是这种表现）。

    这是实测里最隐蔽的一个坑：``jq 'select(...)'`` 匹配不到内容时打印空、退出码 0，
    于是观测是空的，模型以为自己被截断了（或环境坏了），就**再跑一次同样的命令** ——
    重复检测又把它拒掉，浪费两步。这里直接把"选不到东西"翻译出来，并提示 id 可能写错。
    """
    command = str(call.args.get("command") or "")
    if "jq" not in command:
        return "\n\n[(no output). If that is unexpected, re-check the command.]"
    return (
        "\n\n[(no output). For jq this usually means the selector matched NOTHING -- the file may "
        "not contain that id. Remember ids differ by file: inputs/atommem.jsonl records are "
        "session_1_3_1 (with a sequence number), while inputs/msgmem.jsonl MESSAGES are session_1_3 "
        "(without). Print the real ids to check: "
        "jq -r '.metadata.id' inputs/msgmem.jsonl | head -20]"
    )


# ---------------------------------------------------------------------------
# 步 / 结果
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """一次 agent 步：模型输出 + 解析结果 + 工具执行结果。"""

    index: int
    round: int
    reply: str
    call: ToolCall | None
    parsed: bool
    result: ToolResult | None = None
    error: str = ""
    duration_seconds: float = 0.0
    state_before: EvidenceState = field(default_factory=EvidenceState)
    state_after: EvidenceState = field(default_factory=EvidenceState)
    attempt: int = 1
    parse_retry: bool = False
    repeat_rejected: bool = False
    mutated: bool = False

    @property
    def progressed(self) -> bool:
        """这一步有没有可能推进任务：唯一 id 数或字节数变化了。"""
        return self.state_before.signature() != self.state_after.signature()

    @property
    def call_repr(self) -> str:
        return self.call.describe() if self.call else "(no tool call)"

    @property
    def command(self) -> str:
        if not self.call:
            return ""
        value = self.call.args.get("command") or self.call.args.get("path") or ""
        return str(value)

    def to_dict(self, observation_chars: int = 0) -> dict[str, Any]:
        """序列化这一步。

        ``observation_chars`` 截断 ``result_text`` / ``reply``（0 = 不截断）。默认在 harness
        里设成 600：实测单次全量跑 152 个 QA 的轨迹文件 **4.8MB**，其中 ``rounds[].steps[]``
        占 95%，而 ``result_text``（原始工具观测）一项就占 55%、``result_details`` 再占 10%。

        截断而不是删除的理由：观测是**唯一**记录"模型当时看到了什么"的地方
        （``model_calls.jsonl`` 默认不写 messages，只写模型的原始输出），排错时要看。
        但它不需要全文 —— 前 600 字符足以看出是检索结果、报错、还是空输出。
        要全文就把 ``observation_chars`` 设成 0（``--log-full-observations``）。
        """
        result = self.result

        def clip(text: Any) -> Any:
            if not observation_chars or not isinstance(text, str):
                return text
            if len(text) <= observation_chars:
                return text
            return text[:observation_chars] + f"… [截断，共 {len(text)} 字符]"

        return {
            "step": self.index,
            "round": self.round,
            "attempt": self.attempt,
            "reply": clip(self.reply),
            "reply_chars": len(self.reply),
            "call": {"tool": self.call.tool, "args": self.call.args} if self.call else None,
            "parsed": self.parsed,
            "error": self.error,
            "ok": result.ok if result else False,
            "result_text": clip(result.text) if result else "",
            "result_text_chars": len(result.text) if result else 0,
            "result_details": result.details if result else {},
            "evidence_before": vars(self.state_before),
            "evidence_after": vars(self.state_after),
            "progressed": self.progressed,
            "mutated": self.mutated,
            "duration_seconds": self.duration_seconds,
            "parse_retry": self.parse_retry,
            "repeat_rejected": self.repeat_rejected,
        }


def steps_for_record(records: list[Any], observation_chars: int = 600) -> list[dict[str, Any]]:
    """把若干 ``Step`` 序列化成可落盘的 dict（统一在这里决定截断长度）。"""
    return [step.to_dict(observation_chars) for step in records]


@dataclass
class AgentOutcome:
    """agent 在一个 round 里（到它说"我做完了"为止）的完整记录。"""

    steps: list[Step] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)
    done_reason: str = ""
    parse_failures: int = 0
    parse_retries: int = 0
    tool_errors: int = 0
    repeated_calls: int = 0
    no_progress_steps: int = 0
    compactions: int = 0

    @property
    def tool_calls(self) -> int:
        return sum(1 for step in self.steps if step.call is not None and not step.parse_retry)

    def counts_by_tool(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for step in self.steps:
            if step.call is not None:
                counts[step.call.tool] = counts.get(step.call.tool, 0) + 1
        return counts

    def summary(self) -> str:
        by_tool = " ".join(f"{k}={v}" for k, v in sorted(self.counts_by_tool().items()))
        return (
            f"步 {len(self.steps)} 工具 {self.tool_calls}"
            + (f"（{by_tool}）" if by_tool else "")
            + (f" 解析失败 {self.parse_failures}" if self.parse_failures else "")
            + (f" 解析重试 {self.parse_retries}" if self.parse_retries else "")
            + (f" 重复拒绝 {self.repeated_calls}" if self.repeated_calls else "")
            + (f" 工具错误 {self.tool_errors}" if self.tool_errors else "")
            + (f" 压缩 {self.compactions}" if self.compactions else "")
            + f" 结束={self.done_reason}"
        )


class _RepeatTracker:
    """同一个工具+参数重复调用的检测。

    实测：模型会用**完全相同**的 ``jq ... >> evidence.jsonl`` 连做 5~10 次，因为观测是空的
    它以为没成功。直接拒绝（而不是照做）能在第一次重复时就打断这个循环 —— 照做只会让文件
    更脏（QA2 的 119 行重复就是这么来的）。

    **但重复检测必须感知状态。** 一个成熟 code agent 的语义是"重复一次没推进的调用"，
    而不是"这个命令一辈子只能跑一次"：``jq -s 'unique_by(...)' evidence.jsonl > t && mv t``
    在文件变了之后**应该**被允许再跑。所以计数按 **文件状态指纹** 分桶 —— 指纹变了就重来，
    否则才拒绝。
    """

    def __init__(self) -> None:
        self.counts: dict[tuple[str, str], int] = {}

    @staticmethod
    def key(call: ToolCall) -> str:
        try:
            return call.tool + "\x00" + json.dumps(call.args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return call.tool + "\x00" + repr(call.args)

    def hit(self, call: ToolCall, state_key: str = "") -> int:
        """记录一次调用，返回**在同一文件状态**下它被调用了几次（含本次）。"""
        key = (self.key(call), state_key)
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]


# ---------------------------------------------------------------------------
# 上下文压缩
# ---------------------------------------------------------------------------

def estimate_chars(messages: list[dict[str, str]]) -> int:
    return sum(len(message.get("content", "")) for message in messages)


@dataclass
class CompactionResult:
    messages: list[dict[str, str]]
    dropped: int = 0
    kept: int = 0
    digest: str = ""


def compact_messages(
    messages: list[dict[str, str]],
    *,
    max_chars: int,
    keep_recent: int,
) -> CompactionResult:
    """超出预算时把**中段**坍缩成一段摘要，保留 system + 最近 ``keep_recent`` 条。

    为什么要压缩：实测 12 步就从 3.3k tok 涨到 4.5k tok（每步 +~100 tok，因为工具观测
    一直在累积）。长 QA（30 步）会线性涨到 ~7k tok，而其中大部分是**已经用过的旧观测**。

    摘要不是模型生成的（那要额外花钱、还可能编造），而是**从被丢掉的观测里机械提取**：
    调用过的工具、碰过的 id。保留 id 集合是关键 —— 否则模型会忘了自己已经收过哪些记忆，
    转而重复追加（正是我们要修的那个毛病）。
    """
    if max_chars <= 0 or estimate_chars(messages) <= max_chars:
        return CompactionResult(messages=messages, kept=len(messages))

    if len(messages) <= keep_recent + 2:
        return CompactionResult(messages=messages, kept=len(messages))

    head = messages[:1]                       # system
    tail = messages[-keep_recent:]            # 最近 N 条
    middle = messages[1:len(messages) - keep_recent]

    tools_used: list[str] = []
    ids_seen: list[str] = []
    notes: list[str] = []
    for index, message in enumerate(middle):
        content = message.get("content", "")
        if message.get("role") == "assistant":
            call = toolcfg.parse_tool_call(content)
            if call is None:
                continue
            tools_used.append(call.tool)
            command = str(call.args.get("command") or call.args.get("path") or "")
            for name in ("atommem.jsonl", "msgmem.jsonl", "evidence.jsonl"):
                if name in command:
                    notes.append(name)
            # 抓出命令里提到的 id（形如 "session_1_3_1"），它们是最该记住的东西
            for token in command.replace('"', " ").replace("'", " ").split():
                if token.count("_") >= 2 and token[0].isalpha():
                    if token not in ids_seen:
                        ids_seen.append(token)

    counts: dict[str, int] = {}
    for name in tools_used:
        counts[name] = counts.get(name, 0) + 1
    by_tool = " ".join(f"{k}×{v}" for k, v in sorted(counts.items()))
    digest = (
        f"[context compacted: {len(middle)} earlier turns summarised; full text was dropped "
        f"to save space. Re-read evidence.jsonl if you need the current contents.]\n"
        f"- tools used: {by_tool or '(none)'}\n"
        f"- ids your commands referenced: {', '.join(ids_seen[:80]) or '(none)'}"
        + (f"\n- files touched: {', '.join(sorted(set(notes)))}" if notes else "")
    )
    compacted = [*head, {"role": "user", "content": digest}, *tail]
    return CompactionResult(
        messages=compacted, dropped=len(middle), kept=len(compacted), digest=digest
    )


# ---------------------------------------------------------------------------
# 观测渲染
# ---------------------------------------------------------------------------

def render_observation(step: Step, max_chars: int = 8000) -> str:
    """把一次工具执行渲染成给模型看的观测（失败也照渲染，带 ``ERROR:`` 前缀）。

    **成功的空输出渲染成 ``(no output)``** —— 但那是渲染层的决定，不是工具返回的内容。
    分开的理由见 ``tools.create_bash_tool``：占位串如果由工具塞进 ``text``，调用方就再也
    分辨不出"真的没输出"和"输出恰好是这几个字"，`no_output_notice` 也就永远不会触发。
    """
    if step.error or step.result is None:
        return f"ERROR: {step.error or 'tool did not run'}"
    prefix = "" if step.result.ok else "ERROR: "
    text = step.result.text
    if not text.strip():
        text = "(no output)"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n… [observation truncated, {len(step.result.text)} chars total]"
    return prefix + text


def _is_mutation(call: ToolCall) -> bool:
    """这一步是否**可能**改变工作目录内容（用于决定要不要附证据摘要）。"""
    if call.tool in ("write", "edit"):
        return True
    if call.tool == "bash":
        command = str(call.args.get("command") or "")
        return any(
            marker in command
            for marker in (">", ">>", "tee ", "mv ", "cp ", "rm ", "sed -i", "truncate ")
        )
    return False


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

async def run_agent_round(
    *,
    messages: list[dict[str, str]],
    tools: dict[str, Any],
    model_call: ModelCall,
    evidence_path: Path,
    cwd: Path,
    max_steps: int,
    no_progress_patience: int,
    step_offset: int,
    round_index: int,
    log: Logger,
    observation_max_chars: int = 8000,
    write_nudge_after: int = 5,
    evidence_soft_limit: int = 40,
    max_parse_retries: int = 2,
    repeat_limit: int = 1,
    duplicate_warn_at: int = 10,
    context_max_chars: int = 0,
    context_keep_recent: int = 8,
    on_step: Callable[[Step], None] | None = None,
) -> tuple[AgentOutcome, list[dict[str, str]]]:
    """跑一个 round：从 ``messages`` 出发，直到模型**明确回复纯文本**或用完步数。

    返回 ``(outcome, messages)`` —— 更新后的 ``messages`` 由调用方继续往下用（verifier 的
    反馈就是这么接上去的）。
    """
    outcome = AgentOutcome(messages=messages)
    repeats = _RepeatTracker()
    nudged = False
    oversize_warned = False
    consecutive_no_progress = 0
    steps_used = 0
    attempts_used = 0
    warned_duplicates = False

    while steps_used < max_steps:
        # ---- 上下文压缩（在每次模型调用**之前**检查）----
        if context_max_chars > 0:
            compaction = compact_messages(
                messages, max_chars=context_max_chars, keep_recent=context_keep_recent
            )
            if compaction.dropped:
                messages = compaction.messages
                outcome.compactions += 1
                log.info(
                    f"上下文压缩：丢弃中段 {compaction.dropped} 条，保留 system + 最近 "
                    f"{context_keep_recent} 条（{estimate_chars(messages)} 字符）"
                )
                log.debug(block(f"round {round_index} 压缩摘要", compaction.digest, limit=3000))
                outcome.messages = messages

        global_step = step_offset + steps_used + 1
        attempts_used += 1
        reply = await model_call(
            messages, {"step": global_step, "round": round_index, "attempt": attempts_used}
        )
        call = toolcfg.parse_tool_call(reply)

        # ---- 模型调用失败 / 输出无法解析 → **注入纠正消息重试**，而不是当作完成 ----
        if call is None:
            if not reply.strip():
                # 空回复：可能是调用失败，也可能模型真的没话说。重试几次再当完成。
                if outcome.parse_retries < max_parse_retries and attempts_used <= max_steps * 2:
                    outcome.parse_retries += 1
                    messages.append({"role": "user", "content":
                        "Your last reply was empty. Reply with ONE JSON object "
                        '{"tool": "...", "args": {...}} to call a tool, or with plain text '
                        "(no JSON) if evidence.jsonl is already sufficient."})
                    log.warn(f"step {global_step} 空回复，注入纠正消息重试"
                             f"（{outcome.parse_retries}/{max_parse_retries}）")
                    continue
                step = Step(index=global_step, round=round_index, reply=reply, call=None,
                            parsed=False, error="empty reply", attempt=attempts_used,
                            parse_retry=True)
                outcome.steps.append(step)
                if on_step is not None:
                    on_step(step)
                outcome.parse_failures += 1
                outcome.done_reason = "empty_reply"
                messages.append({"role": "assistant", "content": reply})
                return outcome, messages

            # 非空但没有可解析的工具调用：**这就是完成信号**（模型被要求用纯文本表示完成）。
            # 但要和"畸形 JSON"区分开：以 { 开头却解析失败，多半是被截断，值得重试。
            looks_like_broken_json = reply.strip().startswith("{")
            if looks_like_broken_json and outcome.parse_retries < max_parse_retries:
                outcome.parse_retries += 1
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content":
                    "That was not valid JSON (it may have been truncated). Reply with ONE "
                    'complete JSON object: {"tool": "...", "args": {...}}. Keep tool arguments '
                    "short -- if you are writing a large file, write fewer records at a time."})
                log.warn(f"step {global_step} 输出像被截断的 JSON，注入纠正消息重试"
                         f"（{outcome.parse_retries}/{max_parse_retries}）")
                continue

            step = Step(index=global_step, round=round_index, reply=reply, call=None,
                        parsed=False, attempt=attempts_used)
            outcome.steps.append(step)
            if on_step is not None:
                on_step(step)
            outcome.done_reason = "no_tool_call"
            log.debug(block(f"step {global_step} 无工具调用（视为完成）", reply, limit=2000))
            messages.append({"role": "assistant", "content": reply})
            return outcome, messages

        steps_used += 1
        state_before = read_evidence_state(evidence_path)
        step = Step(
            index=global_step, round=round_index, reply=reply, call=call,
            parsed=True, attempt=attempts_used, state_before=state_before,
        )
        step.mutated = _is_mutation(call)

        # ---- 重复检测：同样的工具+参数、**且文件状态没变**时，才拒绝（不执行）----
        hits = repeats.hit(call, state_before.fingerprint)
        if hits > repeat_limit:
            step.repeat_rejected = True
            step.state_after = state_before
            outcome.repeated_calls += 1
            observation = repeat_notice(call, hits)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": observation})
            outcome.steps.append(step)
            if on_step is not None:
                on_step(step)
            log.warn(f"step {global_step} 重复调用第 {hits} 次，已拒绝：{call.describe()}")
            consecutive_no_progress += 1
            outcome.no_progress_steps += 1
            if consecutive_no_progress >= no_progress_patience:
                outcome.done_reason = f"no_progress({consecutive_no_progress})"
                log.warn(f"连续 {consecutive_no_progress} 步无进展，提前收尾这一轮")
                return outcome, messages
            continue

        # ---- 执行工具 ----
        if call.tool not in tools:
            step.error = (
                f"unknown tool {call.tool!r}. Available: {sorted(tools)}. "
                f"Use bash for shell commands (jq, search, timecalc), or read/write/edit for files."
            )
            outcome.tool_errors += 1
            log.warn(f"step {global_step} 未知工具 {call.tool!r}（可用 {sorted(tools)}）")
        else:
            started = _monotonic()
            try:
                step.result = await tools[call.tool].run(call.args)
            except ToolError as exc:
                step.error = str(exc)
                outcome.tool_errors += 1
            except Exception as exc:  # noqa: BLE001 - 工具崩溃不该炸掉整条 QA
                step.error = f"{type(exc).__name__}: {exc}"
                outcome.tool_errors += 1
                log.warn(f"step {global_step} 工具 {call.tool} 抛异常：{step.error}")
            step.duration_seconds = round(_monotonic() - started, 3)

        state_after = read_evidence_state(evidence_path)
        step.state_after = state_after

        observation = render_observation(step, max_chars=observation_max_chars)
        # 机制层面的"该动笔了"：只在还没写过东西、且已经烧掉不少步数时插一次。
        # 只插一次（nudged）—— 反复插会挤占上下文，模型也会开始忽略它。
        if (not nudged and write_nudge_after > 0
                and steps_used >= write_nudge_after and state_after.unique_ids == 0):
            nudged = True
            observation += write_nudge_notice(steps_used, state_after.unique_ids)
            log.warn(
                f"step {global_step} 已 {steps_used} 步、evidence 仍为空，注入动笔提醒"
            )
        # evidence 过大：只提醒，不截断（截断会静默丢证据）。只提醒一次。
        if (not oversize_warned and evidence_soft_limit > 0
                and state_after.unique_ids > evidence_soft_limit):
            oversize_warned = True
            observation += oversize_notice(state_after.unique_ids, evidence_soft_limit)
            log.warn(
                f"step {global_step} evidence 有 {state_after.unique_ids} 条"
                f"（软上限 {evidence_soft_limit}），注入精简提醒"
            )
        # 命令成功但没有任何输出：`jq` 选不到东西就是这种表现，最容易让模型误判后重复重试。
        if (step.result is not None and step.result.ok and not step.error
                and not step.result.text.strip()):
            observation += no_output_notice(call)
        # P0 修复：变更型调用之后**主动告诉模型文件现在长什么样**（重定向让 stdout 为空）
        if step.mutated and not step.error:
            observation += _evidence_digest(evidence_path, state_after)
        if state_after.duplicate_lines >= duplicate_warn_at and not warned_duplicates:
            warned_duplicates = True
            observation += (
                f"\n\n[WARNING: evidence.jsonl has {state_after.duplicate_lines} duplicate lines "
                f"({state_after.lines} lines but only {state_after.unique_ids} unique ids). "
                f"Appending again will not help. Deduplicate first:\n"
                f"  jq -sc 'unique_by(.metadata.id)[]' evidence.jsonl > t && mv t evidence.jsonl]"
            )

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": observation})
        outcome.messages = messages
        outcome.steps.append(step)
        if on_step is not None:
            on_step(step)

        log.debug(block(
            f"step {global_step} {call.describe()}",
            json.dumps(step.to_dict(), ensure_ascii=False, indent=2, default=str),
            limit=4000,
        ))

        if step.error or step.result is None:
            log.warn(f"step {global_step} {call.tool} 失败：{step.error}")
        elif not step.result.ok:
            log.info(f"step {global_step} {call.tool} 非零退出"
                     f"（exit={step.result.details.get('exit_code')}）")
        else:
            log.info(
                f"step {global_step} {call.tool} OK"
                + (f" evidence={state_after.unique_ids}唯一/{state_after.lines}行"
                   if step.progressed else "")
            )

        if step.progressed:
            if consecutive_no_progress:
                log.debug(f"step {global_step} 恢复进展（此前连续 {consecutive_no_progress} 步无进展）")
            consecutive_no_progress = 0
        else:
            consecutive_no_progress += 1
            outcome.no_progress_steps += 1
            if consecutive_no_progress >= no_progress_patience:
                outcome.done_reason = f"no_progress({consecutive_no_progress})"
                log.warn(
                    f"连续 {consecutive_no_progress} 步既没增加唯一 id 也没改动 evidence.jsonl，"
                    f"提前收尾这一轮"
                )
                return outcome, messages

    outcome.done_reason = "steps_exhausted"
    return outcome, messages


def _monotonic() -> float:
    from time import monotonic

    return monotonic()
