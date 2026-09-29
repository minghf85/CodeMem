"""Verifier：对"这份 evidence 能不能回答这个问题"做独立判定。

设计见 ``docs/search.md``。三条不可省的约束：

1. **只看 ``(question, evidence)``**，看不到 agent 的对话、检索过程、编辑历史。一个既能编辑
   又能宣布成功的 agent，最省力的动作永远是宣布成功 —— 旧版 NOOP 泛滥就是这一偏差的温和版本。
2. **不给参考答案**。用 reference 当闸门就是 oracle 泄漏：RL 会直接学会猜答案，推理时也拿不到
   答案键。所以闸门检验的是**可回答性**，不是正确性；正确性只在评测/奖励时用 judge 算。
3. **必须输出 ``missing[]``**。这是下一轮 agent 最有效的观测，也是"探索"的实际含义 ——
   它把"差在哪里"变成一个具体、可检索的目标（下一轮的关键词就从这里来）。

本模块只负责：构造 prompt、调模型、**稳健解析**、以及在解析失败时**保守地判为"不足"**
（而不是"足够"）—— 假阳性会让 agent 提前收工，假阴性只多花一轮。

**prompt 里没有 few-shot 示例**：判定规则本身很短（能否从记录里读出答案），而示例会让
每条 QA 多背几百 token。唯一的例外是下面那条**等价性**说明 —— 它是判错最多的一种情形。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..io import msg_content, msg_id
from ..log import Logger, block

VERIFIER_SYSTEM_PROMPT = """You are a strict evidence auditor. You do not know how the evidence was produced; you see only a question and a set of records. Judge only what is in front of you.

Do this:

1. Answer the QUESTION using ONLY the records. No outside knowledge; assume nothing that is not written.
2. Say whether they are SUFFICIENT: does the answer follow from them, with who/when/where resolved (no dangling pronouns, no missing date when the question asks when, no facts left unjoined)?
3. If not, list what is missing -- concretely enough that someone could search for it: name the entity, the missing date, the fact that must be joined. These become the next search keywords.

Reply with ONLY this JSON object:

{"sufficient": true|false, "answer": "<your best answer from the records, or empty>", "missing": ["<what is missing>", "..."]}

Rules:
- No records, or records unrelated to the question: sufficient is false.
- `answer` must be a direct answer to the question ("about a month before 2023-06-17", "Caroline", "no") -- never a description of the records.
- Treat a date written differently as the SAME date if it denotes the same day: "the weekend before 4 September 2023" and "2023-09-02" are not a mismatch. Judge the fact, not the phrasing.
- Be strict but not pedantic: if the records state the answer and resolve its referents, say sufficient.
- `missing` must be empty when sufficient is true."""


@dataclass
class Verdict:
    """一次判定结果。

    ``parsed=False`` 表示模型输出没能解析成 JSON —— 此时**保守地当作"不足"**，
    并把原始输出留在 ``raw`` 里供排错。``sufficient`` 是唯一的闸门信号。
    """

    sufficient: bool
    answer: str = ""
    missing: list[str] = field(default_factory=list)
    parsed: bool = True
    raw: str = ""
    error: str = ""

    def describe(self) -> str:
        state = "足够" if self.sufficient else "不足"
        if not self.parsed:
            state += "(未解析)"
        extra = f" 缺 {len(self.missing)} 项" if self.missing else ""
        return f"verifier={state}{extra} answer={self.answer!r}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "answer": self.answer,
            "missing": self.missing,
            "parsed": self.parsed,
            "error": self.error,
            "raw": self.raw,
        }

    def to_memory(self) -> dict[str, Any]:
        """落进 trajectory 的紧凑版（不重复 raw）。"""
        return {
            "sufficient": self.sufficient,
            "answer": self.answer,
            "missing": self.missing,
            "parsed": self.parsed,
        }


_JSON_OBJECT_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def _extract_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    candidates = [stripped]
    candidates.extend(match.group(0) for match in _JSON_OBJECT_RE.finditer(text))
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate.startswith("{"):
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_verdict(text: str) -> Verdict:
    """稳健解析。任何失败都退化为 ``sufficient=False``（保守）并保留原文。"""
    if not text or not text.strip():
        return Verdict(sufficient=False, parsed=False, raw=text,
                       error="empty verifier reply")
    parsed = _extract_object(text)
    if parsed is None:
        return Verdict(sufficient=False, parsed=False, raw=text,
                       error="verifier reply is not a JSON object")
    sufficient = parsed.get("sufficient")
    if not isinstance(sufficient, bool):
        return Verdict(sufficient=False, parsed=False, raw=text,
                       error=f"verifier 'sufficient' must be a boolean, got {sufficient!r}")
    answer = parsed.get("answer")
    if not isinstance(answer, str):
        answer = "" if answer is None else json.dumps(answer, ensure_ascii=False)
    missing_raw = parsed.get("missing")
    missing: list[str] = []
    if isinstance(missing_raw, list):
        for item in missing_raw:
            if isinstance(item, str) and item.strip():
                missing.append(item.strip())
            elif item is not None:
                missing.append(json.dumps(item, ensure_ascii=False))
    elif isinstance(missing_raw, str) and missing_raw.strip():
        missing = [missing_raw.strip()]
    return Verdict(sufficient=sufficient, answer=answer.strip(), missing=missing, raw=text)


def render_memories(memories: list[dict[str, Any]], max_chars: int = 600) -> str:
    """把 evidence 记录渲染成 verifier 的输入。

    **必须走 ``io`` 的访问器**。这里曾经硬读 v2 的 ``memory``/``metadata.id``/``type``/``tag``
    字段，而新的 evidence 是 ``content`` + ``metadata.{source,score}`` —— 于是每条记录都
    渲染成 ``[] () type= tag=[] null``，**verifier 实际收到的证据是空的**。

    实测后果：全量 1540 条 QA 的 sufficient 全是 false。verifier 的判定没错 ——
    它确实什么也没看到；是喂给它的输入被渲染坏了。这是一个"看起来像模型能力问题、
    实际是数据管道问题"的典型：0% 这个数字本身就说明不该往能力上归因。
    """
    if not memories:
        return "(no records were provided)"
    lines: list[str] = []
    for memory in memories:
        meta = memory.get("metadata") or {}
        text = msg_content(memory) or json.dumps(memory, ensure_ascii=False)
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        source = meta.get("source")
        source_text = ", ".join(str(s) for s in source) if isinstance(source, list) else ""
        provenance = f" source=[{source_text}]" if source_text else ""
        lines.append(f"[{msg_id(memory)}]{provenance}\n  {text}")
    return "\n".join(lines)


def build_verifier_messages(question: str, memories: list[dict[str, Any]]) -> list[dict[str, str]]:
    user = (
        f"# QUESTION\n{question.strip()}\n\n"
        f"# MEMORY RECORDS ({len(memories)})\n{render_memories(memories)}\n\n"
        f"# YOUR VERDICT\n"
        f'Output only: {{"sufficient": true|false, "answer": "...", "missing": [...]}}'
    )
    return [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


async def verify(
    *,
    question: str,
    memories: list[dict[str, Any]],
    model_call: Callable[[list[dict[str, str]], dict[str, Any]], Awaitable[str]],
    log: Logger,
    round_index: int = 0,
    evidence_present: bool = True,
) -> Verdict:
    """跑一次判定。模型调用失败也退化为"不足"（保守），不抛异常。

    ``evidence_present=False`` 表示 evidence.jsonl 为空或不存在 —— 这时**不需要**问模型：
    没有证据就一定不足，而且最有用的反馈是"你做了 12 步却什么都没写"这个事实本身，
    而不是一句泛泛的"缺东西"。实测这是最常见的一种失败（agent 一直在读、从不动笔）。
    """
    if not memories:
        missing = [
            "evidence.jsonl is empty -- you produced NO records."
            if not evidence_present
            else "evidence.jsonl has no valid records (every line was rejected)."
        ]
        missing.append(
            "Search first, read the records you need, then WRITE the records into "
            "evidence.jsonl. Reading records without writing them down makes no progress."
        )
        missing.append(
            "Write with `write` (whole file at once) or `jq ... >> evidence.jsonl` in bash, "
            "then re-read the file to confirm it is non-empty."
        )
        log.warn(
            f"verifier 跳过：evidence 为空（{'文件不存在/空' if not evidence_present else '全部被拒'}），"
            f"直接判不足并给出可执行反馈"
        )
        raw = (
            '{"sufficient": false, "answer": "", "missing": '
            + json.dumps(missing, ensure_ascii=False) + "}"
        )
        return Verdict(sufficient=False, answer="", missing=missing, parsed=True, raw=raw)

    messages = build_verifier_messages(question, memories)
    log.debug(block(f"verifier 输入（round {round_index}，{len(memories)} 条）", messages[1]["content"],
                    limit=8000))
    try:
        raw = await model_call(messages, {"step": 0, "round": round_index, "role": "verifier"})
    except Exception as exc:  # noqa: BLE001 - 判定失败不该炸掉整条 QA，保守判"不足"
        log.warn(f"verifier 调用失败：{type(exc).__name__}: {exc}")
        return Verdict(sufficient=False, parsed=False, error=f"{type(exc).__name__}: {exc}")
    verdict = parse_verdict(raw)
    log.debug(block(f"verifier 原始输出（round {round_index}）", raw, limit=2000))
    return verdict
