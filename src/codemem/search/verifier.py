"""Verifier：对"这份 evidence 能不能回答这个问题"做独立判定。

设计见 ``docs/evomem_plan_v3.md`` §6。三条不可省的约束：

1. **只看 ``(question, evidence)``**，看不到 agent 的对话、检索过程、编辑历史。一个既能编辑
   又能宣布成功的 agent，最省力的动作永远是宣布成功 —— 旧版 NOOP 泛滥就是这一偏差的温和版本。
2. **不给参考答案**。用 reference 当闸门就是 oracle 泄漏：RL 会直接学会猜答案，推理时也拿不到
   答案键。所以闸门检验的是**可回答性**，不是正确性；正确性只在评测/奖励时用 judge 算。
3. **必须输出 ``missing[]``**。这是下一轮 agent 最有效的观测，也是"探索"的实际含义 ——
   它把"差在哪里"变成一个具体、可检索的目标。

本模块只负责：构造 prompt、调模型、**稳健解析**、以及在解析失败时**保守地判为"不足"**
（而不是"足够"）—— 假阳性会让 agent 提前收工，假阴性只多花一轮。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..log import Logger, block

VERIFIER_SYSTEM_PROMPT = """You are a strict evidence auditor. You do NOT know how the evidence was produced; you only see a question and a set of memory records. Judge only what is in front of you.

Your task, in order:

1. Try to answer the QUESTION using ONLY the records provided. Do not use outside knowledge, and do not assume facts that are not written.
2. Decide whether the records are SUFFICIENT: does the answer follow from them, with who/when/where resolved (no dangling pronouns, no missing time when the question asks for time, no unjoined facts)?
3. If NOT sufficient, list exactly what is missing -- concretely enough that someone could go search for it (name the entity, the missing time, the fact that must be joined).

Output ONLY this JSON object, nothing else:

{"sufficient": true|false, "answer": "<your best answer from the records, or empty if impossible>", "missing": ["<what is missing>", "..."]}

Rules:
- If the records are absent or unrelated to the question, sufficient is false and missing describes what the question needs.
- `answer` must be phrased as a direct answer to the question ("about a month before 2023-06-17", "Caroline", "no") -- never as a description of the records.
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
    """把 evidence 记录渲染成 verifier 的输入。只给 verifier 需要的最小信息面。"""
    if not memories:
        return "(no records were provided)"
    lines: list[str] = []
    for memory in memories:
        meta = memory.get("metadata") or {}
        text = memory.get("memory")
        text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        lines.append(
            f"[{meta.get('id', '')}] ({meta.get('time', '')}) "
            f"type={meta.get('type', '')} tag={meta.get('tag', [])}\n  {text}"
        )
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
