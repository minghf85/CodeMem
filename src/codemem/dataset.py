"""LoCoMo 数据集模型：读原始数据、遍历会话、把 evidence 标记映射成消息 id。

**为什么单独一层**：add 要遍历会话造 msgmem，search 要按目录取该 sample 的 QA，
answer 要拿 question + reference，eval 要算 evidence recall。四者都需要"同一份数据、
同一套 id 约定"，所以数据模型必须共享 —— 否则 evidence 映射在四个地方各写一遍，
迟早不一致（``D1:3`` → ``session_1_3`` 这条规则就是靠这里唯一实现的）。

数据文件：``data/correct_locomo10.json``，一个 sample 数组，每个 sample 形如::

    {
      "conversation": {
        "speaker_a": "Caroline", "speaker_b": "Melanie",
        "session_1": [{"speaker": "...", "text": "..."}, ...],
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        ...
      },
      "qa": [{"question": "...", "answer": "...", "evidence": ["D1:3"], "category": 2}, ...]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .io import DATA_DIR, PROJECT_ROOT

DATA_FILE = PROJECT_ROOT / "data" / "correct_locomo10.json"


# ---------------------------------------------------------------------------
# sample / QA
# ---------------------------------------------------------------------------

def load_samples(path: Path | None = None) -> list[dict[str, Any]]:
    """读全部 sample。文件缺失时抛错（不是静默返回空 —— 那会让上层以为"没有数据要跑"）。"""
    target = path or DATA_FILE
    if not target.exists():
        raise FileNotFoundError(f"找不到数据集：{target}")
    return json.loads(target.read_text(encoding="utf-8"))


def sample_dir(sample: dict[str, Any]) -> str:
    """``Caroline`` + ``Melanie`` -> ``Caroline_Melanie``（即 ``data/`` 下的目录名）。"""
    conversation = sample["conversation"]
    return f"{conversation['speaker_a']}_{conversation['speaker_b']}"


def sample_by_dir(samples: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    """按目录名找 sample；找不到返回 None（调用方据此把该目录标 SKIPPED）。"""
    for sample in samples:
        if sample_dir(sample) == label:
            return sample
    return None


def reference_answer(qa: dict[str, Any]) -> Any:
    """取参考答案。category 5（adversarial）只有 ``adversarial_answer``。"""
    if "answer" in qa:
        return qa["answer"]
    if "adversarial_answer" in qa:
        return qa["adversarial_answer"]
    raise KeyError("QA item has neither answer nor adversarial_answer")


def sample_questions(
    sample: dict[str, Any],
    limit: int = 0,
    skip_category5: bool = False,
) -> list[tuple[int, dict[str, Any]]]:
    """取该 sample 的 ``(原始下标, qa)`` 列表。

    **保留下标而不是用问题文本当键**：问题文本不唯一（conv-48 有 11 组重复 question），
    用文本做键会把同一条 QA 的两份数据当成一条，续跑时静默丢一条。下标才是唯一标识。

    ``skip_category5`` 时过滤掉 category 5（adversarial，与评测口径一致）。
    """
    items = list(enumerate(sample.get("qa") or []))
    if skip_category5:
        items = [(index, qa) for index, qa in items if qa.get("category") != 5]
    return items[:limit] if limit > 0 else items


def questions_for_dir(
    label: str,
    limit: int = 0,
    skip_category5: bool = False,
    path: Path | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    """``questions_for_dir("Caroline_Melanie")`` -> 该目录的 QA 列表。找不到返回空。"""
    try:
        samples = load_samples(path)
    except (OSError, ValueError, FileNotFoundError):
        return []
    sample = sample_by_dir(samples, label)
    if sample is None:
        return []
    return sample_questions(sample, limit=limit, skip_category5=skip_category5)


# ---------------------------------------------------------------------------
# 会话 / 消息
# ---------------------------------------------------------------------------

@dataclass
class RawMessage:
    """一条原始消息。``id`` 用 ``session_{会话号}_{消息号}``，与 msgmem 的 id 约定一致。"""

    id: str
    session: int
    message_index: int
    speaker: str
    text: str
    session_time: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session": self.session,
            "message_index": self.message_index,
            "speaker": self.speaker,
            "text": self.text,
            "session_time": self.session_time,
        }


def session_keys(conversation: dict[str, Any]) -> list[str]:
    """按会话号排序的 ``session_N`` 键。**必须排序**：字典序下 session_10 会排在 session_2 前面。"""
    return sorted(
        (
            key for key in conversation
            if key.startswith("session_") and key.rsplit("_", 1)[-1].isdigit()
        ),
        key=lambda key: int(key.split("_")[1]),
    )


def iter_messages(sample: dict[str, Any]) -> Iterator[RawMessage]:
    """按会话、消息顺序遍历该 sample 的全部原始消息（msgmem 的输入顺序）。"""
    conversation = sample["conversation"]
    for key in session_keys(conversation):
        session_index = int(key.split("_")[1])
        session_time = str(conversation.get(f"{key}_date_time", "") or "")
        for message_index, message in enumerate(conversation[key], 1):
            yield RawMessage(
                id=f"session_{session_index}_{message_index}",
                session=session_index,
                message_index=message_index,
                speaker=message.get("speaker", "unknown"),
                text=message.get("text", ""),
                session_time=session_time,
            )


# ---------------------------------------------------------------------------
# evidence 标记
# ---------------------------------------------------------------------------

def evidence_to_ids(evidence: Iterable[str] | None) -> set[str]:
    """把 LoCoMo 的 evidence 标记映射成消息 id。

    ``D1:3`` -> ``session_1_3``。**这是 id 约定的唯一实现**：eval 用它算 recall，
    search 用它判断"命中的记录覆盖了哪些证据"。

    容错两种写法：空格分隔的多引用（``"D9:1 D4:4"``）与 ``D:11:26``（缺 session 号时
    整段当一个引用）。这两种以前被静默丢弃，影响 4 条 QA。
    """
    result: set[str] = set()
    for item in evidence or []:
        if not isinstance(item, str):
            continue
        for token in item.split():
            result.update(_parse_evidence_token(token))
    return result


def _parse_evidence_token(token: str) -> set[str]:
    token = token.strip()
    if not token:
        return set()
    parts = token.split(":")
    try:
        if len(parts) == 2:
            document, turn = parts
            return {f"session_{int(document.removeprefix('D'))}_{int(turn)}"}
        if len(parts) == 3 and not parts[0].removeprefix("D"):
            # "D:11:26" —— session 号被吞了，退化成 session_11_26
            return {f"session_{int(parts[1])}_{int(parts[2])}"}
    except (ValueError, AttributeError):
        return set()
    return set()


def evidence_ids_for_qa(qa: dict[str, Any]) -> set[str]:
    """某条 QA 的 evidence 消息 id 集合。"""
    return evidence_to_ids(qa.get("evidence"))
