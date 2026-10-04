"""add · 三层记忆文件的公共读/写/分组工具（msgmem / summem / atommem）。

三个文件格式相同（``memory_template.jsonl``），只是 ``type`` 不同：

    msgmem.jsonl    type=msg     原始消息（会话时间戳是锚点）
    summem.jsonl    type=summary 每个会话一份总结
    atommem.jsonl   type=atom    逐消息抽出的原子事实

工具函数都放在这里，避免 msgmem / summem / atommem 三个模块各写一份 id 解析与分组。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from ..io import memory_id, memory_type, read_jsonl, write_jsonl

#: 三层记忆的文件名（相对 ``data/{dir}/``）。定义在共享层，msgmem/summem/atommem 都从这里取。
MSGMEM_NAME = "msgmem.jsonl"
SUMMEM_NAME = "summem.jsonl"
ATOMMEM_NAME = "atommem.jsonl"

#: ``session_3_14`` -> 3；``session_3_summary`` -> 3；``session_3`` -> 3。
SESSION_NUMBER_RE = re.compile(r"^session_(\d+)(?:$|_)")


def session_number(mem_id: str) -> int | None:
    """从记忆 id 里取会话号（``session_3_14`` / ``session_3_summary`` -> 3）。"""
    match = SESSION_NUMBER_RE.match(str(mem_id or ""))
    return int(match.group(1)) if match else None


def msg_session_number(mem_id: str) -> int | None:
    """从 **msg** 记忆 id 取会话号（``session_3_14`` -> 3）。``session_3_summary`` 不算。"""
    match = re.match(r"^session_(\d+)_(\d+)$", str(mem_id or ""))
    return int(match.group(1)) if match else None


def group_by_session(records: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """按会话号分组，组内保持原顺序。解析不出会话号的记录被丢掉。"""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        number = session_number(memory_id(record))
        if number is None:
            continue
        grouped.setdefault(number, []).append(record)
    return dict(sorted(grouped.items()))


def load_memories(path: Path, memory_type_value: str = "") -> list[dict[str, Any]]:
    """读一个记忆文件；``memory_type_value`` 非空时只保留该 type 的记录。"""
    records = read_jsonl(path)
    if not memory_type_value:
        return records
    return [r for r in records if memory_type(r) == memory_type_value]


def recent_window(
    records: list[dict[str, Any]], index: int, window: int
) -> list[dict[str, Any]]:
    """取 ``records[index]`` 之前（不含自己）最近 ``window`` 条。"""
    if window <= 0 or index <= 0:
        return []
    return records[max(0, index - window):index]


def write_memories(path: Path, records: list[dict[str, Any]]) -> None:
    """整文件重写（原子写）。"""
    write_jsonl(path, records)


def parse_dir_label(label: str) -> tuple[str, str]:
    """``"Caroline_Melanie"`` -> ``("Caroline", "Melanie")``（用数据集里的真名）。

    数据集里找不到该目录时退回按 ``_`` 切分。用于给 prompt 填 ``{speaker_a}`` / ``{speaker_b}``。
    """
    from .. import dataset

    try:
        for sample in dataset.load_samples():
            if dataset.sample_dir(sample) == label:
                conversation = sample["conversation"]
                return conversation["speaker_a"], conversation["speaker_b"]
    except (OSError, ValueError, FileNotFoundError):
        pass
    parts = label.split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (label, "other")
