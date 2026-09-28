"""add · 第一步：把原始对话转成 raw message memory（msgmem）。

**这是 add 的输入层**：把 ``data/correct_locomo10.json`` 的每个 session 摊平成一条条
raw memory，写到 ``data/{speaker_a}_{speaker_b}/msgmem.jsonl``。后续 atommem 在这些 raw
之上抽原子记忆。

设计要点（都是为了让下游能正确地做时间推理）：

- ``id`` = ``session_{会话号}_{消息号}`` —— add/search/answer/eval 全都依赖这个约定。
- **会话首条消息**的 ``time`` 是该会话的真实时间（ISO-8601）；其余消息是
  ``few minutes in {会话时间}``，表示"发生在该会话开始后几分钟"。这样 search 阶段的
  agent 能从**任何一条消息**推出所在会话的绝对时间，而不必非去找首条 —— 相对时间推理
  （"about a month ago"）的锚点就是这么来的。
- ``changelog.time`` 始终是**真实创建时间**，与会话时间无关（两者语义不同，别混）。
- ``source`` 为空：raw 记忆没有出处。

用法::

    python -m codemem.add.msgmem                # 全部 sample
    python -m codemem.add.msgmem Caroline_Melanie
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from .. import dataset
from ..io import DATA_DIR, now_iso, write_jsonl

SESSION_INDEX_RE = re.compile(r"^session_(\d+)$")


def parse_session_datetime(date_time: str) -> str:
    """``'1:56 pm on 8 May, 2023'`` -> ISO-8601（无时区）。解析失败原样返回。

    **不加时区**：会话时间在数据里是本地时间，硬套 UTC 会把它平移几个小时甚至一天，
    而下游要拿它做日期算术（``timecalc shift ... -1day``）—— 差一天就是错的答案。
    """
    try:
        return datetime.strptime(date_time, "%I:%M %p on %d %B, %Y").isoformat()
    except (ValueError, TypeError):
        return date_time


def build_memory_item(
    text: str, session_index: int, message_index: int, speaker: str, session_time: str
) -> dict:
    """把一条消息包成 raw memory item。"""
    delivered = session_time if message_index == 1 else f"few minutes in {session_time}"
    return {
        "memory": text,
        "metadata": {
            "id": f"session_{session_index}_{message_index}",
            "type": "raw",
            "time": delivered,
            "tag": [f"speaker:{speaker}"],
            "source": [],
            "changelog": [{"time": now_iso(), "content": "created"}],
        },
    }


def convert_sample(sample: dict) -> tuple[str, list[dict]]:
    """一个 sample -> ``(目录名, raw memory 列表)``。"""
    conversation = sample["conversation"]
    session_times = {
        key.split("_")[1]: parse_session_datetime(str(conversation.get(f"{key}_date_time", "") or ""))
        for key in dataset.session_keys(conversation)
    }

    items: list[dict] = []
    for message in dataset.iter_messages(sample):
        items.append(
            build_memory_item(
                text=message.text,
                session_index=message.session,
                message_index=message.message_index,
                speaker=message.speaker,
                session_time=session_times.get(str(message.session), ""),
            )
        )
    return dataset.sample_dir(sample), items


def run(targets: list[str] | None = None) -> list[tuple[str, Path, int]]:
    """跑全部（或指定）sample，返回 ``[(目录名, 输出路径, 条数)]``。"""
    samples = dataset.load_samples()
    results: list[tuple[str, Path, int]] = []
    for sample in samples:
        dir_name, items = convert_sample(sample)
        if targets and dir_name not in targets:
            continue
        out_path = DATA_DIR / dir_name / "msgmem.jsonl"
        write_jsonl(out_path, items)
        results.append((dir_name, out_path, len(items)))
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.add.msgmem",
        description="把原始对话转成 raw message memory（msgmem）",
    )
    parser.add_argument("dirs", nargs="*", help="只处理这些 speaker 目录；缺省处理全部")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = run(args.dirs)
    if not results:
        print("没有匹配的 sample", file=sys.stderr)
        return
    for dir_name, out_path, count in results:
        print(f"{dir_name} -> {out_path} ({count} items)")


if __name__ == "__main__":
    main()
