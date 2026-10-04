"""add · 第一步：原始对话 → ``msgmem.jsonl``（纯 CPU，不调模型）。

**这是整条链路的输入层。** 把 ``data/correct_locomo10.json`` 的每个 session 摊平成一条条
raw message memory，写到 ``data/{speaker_a}_{speaker_b}/msgmem.jsonl``。一行一条::

    {"memory": "Hey Mel! Good to see you!",
     "metadata": {"id": "session_1_1", "type": "msg", "time": "1:56 pm on 8 May, 2023",
                  "role": "Caroline", "tag": [], "source": [], "changelog": [...]}}

格式沿用 ``memory_template.jsonl``（三条记忆链中 ``type`` 最小的那一层）。设计要点：

- ``id`` = ``session_{会话号}_{消息号}``（1-indexed，与 LoCoMo 的 ``dia_id`` 对齐）——
  add/search/answer/eval 全都依赖这个约定。
- ``time`` 保留**原始会话时间戳**（``"1:56 pm on 8 May, 2023"``）—— 它是下一步把相对时间
  锚定成绝对时间的**锚点**。每条消息都带着它，于是从任何一条都能拿到锚点。
- ``role`` 用**真实人名**（``Caroline`` / ``Melanie``）—— 后面把 ``she`` 消解成真名要靠它。
- ``tag`` / ``source`` 为空：原始消息没有出处、没有标签。
- ``changelog.time`` 是**真实创建时间**，与会话时间无关（两者语义不同，别混）。

用法::

    python -m codemem.add --stage msgmem                # 全部 sample
    python -m codemem.add --stage msgmem Caroline_Melanie
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .. import dataset
from ..io import DATA_DIR, make_memory, now_iso, write_jsonl
from .memory import ATOMMEM_NAME, MSGMEM_NAME, SUMMEM_NAME

__all__ = ["ATOMMEM_NAME", "MSGMEM_NAME", "SUMMEM_NAME", "main", "run"]


def parse_session_datetime(date_time: str) -> str:
    """``'1:56 pm on 8 May, 2023'`` -> ISO-8601（无时区）。解析失败原样返回。

    **不加时区**：会话时间在数据里是本地时间，硬套 UTC 会把它平移几个小时甚至一天，
    而下游要拿它做日期算术 —— 差一天就是错的答案。
    """
    try:
        return datetime.strptime(date_time, "%I:%M %p on %d %B, %Y").isoformat()
    except (ValueError, TypeError):
        return date_time


def build_msg_item(message: dataset.RawMessage) -> dict:
    """把一条原始消息包成 msg 记忆记录（``type=msg``）。"""
    return make_memory(
        memory=message.text,
        memory_type="msg",
        mem_id=message.id,
        time=message.session_time,
        role=message.speaker,
        tag=[],
        source=[],
        changelog=[{"time": now_iso(), "content": "created"}],
    )


def convert_sample(sample: dict) -> tuple[str, list[dict]]:
    """一个 sample -> ``(目录名, msg 记忆列表)``（按会话、消息顺序）。"""
    return dataset.sample_dir(sample), [
        build_msg_item(message) for message in dataset.iter_messages(sample)
    ]


def run(targets: list[str] | None = None) -> list[tuple[str, Path, int]]:
    """跑全部（或指定）sample，返回 ``[(目录名, 输出路径, 条数)]``。"""
    results: list[tuple[str, Path, int]] = []
    for sample in dataset.load_samples():
        dir_name, items = convert_sample(sample)
        if targets and dir_name not in targets:
            continue
        out_path = DATA_DIR / dir_name / MSGMEM_NAME
        write_jsonl(out_path, items)
        results.append((dir_name, out_path, len(items)))
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.add.msgmem",
        description="把原始对话转成 msgmem.jsonl（type=msg 的原始消息记忆）",
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
        print(f"msgmem    {dir_name} -> {out_path} ({count} 条)")


if __name__ == "__main__":
    main()
