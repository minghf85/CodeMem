"""add · 第一步：把原始对话摊成 session 记录（``sessions.jsonl``）。

**这是整条链路的输入层**。格式是 ``session_template.jsonl``：一条消息一行::

    {"msg_id": "session_1_1", "role": "Caroline", "time": "1:56 pm on 8 May, 2023",
     "content": "Hey Mel! Good to see you!"}

三处刻意的取舍：

**① ``msg_id`` 用 1-indexed，与 ``dia_id`` 对齐。**
LoCoMo 的标注是 ``D1:1``（第 1 个 session 的第 1 条），而模板示例写的是 ``session_1_0``。
选 1-indexed 是因为 ``D1:1`` ↔ ``session_1_1`` 是**恒等映射**，evidence 的 recall 计算
不用做偏移；用 0-indexed 就得在每处映射里 ±1，那是 bug 的温床。

**② ``role`` 用真实人名（``Caroline`` / ``Melanie``），不是 ``speaker1`` / ``speaker2``。**
agent 要做指代消解（"she" 是谁、"her kids" 是谁的孩子），名字是它唯一的线索。
换成 speaker1/2 等于把这一步的信息丢掉，再让它从对话里猜。

**③ ``time`` 保留原始形式（``"1:56 pm on 8 May, 2023"``），不转 ISO。**
原始形式是数据里**唯一没有歧义**的时间来源；转 ISO 时任何解析错误都会被固化进下游，
而且转换本身会丢掉"这是本地时间"这个事实。需要绝对日期的地方现场用
agent 用 ``date -d`` 现场折算 —— 折算过程可见、可复核。

所有消息（含首条）都带上所属会话的时间：这样 agent 从**任何一条**消息都能拿到锚点，
不必非去找会话首条。相对时间推理（"about a month ago"）依赖这个。

用法::

    python -m codemem.add.session                # 全部 sample
    python -m codemem.add.session Caroline_Melanie
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import dataset
from ..io import DATA_DIR, make_session_record, write_jsonl

OUTPUT_NAME = "sessions.jsonl"


def convert_sample(sample: dict) -> tuple[str, list[dict]]:
    """一个 sample -> ``(目录名, session 记录列表)``。"""
    conversation = sample["conversation"]
    session_times = {
        key: str(conversation.get(f"{key}_date_time", "") or "")
        for key in dataset.session_keys(conversation)
    }

    records: list[dict] = []
    for message in dataset.iter_messages(sample):
        records.append(
            make_session_record(
                msg_id_value=message.id,
                role=message.speaker,
                content=message.text,
                time=session_times.get(f"session_{message.session}", ""),
            )
        )
    return dataset.sample_dir(sample), records


def run(targets: list[str] | None = None) -> list[tuple[str, Path, int]]:
    """跑全部（或指定）sample，返回 ``[(目录名, 输出路径, 条数)]``。"""
    results: list[tuple[str, Path, int]] = []
    for sample in dataset.load_samples():
        dir_name, records = convert_sample(sample)
        if targets and dir_name not in targets:
            continue
        out_path = DATA_DIR / dir_name / OUTPUT_NAME
        write_jsonl(out_path, records)
        results.append((dir_name, out_path, len(records)))
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.add.session",
        description="把原始对话摊成 session 记录（sessions.jsonl）",
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
        print(f"session   {dir_name} -> {out_path} ({count} 条)")


if __name__ == "__main__":
    main()
