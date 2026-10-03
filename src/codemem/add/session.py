"""add · 第一步：把原始对话摊成 session 记录（``sessions/session_{n}.jsonl``）。

**这是整条链路的输入层**。**每个会话一个文件** —— `data/{dir}/sessions/session_1.jsonl`、
`session_2.jsonl`…… 于是 search 既可以用 `grep -rin` 递归检索全部会话，也可以
`read inputs/sessions/session_5.jsonl` 精读某一次。一行一条消息::

    {"msg_id": "session_1_1", "role": "Caroline", "time": "1:56 pm on 8 May, 2023",
     "content": "Hey Mel! Good to see you!"}

三处刻意的取舍：

**① ``msg_id`` 用 1-indexed，与 ``dia_id`` 对齐。**
LoCoMo 的标注是 ``D1:1``（第 1 个 session 的第 1 条）。选 1-indexed 是因为
``D1:1`` ↔ ``session_1_1`` 是**恒等映射**，evidence 的 recall 计算不用做偏移。

**② ``role`` 用真实人名（``Caroline`` / ``Melanie``），不是 ``speaker1`` / ``speaker2``。**
指代消解（"she" 是谁）依赖它，换成 speaker1/2 等于把信息丢掉再让下游猜。

**③ ``time`` 保留原始形式（``"1:56 pm on 8 May, 2023"``）。**
这是**会话发生的时间**，是下一步（``resolve``）把相对时间折算成绝对时间的锚点。
每条消息都带着它，于是从任何一条消息都能拿到锚点。

**注意**：本阶段产出的是**原始措辞**（未消解）。第二步 ``resolve`` 会把指代与时间消解掉、
给出上下文无关的表述（原话保留在 ``source_content`` / ``source_time``）。

用法::

    python -m codemem.add.session                # 全部 sample
    python -m codemem.add.session Caroline_Melanie
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .. import dataset
from ..io import DATA_DIR, make_session_record, write_jsonl

#: 会话目录名（相对 ``data/{dir}/``）。每个会话一个 ``session_{n}.jsonl``。
SESSIONS_DIR = "sessions"
#: ``session_5.jsonl`` / ``session_5_3`` -> 5
SESSION_NUMBER_RE = re.compile(r"^session_(\d+)")


def session_filename(session_index: int) -> str:
    return f"session_{session_index}.jsonl"


def session_number_of(path: Path) -> int | None:
    """从文件名里取会话号（``session_3.jsonl`` -> 3）。解析不出返回 None。"""
    match = SESSION_NUMBER_RE.match(path.name)
    return int(match.group(1)) if match else None


def convert_sample(sample: dict) -> tuple[str, dict[int, list[dict]]]:
    """一个 sample -> ``(目录名, {会话号: [记录...]})``。"""
    conversation = sample["conversation"]
    session_times = {
        key: str(conversation.get(f"{key}_date_time", "") or "")
        for key in dataset.session_keys(conversation)
    }
    grouped: dict[int, list[dict]] = {}
    for message in dataset.iter_messages(sample):
        grouped.setdefault(message.session, []).append(
            make_session_record(
                msg_id_value=message.id,
                role=message.speaker,
                content=message.text,
                time=session_times.get(f"session_{message.session}", ""),
            )
        )
    return dataset.sample_dir(sample), grouped


def run(targets: list[str] | None = None) -> list[tuple[str, Path, int]]:
    """跑全部（或指定）sample，返回 ``[(目录名, 会话目录, 消息总数)]``。"""
    results: list[tuple[str, Path, int]] = []
    for sample in dataset.load_samples():
        dir_name, grouped = convert_sample(sample)
        if targets and dir_name not in targets:
            continue
        out_dir = DATA_DIR / dir_name / SESSIONS_DIR
        count = 0
        for session_index in sorted(grouped):
            write_jsonl(out_dir / session_filename(session_index), grouped[session_index])
            count += len(grouped[session_index])
        results.append((dir_name, out_dir, count))
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.add.session",
        description="把原始对话摊成 sessions/session_{n}.jsonl（每个会话一个文件）",
    )
    parser.add_argument("dirs", nargs="*", help="只处理这些 speaker 目录；缺省处理全部")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = run(args.dirs)
    if not results:
        print("没有匹配的 sample", file=sys.stderr)
        return
    for dir_name, out_dir, count in results:
        print(f"session   {dir_name} -> {out_dir}/ ({count} 条)")


if __name__ == "__main__":
    main()
