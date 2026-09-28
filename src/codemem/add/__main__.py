"""add 步骤：把一段对话记忆添加进来并初始化。

**这一步做两件事**（顺序固定，因为第二步依赖第一步的产物）：

    1. ``msgmem``  —— 原始对话 → raw message memory
    2. ``atommem`` —— raw message memory → 原子记忆

用法::

    python -m codemem.add                           # 两步都跑，全部目录
    python -m codemem.add Caroline_Melanie          # 只处理指定目录
    python -m codemem.add --stage msgmem            # 只跑第一步
    python -m codemem.add --stage atommem --resume  # 只跑第二步，断点续跑
    python -m codemem.add --limit 50                # 第二步每目录只处理前 50 条（调试）

产物（都在 ``data/{speaker_a}_{speaker_b}/`` 下）::

    msgmem.jsonl    raw message memory（只读输入，后续步骤的基础）
    atommem.jsonl   原子记忆（search 步骤的输入）
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from . import atommem, msgmem

STAGES = ("msgmem", "atommem", "dpo")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.add",
        description="add：把对话记忆添加进来并初始化（msgmem → atommem）",
    )
    parser.add_argument("dirs", nargs="*", help="speaker 目录名；缺省处理全部")
    parser.add_argument("--stage", choices=STAGES, default="",
                        help="只跑某一个阶段；缺省按 msgmem → atommem 顺序全跑")
    parser.add_argument("--limit", type=int, default=None,
                        help="atommem 每个目录只处理前 N 条 raw（调试用）")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖配置里的并发数")
    parser.add_argument("--context-window", type=int, default=None, help="覆盖上下文窗口大小")
    parser.add_argument("--resume", action="store_true",
                        help="atommem 跳过已完成的 raw，只重试失败或未完成项")
    parser.add_argument("--include-identical", action="store_true",
                        help="dpo 阶段保留 chosen == rejected 的样本")
    return parser.parse_args(argv)


def run_stage(stage: str, args: argparse.Namespace) -> None:
    if stage == "msgmem":
        results = msgmem.run(args.dirs)
        if not results:
            print("msgmem: 没有匹配的 sample", file=sys.stderr)
            return
        for dir_name, out_path, count in results:
            print(f"msgmem   {dir_name} -> {out_path} ({count} items)")
        return

    if stage == "atommem":
        config = atommem.load_config()
        if args.concurrency is not None:
            config["concurrency"] = args.concurrency
        if args.context_window is not None:
            config["context_window"] = args.context_window
        asyncio.run(atommem.run(config, args.dirs, args.limit, args.resume))
        return

    if stage == "dpo":
        from . import dpo

        config = dpo.load_dpo_config()
        asyncio.run(
            dpo.run(config, args.dirs, args.limit, args.resume, args.include_identical)
        )
        return

    raise ValueError(f"未知阶段 {stage!r}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stages = [args.stage] if args.stage else ["msgmem", "atommem"]
    for stage in stages:
        run_stage(stage, args)


if __name__ == "__main__":
    main()
