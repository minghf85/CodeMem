"""add 步骤：把一段对话记忆添加进来并初始化。

**三步，顺序固定**（后一步依赖前一步的产物）：

    1. ``session``  原始对话 → ``sessions.jsonl``（session_template 格式，纯 CPU）
    2. ``summary``  每个 session → session summary；全部 summary → speakers summary
 产物（都在 ``data/{speaker_a}_{speaker_b}/`` 下）::

    sessions.jsonl           原始消息，一条一行（只读输入，后续步骤的基础）
    session_summaries.jsonl  每个 session 一份 summary（role="summary"）
    speakers_summary.jsonl   整段关系的顶层 summary（msg_id="speakers_summary"）

**为什么是两层 summary 而不是原子记忆**：原子把会话结构打碎了 —— 一条条孤立事实无法
回答"他们什么时候认识的""这段关系怎么发展的"。session summary 保留"这次谈了什么、
什么变了"；speakers summary 再串成整段关系的轨迹。层级还给了检索一个由粗到细的入口。
详见 ``summary.py`` 的模块注释。

用法::

    python -m codemem.add                       # session + summary，全部目录
    python -m codemem.add Caroline_Melanie      # 只处理指定目录
    python -m codemem.add --stage session       # 只跑第一步（纯 CPU，不调模型）
    python -m codemem.add --stage summary --limit-sessions 3   # 只跑前 3 个 session（调试）
    python -m codemem.add --stage speakers      # 只重跑顶层 summary
"""

from __future__ import annotations

import argparse
import sys

from . import session, summary

STAGES = ("session", "summary", "speakers")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.add",
        description="add：把对话记忆添加进来并初始化（session → 两层 summary）",
    )
    parser.add_argument("dirs", nargs="*", help="speaker 目录名；缺省处理全部")
    parser.add_argument("--stage", choices=STAGES, default="",
                        help="只跑某一阶段；缺省按 session → summary 顺序全跑")
    parser.add_argument("--config", default=str(summary.CONFIG_FILE),
                        help="summary 用的配置（默认 configs/add.yaml）")
    parser.add_argument("--limit-sessions", type=int, default=0,
                        help="summary 阶段每个目录只跑前 N 个 session（调试用）")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖 summary 并发数")
    return parser.parse_args(argv)


def run_stage(stage: str, args: argparse.Namespace) -> None:
    if stage == "session":
        results = session.run(args.dirs)
        if not results:
            print("session: 没有匹配的 sample", file=sys.stderr)
            return
        for dir_name, out_path, count in results:
            print(f"session   {dir_name} -> {out_path} ({count} 条)")
        return

    if stage in ("summary", "speakers"):
        # 复用 summary 模块的 CLI 流程，只换 stage（避免两处各写一套编排）
        from pathlib import Path

        config = summary.load_config(Path(args.config))
        if args.concurrency is not None:
            config["concurrency"] = args.concurrency
        targets = summary.resolve_targets(args.dirs)
        if not targets:
            print("summary: 没有找到含 sessions.jsonl 的目录"
                  "（先跑 `python -m codemem.add --stage session`）", file=sys.stderr)
            return
        import asyncio

        from .. import llm

        # 顶层 --stage summary = 两层都跑；--stage speakers = 只重跑顶层
        # （summary 模块内部把两层分别叫 "session" / "speakers"）
        internal = ("session", "speakers") if stage == "summary" else ("speakers",)
        client = llm.make_client(config)

        async def run_all() -> list[dict]:
            try:
                return [
                    await summary.run_dir(
                        directory, config, client,
                        limit_sessions=args.limit_sessions, stages=internal,
                    )
                    for directory in targets
                ]
            finally:
                await client.close()

        for item in asyncio.run(run_all()):
            if item.get("status") == "OK":
                print(f"summary   {item['dir']}: {item['sessions']} 个 session -> "
                      f"{item['session_summaries']} 条 summary"
                      + (" + speakers summary" if item["speakers_summary"] else ""))
            else:
                print(f"summary   {item['dir']}: {item.get('status')} — {item.get('reason', '')}")
        return

    raise ValueError(f"未知阶段 {stage!r}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stages = [args.stage] if args.stage else ["session", "summary"]
    for stage in stages:
        run_stage(stage, args)


if __name__ == "__main__":
    main()
