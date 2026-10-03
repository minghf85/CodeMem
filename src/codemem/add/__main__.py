"""add 步骤：把一段对话记忆添加进来并初始化。

**两步，顺序固定**（后一步依赖前一步的产物）：

    1. ``session``  原始对话 → ``sessions/session_{n}.jsonl``（原始措辞，纯 CPU）
    2. ``resolve``  逐消息消解 时间 + 指代 → 就地改写 ``sessions/*.jsonl``（上下文无关）

产物（都在 ``data/{speaker_a}_{speaker_b}/`` 下）::

    sessions/session_1.jsonl    一次会话一个文件：原始消息消解后的、上下文无关的形式
    sessions/session_2.jsonl    （每条带 content/time/time_kind + source_content/source_time）

**为什么是"消解"而不是"摘要"**：``session`` 阶段是**原话**（精确措辞、时间锚点），``resolve``
阶段把它改写成**自足**的形式。两种"概览"都被实测淘汰了：顶层 speakers summary 是**浓缩成品**，
agent 看一眼就凑答案、永远不翻原文（152 条 QA 里 150 条一次 grep 都没做）；session summary 是
**逐会话摘要**，agent 只 grep 原文、119/152 条压根不读它。而 `index.jsonl` / 图都是额外造的
导航层。这一版直接把消息本身改好 —— 关键词检索直达、跨会话靠同一真名、通读时文件自足。

时间消解是重点：消解后的 ``time`` 必须"看到它就对应现实中的一个时间、不需要辅助信息" ——
绝对时间或带绝对锚点的相对时间；**精度只减不增**（源说"去年"就只到年）；区分点/段/大概。
原话保留在 ``source_content`` / ``source_time``。

**每个会话一个文件**（``sessions/session_{n}.jsonl``）：search 用 `grep -rin` 递归检索全部会话，
或 `read inputs/sessions/session_5.jsonl` 精读某一次。

用法::

    python -m codemem.add                       # session + resolve，全部目录
    python -m codemem.add Caroline_Melanie      # 只处理指定目录
    python -m codemem.add --stage session       # 只跑第一步（纯 CPU，不调模型）
    python -m codemem.add --stage resolve --limit-sessions 3   # 只跑前 3 个 session（调试）
"""

from __future__ import annotations

import argparse
import sys

from . import resolve, session

STAGES = ("session", "resolve")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.add",
        description="add：把对话记忆添加进来并初始化（session → resolve）",
    )
    parser.add_argument("dirs", nargs="*", help="speaker 目录名；缺省处理全部")
    parser.add_argument("--stage", choices=STAGES, default="",
                        help="只跑某一阶段；缺省按 session → resolve 顺序全跑")
    parser.add_argument("--config", default=str(resolve.CONFIG_FILE),
                        help="resolve 用的配置（默认 configs/add.yaml）")
    parser.add_argument("--limit-sessions", type=int, default=0,
                        help="resolve 阶段每个目录只跑前 N 个 session（调试用）")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖 resolve 会话级并发数")
    return parser.parse_args(argv)


def run_stage(stage: str, args: argparse.Namespace) -> None:
    if stage == "session":
        results = session.run(args.dirs)
        if not results:
            print("session: 没有匹配的 sample", file=sys.stderr)
            return
        for dir_name, out_dir, count in results:
            print(f"session   {dir_name} -> {out_dir}/ ({count} 条)")
        return

    if stage == "resolve":
        # 复用 resolve 模块的 CLI 流程（避免两处各写一套编排）
        import asyncio
        from pathlib import Path

        from .. import llm

        config = resolve.load_config(Path(args.config))
        if args.concurrency is not None:
            config["concurrency"] = args.concurrency
        targets = resolve.resolve_targets(args.dirs)
        if not targets:
            print("resolve: 没有找到含 sessions/ 的目录"
                  "（先跑 `python -m codemem.add --stage session`）", file=sys.stderr)
            return
        client = llm.make_client(config)

        async def run_all() -> list[dict]:
            try:
                return [
                    await resolve.run_dir(
                        directory, config, client, limit_sessions=args.limit_sessions,
                    )
                    for directory in targets
                ]
            finally:
                await client.close()

        for item in asyncio.run(run_all()):
            if item.get("status") == "OK":
                print(f"resolve   {item['dir']}: {item['sessions']} 个 session / "
                      f"{item['messages']} 条消息已消解")
            else:
                print(f"resolve   {item['dir']}: {item.get('status')} — {item.get('reason', '')}")
        return

    raise ValueError(f"未知阶段 {stage!r}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stages = [args.stage] if args.stage else ["session", "resolve"]
    for stage in stages:
        run_stage(stage, args)


if __name__ == "__main__":
    main()
