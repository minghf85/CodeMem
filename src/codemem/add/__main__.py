"""add 步骤：把一段对话记忆添加进来，建成三层记忆。

**三步，顺序固定**（后一步依赖前一步的产物）：

    1. ``msgmem``   原始对话 → ``msgmem.jsonl``（type=msg 的原始消息，纯 CPU）
    2. ``summem``   逐 session 一次 LLM → ``summem.jsonl``（type=summary 的会话总结）
    3. ``atommem``  逐 message 的 **pipeline** 抽取 → ``atommem.jsonl``（type=atom）

产物（都在 ``data/{speaker_a}_{speaker_b}/`` 下）::

    msgmem.jsonl    每条原始消息一行（id=session_{s}_{i}，会话时间戳是锚点）
    summem.jsonl    每个 session 一份总结（id=session_{s}_summary）
    atommem.jsonl   逐 message 抽出的原子事实（id=session_{s}_{i}_{k}）

**为什么是"三层记忆"而不是"改写消息本身"**：原子记忆是检索的主语料 —— 一条 atom 是自足、
指代消解、时间锚定的事实，grep 命中的就是能回答问题的原料；msgmem 保留原话（补具体答案与
溯源），summem 提供会话粒度的路标。三条链格式统一，search 用同一套 grep/read 检索它们。

**atommem 默认走 pipeline**（整目录并行：召回 → 单次 LLM 抽 → 代码级后处理），详见
``docs/add.md``。逐 message 的 **code agent** 变体在 ``add-agent`` 分支，主分支不包含。

用法::

    python -m codemem.add                       # msgmem + summem + atommem，全部目录
    python -m codemem.add Caroline_Melanie      # 只处理指定目录
    python -m codemem.add --stage msgmem        # 只跑第一步（纯 CPU，不调模型）
    python -m codemem.add --stage atommem --limit-messages 20   # 只跑前 20 条 message（调试）
"""

from __future__ import annotations

import argparse
import sys

from . import atommem, msgmem, summem

STAGES = ("msgmem", "summem", "atommem")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.add",
        description="add：把对话记忆建成三层记忆（msgmem → summem → atommem）",
    )
    parser.add_argument("dirs", nargs="*", help="speaker 目录名；缺省处理全部")
    parser.add_argument("--stage", choices=STAGES, default="",
                        help="只跑某一阶段；缺省按 msgmem → summem → atommem 顺序全跑")
    parser.add_argument("--config", default=str(atommem.CONFIG_FILE),
                        help="summem / atommem 用的配置（默认 configs/add.yaml）")
    parser.add_argument("--limit-messages", type=int, default=0,
                        help="atommem 阶段每个目录只处理前 N 条 message（调试用）")
    parser.add_argument("--limit-sessions", type=int, default=0,
                        help="summem 阶段每个目录只处理前 N 个 session（调试用）")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖目录级并发数")
    return parser.parse_args(argv)


def run_stage(stage: str, args: argparse.Namespace) -> None:
    if stage == "msgmem":
        results = msgmem.run(args.dirs)
        if not results:
            print("msgmem: 没有匹配的 sample", file=sys.stderr)
            return
        for dir_name, out_dir, count in results:
            print(f"msgmem    {dir_name} -> {out_dir} ({count} 条)")
        return

    if stage == "summem":
        import asyncio
        from pathlib import Path

        from .. import llm

        config = summem.load_config(Path(args.config))
        if args.concurrency is not None:
            config["concurrency"] = args.concurrency
        targets = summem.targets_with_msgmem(args.dirs)
        if not targets:
            print("summem: 没有找到含 msgmem.jsonl 的目录"
                  "（先跑 `python -m codemem.add --stage msgmem`）", file=sys.stderr)
            return
        client = llm.make_client(summem.gen_call_config(config, "summem"))
        # 目录之间并行（像 atommem 一样），但共享同一个闸门 —— 总并发仍是 concurrency，不随目录数放大。
        semaphore = asyncio.Semaphore(max(1, int(config.get("concurrency", 8))))

        async def run_all() -> list[dict]:
            try:
                return list(await asyncio.gather(*(
                    summem.run_dir(d, config, client, limit_sessions=args.limit_sessions,
                                   semaphore=semaphore)
                    for d in targets
                )))
            finally:
                await client.close()

        for item in asyncio.run(run_all()):
            summem.report_result(item)
        return

    if stage == "atommem":
        argv: list[str] = []
        if args.dirs:
            argv.extend(args.dirs)
        argv.extend(["--config", args.config])
        if args.limit_messages:
            argv.extend(["--limit-messages", str(args.limit_messages)])
        if args.concurrency is not None:
            argv.extend(["--concurrency", str(args.concurrency)])
        atommem.main(argv)
        return

    raise ValueError(f"未知阶段 {stage!r}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stages = [args.stage] if args.stage else ["msgmem", "summem", "atommem"]
    for stage in stages:
        run_stage(stage, args)


if __name__ == "__main__":
    main()
