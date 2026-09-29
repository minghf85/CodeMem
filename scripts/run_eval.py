#!/usr/bin/env python
"""一条命令跑通 search → answer → eval，并把**该看的数字**摆出来。

存在的理由：search 是现在改动最频繁的一步，改完 prompt / 策略 / 时间规则要看效果，
而完整流程是三步、各自有自己的 CLI 与产物路径。手敲三遍太慢，也容易把中间产物接错
（比如 answer 默认取"最近一次 search 运行"，而你可能刚跑完一次无关的 dry-run）。

    python scripts/run_eval.py                       # 默认：2 个目录 × 前 5 条 QA，判分
    python scripts/run_eval.py --limit 20             # 多跑几条
    python scripts/run_eval.py --dirs Caroline_Melanie
    python scripts/run_eval.py --tag v4-time-rule     # 给这次运行打标记，便于对比
    python scripts/run_eval.py --no-answer            # 只跑 search，看 evidence 产出
    python scripts/run_eval.py --no-judge             # 跑完三步但只算 CPU 指标（省模型调用）
    python scripts/run_eval.py --reuse RUN_DIR        # 复用已有 evidence，只重跑 answer+eval
    python scripts/run_eval.py --quick                # = --max-steps 8 --limit 3（冒烟）

**默认参数是"快反馈"配置**，不是全量：小样本、步数偏紧，几十条 QA 就能看出 prompt 改动
的方向对不对。全量要显式给 `--limit 0`（0 = 不限）。

脚本做三件事：

1. 按参数调 ``python -m codemem.search``（子进程，参数原样透传）；
2. 从**这次**的运行目录（而不是"最近一次"）接着跑 answer、eval —— 避免接错产物；
3. 打印一份摘要：judge 准确率 / evidence recall / 未支持率 / 失败归类，
   外加**按 category 拆开**（temporal 那栏是时间规则改动的直接反馈）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data"
SEARCH_RUNS = DATA_DIR / "search_runs"


# ---------------------------------------------------------------------------
# 子进程
# ---------------------------------------------------------------------------

def run_step(label: str, argv: list[str]) -> bool:
    """跑一步，实时把输出透到终端（不吞日志 —— 排错要看的就是它）。"""
    print(f"\n{'=' * 72}\n▶ {label}\n  {' '.join(argv)}\n{'=' * 72}", flush=True)
    completed = subprocess.run(argv, cwd=str(PROJECT_ROOT))
    if completed.returncode != 0:
        print(f"\n✗ {label} 退出码 {completed.returncode}", file=sys.stderr)
        return False
    return True


def newest_run_dir(before: set[str]) -> Path | None:
    """找出这次 search 新建的运行目录。

    为什么不用 ``answer.latest_search_run()``（"最近一次"）：那个判据在调试时**会接错**——
    你可能刚跑过一次无关的 ``--dry-run``（它也会建目录），或者同时开了两个实验。
    这里改成"跑之前记下已有目录，跑完取差集"，指向的一定是这一次。
    """
    after = {p.name for p in SEARCH_RUNS.iterdir() if p.is_dir()}
    created = sorted(after - before)
    if not created:
        return None
    return SEARCH_RUNS / created[-1]


def existing_runs() -> set[str]:
    if not SEARCH_RUNS.exists():
        return set()
    return {p.name for p in SEARCH_RUNS.iterdir() if p.is_dir()}


# ---------------------------------------------------------------------------
# 摘要
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def report(run_dir: Path) -> None:
    """把该看的数字摆出来。"""
    print(f"\n{'=' * 72}\n运行目录 {run_dir}\n{'=' * 72}")

    search_summary = load_json(run_dir / "summary.json")
    dirs = search_summary.get("dirs") or []
    if dirs:
        print("\n-- search --")
        print(f"  {'目录':<22} {'QA':>8} {'evidence':>9} {'步数':>7} {'充分':>7}")
        for item in dirs:
            if item.get("status") != "OK":
                print(f"  {item.get('dir', '?'):<22} {item.get('status')} "
                      f"{item.get('reason', '')}")
                continue
            print(f"  {item['dir']:<22} "
                  f"{item.get('qa_done', 0)}/{item.get('qa_total', 0):<6} "
                  f"{item.get('evidence_total', 0):>9} "
                  f"{item.get('steps_total', 0):>7} "
                  f"{item.get('sufficient_rate', 0):>6.0%}")
        total_ev = sum(item.get("evidence_total", 0) for item in dirs)
        total_steps = sum(item.get("steps_total", 0) for item in dirs)
        calls = sum(item.get("model_calls", 0) for item in dirs)
        if total_ev or total_steps:
            print(f"  合计 evidence {total_ev} / 步 {total_steps} / 模型调用 {calls}")

    eval_summary = load_json(run_dir / "eval_summary.json")
    overall = eval_summary.get("overall") or {}
    if not overall:
        print("\n（没有 eval 结果 —— 用了 --no-answer，或 eval 那一步失败了）")
        return

    print("\n-- eval（整体）--")
    judge = overall.get("judge_accuracy")
    print(f"  judge_accuracy   {fmt(judge)}"
          + ("   <- n/a 表示没跑 judge（--no-judge）" if judge in (None, "n/a") else ""))
    print(f"  evidence_recall  {fmt(overall.get('evidence_recall'))}")
    print(f"  unsupported_rate {fmt(overall.get('unsupported_rate'))}"
          "   <- 高 = search 没找到该找的记忆")
    print(f"  token_f1         {fmt(overall.get('token_f1'))}")

    # by_category 的键**已经是名字**（summarize 里查过 LOCOMO_CATEGORY_NAMES），不是数字
    by_cat = eval_summary.get("by_category") or {}
    if by_cat:
        print("\n-- eval（按 category）--")
        print(f"  {'category':<14} {'n':>5} {'judge':>8} {'recall':>8} {'unsup':>8}")
        for name, item in sorted(by_cat.items(), key=lambda kv: -kv[1].get("count", 0)):
            print(f"  {str(name):<14} {item.get('count', 0):>5} "
                  f"{fmt(item.get('judge_accuracy')):>8} "
                  f"{fmt(item.get('evidence_recall')):>8} "
                  f"{fmt(item.get('unsupported_rate')):>8}")

    # failures 是扁平的 {归类名: 条数}；附一句"该修哪里"，这是看结果的直接目的
    failures = eval_summary.get("failures") or {}
    if failures:
        print("\n-- 失败归类 --")
        hints = {"missing_evidence": "→ 修 search：该找的没找到",
                 "wrong_answer": "→ 修 answer 的 prompt，或 evidence 写歪了",
                 "judge_error": "→ 修 judge",
                 "unjudged": "→ 没跑 judge"}
        for name, value in sorted(failures.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<18} {value:>4}   {hints.get(name, '')}")


def fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一条命令跑通 search → answer → eval，并打印摘要",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("脚本做三件事")[0].split("存在的理由")[1],
    )
    parser.add_argument("--dirs", nargs="*", default=["Caroline_Melanie"],
                        help="交给 search 的 speaker 目录（默认 Caroline_Melanie）")
    parser.add_argument("--limit", type=int, default=5,
                        help="每个目录处理前 N 条 QA（默认 5；0 = 全部）。这是 --max-qa")
    parser.add_argument("--max-steps", type=int, default=12, help="每条 QA 步数上限（默认 12）")
    parser.add_argument("--qa", default="", help="只跑指定 QA 下标（逗号/空格分隔），透传给 search")
    parser.add_argument("--tag", default="", help="给这次运行打标记，写进目录名与摘要")
    parser.add_argument("--config", default="", help="search 的 --config（默认用项目默认）")
    parser.add_argument("--reuse", default="", metavar="RUN_DIR",
                        help="复用已有的 search 运行目录，只重跑 answer + eval")
    parser.add_argument("--no-answer", action="store_true", help="只跑 search")
    parser.add_argument("--no-judge", action="store_true",
                        help="跑 answer 但 eval 只算 CPU 指标（不调 judge 模型）")
    parser.add_argument("--quick", action="store_true",
                        help="冒烟：--max-steps 8 --limit 3")
    parser.add_argument("--dry-run", action="store_true",
                        help="只渲染 prompt 打印出来，不调模型（search 的 --dry-run）")
    parser.add_argument("--log-level", default="info", help="透传给三步的日志级别")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.max_steps = 8
        if args.limit >= 5:
            args.limit = 3

    if args.reuse:
        run_dir = Path(args.reuse)
        if not run_dir.is_absolute():
            run_dir = SEARCH_RUNS / run_dir if (SEARCH_RUNS / run_dir).exists() else Path(run_dir)
        if not run_dir.is_dir():
            print(f"✗ --reuse 指向的目录不存在：{run_dir}", file=sys.stderr)
            return 2
        print(f"复用已有运行目录：{run_dir}")
        return chain(run_dir, args)

    # ---- 1. search ----
    search_argv = [sys.executable, "-m", "codemem.search", *args.dirs,
                   "--max-qa", str(args.limit), "--max-steps", str(args.max_steps),
                   "--log-level", args.log_level]
    if args.qa:
        search_argv += ["--qa", args.qa]
    if args.tag:
        search_argv += ["--tag", args.tag]
    if args.config:
        search_argv += ["--config", args.config]
    if args.dry_run:
        search_argv += ["--dry-run"]

    before = existing_runs()
    if not run_step("search", search_argv):
        return 1
    if args.dry_run:
        return 0

    run_dir = newest_run_dir(before)
    if run_dir is None:
        print("✗ 没找到这次 search 新建的运行目录 —— 无法接 answer。"
              "看上面的 search 日志确认它是否真的跑起来了。", file=sys.stderr)
        return 1
    return chain(run_dir, args)


def chain(run_dir: Path, args: argparse.Namespace) -> int:
    """从 run_dir 接着跑 answer / eval，再打印摘要。"""
    ran_eval = False
    if not args.no_answer:
        answer_argv = [sys.executable, "-m", "codemem.answer",
                       "--evidence-dir", str(run_dir),
                       "--log-level", args.log_level]
        if args.limit:
            # answer 的 --limit 按 answers 条数算，这里留足余量：search 可能产出更少
            answer_argv += ["--limit", str(max(args.limit * len(args.dirs), args.limit))]
        if not run_step("answer", answer_argv):
            return 1

        # --experiment eval 是**必须的**：eval 默认把 summary.json 写到 --output-dir，
        # 而 search 已经在那里放了一份 summary.json —— 不加前缀就会把它覆盖掉，
        # 于是"这次 search 跑了多少步、多少条 evidence"当场丢失，而报告正要读它。
        # 加了前缀写出来的是 eval_summary.json，两份并存。
        eval_argv = [sys.executable, "-m", "codemem.eval",
                     "--answers", str(run_dir / "answers.jsonl"),
                     "--output-dir", str(run_dir),
                     "--experiment", "eval",
                     "--log-level", args.log_level]
        if args.no_judge:
            eval_argv += ["--no-judge"]
        ran_eval = run_step("eval", eval_argv)

    report(run_dir)
    if not ran_eval and not args.no_answer:
        print("\n（eval 没跑成功 —— 摘要只有 search 的部分）")
    print(f"\n运行目录：{run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
