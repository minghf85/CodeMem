"""eval 步骤：判断回答是否正确，并计算各项指标。

**输入是 answer 步骤的产物**（``answers.jsonl``：每条含 ``question`` / ``answer`` /
``reference`` / ``evidence_path``），输出逐条判定 + 汇总指标。整条链路因此闭合：

    add → search（evidence.jsonl）→ answer（答案）→ eval（指标）

指标分三组，各自回答一个不同的问题：

| 组 | 指标 | 回答的问题 |
|---|---|---|
| **答案质量** | ``judge_accuracy`` / ``exact_match`` / ``token_f1`` | 答对了吗？ |
| **证据质量** | ``evidence_recall`` | search 找到的记忆覆盖了标注证据吗？ |
| **行为** | ``unsupported_rate`` / ``evidence_count`` | 是"证据不足"还是"答错了"？塞了多少条？ |

``unsupported_rate`` 是最有诊断价值的一个：它高说明 search 没找到该找的记忆（闸门正确工作），
它高而 judge 也低说明链路真的没跑通；它低而 judge 低则是**推理**错了（证据在，但答歪了）。
这两种失败的修法完全不同。

judge 对 category 5（adversarial，故意不可回答）**只记录不评分** —— 与数据集的评测口径一致。

用法::

    python -m codemem.eval                                  # 评最近一次 answer
    python -m codemem.eval --answers path/to/answers.jsonl
    python -m codemem.eval --no-judge                       # 只算 CPU 指标（不调模型）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import dataset, llm
from ..io import DATA_DIR, PROJECT_ROOT, memory_id, read_jsonl, write_jsonl
from ..log import Logger
from . import judge as judge_mod
from . import metrics

CONFIG_FILE = PROJECT_ROOT / "configs" / "eval.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "judge": {},
    "concurrency": 8,
    "log": {"level": "info", "output": "data/eval_runs"},
}


def load_config(path: Path | None = None) -> dict[str, Any]:
    import yaml

    config = json.loads(json.dumps(DEFAULT_CONFIG))
    target = path or CONFIG_FILE
    if target.exists():
        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                    config[key].update(value)
                else:
                    config[key] = value
    return config


# ---------------------------------------------------------------------------
# 单条评估
# ---------------------------------------------------------------------------

@dataclass
class ScoredItem:
    """一条回答的完整评分。``skipped`` 表示该条不计入汇总（category 5）。"""

    qa_index: int | None
    question: str
    reference: str
    candidate: Any
    unsupported: bool
    category: Any
    judge_label: str = ""
    judge_reason: str = ""
    judge_error: str = ""
    exact_match: float = 0.0
    token_f1: float = 0.0
    evidence_recall: float | None = None
    evidence_count: int = 0
    evidence_expected: int = 0
    skipped: bool = False

    @property
    def correct(self) -> bool:
        """judge 判 CORRECT 才算对。judge 失败（label 空）不计为对 —— 保守。"""
        return self.judge_label.upper() == "CORRECT"

    def to_dict(self) -> dict[str, Any]:
        return {
            "qa_index": self.qa_index,
            "question": self.question,
            "reference": self.reference,
            "candidate": self.candidate,
            "unsupported": self.unsupported,
            "category": self.category,
            "judge_label": self.judge_label,
            "judge_reason": self.judge_reason,
            "judge_error": self.judge_error,
            "correct": self.correct,
            "exact_match": round(self.exact_match, 4),
            "token_f1": round(self.token_f1, 4),
            "evidence_recall": (
                round(self.evidence_recall, 4) if self.evidence_recall is not None else None
            ),
            "evidence_count": self.evidence_count,
            "evidence_expected": self.evidence_expected,
            "skipped": self.skipped,
        }


def load_evidence_ids(evidence_path: str) -> set[str]:
    """从 evidence.jsonl 里取出记录 id 集合（用于算 evidence_recall）。

    比对的基准是 drug evidence 指向的**消息** id（``session_1_3``），而 evidence 里存的
    记录 id 可能是原子记忆 id（``session_1_3_1``）。所以这里把记录 id 映射回去：先取记录
    自己的 id，再取它 ``source`` 里的消息 id —— 二者都算命中，否则原子记忆永远匹配不上
    消息级标注，recall 会恒为 0。
    """
    if not evidence_path:
        return set()
    path = Path(evidence_path)
    if not path.exists():
        return set()
    ids: set[str] = set()
    for record in read_jsonl(path):
        record_id = memory_id(record)
        if record_id:
            ids.add(record_id)
        for source in (record.get("metadata") or {}).get("source") or []:
            if isinstance(source, str) and source:
                ids.add(source)
    return ids


async def score_item(
    payload: dict[str, Any],
    *,
    client: Any,
    judge_config: dict[str, Any],
    use_judge: bool,
    log: Logger,
) -> ScoredItem:
    """给一条回答打分（CPU 指标 + 可选 judge）。"""
    question = str(payload.get("question") or "")
    reference = str(payload.get("reference") or "")
    candidate = payload.get("answer")
    unsupported = bool(payload.get("unsupported"))
    category = payload.get("category")

    item = ScoredItem(
        qa_index=payload.get("qa_index"),
        question=question,
        reference=reference,
        candidate=candidate,
        unsupported=unsupported,
        category=category,
        evidence_count=int(payload.get("evidence_count") or 0),
    )
    # category 5 是 adversarial：只记录，不参与评分（与数据集口径一致）
    item.skipped = category == 5

    item.exact_match = metrics.exact_match(reference, candidate)
    item.token_f1 = metrics.token_f1(reference, candidate)

    expected = dataset.evidence_ids_for_qa({"evidence": payload.get("evidence") or []})
    if not expected:
        # 兼容：answers.jsonl 里可能直接带了 evidence 标记
        expected = dataset.evidence_ids_for_qa(
            {"evidence": payload.get("evidence_ids") or []}
        )
    if expected:
        got = load_evidence_ids(str(payload.get("evidence_path") or ""))
        item.evidence_expected = len(expected)
        item.evidence_recall = metrics.evidence_recall(expected, got)

    if use_judge and not item.skipped:
        try:
            verdict = await judge_mod.judge_answer_async(
                client, question, reference, candidate, unsupported, judge_config
            )
            item.judge_label = str(verdict.get("label") or "")
            item.judge_reason = str(verdict.get("reason") or "")
        except Exception as exc:  # noqa: BLE001 - 单条 judge 失败不该拖垮整批
            item.judge_error = f"{type(exc).__name__}: {exc}"
            log.warn(f"judge 失败（qa{payload.get('qa_index')}）：{item.judge_error}")
    return item


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def summarize(items: list[ScoredItem]) -> dict[str, Any]:
    """整体 + 分 category 的汇总。**跳过被 skip 的条目**（category 5）。"""
    scored = [item for item in items if not item.skipped]
    judged = [item for item in scored if item.judge_label]
    with_recall = [item for item in scored if item.evidence_recall is not None]

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    overall = {
        "count": len(items),
        "scored": len(scored),
        "skipped": len(items) - len(scored),
        "judged": len(judged),
        # 没跑 judge 时给 None 而不是 0.0 —— 0.0 会被读成"全答错了"，那是误导。
        "judge_accuracy": (
            mean([1.0 if item.correct else 0.0 for item in judged]) if judged else None
        ),
        "exact_match": mean([item.exact_match for item in scored]),
        "token_f1": mean([item.token_f1 for item in scored]),
        "evidence_recall": mean([item.evidence_recall for item in with_recall]),
        "unsupported_rate": mean([1.0 if item.unsupported else 0.0 for item in scored]),
        "evidence_count_mean": mean([float(item.evidence_count) for item in scored]),
        "judge_errors": sum(1 for item in scored if item.judge_error),
    }

    by_category: dict[str, dict[str, Any]] = {}
    for category, group in _group_by_category(scored).items():
        name = metrics.LOCOMO_CATEGORY_NAMES.get(
            category, f"category_{category}" if category is not None else "unknown"
        )
        judged_group = [item for item in group if item.judge_label]
        by_category[name] = {
            "count": len(group),
            "judge_accuracy": (
                mean([1.0 if item.correct else 0.0 for item in judged_group])
                if judged_group else None
            ),
            "token_f1": mean([item.token_f1 for item in group]),
            "evidence_recall": mean(
                [item.evidence_recall for item in group if item.evidence_recall is not None]
            ),
            "unsupported_rate": mean([1.0 if item.unsupported else 0.0 for item in group]),
        }
    return {"overall": overall, "by_category": by_category}


def _group_by_category(items: list[ScoredItem]) -> dict[Any, list[ScoredItem]]:
    groups: dict[Any, list[ScoredItem]] = {}
    for item in items:
        groups.setdefault(item.category, []).append(item)
    return dict(sorted(groups.items(), key=lambda pair: (pair[0] is None, pair[0])))


def failure_breakdown(items: list[ScoredItem]) -> dict[str, int]:
    """把失败分类 —— 这是"下一步该修什么"的直接依据。

    - ``missing_evidence``：证据不足（unsupported）→ 该修 search 的检索/推理。
    - ``wrong_answer``：证据在但答错 → 该修 answer 的 prompt，或 evidence 写歪了。
    - ``judge_error``：judge 调用了但判不出来 → 该修 judge。
    - ``unjudged``：**没跑 judge**（``--no-judge``）→ 既不能说对也不能说错，单独计数。
      不能并进 wrong_answer，否则 `--no-judge` 会报出一个假的 0% 准确率。
    """
    counts = Counter()
    for item in items:
        if item.skipped or item.correct:
            continue
        if item.judge_error:
            counts["judge_error"] += 1
        elif not item.judge_label:
            counts["unjudged"] += 1
        elif item.unsupported:
            counts["missing_evidence"] += 1
        else:
            counts["wrong_answer"] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def latest_answer_run(root: Path | None = None) -> Path | None:
    """找最近一次的 answers.jsonl（answer 写到 search 的运行目录里）。"""
    base = root or (DATA_DIR / "search_runs")
    if not base.exists():
        return None
    for directory in sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name,
                            reverse=True):
        candidate = directory / "answers.jsonl"
        if candidate.exists():
            return candidate
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.eval",
        description="eval：判断回答是否正确并计算指标（输入是 answer 的产物）",
    )
    parser.add_argument("--config", default=str(CONFIG_FILE), help="配置（默认 configs/eval.yaml）")
    parser.add_argument("--answers", default="", help="answers.jsonl 路径；缺省取最近一次")
    parser.add_argument("--output-dir", default="", help="结果输出目录；缺省与 answers 同目录")
    parser.add_argument("--experiment", default="", help="实验名前缀（用于输出文件名）")
    parser.add_argument("--no-judge", action="store_true", help="只算 CPU 指标，不调模型")
    parser.add_argument("--limit", type=int, default=0, help="最多评 N 条（调试用）")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖 judge 并发数")
    parser.add_argument("--log-level", default="", help="debug/info/warn/error/silent")
    parser.add_argument("--log-output", default=None, help="日志目录；空串=只打终端")
    return parser.parse_args(argv)


def build_logger(config: dict[str, Any], args: argparse.Namespace) -> Logger:
    section = dict(config.get("log") or {})
    if args.log_level:
        section["level"] = args.log_level
    if args.log_output is not None:
        section["output"] = args.log_output
    return Logger.from_config({"log": section}, prefix="eval")


async def async_main(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    log = build_logger(config, args)

    answers_path = Path(args.answers) if args.answers else latest_answer_run()
    if answers_path is None or not answers_path.exists():
        log.error(
            "找不到 answers.jsonl。先跑 `python -m codemem.answer`，或用 --answers 指定路径。"
        )
        return
    payloads = read_jsonl(answers_path)
    if not payloads:
        log.error(f"{answers_path} 里没有记录")
        return
    if args.limit:
        payloads = payloads[: args.limit]
    log.info(f"评估 {len(payloads)} 条 <- {answers_path}")

    use_judge = not args.no_judge
    client = llm.make_client(config.get("judge") or {}) if use_judge else None
    semaphore = asyncio.Semaphore(max(1, args.concurrency or int(config.get("concurrency", 8))))

    async def worker(payload: dict[str, Any]) -> ScoredItem:
        async with semaphore:
            return await score_item(
                payload, client=client, judge_config=config.get("judge") or {},
                use_judge=use_judge, log=log,
            )

    try:
        items = list(await asyncio.gather(*(worker(p) for p in payloads)))
    finally:
        if client is not None:
            await client.close()

    summary = summarize(items)
    summary["failures"] = failure_breakdown(items)
    summary["source"] = str(answers_path)
    summary["judge_enabled"] = use_judge

    output_dir = Path(args.output_dir) if args.output_dir else answers_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.experiment}_" if args.experiment else ""
    write_jsonl(output_dir / f"{prefix}eval.jsonl", [item.to_dict() for item in items])
    (output_dir / f"{prefix}summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    overall = summary["overall"]
    accuracy = overall["judge_accuracy"]
    accuracy_text = f"{accuracy:.1%}" if accuracy is not None else "n/a（未跑 judge）"
    log.info(
        f"judge_accuracy={accuracy_text} "
        f"token_f1={overall['token_f1']:.3f} "
        f"evidence_recall={overall['evidence_recall']:.1%} "
        f"unsupported={overall['unsupported_rate']:.1%} "
        f"（评 {overall['judged']} 条 / 跳过 {overall['skipped']} 条 category5）"
    )
    if summary["failures"]:
        log.info("失败构成：" + " ".join(f"{k}={v}" for k, v in summary["failures"].items()))
    for name, group in summary["by_category"].items():
        group_accuracy = group["judge_accuracy"]
        log.info(
            f"  {name:18s} n={group['count']:>4} "
            f"acc={f'{group_accuracy:.1%}' if group_accuracy is not None else 'n/a'} "
            f"f1={group['token_f1']:.3f} "
            f"recall={group['evidence_recall']:.1%}"
        )
    log.info(f"结果 -> {output_dir}")
    log.info(log.summary())
    log.close()


def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(parse_args(argv)))


if __name__ == "__main__":
    main()
