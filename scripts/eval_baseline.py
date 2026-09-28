"""Evaluate full-conversation baseline on Locomo QA."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import openai
from tqdm import tqdm  # type: ignore[import-not-found]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from codemem.eval.legacy import (  # type: ignore[import-not-found]  # noqa: E402
    DATA_FILE,
    DEFAULT_LLM_CONFIG,
    build_answer_messages,
    complete_with_client,
    evidence_to_ids,
    format_memories_with_metadata,
    get_conversation_date_context,
    load_msgmem,
    load_samples,
    load_yaml_config,
    parse_answer_output,
    reference_answer,
    resolve_run_paths,
    sample_dir,
)
from codemem.eval.judge import judge_answer_async  # type: ignore[import-not-found]  # noqa: E402
from codemem.eval.metrics import exact_match, summarize, summarize_by_category, token_f1  # type: ignore[import-not-found]  # noqa: E402
from codemem.prompts import BASELINE_ANSWER_PROMPT  # type: ignore[import-not-found]  # noqa: E402

CONFIG_FILE = PROJECT_ROOT / "configs" / "baseline.yaml"


def result_key(result: dict) -> tuple[str, str]:
    return result["sample_id"], result["question"]


def load_existing_results(path: Path) -> tuple[list[dict], set[tuple[str, str]]]:
    results: list[dict] = []
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return results, keys
    with path.open(encoding="utf-8") as file:
        for line in file:
            try:
                if line.strip():
                    item = json.loads(line)
                    results.append(item)
                    keys.add(result_key(item))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return results, keys


async def run_job(job, generator, judge, config, judge_sem, max_context_tokens):
    sample, qa, context, truncation_info, conversation_date = job
    reference = reference_answer(qa)
    raw_candidate = await complete_with_client(
        generator, build_answer_messages(qa["question"], context, BASELINE_ANSWER_PROMPT, conversation_date), config
    )
    candidate = parse_answer_output(raw_candidate)
    result = {
        "sample_id": sample["sample_id"],
        "dir": sample_dir(sample),
        "question": qa["question"],
        "category": qa.get("category"),
        "reference": reference,
        "reference_field": "answer" if "answer" in qa else "adversarial_answer",
        "prediction": candidate,
        "prediction_answer": candidate["answer"],
        "prediction_unsupported": candidate["unsupported"],
        "prediction_reasoning": candidate["reasoning"],
        "prediction_raw": candidate["raw"],
        "evidence": qa.get("evidence", []),
        "evidence_ids": sorted(evidence_to_ids(qa.get("evidence"))),
        "retrieved_ids": [],
        "evidence_recall": 1.0,
        "exact_match": exact_match(reference, candidate),
        "token_f1": token_f1(reference, candidate),
        "context_truncated": truncation_info["truncated"],
        "context_total_messages": truncation_info.get("total_messages", truncation_info.get("total_memories", 0)),
        "context_kept_messages": truncation_info.get("kept_messages", truncation_info.get("kept_memories", 0)),
        "truncated_speakers": sorted(truncation_info["truncated_speakers"]),
    }
    if qa.get("category") == 5:
        result["score_skipped"] = True
    else:
        async with judge_sem:
            try:
                result["judge"] = await judge_answer_async(
                    judge, qa["question"], reference, candidate["answer"], candidate["unsupported"]
                )
            except Exception as exc:  # noqa: BLE001
                result["judge"] = {"label": "INCORRECT", "reason": f"judge failed: {exc}"}
    return result


async def async_main(args: argparse.Namespace) -> None:
    result_path, summary_path = resolve_run_paths(args.experiment, args.output_dir, "eval_baseline.jsonl", args.resume)
    config = dict(DEFAULT_LLM_CONFIG)
    config.update(load_yaml_config(CONFIG_FILE))

    # Get max context tokens from config, default to 32K
    max_context_tokens = int(config.get("max_context_tokens", 32000))

    jobs = []
    for sample in load_samples(args.input):
        if args.sample and sample.get("sample_id") != args.sample:
            continue

        # Load msgmem.jsonl for this sample
        memories = load_msgmem(sample)

        if not memories:
            print(f"[warn] No msgmem.jsonl found for {sample_dir(sample)}, skipping")
            continue

        # Get conversation date context
        conversation_date = get_conversation_date_context(memories)

        # Format with truncation using format_memories_with_metadata
        context, truncation_info = format_memories_with_metadata(memories, max_tokens=max_context_tokens)

        for qa in sample.get("qa", []):
            if args.limit is not None and len(jobs) >= args.limit:
                break
            jobs.append((sample, qa, context, truncation_info, conversation_date))
        if args.limit is not None and len(jobs) >= args.limit:
            break

    results, completed = load_existing_results(result_path) if args.resume else ([], set())
    pending = [
        job for job in jobs
        if (job[0]["sample_id"], job[1]["question"]) not in completed
    ]
    if args.resume:
        print(f"[resume] completed={len(completed)} pending={len(pending)} total={len(jobs)}")
    if not args.resume and result_path.exists():
        result_path.unlink()

    generator = openai.AsyncOpenAI(
        base_url=config["base_url"],
        api_key=config.get("api_key") or "not-needed",
        timeout=float(config.get("timeout", 300)),
        max_retries=0,
    )
    judge_config = dict(DEFAULT_LLM_CONFIG)
    judge_config.update(load_yaml_config(PROJECT_ROOT / "configs/judge.yaml"))
    judge = openai.AsyncOpenAI(
        base_url=judge_config["base_url"],
        api_key=judge_config.get("api_key") or "not-needed",
        timeout=float(judge_config.get("timeout", 300)),
        max_retries=0,
    )
    sem = asyncio.Semaphore(int(config.get("concurrency", 4)))
    judge_sem = asyncio.Semaphore(
        int(config.get("judge_concurrency", config.get("concurrency", 4)))
    )

    async def limited(job):
        async with sem:
            return await run_job(job, generator, judge, config, judge_sem, max_context_tokens)

    tasks = [asyncio.create_task(limited(job)) for job in pending]
    started = time.monotonic()
    try:
        with result_path.open("a" if args.resume else "w", encoding="utf-8") as file, tqdm(
            total=len(jobs),
            initial=len(jobs) - len(pending),
            desc="baseline",
            unit="qa",
        ) as progress:
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    results.append(result)
                    file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    file.flush()
                except Exception as exc:  # noqa: BLE001
                    print(f"\n[error] {exc}", file=sys.stderr)
                progress.update(1)
                judged = [item for item in results if "judge" in item]
                accuracy = (
                    sum(item["judge"].get("label") == "CORRECT" for item in judged) / len(judged)
                    if judged else 0.0
                )
                rate = progress.n / max(time.monotonic() - started, 1e-6)
                eta = max(len(jobs) - progress.n, 0) / rate if rate > 0 else 0.0
                progress.set_postfix_str(
                    f"judge_acc={accuracy:.3f} rate={rate:.2f}/s ETA={eta / 60:.1f}m"
                )
    finally:
        await generator.close()
        await judge.close()

    # Calculate truncation statistics
    truncated_samples = [r for r in results if r.get("context_truncated")]
    all_truncated_speakers = set()
    for r in truncated_samples:
        all_truncated_speakers.update(r.get("truncated_speakers", []))

    truncation_stats = {
        "max_context_tokens": max_context_tokens,
        "truncated_count": len(truncated_samples),
        "total_count": len(results),
        "truncation_rate": len(truncated_samples) / len(results) if results else 0.0,
        "truncated_speakers": sorted(all_truncated_speakers),
    }

    summary = {
        "experiment": args.experiment,
        "result_file": str(result_path),
        **summarize(results),
        "by_category": summarize_by_category(results),
        "truncation_stats": truncation_stats,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full-context baseline on Locomo QA")
    parser.add_argument("--input", type=Path, default=DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    asyncio.run(async_main(parser.parse_args()))


if __name__ == "__main__":
    main()
