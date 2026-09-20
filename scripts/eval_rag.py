"""Evaluate embedding-retrieval RAG on Locomo QA."""

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

from codemem.eval_utils import (  # type: ignore[import-not-found]  # noqa: E402
    DATA_FILE, DEFAULT_LLM_CONFIG, build_answer_messages, complete_with_client,
    evidence_to_ids, format_memories_with_metadata, get_conversation_date_context,
    load_msgmem, load_samples, load_yaml_config, parse_answer_output, reference_answer,
    retrieve_by_embedding_async, sample_dir, resolve_run_paths,
)
from codemem.judge import judge_answer_async  # type: ignore[import-not-found]  # noqa: E402
from codemem.metrics import evidence_recall, exact_match, summarize, summarize_by_category, token_f1  # type: ignore[import-not-found]  # noqa: E402
from codemem.prompts import RAG_ANSWER_PROMPT  # type: ignore[import-not-found]  # noqa: E402

CONFIG_FILE = PROJECT_ROOT / "configs" / "rag.yaml"


def result_key(result: dict) -> tuple[str, str]:
    return result["sample_id"], result["question"]


def load_existing_results(path: Path) -> tuple[list[dict], set[tuple[str, str]]]:
    results: list[dict] = []
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return results, keys
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                keys.add(result_key(result))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                print(f"[warn] skip invalid result {path}:{line_number}: {exc}", file=sys.stderr)
                continue
            results.append(result)
    return results, keys


def update_progress(progress: tqdm, results: list[dict], started: float, total: int) -> None:
    judged = [item for item in results if "judge" in item]
    judge_accuracy = (
        sum(item["judge"].get("label") == "CORRECT" for item in judged) / len(judged)
        if judged else 0.0
    )
    evidence_accuracy = (
        sum(float(item.get("evidence_recall", 0.0)) for item in results) / len(results)
        if results else 0.0
    )
    elapsed = max(time.monotonic() - started, 1e-6)
    rate = progress.n / elapsed
    eta_seconds = max(total - progress.n, 0) / rate if rate > 0 else 0.0
    progress.set_postfix_str(
        f"judge_acc={judge_accuracy:.3f} evidence_recall={evidence_accuracy:.3f} "
        f"rate={rate:.2f}/s ETA={eta_seconds / 60:.1f}m"
    )


async def async_main(args: argparse.Namespace) -> None:
    result_path, summary_path = resolve_run_paths(args.experiment, args.output_dir, "eval_rag.jsonl", args.resume)
    rag_config = load_yaml_config(
        CONFIG_FILE,
        {"embedding_model": "/root/autodl-tmp/Train/model/Qwen3-Embedding-4B", "top_k": 30},
    )
    answer_config = dict(DEFAULT_LLM_CONFIG)
    answer_config.update(rag_config.get("generator", {}))

    jobs: list[tuple[dict, dict, list[dict], str]] = []
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

        for qa in sample.get("qa", []):
            if args.limit is not None and len(jobs) >= args.limit:
                break
            jobs.append((sample, qa, memories, conversation_date))
        if args.limit is not None and len(jobs) >= args.limit:
            break

    results, completed_keys = load_existing_results(result_path) if args.resume else ([], set())
    pending_jobs = [
        job for job in jobs
        if (job[0]["sample_id"], job[1]["question"]) not in completed_keys
    ]
    if args.resume:
        print(f"[resume] completed={len(completed_keys)} pending={len(pending_jobs)} total={len(jobs)}")
    if not args.resume and result_path.exists():
        result_path.unlink()

    embedding_config = dict(rag_config.get("embedding", {}))
    embedding_client = openai.AsyncOpenAI(
        base_url=embedding_config.get("base_url", "http://127.0.0.1:30001/v1"),
        api_key=embedding_config.get("api_key", "sglang"),
    )
    embedding_model = embedding_config.get("model", rag_config["embedding_model"])

    file_mode = "a" if args.resume else "w"
    generator = openai.AsyncOpenAI(
        base_url=answer_config["base_url"], api_key=answer_config.get("api_key") or "not-needed",
        timeout=float(answer_config.get("timeout", 300)), max_retries=0,
    )
    judge_config = dict(DEFAULT_LLM_CONFIG)
    judge_config.update(load_yaml_config(PROJECT_ROOT / "configs/judge.yaml"))
    judge_client = openai.AsyncOpenAI(
        base_url=judge_config["base_url"], api_key=judge_config.get("api_key") or "not-needed",
        timeout=float(judge_config.get("timeout", 300)), max_retries=0,
    )
    generation_sem = asyncio.Semaphore(int(rag_config.get("concurrency", 4)))
    embedding_sem = asyncio.Semaphore(int(rag_config.get("embedding_concurrency", 8)))
    judge_sem = asyncio.Semaphore(int(rag_config.get("judge_concurrency", 4)))

    async def run_job(job):
        sample, qa, memories, conversation_date = job
        async with embedding_sem:
            retrieved = await retrieve_by_embedding_async(
                qa["question"], memories, embedding_model,
                int(rag_config.get("top_k", 30)), embedding_client,
            )
        # Format retrieved memories with metadata
        retrieved_context, _ = format_memories_with_metadata(retrieved)
        async with generation_sem:
            raw_candidate = await complete_with_client(
                generator, build_answer_messages(
                    qa["question"], retrieved_context, RAG_ANSWER_PROMPT, conversation_date
                ), answer_config
            )
        candidate = parse_answer_output(raw_candidate)
        reference = reference_answer(qa)
        expected_ids = evidence_to_ids(qa.get("evidence"))
        # Extract IDs from retrieved memories
        retrieved_ids = set()
        for item in retrieved:
            mem_id = item.get("metadata", {}).get("id", "")
            if mem_id:
                retrieved_ids.add(mem_id)
        result = {
            "sample_id": sample["sample_id"], "dir": sample_dir(sample),
            "question": qa["question"], "category": qa.get("category"), "reference": reference,
            "reference_field": "answer" if "answer" in qa else "adversarial_answer",
            "prediction": candidate,
            "prediction_answer": candidate["answer"],
            "prediction_unsupported": candidate["unsupported"],
            "prediction_reasoning": candidate["reasoning"],
            "prediction_raw": candidate["raw"],
            "evidence": qa.get("evidence", []),
            "evidence_ids": sorted(expected_ids), "retrieved_ids": sorted(retrieved_ids),
            "evidence_recall": evidence_recall(expected_ids, retrieved_ids),
            "exact_match": exact_match(reference, candidate), "token_f1": token_f1(reference, candidate),
        }
        if qa.get("category") == 5:
            result["score_skipped"] = True
        else:
            async with judge_sem:
                try:
                    result["judge"] = await judge_answer_async(
                        judge_client, qa["question"], reference, candidate["answer"], candidate["unsupported"]
                    )
                except Exception as exc:  # noqa: BLE001
                    result["judge"] = {"label": "INCORRECT", "reason": f"judge failed: {exc}"}
        return result

    async def limited(job):
        return await run_job(job)

    tasks = [asyncio.create_task(limited(job)) for job in pending_jobs]
    with result_path.open(file_mode, encoding="utf-8") as file, tqdm(
        pending_jobs,
        total=len(jobs),
        initial=len(jobs) - len(pending_jobs),
        desc="rag",
        unit="qa",
    ) as progress:
        started = time.monotonic()
        update_progress(progress, results, started, len(jobs))
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
                completed_keys.add(result_key(result))
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()
            except Exception as exc:  # noqa: BLE001
                print(f"\n[error] {exc}", file=sys.stderr)
            progress.update(1)
            update_progress(progress, results, started, len(jobs))
    await embedding_client.close()
    await generator.close()
    await judge_client.close()
    summary = {"experiment": args.experiment, "result_file": str(result_path), **summarize(results), "by_category": summarize_by_category(results)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate embedding RAG on Locomo QA")
    parser.add_argument("--input", type=Path, default=DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment", default="rag")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="跳过输出文件中已完成的 QA")
    asyncio.run(async_main(parser.parse_args()))


if __name__ == "__main__":
    main()
