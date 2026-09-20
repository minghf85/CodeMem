"""Evaluate atommem retrieval on Locomo QA - compare msgmem vs atommem recall."""

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
    load_msgmem, load_samples, load_yaml_config, reference_answer,
    retrieve_by_embedding_async, sample_dir, resolve_run_paths,
)
from codemem.judge import judge_answer_async  # type: ignore[import-not-found]  # noqa: E402
from codemem.metrics import evidence_recall, exact_match, summarize, summarize_by_category, token_f1  # type: ignore[import-not-found]  # noqa: E402
from codemem.prompts import RAG_ANSWER_PROMPT  # type: ignore[import-not-found]  # noqa: E402

CONFIG_FILE = PROJECT_ROOT / "configs" / "atommem.yaml"


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


def load_atommem(sample: dict) -> list[dict]:
    """Load atommem.jsonl for a given sample."""
    dir_name = sample_dir(sample)
    atommem_path = PROJECT_ROOT / "data" / dir_name / "atommem.jsonl"

    if not atommem_path.exists():
        return []

    memories = []
    with atommem_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    memories.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return memories


def update_progress(progress: tqdm, results: list[dict], started: float, total: int) -> None:
    judged = [item for item in results if "judge" in item]
    judge_accuracy = (
        sum(item["judge"].get("label") == "CORRECT" for item in judged) / len(judged)
        if judged else 0.0
    )

    # Calculate average evidence recall for both msgmem and atommem
    msgmem_evidence_recall = (
        sum(float(item.get("msgmem_evidence_recall", 0.0)) for item in results) / len(results)
        if results else 0.0
    )
    atommem_evidence_recall = (
        sum(float(item.get("atommem_evidence_recall", 0.0)) for item in results) / len(results)
        if results else 0.0
    )

    elapsed = max(time.monotonic() - started, 1e-6)
    rate = progress.n / elapsed
    eta_seconds = max(total - progress.n, 0) / rate if rate > 0 else 0.0
    progress.set_postfix_str(
        f"judge_acc={judge_accuracy:.3f} msg_recall={msgmem_evidence_recall:.3f} "
        f"atom_recall={atommem_evidence_recall:.3f} rate={rate:.2f}/s ETA={eta_seconds / 60:.1f}m"
    )


async def async_main(args: argparse.Namespace) -> None:
    result_path, summary_path = resolve_run_paths(
        args.experiment, args.output_dir, "eval_atommem.jsonl", args.resume
    )
    config = load_yaml_config(
        CONFIG_FILE,
        {"embedding_model": "/root/autodl-tmp/Train/model/Qwen3-Embedding-4B", "top_k": 30},
    )
    answer_config = dict(DEFAULT_LLM_CONFIG)
    answer_config.update(config.get("generator", {}))

    # Prepare jobs
    jobs: list[tuple[dict, dict, list[dict], list[dict], str]] = []
    for sample in load_samples(args.input):
        if args.sample and sample.get("sample_id") != args.sample:
            continue

        # Load both msgmem and atommem
        msgmem = load_msgmem(sample)
        atommem = load_atommem(sample)

        if not msgmem:
            print(f"[warn] No msgmem.jsonl found for {sample_dir(sample)}, skipping")
            continue

        if not atommem:
            print(f"[warn] No atommem.jsonl found for {sample_dir(sample)}, skipping")
            continue

        conversation_date = get_conversation_date_context(msgmem)

        for qa in sample.get("qa", []):
            if args.limit is not None and len(jobs) >= args.limit:
                break
            jobs.append((sample, qa, msgmem, atommem, conversation_date))
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

    # Setup clients
    embedding_config = dict(config.get("embedding", ))
    embedding_client = openai.AsyncOpenAI(
        base_url=embedding_config.get("base_url", "http://127.0.0.1:30001/v1"),
        api_key=embedding_config.get("api_key", "sglang"),
    )
    embedding_model = embedding_config.get("model", config["embedding_model"])

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

    generation_sem = asyncio.Semaphore(int(config.get("concurrency", 4)))
    embedding_sem = asyncio.Semaphore(int(config.get("embedding_concurrency", 8)))
    judge_sem = asyncio.Semaphore(int(config.get("judge_concurrency", 4)))

    async def run_job(job):
        sample, qa, msgmem, atommem, conversation_date = job

        # Retrieve from msgmem
        async with embedding_sem:
            msgmem_retrieved = await retrieve_by_embedding_async(
                qa["question"], msgmem, embedding_model,
                int(config.get("top_k", 30)), embedding_client,
            )

        # Retrieve from atommem
        async with embedding_sem:
            atommem_retrieved = await retrieve_by_embedding_async(
                qa["question"], atommem, embedding_model,
                int(config.get("top_k", 30)), embedding_client,
            )

        # Format contexts
        msgmem_context, _ = format_memories_with_metadata(msgmem_retrieved)
        atommem_context, _ = format_memories_with_metadata(atommem_retrieved)

        # Generate answers with both contexts
        async with generation_sem:
            msgmem_candidate = await complete_with_client(
                generator, build_answer_messages(
                    qa["question"], msgmem_context, RAG_ANSWER_PROMPT, conversation_date
                ), answer_config
            )

        async with generation_sem:
            atommem_candidate = await complete_with_client(
                generator, build_answer_messages(
                    qa["question"], atommem_context, RAG_ANSWER_PROMPT, conversation_date
                ), answer_config
            )

        reference = reference_answer(qa)
        expected_ids = evidence_to_ids(qa.get("evidence"))

        # Extract IDs from retrieved memories
        msgmem_retrieved_ids = set()
        for item in msgmem_retrieved:
            mem_id = item.get("metadata", {}).get("id", "")
            if mem_id:
                msgmem_retrieved_ids.add(mem_id)

        atommem_retrieved_ids = set()
        for item in atommem_retrieved:
            # Atommem IDs are in format session_X_Y_Z, need to get parent raw ID
            mem_id = item.get("metadata", {}).get("id", "")
            if mem_id:
                # Get source IDs from atommem
                sources = item.get("metadata", {}).get("source", [])
                atommem_retrieved_ids.update(sources)

        result = {
            "sample_id": sample["sample_id"], "dir": sample_dir(sample),
            "question": qa["question"], "category": qa.get("category"), "reference": reference,
            "reference_field": "answer" if "answer" in qa else "adversarial_answer",
            "msgmem_prediction": msgmem_candidate, "atommem_prediction": atommem_candidate,
            "evidence": qa.get("evidence", []),
            "evidence_ids": sorted(expected_ids),
            "msgmem_retrieved_ids": sorted(msgmem_retrieved_ids),
            "atommem_retrieved_ids": sorted(atommem_retrieved_ids),
            "msgmem_evidence_recall": evidence_recall(expected_ids, msgmem_retrieved_ids),
            "atommem_evidence_recall": evidence_recall(expected_ids, atommem_retrieved_ids),
            "msgmem_exact_match": exact_match(reference, msgmem_candidate),
            "atommem_exact_match": exact_match(reference, atommem_candidate),
            "msgmem_token_f1": token_f1(reference, msgmem_candidate),
            "atommem_token_f1": token_f1(reference, atommem_candidate),
        }

        if qa.get("category") == 5:
            result["score_skipped"] = True
        else:
            # Judge both answers
            async with judge_sem:
                try:
                    result["msgmem_judge"] = await judge_answer_async(
                        judge_client, qa["question"], reference, msgmem_candidate
                    )
                except Exception as exc:  # noqa: BLE001
                    result["msgmem_judge"] = {"label": "INCORRECT", "reason": f"judge failed: {exc}"}

            async with judge_sem:
                try:
                    result["atommem_judge"] = await judge_answer_async(
                        judge_client, qa["question"], reference, atommem_candidate
                    )
                except Exception as exc:  # noqa: BLE001
                    result["atommem_judge"] = {"label": "INCORRECT", "reason": f"judge failed: {exc}"}

            # Add a unified judge field for compatibility with existing metrics
            result["judge"] = result["msgmem_judge"]

        return result

    tasks = [asyncio.create_task(run_job(job)) for job in pending_jobs]
    with result_path.open(file_mode, encoding="utf-8") as file, tqdm(
        pending_jobs,
        total=len(jobs),
        initial=len(jobs) - len(pending_jobs),
        desc="atommem",
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

    # Calculate metrics separately for msgmem and atommem
    def summarize_with_prefix(results: list[dict], prefix: str) -> dict:
        """Create summary metrics with field prefix."""
        judged = [r for r in results if f"{prefix}_judge" in r and r.get("category") != 5]
        return {
            f"{prefix}_judge_accuracy": (
                sum(r[f"{prefix}_judge"].get("label") == "CORRECT" for r in judged) / len(judged)
                if judged else 0.0
            ),
            f"{prefix}_evidence_recall": (
                sum(r.get(f"{prefix}_evidence_recall", 0.0) for r in results) / len(results)
                if results else 0.0
            ),
            f"{prefix}_exact_match": (
                sum(r.get(f"{prefix}_exact_match", 0.0) for r in results) / len(results)
                if results else 0.0
            ),
            f"{prefix}_token_f1": (
                sum(r.get(f"{prefix}_token_f1", 0.0) for r in results) / len(results)
                if results else 0.0
            ),
        }

    msgmem_summary = summarize_with_prefix(results, "msgmem")
    atommem_summary = summarize_with_prefix(results, "atommem")

    summary = {
        "experiment": args.experiment,
        "result_file": str(result_path),
        "total_samples": len(results),
        **msgmem_summary,
        **atommem_summary,
        "by_category": summarize_by_category(results),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate atommem retrieval on Locomo QA")
    parser.add_argument("--input", type=Path, default=DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment", default="atommem")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="跳过输出文件中已完成的 QA")
    asyncio.run(async_main(parser.parse_args()))


if __name__ == "__main__":
    main()
