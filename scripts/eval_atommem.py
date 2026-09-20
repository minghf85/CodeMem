"""Evaluate atommem retrieval on Locomo QA - compare msgmem vs atommem recall."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import openai
from tqdm import tqdm  # type: ignore[import-not-found]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from codemem.eval_utils import (  # type: ignore[import-not-found]  # noqa: E402
    DATA_FILE, DEFAULT_LLM_CONFIG, EmbeddingCache, build_answer_messages,
    check_embedding_endpoint, complete_with_client,
    evidence_to_ids, format_memories_with_metadata, get_conversation_date_context,
    interleave_retrieved,
    load_msgmem, load_samples, load_yaml_config, parse_answer_output, reference_answer,
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


def make_logger(enabled: bool):
    """返回一个带时间戳的日志函数；enabled=False 时退化为空操作。"""
    def log(message: str) -> None:
        if enabled:
            stamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{stamp}] {message}", flush=True)
    return log


def update_progress(progress: tqdm, results: list[dict], started: float, total: int,
                    stages: dict[str, int] | None = None) -> None:
    judged = [item for item in results if "judge" in item]
    judge_accuracy = (
        sum(item["judge"].get("label") == "CORRECT" for item in judged) / len(judged)
        if judged else 0.0
    )

    # Calculate merged evidence recall, plus per-source diagnostics
    merged_evidence_recall = (
        sum(float(item.get("evidence_recall", 0.0)) for item in results) / len(results)
        if results else 0.0
    )
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

    # 阶段计数：区分"卡在 embedding"还是"卡在生成/评判"。in-flight = 已进入该阶段
    # 但还没产出结果的 QA 数，这是判断"是否在推进"的关键信号。
    stage_str = ""
    if stages is not None:
        in_flight_embed = stages.get("embed", 0) - stages.get("embed_done", 0)
        in_flight_gen = stages.get("generate", 0) - stages.get("generate_done", 0)
        stage_str = (
            f" | embed {stages.get('embed_done', 0)}/{stages.get('embed', 0)}"
            f"(+{in_flight_embed}) gen {stages.get('generate_done', 0)}/{stages.get('generate', 0)}"
            f"(+{in_flight_gen}) judged={len(judged)}"
        )

    progress.set_postfix_str(
        f"judge_acc={judge_accuracy:.3f} recall={merged_evidence_recall:.3f} "
        f"(msg={msgmem_evidence_recall:.3f} atom={atommem_evidence_recall:.3f}) "
        f"rate={rate:.2f}/s ETA={eta_seconds / 60:.1f}m{stage_str}"
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

    # 启动前给出工作量与成本估算：每条 QA 需要 2 次 embedding（msgmem + atommem，
    # 各自会把该样本的全部记忆重新嵌入一次）和 1 次生成 + 1 次评判。
    msgmem_total = sum(len(job[2]) for job in pending_jobs)
    atommem_total = sum(len(job[3]) for job in pending_jobs)
    print(
        f"[plan] pending QA={len(pending_jobs)} total={len(jobs)}\n"
        f"       embedding calls={len(pending_jobs) * 2} "
        f"(re-embedding {msgmem_total} msgmem + {atommem_total} atommem texts each time)\n"
        f"       generation calls={len(pending_jobs)}, judge calls (excluding category 5) ~{len(pending_jobs)}\n"
        f"       并发: embedding={config.get('embedding_concurrency', 8)} "
        f"generation={config.get('concurrency', 4)} judge={config.get('judge_concurrency', 4)}\n"
        f"       进度条每 10s 自动刷新；加 --verbose 可看每个 QA 的阶段日志。",
        flush=True,
    )

    # Setup clients
    embedding_config = dict(config.get("embedding") or {})
    embedding_client = openai.AsyncOpenAI(
        base_url=embedding_config.get("base_url", "http://127.0.0.1:30001/v1"),
        api_key=embedding_config.get("api_key", "sglang"),
    )
    embedding_model = embedding_config.get("model", config["embedding_model"])

    # 先探测 embedding 端点：配置错误时立刻报错，而不是让 1986 条 QA 各自退避重试
    # 把进度条一直卡在 0。
    print(f"[preflight] embedding endpoint: {embedding_config.get('base_url')} model={embedding_model}")
    await check_embedding_endpoint(embedding_client, embedding_model)
    print("[preflight] embedding endpoint OK")

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

    # 阶段计数与日志：用于区分"卡在 embedding"还是"卡在生成/评判"。
    stages = {"embed": 0, "embed_done": 0, "generate": 0, "generate_done": 0}
    log = make_logger(args.verbose)

    # embedding 缓存：同一样本的记忆只嵌入一次。磁盘缓存按目录存放，源文件改动会因
    # 文本哈希变化而自动重新嵌入。
    cache_dir = None if args.no_embedding_cache else (args.embedding_cache_dir or (PROJECT_ROOT / "data" / "embedding_cache"))
    embedding_cache = EmbeddingCache(embedding_client, embedding_model, cache_dir)

    async def run_job(job):
        sample, qa, msgmem, atommem, conversation_date = job
        question = qa["question"]

        # Split the retrieval budget evenly: half from msgmem, half from atommem.
        top_k = int(config.get("top_k", 30))
        msgmem_k = top_k // 2
        atommem_k = top_k - msgmem_k

        stages["embed"] += 1
        log(f"[embed] {sample['sample_id']} | {question[:50]!r} "
            f"(msgmem={len(msgmem)} atommem={len(atommem)} -> top {msgmem_k}/{atommem_k})")
        async with embedding_sem:
            msgmem_retrieved = await retrieve_by_embedding_async(
                question, msgmem, embedding_model, msgmem_k, embedding_client,
                cache=embedding_cache,
            )

        async with embedding_sem:
            atommem_retrieved = await retrieve_by_embedding_async(
                question, atommem, embedding_model, atommem_k, embedding_client,
                cache=embedding_cache,
            )
        stages["embed_done"] += 1
        log(f"[embed] {sample['sample_id']} | done, got "
            f"{len(msgmem_retrieved)}+{len(atommem_retrieved)} memories")

        # Merge the two halves into ONE context, ordered by similarity to the question.
        merged_retrieved = interleave_retrieved(msgmem_retrieved, atommem_retrieved)
        merged_context, _ = format_memories_with_metadata(merged_retrieved)

        # Generate a single answer from the merged context.
        stages["generate"] += 1
        log(f"[generate] {sample['sample_id']} | {question[:50]!r} "
            f"({len(merged_context)} chars of context)")
        async with generation_sem:
            raw_candidate = await complete_with_client(
                generator, build_answer_messages(
                    question, merged_context, RAG_ANSWER_PROMPT, conversation_date
                ), answer_config
            )
        stages["generate_done"] += 1
        log(f"[generate] {sample['sample_id']} | done ({len(raw_candidate)} chars)")

        candidate = parse_answer_output(raw_candidate)

        reference = reference_answer(qa)
        expected_ids = evidence_to_ids(qa.get("evidence"))

        # Collect retrieved IDs from both halves. Atommem entries carry their parent
        # msgmem ids in metadata.source, so map them back before comparing to evidence.
        msgmem_retrieved_ids = set()
        for item in msgmem_retrieved:
            mem_id = item.get("metadata", {}).get("id", "")
            if mem_id:
                msgmem_retrieved_ids.add(mem_id)

        atommem_retrieved_ids = set()
        for item in atommem_retrieved:
            atommem_retrieved_ids.update(item.get("metadata", {}).get("source", []))

        retrieved_ids = msgmem_retrieved_ids | atommem_retrieved_ids

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
            "evidence_ids": sorted(expected_ids),
            "msgmem_retrieved_ids": sorted(msgmem_retrieved_ids),
            "atommem_retrieved_ids": sorted(atommem_retrieved_ids),
            "retrieved_ids": sorted(retrieved_ids),
            "msgmem_evidence_recall": evidence_recall(expected_ids, msgmem_retrieved_ids),
            "atommem_evidence_recall": evidence_recall(expected_ids, atommem_retrieved_ids),
            "evidence_recall": evidence_recall(expected_ids, retrieved_ids),
            "exact_match": exact_match(reference, candidate),
            "token_f1": token_f1(reference, candidate),
            "msgmem_retrieved_count": len(msgmem_retrieved),
            "atommem_retrieved_count": len(atommem_retrieved),
        }

        if qa.get("category") == 5:
            result["score_skipped"] = True
        else:
            async with judge_sem:
                try:
                    result["judge"] = await judge_answer_async(
                        judge_client, qa["question"], reference,
                        candidate["answer"], candidate["unsupported"],
                    )
                except Exception as exc:  # noqa: BLE001
                    result["judge"] = {"label": "INCORRECT", "reason": f"judge failed: {exc}"}
                    log(f"[judge] {sample['sample_id']} | FAILED {type(exc).__name__}: {exc}")

            verdict = result["judge"].get("label")
            log(
                f"[judge] {sample['sample_id']} | {verdict} | "
                f"unsupported={candidate['unsupported']} | "
                f"pred={candidate['answer']!r} ref={reference!r} | "
                f"{(result['judge'].get('reason') or '')[:120]}"
            )

        return result

    tasks = [asyncio.create_task(run_job(job)) for job in pending_jobs]

    async def heartbeat(progress: tqdm) -> None:
        """定期刷新进度条：即使所有 worker 都阻塞在一个慢调用里，也能看到仍在推进。"""
        while True:
            await asyncio.sleep(10)
            update_progress(progress, results, started, len(jobs), stages)

    with result_path.open(file_mode, encoding="utf-8") as file, tqdm(
        pending_jobs,
        total=len(jobs),
        initial=len(jobs) - len(pending_jobs),
        desc="atommem",
        unit="qa",
    ) as progress:
        started = time.monotonic()
        update_progress(progress, results, started, len(jobs), stages)
        beat = asyncio.create_task(heartbeat(progress))
        try:
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
                update_progress(progress, results, started, len(jobs), stages)
        finally:
            beat.cancel()

    embedding_cache.save()
    print(f"[cache] {embedding_cache.stats()}", flush=True)

    await embedding_client.close()
    await generator.close()
    await judge_client.close()

    # Summary: one merged prediction, so a single set of metrics. The per-source
    # evidence recall is still reported as a diagnostic.
    msgmem_recall = (
        sum(r.get("msgmem_evidence_recall", 0.0) for r in results) / len(results)
        if results else 0.0
    )
    atommem_recall = (
        sum(r.get("atommem_evidence_recall", 0.0) for r in results) / len(results)
        if results else 0.0
    )

    summary = {
        "experiment": args.experiment,
        "result_file": str(result_path),
        "total_samples": len(results),
        **summarize(results),
        "by_category": summarize_by_category(results),
        "retrieval_stats": {
            "top_k": int(config.get("top_k", 30)),
            "msgmem_evidence_recall": msgmem_recall,
            "atommem_evidence_recall": atommem_recall,
        },
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
    parser.add_argument("--verbose", action="store_true", help="打印每个 QA 的 embed/generate 阶段日志")
    parser.add_argument("--no-embedding-cache", action="store_true", help="禁用 embedding 缓存（每次重新嵌入）")
    parser.add_argument("--embedding-cache-dir", type=Path, default=None,
                        help="embedding 磁盘缓存目录（默认 data/embedding_cache）")
    asyncio.run(async_main(parser.parse_args()))


if __name__ == "__main__":
    main()
