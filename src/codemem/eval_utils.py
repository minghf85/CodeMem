"""Shared Locomo evaluation data loading, retrieval, and model helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import openai
import yaml

from . import atommem
from .prompts import ANSWER_PROMPT

PROJECT_ROOT = atommem.PROJECT_ROOT
DATA_FILE = PROJECT_ROOT / "data" / "correct_locomo10.json"


def resolve_run_paths(
    experiment: str,
    output_dir: Path | None,
    result_name: str,
    resume: bool,
) -> tuple[Path, Path]:
    root = PROJECT_ROOT / "data" / "eval_runs"
    if output_dir is None and resume:
        candidates = sorted(root.glob(f"{experiment}_*")) if root.exists() else []
        output_dir = candidates[-1] if candidates else None
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = root / f"{experiment}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / result_name, output_dir / "summary.json"

DEFAULT_LLM_CONFIG: dict[str, Any] = {
    "base_url": "http://127.0.0.1:30000/v1",
    "api_key": "sglang",
    "model": "/root/autodl-tmp/EvoAtomMem/output/qwen3-8b-orpo-merged",
    "temperature": 0.2,
    "max_tokens": 2048,
    "timeout": 300,
    "max_retries": 3,
    "backoff_base": 2.0,
    "rate_limit_backoff": 10.0,
    "max_backoff": 60.0,
    "backoff_jitter": 0.2,
    "enable_thinking": False,
}


def load_yaml_config(path: Path, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(defaults or {})
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            config.update(loaded)
    return config


def load_msgmem(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Load msgmem.jsonl for a given sample.

    Returns list of memory items from msgmem.jsonl file.
    """
    dir_name = sample_dir(sample)
    directory = PROJECT_ROOT / "data" / dir_name
    msgmem_path = directory / "msgmem.jsonl"
    if not msgmem_path.exists():
        # 兼容旧命名：部分目录仍是 memory.jsonl
        msgmem_path = directory / "memory.jsonl"

    if not msgmem_path.exists():
        # Fallback: return empty list if file doesn't exist
        return []

    memories = []
    with msgmem_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    memories.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return memories


def get_conversation_date_context(memories: list[dict[str, Any]]) -> str:
    """Extract the date range or latest date from memories to use as current_date context.

    Returns a date string representing when the conversation took place.
    """
    if not memories:
        return datetime.now().strftime("%Y-%m-%d")

    # Try to find the latest timestamp in the memories
    timestamps = []
    for mem in memories:
        time_str = mem.get("metadata", {}).get("time", "")
        if time_str and not time_str.startswith("few minutes"):
            try:
                # Try to parse ISO format
                if "T" in time_str:
                    dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                    timestamps.append(dt)
            except (ValueError, AttributeError):
                pass

    if timestamps:
        # Use the latest timestamp as the "current date" for the conversation
        latest = max(timestamps)
        return latest.strftime("%Y-%m-%d")

    # Fallback
    return datetime.now().strftime("%Y-%m-%d")


def load_samples(path: Path = DATA_FILE) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_dir(sample: dict[str, Any]) -> str:
    conversation = sample["conversation"]
    return f"{conversation['speaker_a']}_{conversation['speaker_b']}"


def reference_answer(qa: dict[str, Any]) -> Any:
    """Return the standard answer, falling back to Locomo adversarial_answer."""
    if "answer" in qa:
        return qa["answer"]
    if "adversarial_answer" in qa:
        return qa["adversarial_answer"]
    raise KeyError("QA item has neither answer nor adversarial_answer")


def iter_messages(sample: dict[str, Any]) -> Iterable[dict[str, Any]]:
    conversation = sample["conversation"]
    session_keys = sorted(
        (key for key in conversation if key.startswith("session_") and key.rsplit("_", 1)[-1].isdigit()),
        key=lambda key: int(key.split("_")[1]),
    )
    for session_key in session_keys:
        session_index = int(session_key.split("_")[1])
        for message_index, message in enumerate(conversation[session_key], 1):
            yield {
                "id": f"session_{session_index}_{message_index}",
                "session": session_index,
                "message_index": message_index,
                "speaker": message.get("speaker", "unknown"),
                "text": message.get("text", ""),
            }


def evidence_to_ids(evidence: list[str] | None) -> set[str]:
    """Convert Locomo evidence such as D1:3 into generated message IDs."""
    result: set[str] = set()
    for item in evidence or []:
        try:
            document, turn = item.split(":", 1)
            result.add(f"session_{int(document.removeprefix('D'))}_{int(turn)}")
        except (AttributeError, ValueError):
            continue
    return result


def format_messages(messages: Iterable[dict[str, Any]], max_tokens: int | None = None) -> str:
    """Format messages with optional truncation to fit within token limit.

    Args:
        messages: Iterable of message dicts with id, speaker, text
        max_tokens: Maximum approximate tokens (estimated as 4 chars per token).
                   If None, no truncation is applied.

    Returns:
        Formatted string with messages, potentially truncated from the start.
    """
    message_list = list(messages)
    if max_tokens is None:
        return "\n".join(
            f"[{item['id']}] {item['speaker']}: {item['text']}" for item in message_list
        )

    # Estimate tokens (rough: 4 chars ≈ 1 token)
    formatted = [f"[{item['id']}] {item['speaker']}: {item['text']}" for item in message_list]
    total_chars = sum(len(line) for line in formatted)
    estimated_tokens = total_chars / 4

    if estimated_tokens <= max_tokens:
        return "\n".join(formatted)

    # Truncate from the beginning, keep most recent messages
    target_chars = int(max_tokens * 4 * 0.95)  # Keep 95% to leave margin
    kept = []
    current_chars = 0

    for i in range(len(formatted) - 1, -1, -1):
        line_len = len(formatted[i]) + 1  # +1 for newline
        if current_chars + line_len > target_chars and kept:
            break
        kept.insert(0, formatted[i])
        current_chars += line_len

    return "\n".join(kept)


def format_memories_with_metadata(memories: Iterable[dict[str, Any]], max_tokens: int | None = None) -> tuple[str, dict[str, Any]]:
    """Format memories with timestamp and speaker information for context.

    Args:
        memories: Iterable of memory dicts with memory and metadata fields
        max_tokens: Maximum approximate tokens (estimated as 4 chars per token).
                   If None, no truncation is applied.

    Returns:
        Tuple of (formatted_string, truncation_info) where truncation_info contains:
        - truncated: bool, whether truncation occurred
        - total_memories: int, total number of memories
        - kept_memories: int, number of memories kept after truncation
        - truncated_speakers: set of speaker names whose memories were truncated
    """
    memory_list = list(memories)
    truncation_info = {
        "truncated": False,
        "total_memories": len(memory_list),
        "kept_memories": len(memory_list),
        "truncated_speakers": set(),
    }

    lines = []
    for mem in memory_list:
        timestamp = mem.get("metadata", {}).get("time", "")
        speaker_tags = mem.get("metadata", {}).get("tag", [])
        speaker = ""
        for tag in speaker_tags:
            if tag.startswith("speaker:"):
                speaker = tag.replace("speaker:", "")
                break

        # Format: [timestamp] speaker: memory
        if timestamp and speaker:
            lines.append((f"[{timestamp}] {speaker}: {mem['memory']}", speaker))
        elif speaker:
            lines.append((f"{speaker}: {mem['memory']}", speaker))
        else:
            lines.append((f"- {mem['memory']}", ""))

    if max_tokens is None:
        return "\n".join(line for line, _ in lines), truncation_info

    # Estimate tokens (rough: 4 chars ≈ 1 token)
    total_chars = sum(len(line) for line, _ in lines)
    estimated_tokens = total_chars / 4

    if estimated_tokens <= max_tokens:
        return "\n".join(line for line, _ in lines), truncation_info

    # Truncate from the beginning, keep most recent memories
    truncation_info["truncated"] = True
    target_chars = int(max_tokens * 4 * 0.95)  # Keep 95% to leave margin
    kept = []
    current_chars = 0

    for i in range(len(lines) - 1, -1, -1):
        line, speaker = lines[i]
        line_len = len(line) + 1  # +1 for newline
        if current_chars + line_len > target_chars and kept:
            # This memory and all before it are truncated
            if speaker:
                truncation_info["truncated_speakers"].add(speaker)
        else:
            kept.insert(0, line)
            current_chars += line_len

    # Add all speakers from truncated memories
    for i in range(len(lines) - len(kept)):
        _, speaker = lines[i]
        if speaker:
            truncation_info["truncated_speakers"].add(speaker)

    truncation_info["kept_memories"] = len(kept)

    return "\n".join(kept), truncation_info
    """Format memories with timestamp and speaker information for context."""
    lines = []
    for mem in memories:
        timestamp = mem.get("metadata", {}).get("time", "")
        speaker_tags = mem.get("metadata", {}).get("tag", [])
        speaker = ""
        for tag in speaker_tags:
            if tag.startswith("speaker:"):
                speaker = tag.replace("speaker:", "")
                break

        # Format: [timestamp] speaker: memory
        if timestamp and speaker:
            lines.append(f"[{timestamp}] {speaker}: {mem['memory']}")
        elif speaker:
            lines.append(f"{speaker}: {mem['memory']}")
        else:
            lines.append(f"- {mem['memory']}")
    return "\n".join(lines)


def build_answer_messages(
    question: str,
    context: str,
    prompt: str = ANSWER_PROMPT,
    current_date: str | None = None,
) -> list[dict[str, str]]:
    if current_date is None:
        current_date = datetime.now().strftime("%Y-%m-%d")
    return [
        {
            "role": "system",
            "content": prompt.format(question=question, context=context, current_date=current_date),
        },
        {"role": "user", "content": "Return the answer now, as the JSON object described above."},
    ]


def build_answer_prompt(
    question: str,
    context: str,
    prompt: str = ANSWER_PROMPT,
    current_date: str | None = None,
) -> str:
    """Render the answer prompt into a single string (for logging or debugging)."""
    if current_date is None:
        current_date = datetime.now().strftime("%Y-%m-%d")
    return prompt.format(question=question, context=context, current_date=current_date)


def parse_answer_output(text: str) -> dict[str, Any]:
    """Parse a model answer response into a unified dict.

    The answer prompts emit ``{"answer": ..., "unsupported": ..., "reasoning": ...}``.
    Older or fallback responses may instead use ``<answer>...</answer>`` tags, or plain
    prose. Returns a dict with at least:
    - answer: str | None (None means the memories did not support an answer)
    - unsupported: bool
    - reasoning: str
    - raw: the original model output
    """
    raw = text or ""
    parsed = _parse_answer_json(raw)
    if parsed is not None:
        return _normalize_answer_dict(parsed, raw)

    # Fallback 1: legacy <answer>...</answer> tag
    tag_match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.DOTALL | re.IGNORECASE)
    if tag_match:
        answer = tag_match.group(1).strip()
        return _normalize_answer_dict({"answer": answer}, raw)

    # Fallback 2: treated as plain text; strip any stray tags/fences
    cleaned = raw.strip().strip("`").strip()
    return _normalize_answer_dict({"answer": cleaned}, raw)


def _parse_answer_json(raw: str) -> dict[str, Any] | None:
    """Best-effort extraction of the answer JSON object from a model response."""
    if not raw.strip():
        return None
    candidate = raw.strip()
    # Strip markdown fences.
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    for text in (candidate, atommem._repair_json(candidate)):
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return {"answer": value}
    # Last resort: grab the outermost {...} block.
    start, end = candidate.find("{"), candidate.rfind("}")
    if 0 <= start < end:
        block = candidate[start : end + 1]
        for text in (block, atommem._repair_json(block)):
            try:
                value = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(value, dict):
                return value
    return None


def _normalize_answer_dict(parsed: dict[str, Any], raw: str) -> dict[str, Any]:
    answer = parsed.get("answer", parsed.get("prediction", parsed.get("final_answer")))
    if isinstance(answer, str):
        answer = answer.strip()
    if answer == "":
        answer = None
    unsupported = parsed.get("unsupported")
    if not isinstance(unsupported, bool):
        unsupported = answer is None
    reasoning = parsed.get("reasoning", parsed.get("reason", parsed.get("explanation", "")))
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    return {
        "answer": answer,
        "unsupported": bool(unsupported),
        "reasoning": reasoning.strip(),
        "raw": raw,
    }


async def complete(messages: list[dict[str, str]], config: dict[str, Any]) -> str:
    client = openai.AsyncOpenAI(
        base_url=config["base_url"],
        api_key=config.get("api_key") or "not-needed",
        timeout=float(config.get("timeout", 300)),
        max_retries=0,
    )
    request_config = dict(DEFAULT_LLM_CONFIG)
    request_config.update(config)
    try:
        return await atommem.chat_completion(client, request_config, messages)
    finally:
        await client.close()


async def complete_with_client(
    client: openai.AsyncOpenAI,
    messages: list[dict[str, str]],
    config: dict[str, Any],
) -> str:
    request_config = dict(DEFAULT_LLM_CONFIG)
    request_config.update(config)
    return await atommem.chat_completion(client, request_config, messages)


def complete_sync(messages: list[dict[str, str]], config: dict[str, Any]) -> str:
    return asyncio.run(complete(messages, config))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def retrieve_by_embedding(
    query: str,
    messages: list[dict[str, Any]],
    model_name: str,
    top_k: int,
    embedding_client: Any | None = None,
) -> list[dict[str, Any]]:
    if embedding_client is None:
        embedding_client = openai.OpenAI(base_url="http://127.0.0.1:30001/v1", api_key="sglang")
    texts = [item["text"] for item in messages]
    try:
        response = embedding_client.embeddings.create(model=model_name, input=[query, *texts])
    except Exception as exc:  # noqa: BLE001
        _raise_embedding_error(exc)
    vectors = [item.embedding for item in response.data]
    query_vector = vectors[0]
    scored = [
        (cosine_similarity(query_vector, vector), item)
        for vector, item in zip(vectors[1:], messages)
    ]
    return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:top_k]]


def interleave_retrieved(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge two similarity-sorted retrieval lists into one, preserving rank order.

    Both inputs are assumed to be ordered best-first. Interleaving keeps each list's
    own ranking intact while avoiding a positional bias that concatenating (all of A
    then all of B) would introduce in the merged context.
    """
    merged: list[dict[str, Any]] = []
    for index in range(max(len(first), len(second))):
        if index < len(first):
            merged.append(first[index])
        if index < len(second):
            merged.append(second[index])
    return merged


class EmbeddingUnavailable(RuntimeError):
    """Embedding 端点不可用（连接失败/超时/未按 embedding 模式启动）。

    这类错误单独成类，方便调用方快速失败，而不是在每条 QA 上反复退避重试——
    那种情况下进度条会长时间停在 0，看起来像"卡住"。
    """


def _raise_embedding_error(exc: Exception) -> None:
    """把 embedding 调用异常转成带有排查提示的 EmbeddingUnavailable。"""
    message = str(exc)
    if "--is-embedding" in message or "embedding model" in message.lower():
        raise EmbeddingUnavailable(
            "Embedding API 未按 embedding 模式启动，请在 sglang 命令中加入 --is-embedding"
        ) from exc
    raise EmbeddingUnavailable(
        f"Embedding 端点不可用：{type(exc).__name__}: {message[:200]}\n"
        "  请检查 embedding.base_url / api_key / model，并确认该服务可访问。"
    ) from exc


async def check_embedding_endpoint(client: Any, model_name: str) -> None:
    """启动前探测 embedding 端点，尽早暴露配置错误。

    失败直接抛 EmbeddingUnavailable，避免大批任务各自退避重试导致进度条长时间停在 0。
    """
    try:
        await client.embeddings.create(model=model_name, input=["ping"])
    except Exception as exc:  # noqa: BLE001
        _raise_embedding_error(exc)


def memory_text(item: dict[str, Any]) -> str:
    """取出用于 embedding 的文本，兼容 memory 格式与 message 格式。"""
    if "memory" in item:
        return item["memory"]
    if "text" in item:
        return item["text"]
    return str(item)


class EmbeddingCache:
    """把 (model, text) -> vector 缓存在内存里，避免同一批记忆被反复嵌入。

    之前的实现每条 QA 都调用一次 embeddings.create，把该样本的全部记忆重新嵌入一遍，
    1986 条 QA 会产生约 4000 次调用、重复嵌入大量完全相同的文本。这里缓存之后，
    同一样本的记忆只在第一次用到时嵌入一次。

    磁盘缓存可选：key 里带上文本内容的哈希，源文件一旦变化会重新嵌入，不会读到过期向量。
    """

    def __init__(self, client: Any, model_name: str, disk_dir: Path | None = None) -> None:
        self.client = client
        self.model_name = model_name
        self.disk_dir = disk_dir
        self.vectors: dict[str, list[float]] = {}
        self.hits = 0
        self.misses = 0
        self.requests = 0
        self._disk_dirty = False
        if disk_dir is not None:
            self._load_disk()

    def _cache_key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model_name}\x00{text}".encode("utf-8")).hexdigest()

    def _disk_path(self) -> Path | None:
        if self.disk_dir is None:
            return None
        return self.disk_dir / "embedding_cache.jsonl"

    def _load_disk(self) -> None:
        path = self._disk_path()
        if path is None or not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        key, vector = record["key"], record["vector"]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
                    if isinstance(key, str) and isinstance(vector, list) and vector:
                        self.vectors[key] = vector
        except OSError as exc:
            print(f"[warn] 无法读取 embedding 缓存 {path}: {exc}", file=sys.stderr)
        if self.vectors:
            print(f"[cache] 从 {path} 载入 {len(self.vectors)} 条向量")

    def _save_disk(self) -> None:
        if not self._disk_dirty:
            return
        path = self._disk_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as file:
                for key, vector in self.vectors.items():
                    file.write(json.dumps({"key": key, "vector": vector}) + "\n")
        except OSError as exc:
            print(f"[warn] 无法写入 embedding 缓存 {path}: {exc}", file=sys.stderr)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """返回与 texts 等长的向量列表，命中缓存的不再请求。"""
        keys = [self._cache_key(text) for text in texts]

        # 同一批里去重，只请求一次未命中的文本（QA 之间常有完全相同的记忆）。
        pending: dict[str, str] = {}
        for key, text in zip(keys, texts):
            if key in self.vectors:
                self.hits += 1
            else:
                pending.setdefault(key, text)

        if pending:
            self.misses += len(pending)
            self.requests += 1
            unique_keys = list(pending)
            try:
                response = await self.client.embeddings.create(
                    model=self.model_name, input=[pending[key] for key in unique_keys]
                )
            except Exception as exc:  # noqa: BLE001
                _raise_embedding_error(exc)
            if len(response.data) != len(unique_keys):
                raise EmbeddingUnavailable(
                    f"embedding 返回数量不匹配：请求 {len(unique_keys)} 条，返回 {len(response.data)} 条"
                )
            for key, item in zip(unique_keys, response.data):
                self.vectors[key] = item.embedding
            self._disk_dirty = True

        return [self.vectors[key] for key in keys]

    def stats(self) -> str:
        total = self.hits + self.misses
        ratio = (self.hits / total * 100) if total else 0.0
        return (
            f"embedding 缓存: 命中 {self.hits}/{total} ({ratio:.1f}%), "
            f"实际请求 {self.requests} 次"
        )

    def save(self) -> None:
        self._save_disk()


def rank_by_query_vector(
    query_vector: list[float],
    memories: list[dict[str, Any]],
    vectors: list[list[float]],
    top_k: int,
) -> list[dict[str, Any]]:
    """按 query 向量与记忆向量的余弦相似度取 top_k（检索的唯一实现）。"""
    scored = [
        (cosine_similarity(query_vector, vector), item)
        for vector, item in zip(vectors, memories)
    ]
    return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:top_k]]


async def retrieve_by_embedding_async(
    query: str,
    memories: list[dict[str, Any]],
    model_name: str,
    top_k: int,
    embedding_client: Any,
    cache: "EmbeddingCache | None" = None,
) -> list[dict[str, Any]]:
    """按 embedding 相似度检索 top_k。

    传入 cache 时，记忆向量走缓存（同一批记忆只嵌入一次）；否则保持旧行为
    （每次调用把 query 和全部记忆一起嵌入）。
    """
    if cache is not None:
        query_vector, *memory_vectors = await cache.embed([query, *[memory_text(m) for m in memories]])
        return rank_by_query_vector(query_vector, memories, memory_vectors, top_k)

    # Extract text from memory objects - handle both memory format and message format
    texts = [memory_text(item) for item in memories]

    try:
        response = await embedding_client.embeddings.create(model=model_name, input=[query, *texts])
    except Exception as exc:  # noqa: BLE001
        _raise_embedding_error(exc)
    vectors = [item.embedding for item in response.data]
    scored = [
        (cosine_similarity(vectors[0], vector), item)
        for vector, item in zip(vectors[1:], memories)
    ]
    return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:top_k]]


async def retrieve_by_embedding_async_old(
    query: str,
    messages: list[dict[str, Any]],
    model_name: str,
    top_k: int,
    embedding_client: Any,
) -> list[dict[str, Any]]:
    texts = [item["text"] for item in messages]
    try:
        response = await embedding_client.embeddings.create(model=model_name, input=[query, *texts])
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "--is-embedding" in message or "embedding model" in message.lower():
            raise RuntimeError(
                "Embedding API 未按 embedding 模式启动，请在 sglang 命令中加入 --is-embedding"
            ) from exc
        raise
    vectors = [item.embedding for item in response.data]
    scored = [
        (cosine_similarity(vectors[0], vector), item)
        for vector, item in zip(vectors[1:], messages)
    ]
    return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:top_k]]
