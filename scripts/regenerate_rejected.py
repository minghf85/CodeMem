"""重新生成 DPO JSONL 中的 rejected，保留 chosen 和其它字段不变。

默认使用保存于每条样本中的 ``messages``，因此重跑时与原始 rejected
生成时的 prompt 分布一致。输出先写入临时文件，全部成功后再替换原文件。

用法：
	python scripts/regenerate_rejected.py
	python scripts/regenerate_rejected.py --limit 10 --concurrency 1
	python scripts/regenerate_rejected.py --input data/dpo/atom_dpo.jsonl \
		--output data/dpo/atom_dpo_regenerated.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

import openai

from codemem import atommem  # type: ignore[import-not-found]


DEFAULT_INPUT = PROJECT_ROOT / "data/dpo/atom_dpo.jsonl"
DEFAULT_OUTPUT = DEFAULT_INPUT
MODEL_CONFIG: dict[str, Any] = {
	"base_url": "http://127.0.0.1:30000/v1",
	"api_key": "sglang",
	"model": "/root/autodl-tmp/Train/model/Qwen3-4B",
	"temperature": 0.7,
	"max_tokens": 8192,
	"timeout": 300,
	"max_retries": 6,
	"backoff_base": 2.0,
	"rate_limit_backoff": 10.0,
	"max_backoff": 120.0,
	"backoff_jitter": 0.5,
	"enable_thinking": False,
}


def load_records(path: Path) -> list[dict[str, Any]]:
	records: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as file:
		for line_number, line in enumerate(file, 1):
			if not line.strip():
				continue
			try:
				record = json.loads(line)
			except json.JSONDecodeError as exc:
				raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
			if not isinstance(record, dict):
				raise ValueError(f"{path}:{line_number}: expected a JSON object")
			if not isinstance(record.get("messages"), list):
				raise ValueError(f"{path}:{line_number}: missing messages")
			records.append(record)
	return records


async def regenerate_one(
	client: openai.AsyncOpenAI,
	semaphore: asyncio.Semaphore,
	record: dict[str, Any],
	config: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
	async with semaphore:
		try:
			rejected = await atommem.chat_completion(client, config, record["messages"])
		except Exception as exc:  # noqa: BLE001 - 单条失败需保留原 rejected
			return record, f"{type(exc).__name__}: {exc}"

	updated = dict(record)
	updated["rejected"] = rejected.strip()
	meta = dict(updated.get("meta") or {})
	meta["rejected_model"] = config["model"]
	meta["rejected_parse_failed"] = not is_valid_atom_json(rejected)
	updated["meta"] = meta
	return updated, None


def is_valid_atom_json(text: str | None) -> bool:
	if not text or not text.strip():
		return False
	try:
		parsed = atommem.extract_json(text)
	except Exception:  # noqa: BLE001
		return False
	return isinstance(parsed.get("atoms"), list)


async def regenerate(
	records: list[dict[str, Any]],
	config: dict[str, Any],
	concurrency: int,
	limit: int | None,
) -> tuple[list[dict[str, Any]], list[str]]:
	client = openai.AsyncOpenAI(
		base_url=config["base_url"],
		api_key=config["api_key"],
		timeout=float(config["timeout"]),
		max_retries=0,
	)
	semaphore = asyncio.Semaphore(concurrency)
	selected = records if limit is None else records[:limit]
	untouched = records[len(selected):]
	async def regenerate_at(
		index: int, record: dict[str, Any]
	) -> tuple[int, dict[str, Any], str | None]:
		updated, error = await regenerate_one(client, semaphore, record, config)
		return index, updated, error

	tasks = [
		asyncio.create_task(regenerate_at(index, record))
		for index, record in enumerate(selected)
	]
	regenerated: dict[int, dict[str, Any]] = {}
	errors: list[str] = []
	try:
		for index, task in enumerate(asyncio.as_completed(tasks), 1):
			original_index, record, error = await task
			regenerated[original_index] = record
			if error:
				sample_id = (record.get("meta") or {}).get("id", f"index-{index}")
				errors.append(f"{sample_id}: {error}")
			if index % 20 == 0 or index == len(tasks):
				print(f"processed {index}/{len(tasks)}")
	finally:
		await client.close()

	# as_completed 返回完成顺序，按任务携带的原始位置恢复输出顺序。
	ordered = [regenerated[index] for index in range(len(selected))]
	return ordered + untouched, errors


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
	temporary = path.with_suffix(path.suffix + ".tmp")
	path.parent.mkdir(parents=True, exist_ok=True)
	with temporary.open("w", encoding="utf-8") as file:
		for record in records:
			file.write(json.dumps(record, ensure_ascii=False) + "\n")
	temporary.replace(path)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="重新生成 DPO 数据的 rejected 字段")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--concurrency", type=int, default=4)
	parser.add_argument("--limit", type=int, default=None, help="只重跑前 N 条")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.concurrency < 1:
		raise SystemExit("--concurrency must be >= 1")
	records = load_records(args.input)
	print(f"input: {args.input}")
	print(f"output: {args.output}")
	print(f"model: {MODEL_CONFIG['model']} (thinking disabled)")
	print(f"records: {len(records) if args.limit is None else min(args.limit, len(records))}/{len(records)}")

	updated, errors = asyncio.run(
		regenerate(records, MODEL_CONFIG, args.concurrency, args.limit)
	)
	write_jsonl_atomic(args.output, updated)
	print(f"written: {len(updated)} records")
	if errors:
		print(f"warning: {len(errors)} request(s) failed; original rejected was retained")
		for error in errors:
			print(f"  {error}")


if __name__ == "__main__":
	main()
