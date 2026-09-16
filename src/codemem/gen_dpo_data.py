"""
生成原子记忆抽取模型的 DPO 训练数据。

每个 raw message memory 构造一对偏好数据：
    - chosen   : 由较强的模型（configs/model.yaml 里的 dpo.strong）按 atommem 的 prompt 生成
    - rejected : 由待训练的小模型（dpo.weak）按同一 prompt 生成
两者使用完全相同的 system/user prompt，只有模型不同，保证 DPO 的对比只来自模型能力差异。

输出 JSONL，每行一条（TRL / LLaMA-Factory 兼容的 messages 格式）：

    {
      "messages": [{"role": "system", ...}, {"role": "user", ...}],
      "chosen":   "<强模型输出的 JSON 字符串>",
      "rejected": "<小模型输出的 JSON 字符串>",
      "meta": {"id": "session_1_2", "dir": "Caroline_Melanie", "chosen_model": ..., "rejected_model": ...}
    }

用法：
    python -m codemem.gen_dpo_data                      # 处理 data/ 下全部 speaker 目录
    python -m codemem.gen_dpo_data Caroline_Melanie     # 只处理指定目录
    python -m codemem.gen_dpo_data --limit 50           # 每个目录只取前 50 条 raw（调试）
    python -m codemem.gen_dpo_data --include-identical  # 保留 chosen==rejected 的样本
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import openai
import yaml

from . import atommem
from .prompts import ATOM_EXTRACTION_SYSTEM_PROMPT

PROJECT_ROOT = atommem.PROJECT_ROOT
DATA_DIR = atommem.DATA_DIR
CONFIG_FILE = atommem.CONFIG_FILE

DEFAULT_DPO_CONFIG: dict[str, Any] = {
    "output": "data/dpo/atom_dpo.jsonl",
    "error_log": "data/dpo/atom_dpo_errors.log",
    "strong": {"base_url": "", "api_key": "", "model": "", "temperature": 0.2},
    "weak": {"base_url": "", "api_key": "", "model": "", "temperature": 0.7},
}


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def load_dpo_config() -> tuple[dict, dict]:
    """返回 (base 配置, dpo 配置)。base 用于并发/超时/重试等通用参数。"""
    base = atommem.load_config()
    dpo = json.loads(json.dumps(DEFAULT_DPO_CONFIG))  # 深拷贝
    if CONFIG_FILE.exists():
        loaded = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")) or {}
        user_dpo = loaded.get("dpo") or {}
        for key, value in user_dpo.items():
            if key in ("strong", "weak") and isinstance(value, dict):
                dpo[key].update(value)
            else:
                dpo[key] = value
    return base, dpo


def _profile_config(base: dict, profile: dict) -> dict:
    """把某个模型 profile 合并进 base 配置，得到 atommem.chat_completion 可用的 config。"""
    config = dict(base)
    for key in ("base_url", "api_key", "model", "temperature"):
        if profile.get(key) not in (None, ""):
            config[key] = profile[key]
    return config


# ---------------------------------------------------------------------------
# 单条样本生成
# ---------------------------------------------------------------------------

async def _generate_one(
    client: openai.AsyncOpenAI, config: dict, messages: list[dict]
) -> str | None:
    try:
        return await atommem.chat_completion(client, config, messages)
    except Exception:  # noqa: BLE001 - 单条失败不影响整体
        return None


def _valid_json(text: str | None) -> bool:
    """基本校验：能解析成含 atoms 的 JSON 才算有效输出。"""
    if not text or not text.strip():
        return False
    try:
        parsed = atommem.extract_json(text)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(parsed.get("atoms"), list)


def _normalize_text(text: str) -> str:
    """用于判断 chosen/rejected 是否实质相同的归一化。"""
    try:
        parsed = atommem.extract_json(text)
    except Exception:  # noqa: BLE001
        return text.strip()
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True)


async def build_pair(
    strong_client: openai.AsyncOpenAI,
    weak_client: openai.AsyncOpenAI,
    strong_config: dict,
    weak_config: dict,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    raws: list[dict],
    index: int,
    window: int,
) -> tuple[dict | None, str | None]:
    """生成一条 (样本, 错误信息)。chosen 来自强模型，rejected 来自小模型。"""
    raw_id = raws[index]["metadata"]["id"]
    messages = atommem.build_messages(system_prompt, raws, index, window)

    async with semaphore:
        chosen, rejected = await asyncio.gather(
            _generate_one(strong_client, strong_config, messages),
            _generate_one(weak_client, weak_config, messages),
        )

    if chosen is None or not _valid_json(chosen):
        return None, f"{raw_id}: strong model produced invalid JSON"
    if rejected is None or not _valid_json(rejected):
        return None, f"{raw_id}: weak model produced invalid JSON"

    return {
        "messages": messages,
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
        "meta": {
            "id": raw_id,
            "chosen_model": strong_config["model"],
            "rejected_model": weak_config["model"],
        },
    }, None


# ---------------------------------------------------------------------------
# 目录处理
# ---------------------------------------------------------------------------

async def process_dir(
    strong_client: openai.AsyncOpenAI,
    weak_client: openai.AsyncOpenAI,
    base_config: dict,
    strong_config: dict,
    weak_config: dict,
    dir_name: str,
    limit: int | None,
) -> tuple[list[dict], list[str]]:
    directory = atommem.resolve_dir(dir_name)
    label = directory.name
    memory_file = directory / "memory.jsonl"
    if not memory_file.exists():
        print(f"[skip] {label}: no memory.jsonl")
        return [], []

    existing = atommem.read_jsonl(memory_file)
    raws = [item for item in existing if item["metadata"].get("type") == "raw"]
    if limit is not None:
        raws = raws[:limit]
    if not raws:
        print(f"[skip] {label}: no raw memory")
        return [], []

    window = int(base_config["context_window"])
    system_prompt = ATOM_EXTRACTION_SYSTEM_PROMPT.replace(
        "{{SCHEMA}}", atommem.build_schema_prompt(atommem.load_template())
    )
    semaphore = asyncio.Semaphore(int(base_config["concurrency"]))

    print(f"[run ] {label}: {len(raws)} raw messages, concurrency={base_config['concurrency']}, window={window}")

    tasks = [
        asyncio.create_task(
            build_pair(
                strong_client, weak_client, strong_config, weak_config,
                semaphore, system_prompt, raws, i, window,
            )
        )
        for i in range(len(raws))
    ]

    samples: list[dict] = []
    errors: list[str] = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        sample, error = await coro
        done += 1
        if error:
            errors.append(error)
        elif sample:
            sample["meta"]["dir"] = label
            samples.append(sample)
        if done % 20 == 0 or done == len(raws):
            print(f"        {done}/{len(raws)} pairs processed")

    # 保持原始顺序
    order = {r["metadata"]["id"]: i for i, r in enumerate(raws)}
    samples.sort(key=lambda s: order.get(s["meta"]["id"], 0))
    return samples, errors


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


async def run(base_config: dict, dpo_config: dict, targets: list[str], limit: int | None,
              include_identical: bool) -> None:
    if targets:
        dir_names = targets
    else:
        dir_names = sorted(p.name for p in DATA_DIR.iterdir() if (p / "memory.jsonl").exists())

    strong_config = _profile_config(base_config, dpo_config["strong"])
    weak_config = _profile_config(base_config, dpo_config["weak"])
    if not strong_config.get("model") or not weak_config.get("model"):
        raise SystemExit("dpo.strong / dpo.weak 未配置 model，请检查 configs/model.yaml")

    print(f"strong: {strong_config['model']} @ {strong_config['base_url']}")
    print(f"weak  : {weak_config['model']} @ {weak_config['base_url']}")

    strong_client = openai.AsyncOpenAI(
        base_url=strong_config["base_url"],
        api_key=strong_config.get("api_key") or "not-needed",
        timeout=float(base_config["timeout"]),
        max_retries=0,
    )
    weak_client = openai.AsyncOpenAI(
        base_url=weak_config["base_url"],
        api_key=weak_config.get("api_key") or "not-needed",
        timeout=float(base_config["timeout"]),
        max_retries=0,
    )

    all_samples: list[dict] = []
    all_errors: list[str] = []
    try:
        for dir_name in dir_names:
            samples, errors = await process_dir(
                strong_client, weak_client, base_config,
                strong_config, weak_config, dir_name, limit,
            )
            all_samples.extend(samples)
            all_errors.extend(errors)
    finally:
        await strong_client.close()
        await weak_client.close()

    # 过滤 chosen == rejected 的样本（没有偏好信号）
    kept = all_samples
    dropped = 0
    if not include_identical:
        kept = [s for s in all_samples if _normalize_text(s["chosen"]) != _normalize_text(s["rejected"])]
        dropped = len(all_samples) - len(kept)

    output = PROJECT_ROOT / dpo_config["output"]
    _write_jsonl(output, kept)
    print(
        f"\n[done] {len(kept)} DPO pairs -> {output}"
        + (f"（丢弃 {dropped} 条 chosen==rejected）" if dropped else "")
    )

    if all_errors:
        error_file = PROJECT_ROOT / dpo_config["error_log"]
        error_file.parent.mkdir(parents=True, exist_ok=True)
        error_file.write_text("\n".join(all_errors) + "\n", encoding="utf-8")
        print(f"[warn] {len(all_errors)} failed pair(s) -> {error_file}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成原子记忆抽取的 DPO 偏好数据")
    parser.add_argument("dirs", nargs="*", help="speaker 目录名；缺省处理全部")
    parser.add_argument("--limit", type=int, default=None, help="每个目录只处理前 N 条 raw（调试用）")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖配置里的并发数")
    parser.add_argument("--context-window", type=int, default=None, help="覆盖配置里的上下文窗口大小")
    parser.add_argument("--output", type=str, default=None, help="覆盖输出文件路径")
    parser.add_argument("--include-identical", action="store_true", help="保留 chosen==rejected 的样本")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    base_config, dpo_config = load_dpo_config()
    if args.concurrency is not None:
        base_config["concurrency"] = args.concurrency
    if args.context_window is not None:
        base_config["context_window"] = args.context_window
    if args.output is not None:
        dpo_config["output"] = args.output
    asyncio.run(
        run(base_config, dpo_config, args.dirs, args.limit, args.include_identical)
    )


if __name__ == "__main__":
    main()
