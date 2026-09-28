"""
生成原子记忆抽取模型的 DPO 训练数据。

每个 raw message memory 构造一对偏好数据：
    - chosen   : 由较强的模型（configs/model.yaml 里的 dpo.strong）+ 完整 prompt 生成
    - rejected : 由待训练的小模型（dpo.weak）+ 简化 prompt 生成；若重试后仍无法解析出
                 合法 JSON，则保留其原始输出作为负样本（这正是 DPO 要抑制的行为）
    - messages : 存强模型使用的完整 prompt，使训练输入遵循部署时的完整规则
强模型用完整 prompt，小模型用简化 prompt（prompts.WEAK_ATOM_EXTRACTION_SYSTEM_PROMPT）。

输出 JSONL，每行一条（TRL / LLaMA-Factory 兼容的 messages 格式）：

    {
      "messages": [{"role": "system", ...}, {"role": "user", ...}],
      "chosen":   "<强模型输出的 JSON 字符串>",
      "rejected": "<小模型输出的 JSON 字符串，解析失败时为原始错误输出>",
      "meta": {"id": "session_1_2", "dir": "Caroline_Melanie",
               "chosen_model": ..., "rejected_model": ..., "rejected_parse_failed": false}
    }

用法：
    python -m codemem.add --stage dpo                      # 处理 data/ 下全部 speaker 目录
    python -m codemem.add --stage dpo Caroline_Melanie     # 只处理指定目录
    python -m codemem.add --stage dpo --limit 50           # 每个目录只取前 50 条 raw（调试）
    python -m codemem.add --stage dpo --resume             # 跳过已生成过的样本，追加到已有输出
    python -m codemem.add --stage dpo --include-identical  # 保留 chosen==rejected 的样本
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import openai
import yaml

from .. import llm
from ..io import DATA_DIR, PROJECT_ROOT, append_jsonl, read_jsonl, resolve_dir
from ..prompts import ATOM_EXTRACTION_SYSTEM_PROMPT, WEAK_ATOM_EXTRACTION_SYSTEM_PROMPT
from . import atommem

DPO_CONFIG_FILE = PROJECT_ROOT / "configs" / "add.yaml"

DEFAULT_DPO_CONFIG: dict[str, Any] = {
    "strong": {"base_url": "", "api_key": "", "model": "", "temperature": 0.2, "concurrency": 4},
    "weak": {"base_url": "", "api_key": "", "model": "", "temperature": 0.7, "concurrency": 1},
    "max_tokens": 8192,
    "timeout": 300,
    "max_retries": 6,
    "backoff_base": 2.0,
    "rate_limit_backoff": 10.0,
    "max_backoff": 120.0,
    "backoff_jitter": 0.5,
    "context_window": 4,
    "enable_thinking": False,
    "output": "data/dpo/atom_dpo.jsonl",
    "error_log": "data/dpo/atom_dpo_errors.log",
}


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def load_dpo_config() -> dict:
    """读取 configs/dpo_data.yaml，缺失字段用 DEFAULT_DPO_CONFIG 补齐。"""
    config = json.loads(json.dumps(DEFAULT_DPO_CONFIG))  # 深拷贝
    if DPO_CONFIG_FILE.exists():
        loaded = yaml.safe_load(DPO_CONFIG_FILE.read_text(encoding="utf-8")) or {}
        for key, value in loaded.items():
            if key in ("strong", "weak") and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    return config


def _model_config(dpo_config: dict, which: str) -> dict:
    """把某个模型 profile + 通用参数组装成 ``llm.chat_completion`` 可用的 config。"""
    profile = dpo_config[which]
    config = {
        "base_url": profile.get("base_url", ""),
        "api_key": profile.get("api_key", ""),
        "model": profile.get("model", ""),
        "temperature": profile.get("temperature", 0.2),
        "enable_thinking": dpo_config.get("enable_thinking", False),
    }
    for key in ("max_tokens", "timeout", "max_retries", "backoff_base",
                "rate_limit_backoff", "max_backoff", "backoff_jitter"):
        config[key] = dpo_config[key]
    return config


# ---------------------------------------------------------------------------
# 单条样本生成
# ---------------------------------------------------------------------------

async def _generate_one(
    client: openai.AsyncOpenAI, config: dict, messages: list[dict]
) -> str | None:
    try:
        return await llm.chat_completion(client, config, messages)
    except Exception:  # noqa: BLE001 - 单条失败不影响整体
        return None


def _valid_json(text: str | None) -> bool:
    """基本校验：能解析成含 atoms 的 JSON 才算有效输出。"""
    if not text or not text.strip():
        return False
    try:
        parsed = atommem.extract_json(text)
        # 确保 atoms 字段存在且为列表（可以为空列表）
        return isinstance(parsed.get("atoms"), list)
    except Exception:  # noqa: BLE001
        return False


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
    strong_semaphore: asyncio.Semaphore,
    weak_semaphore: asyncio.Semaphore,
    strong_prompt: str,
    weak_prompt: str,
    raws: list[dict],
    index: int,
    window: int,
) -> tuple[dict | None, str | None]:
    """生成一条 (样本, 错误信息)。

    - chosen  : 强模型 + 完整 prompt
    - rejected: 小模型 + 简化 prompt；若重试后仍解析失败，保留其原始输出作为负样本
    - messages: 存强模型使用的完整 prompt，作为训练时的输入 prompt
    强/弱模型各自用自己的并发信号量。
    """
    raw_id = raws[index]["metadata"]["id"]
    strong_messages = atommem.build_messages(strong_prompt, raws, index, window)
    weak_messages = atommem.build_messages(weak_prompt, raws, index, window)

    async def _strong():
        async with strong_semaphore:
            return await _generate_one(strong_client, strong_config, strong_messages)

    async def _weak():
        async with weak_semaphore:
            return await _generate_one(weak_client, weak_config, weak_messages)

    chosen, rejected = await asyncio.gather(_strong(), _weak())

    # 强模型必须产出合法 JSON，否则这条没有可靠的正样本
    if chosen is None or not _valid_json(chosen):
        error_detail = f"chosen output: {(chosen or '')[:300]!r}" if chosen else "chosen is None"
        return None, f"{raw_id}: strong model produced invalid JSON ({error_detail})"

    # 小模型失败（空输出 / JSON 无法解析）时，原始输出本身就是一个有效的 rejected 负样本
    weak_failed = rejected is None or not _valid_json(rejected)

    return {
        "messages": strong_messages,
        "chosen": chosen.strip(),
        "rejected": (rejected or "").strip(),
        "meta": {
            "id": raw_id,
            "chosen_model": strong_config["model"],
            "rejected_model": weak_config["model"],
            # 标记该 rejected 是解析失败的原始输出（DPO 中正是要抑制的行为）
            "rejected_parse_failed": weak_failed,
        },
    }, None


# ---------------------------------------------------------------------------
# 目录处理
# ---------------------------------------------------------------------------

async def process_dir(
    strong_client: openai.AsyncOpenAI,
    weak_client: openai.AsyncOpenAI,
    dpo_config: dict,
    strong_config: dict,
    weak_config: dict,
    dir_name: str,
    limit: int | None,
    done_keys: set[tuple[str, str]],
    output: Path,
    include_identical: bool,
) -> tuple[list[dict], list[str]]:
    directory = resolve_dir(dir_name)
    label = directory.name
    msgmem_file = directory / "msgmem.jsonl"
    legacy_memory_file = directory / "memory.jsonl"
    source_file = msgmem_file if msgmem_file.exists() else legacy_memory_file
    if not source_file.exists():
        print(f"[skip] {label}: no msgmem.jsonl")
        return [], []

    existing = read_jsonl(source_file)
    raws = [item for item in existing if item["metadata"].get("type") == "raw"]
    if limit is not None:
        raws = raws[:limit]
    if not raws:
        print(f"[skip] {label}: no raw memory")
        return [], []

    # resume：跳过已经生成过的样本
    pending = [(i, r) for i, r in enumerate(raws) if (label, r["metadata"]["id"]) not in done_keys]
    skipped = len(raws) - len(pending)
    if skipped:
        print(f"[skip] {label}: {skipped} 条已在输出中，跳过")
    if not pending:
        print(f"[skip] {label}: 全部已生成")
        return [], []

    window = int(dpo_config["context_window"])
    schema = atommem.build_schema_prompt(atommem.load_template())
    strong_prompt = ATOM_EXTRACTION_SYSTEM_PROMPT.replace("{{SCHEMA}}", schema)
    weak_prompt = WEAK_ATOM_EXTRACTION_SYSTEM_PROMPT.replace("{{SCHEMA}}", schema)
    strong_semaphore = asyncio.Semaphore(int(dpo_config["strong"].get("concurrency", 4)))
    weak_semaphore = asyncio.Semaphore(int(dpo_config["weak"].get("concurrency", 1)))

    print(
        f"[run ] {label}: {len(pending)} raw messages "
        f"(strong并发={dpo_config['strong'].get('concurrency', 4)}, "
        f"weak并发={dpo_config['weak'].get('concurrency', 1)}, window={window})"
    )

    tasks = [
        asyncio.create_task(
            build_pair(
                strong_client, weak_client, strong_config, weak_config,
                strong_semaphore, weak_semaphore, strong_prompt, weak_prompt,
                raws, i, window,
            )
        )
        for i, _ in pending
    ]

    samples: list[dict] = []
    errors: list[str] = []
    filtered_identical = 0
    done = 0
    for coro in asyncio.as_completed(tasks):
        sample, error = await coro
        done += 1
        if error:
            errors.append(error)
        elif sample:
            sample["meta"]["dir"] = label
            # 实时追加：每生成一条就立即写入到 output
            kept = _filter_identical([sample], include_identical)
            if kept:
                samples.append(sample)
                append_jsonl(output, kept)
                done_keys.add((sample["meta"]["dir"], sample["meta"]["id"]))
            else:
                filtered_identical += 1
        if done % 20 == 0 or done == len(pending):
            print(f"        {done}/{len(pending)} pairs processed, {len(samples)} written")

    if filtered_identical:
        print(
            f"[skip] {label}: {filtered_identical} 条 chosen/rejected 相同，"
            "未写入（如需保留请使用 --include-identical）"
        )

    # 保持原始顺序（用于返回统计信息）
    order = {r["metadata"]["id"]: i for i, r in enumerate(raws)}
    samples.sort(key=lambda s: order.get(s["meta"]["id"], 0))
    return samples, errors


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def _load_done_keys(
    path: Path, expected_models: tuple[str, str] | None = None
) -> tuple[set[tuple[str, str]], list[dict]]:
    """读取已有输出，返回 (已生成样本的 (dir, id) 集合, 已有样本列表)。

    注意：message id 只在单个 speaker 目录内唯一（不同目录都有 session_1_1），
    因此 resume 去重必须以 (dir, id) 为键。
    如果提供 expected_models，则要求已有样本全部由同一 strong/weak 模型对生成。
    """
    if not path.exists():
        return set(), []
    existing = []
    keys: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            existing.append(item)
            meta = item.get("meta") or {}
            sid = meta.get("id")
            sdir = meta.get("dir")
            if sid and sdir:
                keys.add((sdir, sid))
            if expected_models is not None:
                actual_models = (meta.get("chosen_model"), meta.get("rejected_model"))
                if actual_models != expected_models:
                    raise ValueError(
                        "已有 DPO 数据的模型与当前配置不一致："
                        f"已有 strong={actual_models[0]!r}, weak={actual_models[1]!r}; "
                        f"当前 strong={expected_models[0]!r}, weak={expected_models[1]!r}。"
                        "请更换输出文件，或使用与已有数据一致的模型配置。"
                    )
    return keys, existing


async def run(dpo_config: dict, targets: list[str], limit: int | None,
              include_identical: bool, resume: bool) -> None:
    if targets:
        dir_names = targets
    else:
        dir_names = sorted(
            p.name
            for p in DATA_DIR.iterdir()
            if (p / "msgmem.jsonl").exists() or (p / "memory.jsonl").exists()
        )

    strong_config = _model_config(dpo_config, "strong")
    weak_config = _model_config(dpo_config, "weak")
    if not strong_config.get("model") or not weak_config.get("model"):
        raise SystemExit("dpo_data.yaml 里的 strong/weak 未配置 model")

    output = PROJECT_ROOT / dpo_config["output"]
    print(f"strong: {strong_config['model']} @ {strong_config['base_url']}")
    print(f"weak  : {weak_config['model']} @ {weak_config['base_url']}")
    print(f"output: {output}")
    print(f"[info] 实时追加模式：每生成一条样本立即写入 {output}")

    # resume：载入已生成的 (dir, id)，跳过重复生成
    done_keys: set[tuple[str, str]] = set()
    existing_samples: list[dict] = []
    if resume:
        expected_models = (strong_config["model"], weak_config["model"])
        done_keys, existing_samples = _load_done_keys(output, expected_models)
        print(f"[resume] 已有 {len(done_keys)} 条样本，将跳过这些 message")
    else:
        # 非 resume 模式：清空旧输出，从头生成
        if output.exists():
            output.unlink()

    strong_client = openai.AsyncOpenAI(
        base_url=strong_config["base_url"],
        api_key=strong_config.get("api_key") or "not-needed",
        timeout=float(dpo_config["timeout"]),
        max_retries=0,
    )
    weak_client = openai.AsyncOpenAI(
        base_url=weak_config["base_url"],
        api_key=weak_config.get("api_key") or "not-needed",
        timeout=float(dpo_config["timeout"]),
        max_retries=0,
    )

    all_samples: list[dict] = []
    all_errors: list[str] = []
    try:
        for dir_name in dir_names:
            samples, errors = await process_dir(
                strong_client, weak_client, dpo_config,
                strong_config, weak_config, dir_name, limit, done_keys,
                output, include_identical,
            )
            all_samples.extend(samples)
            all_errors.extend(errors)
    finally:
        await strong_client.close()
        await weak_client.close()

    # 统计：已有 + 本次新增
    total = len(existing_samples) + len(all_samples)
    print(f"\n[done] 本次新增 {len(all_samples)} 条，输出共 {total} 条 -> {output}")
    if resume and existing_samples:
        print(f"[resume] 其中已有 {len(existing_samples)} 条")

    if all_errors:
        error_file = PROJECT_ROOT / dpo_config["error_log"]
        error_file.parent.mkdir(parents=True, exist_ok=True)
        error_file.write_text("\n".join(all_errors) + "\n", encoding="utf-8")
        print(f"[warn] {len(all_errors)} failed pair(s) -> {error_file}")


def _filter_identical(samples: list[dict], include_identical: bool) -> list[dict]:
    """过滤 chosen == rejected 的样本（没有偏好信号）。"""
    if include_identical:
        return samples
    return [s for s in samples if _normalize_text(s["chosen"]) != _normalize_text(s["rejected"])]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成原子记忆抽取的 DPO 偏好数据")
    parser.add_argument("dirs", nargs="*", help="speaker 目录名；缺省处理全部")
    parser.add_argument("--limit", type=int, default=None, help="每个目录只处理前 N 条 raw（调试用）")
    parser.add_argument("--strong-concurrency", type=int, default=None, help="覆盖强模型并发数")
    parser.add_argument("--weak-concurrency", type=int, default=None, help="覆盖小模型并发数")
    parser.add_argument("--context-window", type=int, default=None, help="覆盖配置里的上下文窗口大小")
    parser.add_argument("--output", type=str, default=None, help="覆盖输出文件路径")
    parser.add_argument("--resume", action="store_true", help="跳过已生成过的样本，追加到已有输出")
    parser.add_argument("--include-identical", action="store_true", help="保留 chosen==rejected 的样本")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    dpo_config = load_dpo_config()
    if args.strong_concurrency is not None:
        dpo_config["strong"]["concurrency"] = args.strong_concurrency
    if args.weak_concurrency is not None:
        dpo_config["weak"]["concurrency"] = args.weak_concurrency
    if args.context_window is not None:
        dpo_config["context_window"] = args.context_window
    if args.output is not None:
        dpo_config["output"] = args.output
    asyncio.run(
        run(dpo_config, args.dirs, args.limit, args.include_identical, args.resume)
    )


if __name__ == "__main__":
    main()
