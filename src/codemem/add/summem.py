"""add · 第二步：会话总结 → ``summem.jsonl``（逐 session 一次 LLM 调用）。

**每个 session 一份总结**：把该会话的全部 msgmem 消息交给模型，产出一段自足的叙述，
定位"哪次会话谈过这件事"。它写进 ``data/{dir}/summem.jsonl``，``id`` = ``session_{s}_summary``。

三条硬约束与消解/atommem 一致：**自足**（时间绝对或带锚点、指代消解）、**精度只减不增**、
不编造。会话之间**并行**（各自独立）。

用法::

    python -m codemem.add --stage summem                # 全部目录
    python -m codemem.add --stage summem Caroline_Melanie
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import openai
import yaml

from .. import llm
from ..io import DATA_DIR, PROJECT_ROOT, memory_id, memory_role, memory_text, now_iso
from ..log import Logger
from ..prompts import SUMMEM_SYSTEM_PROMPT, SUMMEM_USER_PROMPT, salvage_json_object
from .memory import MSGMEM_NAME, SUMMEM_NAME, group_by_session, parse_dir_label, write_memories

CONFIG_FILE = PROJECT_ROOT / "configs" / "add.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "model": "local",
    "base_url": "http://127.0.0.1:30000/v1",
    "api_key": "sglang",
    "temperature": 0.2,
    "max_tokens": 4096,
    "concurrency": 8,
    "timeout": 300,
    "max_retries": 3,
    "backoff_base": 2.0,
    "rate_limit_backoff": 10.0,
    "max_backoff": 60.0,
    "backoff_jitter": 0.2,
    "enable_thinking": False,
}

CALL_KEYS = (
    "timeout", "max_retries", "backoff_base", "rate_limit_backoff", "max_backoff",
    "backoff_jitter",
)


def load_config(path: Path | None = None) -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    target = path or CONFIG_FILE
    if target.exists():
        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                    config[key].update(value)
                elif key in config:
                    config[key] = value
    return config


def gen_call_config(config: dict[str, Any], section: str = "summem") -> dict[str, Any]:
    """把 ``summem`` 段（可选）与顶层重试参数合成一份 ``chat_completion`` 能吃的配置。"""
    merged = dict(config.get(section) or {})
    for key in CALL_KEYS + ("model", "base_url", "api_key", "temperature", "max_tokens",
                            "enable_thinking"):
        if key not in merged and key in config:
            merged[key] = config[key]
    merged.setdefault("max_retries", 3)
    merged.setdefault("timeout", 300)
    merged.setdefault("temperature", 0.2)
    merged.setdefault("max_tokens", 4096)
    return merged


def render_messages(records: list[dict[str, Any]]) -> str:
    lines = []
    for record in records:
        when = str((record.get("metadata") or {}).get("time") or "")
        role = memory_role(record) or "unknown"
        stamp = f" ({when})" if when else ""
        lines.append(f"[{memory_id(record)}]{stamp} {role}: {memory_text(record)}")
    return "\n".join(lines)


def build_messages(
    *, session_index: int, session_time: str, records: list[dict[str, Any]],
    speaker_a: str, speaker_b: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SUMMEM_SYSTEM_PROMPT
            .replace("{speaker_a}", speaker_a).replace("{speaker_b}", speaker_b)},
        {"role": "user", "content": SUMMEM_USER_PROMPT
            .replace("{session_index}", str(session_index))
            .replace("{session_time}", session_time or "(unknown)")
            .replace("{messages}", render_messages(records))},
    ]


def parse_summary(raw: str) -> str:
    """从模型输出里取 ``memory`` 字段（空/解析失败返回空串，调用方保留原状）。"""
    parsed = salvage_json_object(raw)
    if parsed:
        for key in ("memory", "summary", "text", "content"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
        return ""
    # 兜底：模型直接吐了一段纯文本
    return " ".join((raw or "").split())


async def summarize_session(
    *,
    session_index: int,
    session_time: str,
    records: list[dict[str, Any]],
    client: openai.AsyncOpenAI,
    config: dict[str, Any],
    speaker_a: str,
    speaker_b: str,
) -> dict[str, Any]:
    """对一个 session 跑一次总结，返回 summem 记录。失败保留空 memory（不写坏）。"""
    messages = build_messages(
        session_index=session_index, session_time=session_time, records=records,
        speaker_a=speaker_a, speaker_b=speaker_b,
    )
    summary = ""
    try:
        raw = await llm.chat_completion(client, gen_call_config(config, "summem"), messages)
        summary = parse_summary(raw)
    except Exception as exc:  # noqa: BLE001 - 一个 session 失败不该让整目录没有输出
        print(f"  [summem] session {session_index} 总结失败：{type(exc).__name__}: {exc}",
              file=sys.stderr)
    source = [memory_id(record) for record in records if memory_id(record)]
    return {
        "memory": summary,
        "metadata": {
            "id": f"session_{session_index}_summary",
            "type": "summary",
            "time": session_time,
            "role": "summary",
            "tag": [],
            "source": source,
            "changelog": [{"time": now_iso(), "content": "created"}],
        },
    }


async def run_dir(
    directory: Path,
    config: dict[str, Any],
    client: openai.AsyncOpenAI,
    *,
    limit_sessions: int = 0,
) -> dict[str, Any]:
    """对一个目录的全部 session 并行做总结，写 summem.jsonl。"""
    from ..io import read_jsonl

    label = directory.name
    msg_path = directory / MSGMEM_NAME
    if not msg_path.exists():
        return {"dir": label, "status": "SKIPPED",
                "reason": f"no {MSGMEM_NAME}（先跑 add --stage msgmem）"}

    grouped = group_by_session(read_jsonl(msg_path))
    numbers = sorted(grouped)
    if limit_sessions:
        numbers = numbers[:limit_sessions]
    speaker_a, speaker_b = parse_dir_label(label)
    semaphore = asyncio.Semaphore(max(1, int(config.get("concurrency", 8))))

    async def one(number: int) -> dict[str, Any]:
        records = grouped[number]
        session_time = str((records[0].get("metadata") or {}).get("time") or "") if records else ""
        async with semaphore:
            return await summarize_session(
                session_index=number, session_time=session_time, records=records,
                client=client, config=config, speaker_a=speaker_a, speaker_b=speaker_b,
            )

    summaries = list(await asyncio.gather(*(one(n) for n in numbers)))
    write_memories(directory / SUMMEM_NAME, summaries)
    return {"dir": label, "status": "OK", "sessions": len(summaries)}


def targets_with_msgmem(dirs: list[str]) -> list[Path]:
    if dirs:
        resolved: list[Path] = []
        for item in dirs:
            path = Path(item)
            if not path.is_absolute():
                candidate = DATA_DIR / item
                path = candidate if candidate.exists() else PROJECT_ROOT / item
            resolved.append(path)
        return resolved
    return sorted(p for p in DATA_DIR.iterdir() if p.is_dir() and (p / MSGMEM_NAME).exists())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.add.summem",
        description="逐 session 生成会话总结，写 summem.jsonl",
    )
    parser.add_argument("dirs", nargs="*", help="speaker 目录名；缺省处理全部")
    parser.add_argument("--config", default=str(CONFIG_FILE), help="配置（默认 configs/add.yaml）")
    parser.add_argument("--limit-sessions", type=int, default=0,
                        help="每个目录只跑前 N 个 session（调试用）")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖会话级并发数")
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    if args.concurrency is not None:
        config["concurrency"] = args.concurrency
    targets = targets_with_msgmem(args.dirs)
    if not targets:
        print(f"没有找到含 {MSGMEM_NAME} 的目录（先跑 `python -m codemem.add --stage msgmem`）",
              file=sys.stderr)
        return
    log = Logger(level="info")
    client = llm.make_client(gen_call_config(config, "summem"))
    try:
        results = [await run_dir(d, config, client, limit_sessions=args.limit_sessions)
                   for d in targets]
    finally:
        await client.close()
    for item in results:
        if item.get("status") == "OK":
            print(f"summem    {item['dir']}: {item['sessions']} 个 session 总结")
        else:
            print(f"summem    {item['dir']}: {item.get('status')} — {item.get('reason', '')}")


def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(parse_args(argv)))


if __name__ == "__main__":
    main()
