"""add · 第二步：消解 —— 把每条消息改写成**上下文无关**的形式。

      sessions/session_N.jsonl（原始措辞）  ──►  sessions/session_N.jsonl（消解后，就地覆盖）

**为什么是"消解"而不是"另建一层导航"。** 之前的 `index.jsonl`（三元组）/ `nodes+edges`
（图）都是为了"跨会话定位"额外造一层结构。这一版换个更直接的做法：**把原始消息本身改写成
自足的** —— 时间锚定成绝对时间或带绝对锚点的相对时间，指代消解成原始含义。于是：

- **关键词检索直达**：grep 命中的就是消解后的文本，本身就是能回答问题的原料；
- **跨会话联系**：同一实体每次都写成同一个真名，grep 那个名字就命中它的全部会话
  （这正是图里"边带 session 号"想干的事，但不需要单独一层）；
- **通读**时每个 session 文件自足，不会"读到后面忘了前面"（multi_hop 平均跨 2.41 个
  session，最多 7 个）。

**时间是重点**：消解后的 `time` 必须"看到它就对应现实中的一个时间、不需要任何辅助信息"。
即绝对时间（`7 May 2023`）或带绝对锚点的相对时间（`the week before 9 June 2023`）。
三条硬约束（写进 `RESOLVE_SYSTEM_PROMPT`）：

1. **自足** —— 禁止无锚点的相对词（`yesterday`/`last week` 单说）；
2. **精度只减不增** —— 源说 `last year` 就只能到年（`2022`），不许编月日；
3. **区分点/段/大概** —— `time_kind` = `point` / `range` / `approx`。

**指代消解**同理：`it`/`she`/`there` 换成真名 / 真实地点 / 真实物件 —— 逐消息处理，
并喂前文消息作为消解指代的上下文（见 `extract`）。

**原话保留**：消解后 `content`/`time` 是消解结果，`source_content`/`source_time` 保留
原始措辞与原始时间词（溯源 + 精度备份）。re-resolve 时用 `source_*`（缺则用 `content`）
作为输入，所以**就地覆盖是幂等的**，不需要另存一份 raw。

逐消息、会话内串行（后面的消息能看到前面）、会话间并行。见 `docs/core.md` §1。

用法::

    python -m codemem.add --stage resolve                 # 全部目录
    python -m codemem.add --stage resolve Caroline_Melanie
    python -m codemem.add --stage resolve --limit-sessions 3   # 调试
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from .. import llm
from ..io import (
    DATA_DIR,
    PROJECT_ROOT,
    make_session_record,
    msg_content,
    msg_id,
    msg_role,
    msg_time,
    read_jsonl,
    write_jsonl,
)
from ..prompts import RESOLVE_SYSTEM_PROMPT, RESOLVE_USER_PROMPT
from .session import SESSIONS_DIR, session_filename, session_number_of

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
    # 同一条消息的 prompt 里附上前 N 条同会话消息作为消解指代的上下文窗口。
    "context_window": 8,
}

VALID_TIME_KINDS = ("point", "range", "approx")


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 原始视图：source_* 优先（就地覆盖是幂等的，re-resolve 靠它拿到原话）
# ---------------------------------------------------------------------------

def raw_content(record: dict[str, Any]) -> str:
    """原始措辞：``source_content`` 优先，否则 ``content``（尚未消解过的记录）。"""
    for key in ("source_content", "content"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def raw_time(record: dict[str, Any]) -> str:
    """原始时间词（会话时间戳，锚点）：``source_time`` 优先，否则 ``time``。"""
    for key in ("source_time", "time"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


# ---------------------------------------------------------------------------
# 会话文件枚举
# ---------------------------------------------------------------------------

def sessions_dir(directory: Path) -> Path:
    return directory / SESSIONS_DIR


def list_session_files(directory: Path) -> list[tuple[int, Path]]:
    """``data/{dir}/sessions/`` 下全部 ``session_N.jsonl``，按会话号排序。"""
    base = sessions_dir(directory)
    if not base.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for path in base.glob("session_*.jsonl"):
        number = session_number_of(path)
        if number is not None:
            found.append((number, path))
    return sorted(found, key=lambda pair: pair[0])


# ---------------------------------------------------------------------------
# 渲染 / 解析
# ---------------------------------------------------------------------------

def _short(value: Any, limit: int = 2000) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _salvage(raw: str) -> dict[str, Any] | None:
    """从模型输出里取一个 JSON 对象；容忍围栏、前后说明、被截断的尾巴。"""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    candidates = [text]
    if "{" in text:
        candidates.append(text[text.find("{"): text.rfind("}") + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    from ..io import salvage_json_records
    salvaged, _ = salvage_json_records(text)
    return salvaged[0] if salvaged else None


def parse_resolution(raw: str) -> dict[str, str]:
    """解析一条消解结果：``{content, time, time_kind, time_raw}``。

    解析失败返回空 dict —— 调用方据此**保留原话**（绝不写坏记录）。``time_kind`` 非法一律
    归 ``""``。
    """
    parsed = _salvage(raw)
    if not parsed:
        return {}
    content = _short(parsed.get("content") or parsed.get("text") or parsed.get("memory"))
    kind = _short(parsed.get("time_kind") or parsed.get("kind"), 20).lower()
    if kind not in VALID_TIME_KINDS:
        kind = ""
    return {
        "content": content,
        "time": _short(parsed.get("time"), 200),
        "time_kind": kind,
        "time_raw": _short(parsed.get("time_raw") or parsed.get("when"), 200),
    }


def render_message(record: dict[str, Any]) -> str:
    role = msg_role(record) or "unknown"
    when = raw_time(record)
    stamp = f" ({when})" if when else ""
    return f"[{msg_id(record)}]{stamp} {role}: {raw_content(record)}"


def render_window(prior: list[dict[str, Any]], config: dict[str, Any]) -> str:
    window = max(0, int(config.get("context_window", 8)))
    if window == 0 or not prior:
        return "(this is the first message of the session)"
    return "\n".join(render_message(record) for record in prior[-window:])


def build_message_prompt(
    *,
    target: dict[str, Any],
    prior: list[dict[str, Any]],
    session_index: int,
    session_time: str,
    speaker_a: str,
    speaker_b: str,
    config: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RESOLVE_SYSTEM_PROMPT
            .replace("{speaker_a}", speaker_a).replace("{speaker_b}", speaker_b)},
        {"role": "user", "content": RESOLVE_USER_PROMPT
            .replace("{session_index}", str(session_index))
            .replace("{session_time}", session_time or "(unknown)")
            .replace("{window}", render_window(prior, config))
            .replace("{msg_id}", msg_id(target))
            .replace("{msg_time}", raw_time(target) or session_time or "(unknown)")
            .replace("{role}", msg_role(target) or "unknown")
            .replace("{content}", raw_content(target))},
    ]


async def resolve_one(
    *,
    target: dict[str, Any],
    prior: list[dict[str, Any]],
    session_index: int,
    session_time: str,
    speaker_a: str,
    speaker_b: str,
    config: dict[str, Any],
    client: Any,
) -> dict[str, str]:
    """消解一条消息。失败/解析不出**返回空 dict**（调用方保留原话）。"""
    prompt = build_message_prompt(
        target=target, prior=prior, session_index=session_index,
        session_time=session_time, speaker_a=speaker_a, speaker_b=speaker_b, config=config,
    )
    try:
        raw = await llm.chat_completion(client, config, prompt)
    except Exception as exc:  # noqa: BLE001 - 一条消息失败不该让整个会话没有输出
        print(f"  [resolve] session {session_index} msg {msg_id(target)} 消解失败："
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return {}
    return parse_resolution(raw)


def to_resolved_record(
    target: dict[str, Any], resolved: dict[str, str], session_time: str
) -> dict[str, Any]:
    """把一条消息 + 消解结果落成新记录（原话进 ``source_*``）。

    消解失败（``resolved`` 为空）时 **content 保留原话**、``time`` 留空、``time_kind`` 留空 ——
    宁可这条消息没被消解，也不能把它写坏。
    """
    src_content = raw_content(target)
    src_time = raw_time(target) or session_time
    content = resolved.get("content") or src_content
    record = make_session_record(
        msg_id_value=msg_id(target),
        role=msg_role(target),
        content=content,
        time=resolved.get("time", ""),
    )
    record["time_kind"] = resolved.get("time_kind", "")
    record["source_content"] = src_content
    record["source_time"] = src_time
    return record


# ---------------------------------------------------------------------------
# 逐会话消解
# ---------------------------------------------------------------------------

async def resolve_session_file(
    path: Path,
    session_index: int,
    config: dict[str, Any],
    client: Any,
    *,
    speaker_a: str,
    speaker_b: str,
) -> int:
    """消解一个 session 文件（**就地覆盖**），返回消解的消息条数。

    输入取每条记录的**原始视图**（``source_content``/``source_time`` 优先），所以对已经
    消解过的文件再跑一遍是幂等的（不会把消解结果当原话再消一次）。会话内**串行**：后面的
    消息要能看到前面的原话，指代才消解得了。
    """
    raw_records = read_jsonl(path)
    if not raw_records:
        return 0
    session_time = next((raw_time(r) for r in raw_records if raw_time(r)), "")

    resolved_records: list[dict[str, Any]] = []
    for position, target in enumerate(raw_records):
        resolved = await resolve_one(
            target=target, prior=raw_records[:position], session_index=session_index,
            session_time=session_time, speaker_a=speaker_a, speaker_b=speaker_b,
            config=config, client=client,
        )
        resolved_records.append(to_resolved_record(target, resolved, session_time))

    write_jsonl(path, resolved_records)
    return len(resolved_records)


# ---------------------------------------------------------------------------
# 单目录 / 增量
# ---------------------------------------------------------------------------

def parse_dir_label(label: str) -> tuple[str, str]:
    """``"Caroline_Melanie"`` -> ``("Caroline", "Melanie")``（用数据集里的真名）。"""
    from .. import dataset

    try:
        for sample in dataset.load_samples():
            if dataset.sample_dir(sample) == label:
                conversation = sample["conversation"]
                return conversation["speaker_a"], conversation["speaker_b"]
    except (OSError, ValueError, FileNotFoundError):
        pass
    parts = label.split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (label, "other")


async def run_dir(
    directory: Path,
    config: dict[str, Any],
    client: Any,
    limit_sessions: int = 0,
) -> dict[str, Any]:
    """对一个目录的全部会话做消解（就地覆盖 sessions/*.jsonl）。会话之间并行。"""
    label = directory.name
    targets = list_session_files(directory)
    if not targets:
        return {"dir": label, "status": "SKIPPED",
                "reason": f"no {SESSIONS_DIR}/session_*.jsonl（先跑 add --stage session）"}
    if limit_sessions:
        targets = targets[:limit_sessions]

    speaker_a, speaker_b = parse_dir_label(label)
    concurrency = max(1, int(config.get("concurrency", 8)))
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int, path: Path) -> tuple[int, int]:
        async with semaphore:
            count = await resolve_session_file(
                path, index, config, client, speaker_a=speaker_a, speaker_b=speaker_b
            )
            print(f"  [{label}] session {index}: 消解 {count} 条", flush=True)
            return index, count

    results = await asyncio.gather(*(one(i, p) for i, p in targets))
    return {
        "dir": label, "status": "OK",
        "sessions": len(results), "messages": sum(count for _, count in results),
    }


async def refresh_session(
    directory: Path,
    session_index: int,
    config: dict[str, Any],
    client: Any,
    *,
    speaker_a: str,
    speaker_b: str,
) -> bool:
    """消解**一个**会话文件（就地覆盖）。供 API 的 Add 用：写入返回即"立即可检索"。

    输入取记录的原始视图（``source_content``/``source_time``），所以新追加的原始消息与
    已有消解记录能一起重新消解。失败**返回 False 而不抛**。
    """
    path = sessions_dir(directory) / session_filename(session_index)
    if not path.exists():
        return False
    try:
        await resolve_session_file(
            path, session_index, config, client,
            speaker_a=speaker_a, speaker_b=speaker_b,
        )
    except Exception:  # noqa: BLE001 - 消解失败不该让 Add 失败
        return False
    return True


def resolve_targets(dirs: list[str]) -> list[Path]:
    """把目录名/路径解析成实际目录（只保留含 sessions/ 的）。"""
    if dirs:
        resolved: list[Path] = []
        for item in dirs:
            path = Path(item)
            if not path.is_absolute():
                candidate = DATA_DIR / item
                path = candidate if candidate.exists() else PROJECT_ROOT / item
            resolved.append(path)
        return resolved
    return sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_dir() and sessions_dir(p).is_dir()
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.add.resolve",
        description="逐消息消解时间与指代，把 sessions/*.jsonl 改写成上下文无关的形式",
    )
    parser.add_argument("dirs", nargs="*", help="speaker 目录名；缺省处理全部")
    parser.add_argument("--config", default=str(CONFIG_FILE), help="配置（默认 configs/add.yaml）")
    parser.add_argument("--limit-sessions", type=int, default=0,
                        help="每个目录只跑前 N 个 session（调试用）")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖会话级并发数")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(Path(args.config))
    if args.concurrency is not None:
        config["concurrency"] = args.concurrency

    targets = resolve_targets(args.dirs)
    if not targets:
        print(f"没有找到含 {SESSIONS_DIR}/ 的目录（先跑 `python -m codemem.add --stage session`）",
              file=sys.stderr)
        return

    client = llm.make_client(config)

    async def run_all() -> list[dict[str, Any]]:
        try:
            return [await run_dir(d, config, client, limit_sessions=args.limit_sessions)
                    for d in targets]
        finally:
            await client.close()

    for item in asyncio.run(run_all()):
        if item.get("status") == "OK":
            print(f"resolve   {item['dir']}: {item['sessions']} 个 session / "
                  f"{item['messages']} 条消息已消解")
        else:
            print(f"resolve   {item['dir']}: {item.get('status')} — {item.get('reason', '')}")


if __name__ == "__main__":
    main()
