"""add · 第二步：两层 summary（session → speakers）。

     每个 session  ──►  session summary（session_{n}_summary）
     全部 session summaries ──►  speakers summary（speakers_summary）

**为什么要两层、而不是继续抽原子**：实测 152 条 QA 里 multi_hop 只有 43%。根因是原子
**丢掉了会话结构** —— 一条条孤立事实无法回答"他们什么时候认识的""这段关系怎么发展的"。
session summary 保留"这次谈了什么、什么变了、谁对谁说了什么"；speakers summary 再把
这些串成整段关系的**轨迹**。层级同时给了检索一个由粗到细的入口：
先读 speakers summary 定位到哪次会话，再进那次会话的 summary 与原文。

**两层都用同一份 session 记录格式**（``session_template.jsonl``），只把 ``role`` 标记成
``"summary"``。好处是 agent 只需要理解一种记录格式，工具/模板/解析全部复用；代价是
读的时候要靠 ``role`` 区分粒度 —— 比让 agent 学两种格式划算。

**speakers summary 耗时相关**：它需要全部 session summary 都就绪，所以这一层必须等
第一层全部完成。给定 ``--limit-sessions`` 时只用前 N 个 session（调试用）。

用法::

    python -m codemem.add.summary                       # 全部目录，两层都跑
    python -m codemem.add.summary --stage session        # 只跑第一层
    python -m codemem.add.summary Caroline_Melanie
    python -m codemem.add.summary --limit-sessions 3     # 每个目录只跑前 3 个 session
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
from ..prompts import (
    SESSION_SUMMARY_SYSTEM_PROMPT,
    SESSION_SUMMARY_USER_PROMPT,
    SPEAKERS_SUMMARY_SYSTEM_PROMPT,
    SPEAKERS_SUMMARY_USER_PROMPT,
)
from .session import OUTPUT_NAME as SESSIONS_NAME

CONFIG_FILE = PROJECT_ROOT / "configs" / "add.yaml"
SESSION_OUTPUT = "session_summaries.jsonl"
SPEAKERS_OUTPUT = "speakers_summary.jsonl"

DEFAULT_CONFIG: dict[str, Any] = {
    "model": "local",
    "base_url": "http://127.0.0.1:30000/v1",
    "api_key": "sglang",
    "temperature": 0.2,
    # summary 比原子抽取长得多（session 120-300 词、speakers 250-500 词），给足空间
    "max_tokens": 4096,
    "concurrency": 4,
    "timeout": 300,
    "max_retries": 3,
    "backoff_base": 2.0,
    "rate_limit_backoff": 10.0,
    "max_backoff": 60.0,
    "backoff_jitter": 0.2,
    "enable_thinking": False,
}


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
# 输入：按 session 分组
# ---------------------------------------------------------------------------

SESSION_NUMBER_RE = re.compile(r"^session_(\d+)_")


def group_by_session(records: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    """把 session 记录按会话分组，返回 ``[(会话号, [记录...])]``（按会话号排序）。

    ``msg_id`` 形如 ``session_{会话号}_{消息号}``；解析不出的记录**按物理顺序**归入
    当前会话（而不是丢掉）—— 丢弃会让 summary 凭空少一段对话，而那是静默的错误。
    """
    groups: dict[int, list[dict[str, Any]]] = {}
    current: int | None = None
    fallback = 0

    for record in records:
        match = SESSION_NUMBER_RE.match(msg_id(record))
        if match:
            current = int(match.group(1))
        elif current is None:
            fallback += 1
            current = fallback          # 头部就无法解析时，造一个序号兜底
        groups.setdefault(current, []).append(record)

    return sorted(groups.items(), key=lambda pair: pair[0])


def render_session(messages: list[dict[str, Any]]) -> str:
    """把一次会话渲染成给模型看的文本。``role`` + ``time`` 都要给 ——
    时间是相对时间推理的锚点，名字是指代消解的唯一线索（见 ``session.py``）。"""
    lines: list[str] = []
    for record in messages:
        role = msg_role(record) or "unknown"
        when = msg_time(record)
        stamp = f" ({when})" if when else ""
        lines.append(f"[{msg_id(record)}]{stamp} {role}: {msg_content(record)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 输出解析
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def parse_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出里取 JSON 对象。容忍围栏、前后说明文字、以及**被截断的尾巴**。"""
    if not text or not text.strip():
        return None
    candidate = _FENCE_RE.sub("", text.strip()).strip()
    for attempt in (candidate, candidate[: candidate.rfind("}") + 1] if "}" in candidate else ""):
        if not attempt or not attempt.startswith("{"):
            continue
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    start, end = candidate.find("{"), candidate.rfind("}")
    if 0 <= start < end:
        try:
            parsed = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def summary_text(parsed: dict[str, Any]) -> str:
    """取出 summary 正文。模型偶尔会用别的键名，逐个兜底。"""
    for key in ("summary", "content", "text", "session_summary", "speakers_summary"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def key_facts(parsed: dict[str, Any]) -> list[str]:
    """取出 key_facts（检索钩子）。非法形态一律丢弃，不猜。"""
    value = parsed.get("key_facts") or parsed.get("facts") or []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, (str, int, float))
            and str(item).strip()]


# ---------------------------------------------------------------------------
# 第一层：session summary
# ---------------------------------------------------------------------------

def build_session_messages(
    messages: list[dict[str, Any]],
    session_index: int,
    session_time: str,
    speaker_a: str,
    speaker_b: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SESSION_SUMMARY_SYSTEM_PROMPT.format(
                speaker_a=speaker_a, speaker_b=speaker_b
            ),
        },
        {
            "role": "user",
            "content": SESSION_SUMMARY_USER_PROMPT.format(
                session_index=session_index,
                session_time=session_time or "(unknown)",
                message_count=len(messages),
                messages=render_session(messages),
            ),
        },
    ]


async def summarize_session(
    messages: list[dict[str, Any]],
    session_index: int,
    session_time: str,
    speaker_a: str,
    speaker_b: str,
    config: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    """跑一个 session 的 summary，返回 session 记录（``role="summary"``）。

    失败时返回一条**说明失败**的记录而不是抛异常 —— 一个 session 失败不该让整个目录
    没有输出；失败会体现在 summary 正文里，读的时候一眼可见。
    """
    prompt = build_session_messages(
        messages, session_index, session_time, speaker_a, speaker_b
    )
    try:
        raw = await llm.chat_completion(client, config, prompt)
    except Exception as exc:  # noqa: BLE001
        return make_session_record(
            f"session_{session_index}_summary", "summary",
            f"[summary failed: {type(exc).__name__}: {exc}]", session_time,
        )

    parsed = parse_json_object(raw) or {}
    text = summary_text(parsed)
    facts = key_facts(parsed)
    if not text:
        # 解析不出结构化结果时，退回原始输出 —— 总比丢掉整次会话好
        text = raw.strip()
    body = text if not facts else text + "\n\nKey facts:\n" + "\n".join(f"- {f}" for f in facts)
    return make_session_record(
        f"session_{session_index}_summary", "summary", body,
        str(parsed.get("time") or session_time or ""),
    )


# ---------------------------------------------------------------------------
# 第二层：speakers summary
# ---------------------------------------------------------------------------

def build_speakers_messages(
    session_summaries: list[dict[str, Any]],
    speaker_a: str,
    speaker_b: str,
) -> list[dict[str, str]]:
    numbers: list[int] = []
    for record in session_summaries:
        match = SESSION_NUMBER_RE.match(msg_id(record))
        if match:
            numbers.append(int(match.group(1)))
    blocks = [
        f"### {msg_id(record)}  ({msg_time(record) or 'unknown time'})\n{msg_content(record)}"
        for record in session_summaries
    ]
    first = min(numbers) if numbers else 1
    last = max(numbers) if numbers else len(session_summaries)
    times = [msg_time(r) for r in session_summaries if msg_time(r)]
    return [
        {
            "role": "system",
            "content": SPEAKERS_SUMMARY_SYSTEM_PROMPT.format(
                speaker_a=speaker_a, speaker_b=speaker_b
            ),
        },
        {
            "role": "user",
            "content": SPEAKERS_SUMMARY_USER_PROMPT.format(
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                session_count=len(session_summaries),
                first_session=first,
                last_session=last,
                time_span=f"{times[0]} .. {times[-1]}" if times else "(unknown)",
                session_summaries="\n\n".join(blocks),
            ),
        },
    ]


async def summarize_speakers(
    session_summaries: list[dict[str, Any]],
    speaker_a: str,
    speaker_b: str,
    config: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    """跑 speakers summary，返回 session 记录（``msg_id="speakers_summary"``）。"""
    prompt = build_speakers_messages(session_summaries, speaker_a, speaker_b)
    try:
        raw = await llm.chat_completion(client, config, prompt)
    except Exception as exc:  # noqa: BLE001
        return make_session_record(
            "speakers_summary", "summary",
            f"[speakers summary failed: {type(exc).__name__}: {exc}]", "",
        )
    parsed = parse_json_object(raw) or {}
    text = summary_text(parsed)
    if not text:
        text = raw.strip()
    people = parsed.get("people")
    if isinstance(people, list) and people:
        text += "\n\nPeople:\n" + "\n".join(f"- {p}" for p in people)
    arc = parsed.get("arc")
    if isinstance(arc, str) and arc.strip():
        text += f"\n\nArc: {arc.strip()}"
    return make_session_record("speakers_summary", "summary", text, "")


# ---------------------------------------------------------------------------
# 单个目录
# ---------------------------------------------------------------------------

def parse_dir_label(label: str) -> tuple[str, str]:
    """``"Caroline_Melanie"`` -> ``("Caroline", "Melanie")``。

    用数据集里的真实名字（而不是靠下划线切），因为名字本身可能含下划线；
    找不到就退回切分结果。
    """
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
    stages: tuple[str, ...] = ("session", "speakers"),
    log_prefix: str = "",
) -> dict[str, Any]:
    """对一个目录跑两层 summary。返回小结 dict。"""
    label = directory.name
    sessions_path = directory / SESSIONS_NAME
    if not sessions_path.exists():
        print(f"[skip] {label}: 没有 {SESSIONS_NAME}（先跑 add --stage session）", file=sys.stderr)
        return {"dir": label, "status": "SKIPPED", "reason": f"no {SESSIONS_NAME}"}

    records = read_jsonl(sessions_path)
    # 只对**原始消息**做 summary；文件里若已有 summary，不作为输入（幂等重跑）
    raw_records = [r for r in records if not _is_summary_record(r)]
    groups = group_by_session(raw_records)
    if limit_sessions:
        groups = groups[:limit_sessions]
    if not groups:
        return {"dir": label, "status": "SKIPPED", "reason": "no sessions"}

    speaker_a, speaker_b = parse_dir_label(label)
    concurrency = max(1, int(config.get("concurrency", 4)))
    summary_count = 0

    if "session" in stages:
        # 第一层：session summary（会话之间互相独立，可并发）
        semaphore = asyncio.Semaphore(concurrency)

        async def one(index: int, messages: list[dict[str, Any]]) -> dict[str, Any]:
            async with semaphore:
                session_time = next((msg_time(m) for m in messages if msg_time(m)), "")
                result = await summarize_session(
                    messages, index, session_time, speaker_a, speaker_b, config, client
                )
                print(f"  [{label}] session {index} summary ({len(messages)} 条消息)", flush=True)
                return result

        session_summaries = list(
            await asyncio.gather(*(one(index, messages) for index, messages in groups))
        )
        write_jsonl(directory / SESSION_OUTPUT, session_summaries)
        summary_count = len(session_summaries)
    else:
        session_summaries = read_jsonl(directory / SESSION_OUTPUT)
        if not session_summaries:
            session_summaries = [
                make_session_record(f"session_{index}_summary", "summary", "", "")
                for index, _ in groups
            ]

    speakers_path = directory / SPEAKERS_OUTPUT
    if "speakers" in stages:
        # 第二层：必须等第一层全部就绪（它消费所有 session summary）
        result = await summarize_speakers(
            session_summaries, speaker_a, speaker_b, config, client
        )
        write_jsonl(speakers_path, [result])
        print(f"  [{label}] speakers summary（基于 {len(session_summaries)} 个 session summary）",
              flush=True)
    else:
        result = None

    return {
        "dir": label,
        "status": "OK",
        "sessions": len(groups),
        "session_summaries": summary_count,
        "speakers_summary": bool(result),
    }


def _is_summary_record(record: dict[str, Any]) -> bool:
    return msg_role(record).lower() == "summary" or msg_id(record).endswith("_summary")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def resolve_targets(dirs: list[str]) -> list[Path]:
    """把目录名/路径解析成实际目录（只保留含 sessions.jsonl 的）。"""
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
        if p.is_dir() and (p / SESSIONS_NAME).exists()
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.add.summary",
        description="两层 summary：session summary → speakers summary",
    )
    parser.add_argument("dirs", nargs="*", help="speaker 目录名；缺省处理全部")
    parser.add_argument("--config", default=str(CONFIG_FILE), help="配置（默认 configs/add.yaml）")
    parser.add_argument("--stage", choices=("session", "speakers", "all"), default="all",
                        help="只跑某一层（speakers 需要 session 层已就绪）")
    parser.add_argument("--limit-sessions", type=int, default=0,
                        help="每个目录只跑前 N 个 session（调试用）")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖并发数")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(Path(args.config))
    if args.concurrency is not None:
        config["concurrency"] = args.concurrency

    targets = resolve_targets(args.dirs)
    if not targets:
        print("没有找到含 sessions.jsonl 的目录（先跑 `python -m codemem.add --stage session`）",
              file=sys.stderr)
        return

    stages = ("session", "speakers") if args.stage == "all" else (args.stage,)
    client = llm.make_client(config)

    async def run_all() -> list[dict[str, Any]]:
        try:
            return [
                await run_dir(
                    directory, config, client,
                    limit_sessions=args.limit_sessions, stages=stages,
                )
                for directory in targets
            ]
        finally:
            await client.close()

    results = asyncio.run(run_all())
    for item in results:
        if item.get("status") == "OK":
            print(
                f"summary   {item['dir']}: {item['sessions']} 个 session -> "
                f"{item['session_summaries']} 条 summary"
                + (" + speakers summary" if item["speakers_summary"] else "")
            )
        else:
            print(f"summary   {item['dir']}: {item.get('status')} — {item.get('reason', '')}")


if __name__ == "__main__":
    main()
