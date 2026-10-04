"""add · 第三步：原子事实抽取（atommem.jsonl）—— **逐 message 的 add agent**。

     msgmem.jsonl / summem.jsonl / atommem.jsonl  ─►  atommem.jsonl（追加本 message 的 atom）

**核心原理**：逐个 message 处理。后面的 message 的信息可能受**前面内容**影响、可能**跨
session 影响**，所以处理一条 message 时，agent 需要看到前面**相关的**信息才能把指代消解对、
把隐含事实补全。于是这里不是一个单次 LLM 调用，而是一个**朴素的 agent loop**（与 search
同形）：给 agent 三个工具 `read` / `edit` / `bash`、一个装着三层记忆的工作目录，让它自己
`grep` 前文、`read` 相关 session，把原子事实 `edit append` 进 `atommem.jsonl`。默认 `max_step=6`。

**agent 只填四个字段**（``memory`` + ``metadata.{time, role, tag}``）；``id`` / ``source`` /
``type`` / ``changelog`` 由系统在收尾时补上 —— 见 ``finalize_atoms``。这既省了 agent 的
力气，也保证这些字段永远正确（id 唯一、source 指向本 message、type 恒为 atom）。

**串行**：同一目录内按 message 顺序**串行**（后面的 message 要看到前面已生成的 atommem）；
**不同目录并行**。

**增量/幂等**：重跑一条 message 时，先把 atommem 里 ``source`` 指向它的旧 atom 去掉，再重新
生成 —— 于是重复跑不会累积重复 atom（"就地覆盖是幂等的"的 agent 版）。

复用 search 的组件（``search/tools.py`` 的 ``build_tools``、``search/toolcfg.py`` 的
``ToolCatalog``、``search/agent.py`` 的 ``run_agent_round``），配置在 ``configs/add_tool.json``。

用法::

    python -m codemem.add --stage atommem                      # 全部目录
    python -m codemem.add --stage atommem Caroline_Melanie     # 指定目录
    python -m codemem.add --stage atommem --limit-messages 20  # 每目录只跑前 20 条（调试）
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
from ..io import DATA_DIR, PROJECT_ROOT, memory_id, memory_role, memory_text, now_iso, read_jsonl
from ..log import Logger
from ..prompts import ATOM_AGENT_STANDARD, ATOM_AGENT_SYSTEM_PROMPT, ATOM_AGENT_USER_PROMPT
from ..search import agent as agent_mod
from ..search import toolcfg
from ..search import tools as tools_mod
from .memory import (
    ATOMMEM_NAME,
    MSGMEM_NAME,
    SUMMEM_NAME,
    group_by_session,
    msg_session_number,
    parse_dir_label,
    write_memories,
)

CONFIG_FILE = PROJECT_ROOT / "configs" / "add.yaml"
TOOL_CONFIG_FILE = PROJECT_ROOT / "configs" / "add_tool.json"

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
    # agent loop
    "max_steps": 6,
    "no_progress_patience": 4,
    "observation_max_chars": 8000,
    "max_parse_retries": 2,
    "repeat_limit": 1,
    "context_max_chars": 120000,
    "context_keep_recent": 6,
    "bash_timeout": 60,
    # 同一条 message 的 prompt 里附上前 N 条同会话消息作为消解指代的上下文窗口（原始措辞）。
    "context_window": 8,
}

CALL_KEYS = (
    "timeout", "max_retries", "backoff_base", "rate_limit_backoff", "max_backoff",
    "backoff_jitter",
)


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


def gen_call_config(config: dict[str, Any], section: str = "atommem") -> dict[str, Any]:
    """把某个模型子段（可选）与顶层重试参数合成 ``chat_completion`` 能吃的配置。"""
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


# ---------------------------------------------------------------------------
# 目录 / 输入
# ---------------------------------------------------------------------------

def directory_files(directory: Path) -> dict[str, Path]:
    return {
        "msgmem": directory / MSGMEM_NAME,
        "summem": directory / SUMMEM_NAME,
        "atommem": directory / ATOMMEM_NAME,
    }


def targets_with_msgmem(dirs: list[str]) -> list[Path]:
    """把目录名/路径解析成实际目录（只保留含 msgmem.jsonl 的）。"""
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
        if p.is_dir() and (p / MSGMEM_NAME).exists()
    )


# ---------------------------------------------------------------------------
# 渲染：会话清单 / 会话内窗口 / 目标 message
# ---------------------------------------------------------------------------

def render_sessions(summaries: dict[int, dict[str, Any]]) -> str:
    """会话清单：每行 ``session_N (时间)``，**不含事实内容**（克制，防止跳过检索）。"""
    if not summaries:
        return "(no sessions found -- check msgmem.jsonl)"
    lines = ["The conversation is split into these sessions (each has a summary in summem.jsonl):"]
    for number in sorted(summaries):
        when = str((summaries[number].get("metadata") or {}).get("time") or "")
        lines.append(f"  session_{number}" + (f"  ({when})" if when else ""))
    return "\n".join(lines)


def _render_msg_line(record: dict[str, Any], *, anchor: str) -> str:
    mid = memory_id(record)
    role = memory_role(record) or "unknown"
    when = str((record.get("metadata") or {}).get("time") or "")
    stamp = f" ({when})" if when else ""
    return f"[{mid}]{stamp} {role}: {memory_text(record)}"


def render_window(prior: list[dict[str, Any]], window: int) -> str:
    if window <= 0 or not prior:
        return "(this is the first message of the session)"
    return "\n".join(_render_msg_line(r, anchor="") for r in prior[-window:])


def build_messages(
    *,
    target: dict[str, Any],
    prior: list[dict[str, Any]],
    summaries: dict[int, dict[str, Any]],
    config: dict[str, Any],
    catalog: toolcfg.ToolCatalog,
    speaker_a: str,
    speaker_b: str,
) -> list[dict[str, str]]:
    when = str((target.get("metadata") or {}).get("time") or "")
    target_line = _render_msg_line(target, anchor="")
    system = catalog.render_prompt("").replace("{speaker_a}", speaker_a).replace(
        "{speaker_b}", speaker_b
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": ATOM_AGENT_USER_PROMPT
            .replace("{sessions}", render_sessions(summaries))
            .replace("{msg_id}", memory_id(target))
            .replace("{window}", render_window(prior, int(config.get("context_window", 8))))
            .replace("{target}", target_line)},
    ]


# ---------------------------------------------------------------------------
# agent 驱动的单条 message
# ---------------------------------------------------------------------------

class _Runner:
    """把整体的重试参数与模型配置合成一份，供 ``ModelRunner`` 使用。"""

    def __init__(self, client: openai.AsyncOpenAI, config: dict[str, Any], log: Logger,
                 raw_path: Path | None = None) -> None:
        from ..search.runner import ModelRunner

        self.model = ModelRunner(client, gen_call_config(config, "atommem"), log, raw_path)


async def run_message(
    *,
    target: dict[str, Any],
    prior: list[dict[str, Any]],
    summaries: dict[int, dict[str, Any]],
    workspace: Path,
    atommem_path: Path,
    config: dict[str, Any],
    catalog: toolcfg.ToolCatalog,
    runner: _Runner,
    log: Logger,
    speaker_a: str = "speaker_a",
    speaker_b: str = "speaker_b",
) -> list[dict[str, Any]]:
    """处理一条 message：跑一轮 agent loop，返回它为本 message 写出的 atom 列表。

    返回的是**已补齐系统字段**的完整记录（``finalize_atoms`` 的产物）。
    """
    # 记住文件在处理前的行数：agent 会 append；收尾时把本 message 的新增行捞出来。
    before_lines = _line_count(atommem_path)

    shell_prefix = _shell_prefix(workspace)
    tools = tools_mod.build_tools(
        cwd=workspace, shell_command_prefix=shell_prefix,
        bash_timeout=float(config.get("bash_timeout", 60)),
        default_write_file=ATOMMEM_NAME,
    )
    catalog.apply(tools)

    messages = build_messages(
        target=target, prior=prior, summaries=summaries, config=config,
        catalog=catalog, speaker_a=speaker_a, speaker_b=speaker_b,
    )
    outcome, _ = await agent_mod.run_agent_round(
        messages=messages,
        tools=tools,
        model_call=runner.model,
        evidence_path=atommem_path,
        cwd=workspace,
        max_steps=int(config.get("max_steps", 6)),
        no_progress_patience=int(config.get("no_progress_patience", 4)),
        step_offset=0,
        round_index=1,
        log=log,
        observation_max_chars=int(config.get("observation_max_chars", 8000)),
        write_nudge_after=max(2, int(config.get("max_steps", 6)) // 2),
        evidence_soft_limit=0,          # 不提醒"太多"：一条 message 的 atom 数由内容决定
        max_parse_retries=int(config.get("max_parse_retries", 2)),
        repeat_limit=int(config.get("repeat_limit", 1)),
        duplicate_warn_at=10,
        context_max_chars=int(config.get("context_max_chars", 120000)),
        context_keep_recent=int(config.get("context_keep_recent", 6)),
    )

    # 取 agent 新写的行（行号 > 处理前的行数）。id 由系统补，所以只能按行位置取。
    new_records = _records_after(atommem_path, before_lines)
    atoms = finalize_atoms(
        new_records,
        source_id=memory_id(target),
        default_time=str((target.get("metadata") or {}).get("time") or ""),
        default_role=memory_role(target),
    )
    # 把 agent 写的**原始行**（只有 4 个字段）替换成补齐系统字段的版本，落回文件 ——
    # 这样 on-disk 的 atommem.jsonl 永远带 id/source/type/changelog（下游只认这个）。
    _rewrite_tail(atommem_path, before_lines, atoms)
    return atoms


def _rewrite_tail(path: Path, keep_lines: int, new_records: list[dict[str, Any]]) -> None:
    """保留文件前 ``keep_lines`` 行，其余替换成 ``new_records``（追加到末尾）。"""
    lines = []
    if path.exists():
        lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                 if line.strip()]
    kept = lines[:keep_lines]
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new_records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in kept) + body, encoding="utf-8")


def _shell_prefix(workspace: Path) -> str:
    from ..search.runner import build_shell_prefix

    return build_shell_prefix(inputs_dir=workspace, config={})


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
               if line.strip())


def _records_after(path: Path, skip_lines: int) -> list[dict[str, Any]]:
    """取文件里**第 skip_lines 行之后**的、能解析成对象的记录（agent 新追加的 atom）。"""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
             if line.strip()]
    for line in lines[skip_lines:]:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


# ---------------------------------------------------------------------------
# 系统字段补全
# ---------------------------------------------------------------------------

def _normalize_tags(value: Any) -> list[str]:
    """把 agent 写的 ``tag`` 归一成 ``list[str]``。

    模型常把 ``tag`` 写成**字符串**（``"activity:swimming"``）而不是数组 —— 直接迭代字符串
    会把每个字符拆成一个 tag（``["a","c","t",...]``）。所以：字符串按整体当一个 tag（逗号
    分隔时再拆），列表逐项保留，非字符串一律丢弃。空白 tag 丢掉。
    """
    if isinstance(value, str):
        raw_items = value.replace("，", ",").split(",")
    elif isinstance(value, list):
        raw_items = value
    else:
        return []
    tags: list[str] = []
    for item in raw_items:
        text = item.strip() if isinstance(item, str) else ""
        if text and text not in tags:
            tags.append(text)
    return tags


def finalize_atoms(
    raw_atoms: list[dict[str, Any]],
    *,
    source_id: str,
    default_time: str,
    default_role: str,
) -> list[dict[str, Any]]:
    """给 agent 写的 atom 补上 ``id`` / ``source`` / ``type`` / ``changelog``。

    模型（若违规）可能带了 ``id``/``source``/``type`` —— 一律**覆盖**成系统值：
    id 顺序编号 ``{source_id}_{k}``，source 恒为 ``[本 message 的 id]``，type 恒为 ``atom``。
    ``time`` 缺省时回落到本 message 的会话时间；``role`` 缺省回落到说话人。空 memory 丢弃。
    同一 message 内按 memory 去重。
    """
    atoms: list[dict[str, Any]] = []
    seen: set[str] = set()
    counter = 0
    for raw in raw_atoms:
        text = memory_text(raw).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        meta = raw.get("metadata") or {}
        counter += 1
        time_value = meta.get("time")
        if not isinstance(time_value, str) or not time_value.strip():
            time_value = default_time
        role_value = meta.get("role")
        if not isinstance(role_value, str) or not role_value.strip():
            role_value = default_role
        tags = _normalize_tags(meta.get("tag"))
        atoms.append({
            "memory": text,
            "metadata": {
                "id": f"{source_id}_{counter}",
                "type": "atom",
                "time": time_value,
                "role": role_value,
                "tag": tags,
                "source": [source_id],
                "changelog": [{"time": now_iso(), "content": "created"}],
            },
        })
    return atoms


# ---------------------------------------------------------------------------
# 单目录
# ---------------------------------------------------------------------------
async def run_dir(
    directory: Path,
    config: dict[str, Any],
    client: openai.AsyncOpenAI,
    catalog: toolcfg.ToolCatalog,
    log: Logger,
    *,
    limit_messages: int = 0,
) -> dict[str, Any]:
    """对一个目录的 msgmem **逐 message 串行**跑 atom 抽取，就地写 atommem.jsonl。

    串行的意义：处理第 N 条 message 时，`atommem.jsonl` 里已经有前 N-1 条产出的 atom，
    agent 能 grep 到它们（"跨 session 影响"就靠这个）。所以每处理一条就**增量落盘**一次。
    """
    label = directory.name
    files = directory_files(directory)
    msg_path = files["msgmem"]
    if not msg_path.exists():
        return {"dir": label, "status": "SKIPPED",
                "reason": f"no {MSGMEM_NAME}（先跑 add --stage msgmem）"}
    messages = read_jsonl(msg_path)
    summaries_by_session = group_by_session(read_jsonl(files["summem"]))
    summaries = {number: records[0] for number, records in summaries_by_session.items() if records}

    atommem_path = files["atommem"]
    existing = read_jsonl(atommem_path)
    # 增量：按 source（本 message 的 id）分组旧 atom；重跑一条 message 就覆盖它那一组。
    existing_by_source: dict[str, list[dict[str, Any]]] = {}
    for record in existing:
        for src in (record.get("metadata") or {}).get("source") or []:
            existing_by_source.setdefault(str(src), []).append(record)

    # source 不在 msgmem 里的历史 atom（例如手动标的）——保留，放到末尾。
    known = {memory_id(m) for m in messages}
    orphans = [r for r in existing if not (
        set((r.get("metadata") or {}).get("source") or []) & known
    )]

    speaker_a, speaker_b = parse_dir_label(label)
    workspace = directory
    runner = _Runner(client, config, log, directory / ".atommem.model_calls.jsonl")

    targets = messages if not limit_messages else messages[:limit_messages]
    for position, target in enumerate(targets):
        session = msg_session_number(memory_id(target))
        prior = _session_prior(messages, position, session)
        node_log = log.bind(f"{label}:{memory_id(target)}")

        def write_current() -> None:
            rebuilt: list[dict[str, Any]] = []
            for msg in messages:
                rebuilt.extend(existing_by_source.get(memory_id(msg), []))
            write_memories(atommem_path, rebuilt + orphans)

        # 处理前先把当前视图落盘一次（增量：第 N 条看见前 N-1 条的 atom）。
        write_current()
        try:
            atoms = await run_message(
                target=target, prior=prior, summaries=summaries, workspace=workspace,
                atommem_path=atommem_path, config=config, catalog=catalog, runner=runner,
                log=node_log, speaker_a=speaker_a, speaker_b=speaker_b,
            )
        except Exception as exc:  # noqa: BLE001 - 单条 message 失败不该让整目录没有输出
            node_log.warn(f"处理失败：{type(exc).__name__}: {exc}")
            atoms = []
        existing_by_source[memory_id(target)] = atoms
        write_current()
        if atoms:
            node_log.info(f"写出 {len(atoms)} 条 atom")

    runner.model.close()

    return {
        "dir": label, "status": "OK",
        "messages": len(targets),
        "atoms": sum(len(existing_by_source.get(memory_id(m), [])) for m in messages),
    }


def _session_prior(
    messages: list[dict[str, Any]], position: int, session: int | None
) -> list[dict[str, Any]]:
    """取 target 之前、**同一会话**的消息作为消解指代的窗口。"""
    if session is None:
        return messages[:position]
    prior: list[dict[str, Any]] = []
    for record in messages[:position]:
        if msg_session_number(memory_id(record)) == session:
            prior.append(record)
    return prior


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.add.atommem",
        description="逐 message 用 agent 抽原子事实，写 atommem.jsonl",
    )
    parser.add_argument("dirs", nargs="*", help="speaker 目录名；缺省处理全部")
    parser.add_argument("--config", default=str(CONFIG_FILE), help="配置（默认 configs/add.yaml）")
    parser.add_argument("--tool-config", default=str(TOOL_CONFIG_FILE),
                        help="add agent 工具配置（默认 configs/add_tool.json）")
    parser.add_argument("--limit-messages", type=int, default=0,
                        help="每个目录只处理前 N 条 message（调试用）")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖目录级并发数")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖 agent 步数上限")
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    if args.concurrency is not None:
        config["concurrency"] = args.concurrency
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps

    targets = targets_with_msgmem(args.dirs)
    if not targets:
        print(f"没有找到含 {MSGMEM_NAME} 的目录（先跑 `python -m codemem.add --stage msgmem`）",
              file=sys.stderr)
        return

    catalog = toolcfg.load_catalog(Path(args.tool_config))
    catalog.validate(["read", "edit", "bash"])
    log = Logger(level="info")
    client = llm.make_client(gen_call_config(config, "atommem"))
    semaphore = asyncio.Semaphore(max(1, int(config.get("concurrency", 8))))

    async def one(directory: Path) -> dict[str, Any]:
        async with semaphore:
            return await run_dir(
                directory, config, client, catalog, log, limit_messages=args.limit_messages,
            )

    try:
        results = list(await asyncio.gather(*(one(d) for d in targets)))
    finally:
        await client.close()

    for item in results:
        if item.get("status") == "OK":
            print(f"atommem   {item['dir']}: {item['messages']} 条 message / "
                  f"{item['atoms']} 条 atom")
        else:
            print(f"atommem   {item['dir']}: {item.get('status')} — {item.get('reason', '')}")


def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(parse_args(argv)))


if __name__ == "__main__":
    main()
