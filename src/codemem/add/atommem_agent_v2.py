"""add · atommem agent v2：并行处理 + 充分上下文。

**设计原则**：
- 每个 message 看到它**之前的全部内容**（前面所有 session + 本 session 前面的 message）
- 并行处理（不依赖前面 message 的产出）
- 保证稳定产出：工具调用简化、错误重试

**核心改进**：
1. 上下文充分：处理 session_3_8 时，能看到 session 1、2 全部 + session 3 的 1-7
2. 并行安全：每个 message 的 agent 独立运行，不依赖其他 message 的产出
3. 工具简化：只用 read/edit/bash，默认写到 {msg_id}_atoms.jsonl（独立文件，后合并）
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import openai

from .. import llm
from ..io import memory_id, memory_role, memory_text, now_iso, read_jsonl
from ..log import Logger
from ..search import agent as agent_mod
from ..search import toolcfg, tools as tools_mod
from .atommem import (
    TAG_KEYS,
    _TAG_ALIASES,
    _dedup_atoms,
    _max_suffix,
    _merge_ids,
    _normalize_tags,
    _strip_relative_time,
    gen_call_config,
    parse_dir_label,
)
from .memory import ATOMMEM_NAME, MSGMEM_NAME, SUMMEM_NAME, group_by_session, write_memories

# ============================================================================
# Agent prompt（使用 ToolCatalog 渲染完整 system prompt）
# ============================================================================

def _build_system_prompt(catalog: toolcfg.ToolCatalog, output_file: str, msg_id: str) -> str:
    """构建 system prompt：**直接用 catalog 渲染**（与串行 agent 同一份，含强拆分规则）。

    只额外加一句"产出写到哪个文件"—— 绝不覆盖/弱化 catalog 里的原子标准
    （曾经加过一版"One atom = one sentence"前缀，实测把 `ATOM_AGENT_STANDARD` 的
    "拆分复合句"规则压掉了，导致一条消息只出一个合并原子 → 检索命中率下降）。
    """
    base = catalog.render_prompt(f"Extract the atomic facts of message {msg_id}")
    head = (
        f"Write the target message's atoms into `{output_file}` (default for `edit`).\n\n"
    )
    return head + base

AGENT_USER = """**Target message to extract**:
[{msg_id}] ({msg_time}) {role}: {content}

**Session list** (for reference):
{sessions}

**Output to**: {output_file}

Extract atomic facts from this message. Use tools to search context if needed. When done, reply "Done"."""


# ============================================================================
# 并行 agent
# ============================================================================

async def run_message_agent(
    target: dict[str, Any],
    workspace: Path,
    output_file: str,
    summaries: dict[int, dict[str, Any]],
    config: dict[str, Any],
    catalog: toolcfg.ToolCatalog,
    client: openai.AsyncOpenAI,
    log: Logger,
    speaker_a: str = "speaker_a",
    speaker_b: str = "speaker_b",
) -> list[dict[str, Any]]:
    """并行处理一条 message：agent 独立搜索上下文、产出 atom 到独立文件。

    返回：该 message 产出的 atom 列表（已补全系统字段）。
    """
    msg_id = memory_id(target)
    msg_time = (target.get("metadata") or {}).get("time", "")
    role = memory_role(target) or "unknown"
    content = memory_text(target)

    log.debug(f"处理 {msg_id}: {content[:50]}...")

    # 渲染 prompt（使用 catalog 生成完整的工具描述）
    sessions_text = _render_sessions(summaries)
    system = _build_system_prompt(catalog, output_file, msg_id).replace(
        "Caroline", speaker_a
    ).replace("Melanie", speaker_b)
    user = AGENT_USER.format(
        msg_id=msg_id,
        msg_time=msg_time,
        role=role,
        content=content,
        sessions=sessions_text,
        output_file=output_file,
    )

    # 构建工具
    from ..search.runner import build_shell_prefix

    shell_prefix = build_shell_prefix(inputs_dir=workspace / "inputs", config={})
    tools = tools_mod.build_tools(
        cwd=workspace,
        shell_command_prefix=shell_prefix,
        bash_timeout=float(config.get("bash_timeout", 60)),
        default_write_file=output_file,
    )
    catalog.apply(tools)

    # Agent loop
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    output_path = workspace / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # **总是清空**：重跑同一条 message 时，残留的旧产出会被当成新原子读入，导致重复。
    output_path.write_text("", encoding="utf-8")

    from ..search.runner import ModelRunner

    call_config = gen_call_config(config, "atommem")
    log.debug(f"模型配置: enable_thinking={call_config.get('enable_thinking')}, model={call_config.get('model')}")
    model_runner = ModelRunner(client, call_config, log, None)

    try:
        # 使用标准的 agent round
        from ..search import agent as agent_mod

        outcome, _ = await agent_mod.run_agent_round(
            messages=messages,
            tools=tools,
            model_call=model_runner,
            evidence_path=output_path,
            cwd=workspace,
            max_steps=int(config.get("max_steps", 8)),
            no_progress_patience=int(config.get("no_progress_patience", 8)),
            step_offset=0,
            round_index=1,
            log=log,
            observation_max_chars=int(config.get("observation_max_chars", 8000)),
            write_nudge_after=int(config.get("write_nudge_after", 3)),
            nudge_filename=output_file,
            evidence_soft_limit=0,
            max_parse_retries=int(config.get("max_parse_retries", 2)),
            repeat_limit=int(config.get("repeat_limit", 2)),
            duplicate_warn_at=10,
            context_max_chars=int(config.get("context_max_chars", 120000)),
            context_keep_recent=int(config.get("context_keep_recent", 6)),
            forget_after_turns=0,
            long_observation_chars=0,
            forget_keep_recent_turns=2,
        )

    finally:
        pass  # model_runner 由调用方统一关闭

    # 读取产出并补全字段
    raw_atoms = read_jsonl(output_path)
    processed = []
    counter = 0

    for raw in raw_atoms:
        memory = (raw.get("memory") or "").strip()
        if not memory:
            continue

        meta = dict(raw.get("metadata") or {})
        time_value = str(meta.get("time", "")).strip()
        role_value = str(meta.get("role", "")).strip() or role
        tags = _normalize_tags(meta.get("tag"))

        # 解析时间
        from .atommem_time import parse_time_expression
        import re
        import subprocess

        # 1. 如果 agent 直接复制了会话时间戳（含 "pm on" / "am on"），提取日期
        if time_value and ('pm on' in time_value.lower() or 'am on' in time_value.lower()):
            match = re.search(r'(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})', time_value)
            if match:
                day, month, year = match.groups()
                time_value = f"{day} {month} {year}"

        # 2. 规范化所有日期格式为 YYYY-MM-DD
        # 匹配: "8 May 2023", "8 May, 2023", "May 8 2023" 等
        if time_value and not time_value.startswith('20') and re.search(r'\d+.*[A-Za-z].*\d{4}', time_value):
            try:
                # 去掉逗号（date -d 不接受 "8 May, 2023"）
                normalized = time_value.replace(',', '')
                result = subprocess.run(
                    ['date', '-d', normalized, '+%Y-%m-%d'],
                    capture_output=True, text=True, timeout=5, env={'LC_ALL': 'C'}
                )
                if result.returncode == 0:
                    time_value = result.stdout.strip()
            except:
                pass

        # 3. 如果为空或含相对词，用 parse_time_expression 解析
        if not time_value or any(word in time_value.lower() for word in
                                   ['yesterday', 'last', 'next', 'ago', 'recently', 'soon']):
            # 尝试从 memory 中提取时间表述
            time_expr = time_value or ""
            if not time_expr:
                # 简单启发：从 memory 里找常见时间词
                for pattern in ['yesterday', 'last week', 'last month', 'last year']:
                    if pattern in memory.lower():
                        time_expr = pattern
                        break

            if time_expr:
                parsed = parse_time_expression(time_expr, msg_time)
                time_value = parsed

        # 4. harness 兜底：memory 正文不得留裸相对时间词（8B 实测 7.5% 会犯）。
        #    此时 time 已解析好，直接把这些措辞从句子里剥掉即可（不是改写事实）。
        memory = _strip_relative_time(memory)

        counter += 1
        atom_id = f"{msg_id}_{counter}"

        processed.append({
            "memory": memory,
            "metadata": {
                "id": atom_id,
                "type": "atom",
                "time": time_value,
                "role": role_value,
                "tag": tags,
                "source": [msg_id],
                "changelog": [{"time": now_iso(), "content": "created"}],
            },
        })

    log.info(f"{msg_id}: 产出 {len(processed)} 条 atom")
    return processed


# 裸相对时间剥离：唯一实现已上移到 ``atommem._strip_relative_time``（serial agent 与 v2 共用），
# 见文件顶部的 import。


def _render_sessions(summaries: dict[int, dict[str, Any]]) -> str:
    """会话清单：每行 ``session_N (时间)``，不含事实内容（只给 agent 一个定位路标）。"""
    if not summaries:
        return "(no sessions)"
    lines = []
    for number in sorted(summaries):
        when = str((summaries[number].get("metadata") or {}).get("time") or "")
        lines.append(f"  session_{number}" + (f"  ({when})" if when else ""))
    return "\n".join(lines)


# ============================================================================
# 目录级并行处理
# ============================================================================

async def run_dir_parallel(
    directory: Path,
    config: dict[str, Any],
    client: openai.AsyncOpenAI,
    catalog: toolcfg.ToolCatalog,
    log: Logger,
    *,
    limit_messages: int = 0,
) -> dict[str, Any]:
    """并行处理整个目录的 message。"""
    label = directory.name
    msg_path = directory / MSGMEM_NAME
    if not msg_path.exists():
        return {"dir": label, "status": "SKIPPED", "reason": f"no {MSGMEM_NAME}"}

    messages = read_jsonl(msg_path)
    summaries_by_session = group_by_session(read_jsonl(directory / SUMMEM_NAME))
    summaries = {
        number: records[0] for number, records in summaries_by_session.items() if records
    }

    speaker_a, speaker_b = parse_dir_label(label)
    workspace = directory / ".atommem_workspace"
    workspace.mkdir(exist_ok=True)

    # 建立 inputs 链接（所有 agent 共享）
    inputs_dir = workspace / "inputs"
    if not inputs_dir.exists():
        inputs_dir.symlink_to(directory)

    targets = messages if not limit_messages else messages[:limit_messages]

    # 并行处理每个 message
    semaphore = asyncio.Semaphore(int(config.get("concurrency", 4)))

    async def one(target: dict[str, Any]) -> list[dict[str, Any]]:
        async with semaphore:
            msg_id = memory_id(target)
            output_file = f"{msg_id}_atoms.jsonl"
            node_log = log.bind(f"{label}:{msg_id}")
            try:
                return await run_message_agent(
                    target=target,
                    workspace=workspace,
                    output_file=output_file,
                    summaries=summaries,
                    config=config,
                    catalog=catalog,
                    client=client,
                    log=node_log,
                    speaker_a=speaker_a,
                    speaker_b=speaker_b,
                )
            except Exception as exc:  # noqa: BLE001
                node_log.warn(f"处理失败：{type(exc).__name__}: {exc}")
                return []

    results = await asyncio.gather(*(one(t) for t in targets))

    # 合并所有 atom
    all_atoms = []
    for atoms in results:
        all_atoms.extend(atoms)

    # 去重
    all_atoms = _dedup_atoms(all_atoms)

    # 写回 atommem.jsonl
    atommem_path = directory / ATOMMEM_NAME
    write_memories(atommem_path, all_atoms)

    return {
        "dir": label,
        "status": "OK",
        "messages": len(targets),
        "atoms": len(all_atoms),
    }
