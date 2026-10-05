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
    gen_call_config,
    parse_dir_label,
)
from .memory import ATOMMEM_NAME, MSGMEM_NAME, SUMMEM_NAME, group_by_session, write_memories

# ============================================================================
# Agent prompt（使用 ToolCatalog 渲染完整 system prompt）
# ============================================================================

def _build_system_prompt(catalog: toolcfg.ToolCatalog, output_file: str, msg_id: str) -> str:
    """构建完整的 system prompt（使用 catalog 渲染工具描述）。

    catalog.render_prompt() 会替换 {{TOOLS}} 等槽位，但它期望 {{QUESTION}} ——
    这里我们把"提取目标"当作问题来用。
    """
    # 用一个伪问题占位，后面会被 user message 里的实际内容覆盖
    base = catalog.render_prompt(f"Extract atomic facts from message {msg_id}")

    # 在 base 前面加上我们的特定指令（工具描述已经在 base 里了）
    prefix = f"""You are extracting atomic facts from ONE conversation message into `{output_file}`.

**CRITICAL RULES**:
1. `memory` field: Pure factual statement WITHOUT any time words (yesterday, ago, last, next, recently, soon, etc.)
2. `metadata.time`: Original time expression from source OR empty string
3. `metadata.role`: speaker name (Caroline or Melanie)
4. `metadata.tag`: Array of "key:value". Keys MUST be from: {', '.join(TAG_KEYS)}
5. One atom = one sentence, one fact. NO multiple sentences.
6. Resolve pronouns (she/he/it/there → real names) using prior context.

**WORKFLOW**:
1. Use `bash`/`read` to search prior messages/summaries if you need context (to resolve pronouns, references)
2. Extract facts from the target message
3. For each fact, `edit {output_file}` with `append`:
   {{"memory": "Pure fact without time words", "metadata": {{"time": "original time expr or empty", "role": "speaker", "tag": ["key:value", ...]}}}}
4. When done, reply with plain text "Done" (no JSON)

**EXAMPLES**:

Target: "I got married yesterday!" (speaker: Melanie)
→ {{"tool":"edit","args":{{"append":'{{"memory":"Melanie got married","metadata":{{"time":"yesterday","role":"Melanie","tag":["event:marriage"]}}}}\\n'}}}}

Target: "She went camping last week." (need to find who "she" is)
→ First: {{"tool":"bash","args":{{"command":"grep -i 'caroline\\|melanie' inputs/msgmem.jsonl | tail -5"}}}}
→ Then: {{"tool":"edit","args":{{"append":'{{"memory":"Caroline went camping","metadata":{{"time":"last week","role":"Caroline","tag":["activity:camping"]}}}}\\n'}}}}

**IMPORTANT**:
- NO time words in `memory` (put them in `metadata.time`)
- Resolve ALL pronouns using context
- One sentence per atom
- When done extracting, reply "Done" in plain text (not JSON)

---

"""
    return prefix + base

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
    if not output_path.exists():
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
        from .atommem_pipeline import parse_time_expression
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


# 裸相对时间词/短语：memory 正文里出现即剥离（时间已进 metadata.time）。
# 顺序重要——先长后短，避免 "last week" 被 "last" 提前吃掉留下残词。
_RELATIVE_TIME_PATTERNS = (
    r"\s*\bthe\s+(?:week|month|year|day|weekend|morning|afternoon|evening)\s+before\b[^.,;]*",
    r"\s*\b\d+\s+(?:day|week|month|year|weekend)s?\s+(?:before|after|ago)\b[^.,;]*",
    r"\s*\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|several|few|couple of)\s+"
    r"(?:day|week|month|year|weekend)s?\s+ago\b",
    r"\s*\blast\s+(?:night|week|weekend|month|year|summer|winter|spring|autumn|fall|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\s*\bnext\s+(?:week|weekend|month|year|summer|winter|spring|autumn|fall|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\s*\b(?:yesterday|today|tomorrow|recently|currently|presently)\b",
    r"\s*\bsince\s+(?:starting|coming|their|her|his|the|last)\b[^.,;]*",
    r"\s*\bin\s+the\s+past\s+(?:few|several|\d+)\s+(?:day|week|month|year)s?\b",
)
_RELATIVE_TIME_RE = tuple(re.compile(p, re.IGNORECASE) for p in _RELATIVE_TIME_PATTERNS)


def _strip_relative_time(memory: str) -> str:
    """剥掉 memory 正文里的裸相对时间词（时间信息已由 metadata.time 承载）。

    8B 实测约 7.5% 的 atom 会把 "last year" / "4 years ago" 留在正文里，而 prompt 的软约束
    拦不住（add 阶段的老问题：纪律要下沉到 harness）。这里只在**时间已解析成功**时才调用，
    所以剥掉这些措辞不丢信息 —— 读者从 ``time`` 就能拿到同一件事的时间。

    只做**删除**，不做改写/替换：删完压一下多余空白与句首标点。三条护栏保证不制造残句 ——
    太短（时间即主体）、删得太多（>50%）、或删完以虚词/悬空分词结尾（"figurines bought"、
    "a lot happened since"）时一律**保留原文**：宁可留个时间词，也别留个读不通的句子。
    """
    if not memory:
        return memory
    cleaned = memory
    for pattern in _RELATIVE_TIME_RE:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;!?])", r"\1", cleaned).strip()
    if len(cleaned.split()) < 3 or len(cleaned) < len(memory) * 0.5:
        return memory
    tail = cleaned.rsplit(" ", 1)[-1].strip(".,;!?").lower()
    if tail in _DANGLING_TAIL_WORDS:
        return memory
    return cleaned


# 剥离后若以这些词结尾，句子多半悬空了（"Caroline mentions a lot happened" / "figurines bought"）——
# 这是"删掉了时间状语，剩下的补语没了宾语"的形态，直接保留原文。
_DANGLING_TAIL_WORDS: frozenset[str] = frozenset(
    "of for in at on to the a an and with since after before by from into over under about "
    "bought bought given made taken done seen gone come known used held told sent kept found "
    "left set put run cut hurt read meant spent built brought thought caught taught paid "
    "happened occurred started began ended continued mentioned discussed".split()
)


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
