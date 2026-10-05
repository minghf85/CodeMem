"""add · atommem pipeline 版：三阶段处理，无 agent loop。

**设计原则**：
- memory 字段**完全不含时间词**（yesterday/ago/next/last/recently/soon/future）
- 时间信息由后处理解析到 metadata.time
- 一次 LLM 调用产出完整 JSON array
- 召回与后处理都是纯 CPU

**三阶段**：
1. 召回（grep msgmem/atommem）
2. 抽取（单次 LLM → [{memory, time_expr, role, tag}, ...]）
3. 后处理（时间解析 + 去重 + 补字段）
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .. import llm
from ..io import memory_id, memory_role, memory_text, now_iso, read_jsonl
from ..log import Logger
from .atommem import (
    TAG_KEYS,
    _TAG_ALIASES,
    _dedup_atoms,
    _dedup_key,
    _max_suffix,
    _merge_ids,
    _normalize_tags,
)

# ============================================================================
# Prompt
# ============================================================================

ATOM_EXTRACTION_SYSTEM = """You are extracting atomic facts from a conversation message.

**CRITICAL RULE**: The `memory` field must NEVER contain ANY time expressions.
Forbidden words in memory: yesterday, today, tomorrow, ago, last, next, past, recent, recently, soon, future, now, currently, presently, before, after, since, until, during, N days, N weeks, N months, N years

Instead:
- Put the original time expression in `time_expr` (e.g., "yesterday", "last year")
- Write the memory as a timeless fact (e.g., "Melanie got married" NOT "Melanie got married 5 years ago")

**Output format**: JSON array of atoms. Each atom:
```json
{
  "memory": "Pure factual statement WITHOUT any time words",
  "time_expr": "Original time expression from the source message, or empty string if no time",
  "role": "speaker_a or speaker_b",
  "tag": ["key:value", ...]
}
```

**Rules**:
1. `memory`: Timeless fact. Resolve pronouns (she/he/it/there → real names). NO time words.
2. `time_expr`: Extract original time expression ("yesterday", "last week", "5 years ago"). Leave empty if message has no time reference.
3. `role`: "speaker_a" or "speaker_b"
4. `tag`: Array of "key:value". Keys MUST be from: speaker, topic, activity, action, event, state, sentiment, relation, entity, possession, time, other
5. One atom = one sentence, one fact. NO multiple sentences in one memory.
6. Capture specifics (names, numbers, titles). Don't abstract "read a book" to "enjoys reading".
7. Skip greetings, filler, pure questions without content.

**Examples**:

Message: "I got married yesterday!"
→ [{"memory": "Melanie got married", "time_expr": "yesterday", "role": "speaker_b", "tag": ["event:marriage", "relation:spouse"]}]

Message: "I've been working at Google for 3 years."
→ [{"memory": "Melanie works at Google", "time_expr": "for 3 years", "role": "speaker_b", "tag": ["topic:job", "entity:Google"]}]

Message: "She went to the lake last summer."
→ [{"memory": "Caroline went to the lake", "time_expr": "last summer", "role": "speaker_a", "tag": ["activity:travel", "entity:lake"]}]

Message: "I love hiking and my daughter loves it too."
→ [
  {"memory": "Melanie loves hiking", "time_expr": "", "role": "speaker_b", "tag": ["activity:hiking", "sentiment:love"]},
  {"memory": "Melanie's daughter loves hiking", "time_expr": "", "role": "speaker_b", "tag": ["activity:hiking", "relation:daughter", "sentiment:love"]}
]
"""

ATOM_EXTRACTION_USER = """**Conversation context** (for resolving pronouns and references):
{context}

**Target message to extract**:
[{msg_id}] ({msg_time}) {role}: {content}

Output a JSON array of atomic facts following the rules. Remember: NO time words in `memory`."""


# ============================================================================
# Stage 1: 召回
# ============================================================================

def recall_context(
    target_id: str,
    msgmem: list[dict[str, Any]],
    atommem: list[dict[str, Any]],
    window: int = 5,
) -> str:
    """召回与 target 相关的上下文（同会话前 N 条消息）。"""
    # 找到 target 的位置
    target_pos = -1
    for i, msg in enumerate(msgmem):
        if memory_id(msg) == target_id:
            target_pos = i
            break

    if target_pos < 0:
        return "(no prior context)"

    # 同会话前 window 条
    target_session = _session_number(target_id)
    prior: list[dict[str, Any]] = []
    for msg in msgmem[max(0, target_pos - window * 3):target_pos]:
        if _session_number(memory_id(msg)) == target_session:
            prior.append(msg)

    if not prior:
        return "(first message of session)"

    # 渲染成文本
    lines = []
    for msg in prior[-window:]:
        mid = memory_id(msg)
        role = memory_role(msg) or "unknown"
        when = (msg.get("metadata") or {}).get("time", "")
        text = memory_text(msg)
        lines.append(f"[{mid}] ({when}) {role}: {text}")

    return "\n".join(lines)


def _session_number(msg_id: str) -> int | None:
    """从 session_1_3 提取 1"""
    match = re.match(r"session_(\d+)_", msg_id)
    return int(match.group(1)) if match else None


# ============================================================================
# Stage 2: 抽取（单次 LLM）
# ============================================================================

async def extract_atoms_llm(
    target: dict[str, Any],
    context: str,
    client: Any,
    config: dict[str, Any],
    log: Logger,
    speaker_a: str = "speaker_a",
    speaker_b: str = "speaker_b",
) -> list[dict[str, Any]]:
    """单次 LLM 调用，产出 atom 数组。"""
    msg_id = memory_id(target)
    msg_time = (target.get("metadata") or {}).get("time", "")
    role = memory_role(target) or "unknown"
    content = memory_text(target)

    user_prompt = ATOM_EXTRACTION_USER.format(
        context=context,
        msg_id=msg_id,
        msg_time=msg_time,
        role=role,
        content=content,
    )

    system_prompt = ATOM_EXTRACTION_SYSTEM.replace("speaker_a", speaker_a).replace("speaker_b", speaker_b)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_text = await llm.chat_completion(
            client=client,
            config=config,
            messages=messages,
        )

        # 解析 JSON（可能被 markdown 代码块包裹）
        cleaned = re.sub(r"```json\s*", "", raw_text)
        cleaned = re.sub(r"```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        atoms = json.loads(cleaned)
        if not isinstance(atoms, list):
            log.warn(f"{msg_id}: LLM 返回不是数组，尝试包装")
            atoms = [atoms]

        return atoms

    except json.JSONDecodeError as e:
        log.warn(f"{msg_id}: JSON 解析失败 - {e}")
        return []
    except Exception as e:
        log.warn(f"{msg_id}: LLM 调用失败 - {type(e).__name__}: {e}")
        return []


# ============================================================================
# Stage 3: 后处理
# ============================================================================

def parse_time_expression(time_expr: str, anchor_time: str) -> str:
    """把 time_expr + 锚点时间 → 绝对时间（用 date -d）。

    返回：
    - ISO 日期（"2023-05-07"）
    - 带锚点的相对表述（"the week before 2023-05-08"）
    - 区间（"2022"、"2023-05-01/2023-05-07"）
    - 空字符串（无法解析或无时间）
    """
    if not time_expr or not time_expr.strip():
        return ""

    expr = time_expr.strip().lower()
    if not anchor_time or not anchor_time.strip():
        # 没有锚点，保留原表述
        return time_expr

    # 提取锚点里的日期部分（"1:56 pm on 8 May, 2023" → "8 May 2023"）
    anchor_date = _extract_date_from_anchor(anchor_time)
    if not anchor_date:
        return f"{time_expr} (anchor: {anchor_time})"

    # 尝试用 date -d 解析
    # 支持的格式：yesterday, last week, 3 days ago, next month 等
    normalized = _normalize_expr(expr)
    try:
        cmd = ["date", "-d", f"{anchor_date} {normalized}", "+%Y-%m-%d"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
            env={"LC_ALL": "C"},  # 确保英文输出
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    # date 解析失败，尝试模式匹配提取粒度
    # "last year" → 年份
    if "last year" in expr:
        match = re.search(r"(\d{4})", anchor_date)
        if match:
            year = int(match.group(1)) - 1
            return str(year)

    # "N years ago"
    match = re.search(r"(\d+)\s+years?\s+ago", expr)
    if match:
        n = int(match.group(1))
        try:
            cmd = ["date", "-d", f"{anchor_date} -{n} year", "+%Y"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, env={"LC_ALL": "C"})
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

    # "for N years" → 区间
    match = re.search(r"for\s+(\d+)\s+years?", expr)
    if match:
        n = int(match.group(1))
        try:
            cmd = ["date", "-d", f"{anchor_date} -{n} year", "+%Y-%m-%d"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, env={"LC_ALL": "C"})
            if result.returncode == 0:
                start = result.stdout.strip()
                cmd2 = ["date", "-d", f"{anchor_date}", "+%Y-%m-%d"]
                result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=5, env={"LC_ALL": "C"})
                if result2.returncode == 0:
                    end = result2.stdout.strip()
                    return f"{start}/{end}"
        except Exception:
            pass

    # 无法解析，保留原表述 + 锚点
    return f"{time_expr} (anchor: {anchor_time})"


def _extract_date_from_anchor(anchor_time: str) -> str:
    """从锚点时间字符串提取 date 命令能识别的日期部分。

    "1:56 pm on 8 May, 2023" → "8 May 2023"
    "8 May 2023" → "8 May 2023"
    """
    # 尝试提取 "on <date>" 格式
    match = re.search(r"on\s+(.+)$", anchor_time)
    if match:
        date_part = match.group(1).strip()
        # 去掉逗号（date 命令不需要）
        return date_part.replace(",", "")

    # 尝试直接匹配日期格式 "8 May 2023"
    match = re.search(r"(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})", anchor_time)
    if match:
        return match.group(1).replace(",", "")

    # 尝试 ISO 格式
    match = re.search(r"(\d{4}-\d{2}-\d{2})", anchor_time)
    if match:
        return match.group(1)

    return anchor_time


def _normalize_expr(expr: str) -> str:
    """把口语化表述转成 date -d 能理解的格式。"""
    # yesterday → -1 day
    # last week → -1 week
    # 3 days ago → -3 days
    # next month → +1 month

    expr = expr.replace("yesterday", "-1 day")
    expr = expr.replace("last week", "-1 week")
    expr = expr.replace("last month", "-1 month")
    expr = expr.replace("last year", "-1 year")
    expr = expr.replace("next week", "+1 week")
    expr = expr.replace("next month", "+1 month")
    expr = expr.replace("next year", "+1 year")

    # N days/weeks/months/years ago → -N days/weeks/months/years
    expr = re.sub(r"(\d+)\s+(day|week|month|year)s?\s+ago", r"-\1 \2", expr)

    return expr


def validate_memory_no_time(memory: str) -> bool:
    """检查 memory 是否含有禁止的时间词。"""
    forbidden = [
        r"\byesterday\b", r"\btoday\b", r"\btomorrow\b",
        r"\bago\b", r"\blast\b", r"\bnext\b", r"\bpast\b",
        r"\brecent\b", r"\brecently\b", r"\bsoon\b",
        r"\bfuture\b", r"\bnow\b", r"\bcurrently\b", r"\bpresently\b",
        r"\bbefore\b", r"\bafter\b", r"\bsince\b", r"\buntil\b", r"\bduring\b",
        r"\d+\s+(day|week|month|year)s?(\s+ago)?",
    ]
    for pattern in forbidden:
        if re.search(pattern, memory, re.IGNORECASE):
            return False
    return True


def postprocess_atoms(
    raw_atoms: list[dict[str, Any]],
    source_id: str,
    anchor_time: str,
    default_role: str,
    existing: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """后处理：时间解析 + 验证 + 补字段。

    返回完整的 atom 记录（带 metadata.{id, type, time, role, tag, source, changelog}）。
    """
    counter = _max_suffix(set(existing.keys()), source_id)
    processed: list[dict[str, Any]] = []

    for raw in raw_atoms:
        memory = (raw.get("memory") or "").strip()
        if not memory:
            continue

        # 验证 memory 不含时间词
        if not validate_memory_no_time(memory):
            # 违规但先不丢弃，只标记警告
            pass

        # 解析时间
        time_expr = (raw.get("time_expr") or "").strip()
        parsed_time = parse_time_expression(time_expr, anchor_time) if time_expr else ""

        # role
        role = (raw.get("role") or "").strip() or default_role

        # tag
        tags = _normalize_tags(raw.get("tag"))

        # 分配 id
        counter += 1
        atom_id = f"{source_id}_{counter}"

        # 组装
        atom = {
            "memory": memory,
            "metadata": {
                "id": atom_id,
                "type": "atom",
                "time": parsed_time,
                "role": role,
                "tag": tags,
                "source": [source_id],
                "changelog": [{"time": now_iso(), "content": "created"}],
            },
        }
        processed.append(atom)

    return processed


# ============================================================================
# 主流程
# ============================================================================

async def run_message_pipeline(
    target: dict[str, Any],
    msgmem: list[dict[str, Any]],
    existing_atoms: list[dict[str, Any]],
    client: Any,
    config: dict[str, Any],
    log: Logger,
    speaker_a: str = "speaker_a",
    speaker_b: str = "speaker_b",
) -> list[dict[str, Any]]:
    """三阶段 pipeline 处理一条 message。

    返回：新产出的 atom 列表（已补全系统字段，未去重）。
    """
    msg_id = memory_id(target)
    anchor_time = (target.get("metadata") or {}).get("time", "")
    default_role = memory_role(target) or "unknown"

    # Stage 1: 召回
    context = recall_context(
        msg_id,
        msgmem,
        existing_atoms,
        window=int(config.get("context_window", 5)),
    )

    # Stage 2: 抽取
    raw_atoms = await extract_atoms_llm(
        target, context, client, config, log, speaker_a, speaker_b
    )

    if not raw_atoms:
        log.info(f"{msg_id}: 未抽取到 atom")
        return []

    # Stage 3: 后处理
    existing_by_id = {memory_id(a): a for a in existing_atoms if memory_id(a)}
    processed = postprocess_atoms(
        raw_atoms, msg_id, anchor_time, default_role, existing_by_id
    )

    log.info(f"{msg_id}: 产出 {len(processed)} 条 atom")
    return processed
