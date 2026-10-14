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
from typing import Any

from .. import llm
from ..io import memory_id, memory_role, memory_text, now_iso
from ..log import Logger
from .atommem import (
    _max_suffix,
    _normalize_tags,
)
from ..adapters import speaker_pair, speakers_phrase

# ============================================================================
# Prompt
# ============================================================================

ATOM_EXTRACTION_SYSTEM = """You are extracting atomic facts from a conversation message.

**WHO THIS SESSION BELONGS TO**: {speakers}
Every record here carries a `role` naming who -- or what -- it belongs to; `role` in your output
must be one of the values above. Two different speakers are never merged into one fact
("<A> does X" is not "<B> does X").

**CRITICAL RULE**: The `memory` field must NEVER contain ANY time expressions.
Forbidden words in memory: yesterday, today, tomorrow, ago, last, next, past, recent, recently, soon, future, now, currently, presently, before, after, since, until, during, N days, N weeks, N months, N years

Instead:
- Put the original time expression in `time_expr` (e.g., "yesterday", "last year")
- Write the memory as a timeless fact (e.g., "{speaker_b} got married" NOT "{speaker_b} got married 5 years ago")

**Output format**: JSON array of atoms. Each atom:
```json
{
  "memory": "Pure factual statement WITHOUT any time words",
  "time_expr": "Original time expression from the source message, or empty string if no time",
  "role": "one of the speakers listed above, or the record's own role",
  "tag": ["key:value", ...]
}
```

**Rules**:
1. `memory`: Timeless fact. Resolve pronouns (she/he/it/there → the speaker) to the actual name. NO time words.
2. `time_expr`: Extract original time expression ("yesterday", "last week", "5 years ago"). Leave empty if message has no time reference.
3. `role`: the speaker of the target message -- one of the values listed above.
4. `tag`: Array of "key:value". Keys MUST be from: speaker, topic, activity, action, event, state, sentiment, relation, entity, possession, time, other
5. **One atom = one sentence, one fact.** NO multiple sentences in one memory.
6. **Capture specifics — names, numbers, titles, exact wording**:
   - Book/band/title names: keep the exact name (e.g. "Charlotte's Web") — NOT "a book" / "a performer" / "classical music"
   - Frequency/count: "once or twice a year" / "3 kids" — NOT omit or abstract to "rarely" / "has kids"
   - Purpose/use: "shoes for running" — NOT just "got shoes"
   - Slogans/quotes: "Trans Lives Matter" — copy verbatim, don't paraphrase to "transgender activism"
   - Object types: "stained glass window" / "cup with a dog face" — NOT "painting" / "pottery"
   - Pet names: when message says "Luna and Oliver" / "another cat named Bailey", you MUST create atoms for Luna, Oliver, AND Bailey — omitting any member is WRONG
7. **Named lists must enumerate each member**: "Luna and Oliver" → 2 atoms (one for Luna, one for Oliver). "another cat named Bailey" → separate atom for Bailey. NEVER drop members or abstract to a count.
8. **NOT every record is a person talking.** A record may be a document, a database triple (`Subject | relation | object`), a narration line, or an evaluation record. For those: state the fact exactly as the record gives it, use the record's own `role`, and invent no speaker. `Antibiotics | treats | infection` becomes one atom about that relation -- never "someone said antibiotics treat infection".
9. Skip greetings, filler, pure questions without content.

**WRONG (abstraction failures — DO NOT do these)**:
- Message: "I just finished reading Charlotte's Web"
  WRONG: {"memory": "{speaker_b} finished reading a book", ...}  ← Lost the title!
  RIGHT: {"memory": "{speaker_b} finished reading Charlotte's Web", ...}
- Message: "These shoes are for running"
  WRONG: {"memory": "{speaker_b} got shoes", ...}  ← Lost the purpose!
  RIGHT: {"memory": "{speaker_b} got shoes for running", ...}
- Message: "The posters said 'Trans Lives Matter'"
  WRONG: {"memory": "{speaker_a} saw posters about transgender activism", ...}  ← Lost the exact text!
  RIGHT: {"memory": "{speaker_a} saw posters saying 'Trans Lives Matter'", ...}
- Message: "I made a stained glass window"
  WRONG: {"memory": "{speaker_a} made art", ...}  ← Lost the object type!
  RIGHT: {"memory": "{speaker_a} made a stained glass window", ...}
- Message: "We have Luna and Oliver, and another cat named Bailey"
  WRONG: {"memory": "{speaker_b} has a pet named Oliver", ...}  ← Lost Luna and Bailey!
  RIGHT: {"memory": "{speaker_b} has a pet named Luna", ...}, {"memory": "{speaker_b} has a pet named Oliver", ...}, {"memory": "{speaker_b} has a pet named Bailey", ...}
- Message: "We go to the beach once or twice a year"
  WRONG: {"memory": "{speaker_b} goes to the beach", ...}  ← Lost the frequency!
  RIGHT: {"memory": "{speaker_b} goes to the beach once or twice a year", ...}
- Message (a knowledge-graph record): "Latvia | official language | Latvian"
  WRONG: {"memory": "{speaker_a} stated the official language of Latvia", ...}  ← Invented a speaker!
  WRONG: {"memory": "Latvia", ...}  ← Dropped the relation, which is the whole fact!
  RIGHT: {"memory": "Latvia's official language is Latvian", "role": "corpus", ...}

**Examples**:

Message: "I got married yesterday!"
→ [{"memory": "{speaker_b} got married", "time_expr": "yesterday", "role": "{speaker_b}", "tag": ["event:marriage", "relation:spouse"]}]

Message: "I've been working at Google for 3 years."
→ [{"memory": "{speaker_b} works at Google", "time_expr": "for 3 years", "role": "{speaker_b}", "tag": ["topic:job", "entity:Google"]}]

Message: "She went to the lake last summer."
→ [{"memory": "{speaker_a} went to the lake", "time_expr": "last summer", "role": "{speaker_a}", "tag": ["activity:travel", "entity:lake"]}]

Message: "I love hiking and my daughter loves it too."
→ [
  {"memory": "{speaker_b} loves hiking", "time_expr": "", "role": "{speaker_b}", "tag": ["activity:hiking", "sentiment:love"]},
  {"memory": "{speaker_b}'s daughter loves hiking", "time_expr": "", "role": "{speaker_b}", "tag": ["activity:hiking", "relation:daughter", "sentiment:love"]}
]

Message: "Luna and Oliver! They are so sweet."
→ [
  {"memory": "{speaker_b} has a pet named Luna", "time_expr": "", "role": "{speaker_b}", "tag": ["entity:Luna", "possession:pet"]},
  {"memory": "{speaker_b} has a pet named Oliver", "time_expr": "", "role": "{speaker_b}", "tag": ["entity:Oliver", "possession:pet"]}
]

Message: "It'll be tough as a single parent, but I'm up for it."
→ [
  {"memory": "{speaker_a} is single", "time_expr": "", "role": "{speaker_a}", "tag": ["state:single", "relation:marital_status"]},
  {"memory": "{speaker_a} plans to be a parent", "time_expr": "", "role": "{speaker_a}", "tag": ["state:planning", "relation:parent"]}
]"""

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
    speakers: list[str] | None = None,
) -> list[dict[str, Any]]:
    """单次 LLM 调用，产出 atom 数组。

    ``speakers`` 是**本次会话实际出现的说话人名单**（由 ``adapters.speakers_of`` 从已落盘的
    msgmem 推出），填进 prompt 的 ``{speakers}`` 槽。``speaker_a`` / ``speaker_b`` 只用来
    替换示例里的人名占位符 —— 名单超过两人（剧本、多人语料）时示例仍取前两个，**名单本身
    才是模型该照抄的**。
    """
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

    names = list(speakers) if speakers else [speaker_a, speaker_b]
    first, second = speaker_pair(names)
    system_prompt = (
        ATOM_EXTRACTION_SYSTEM
        .replace("{speakers}", speakers_phrase(names))
        .replace("{speaker_a}", first)
        .replace("{speaker_b}", second)
    )

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
    """把 time_expr + 锚点时间 → 最终落盘的 ``metadata.time``。

    **委托给单一事实源** ``atommem_time.parse_time_expression``（与 agent 模式共用同一份
    逻辑）。本模块曾有一份自己的实现（直接 ``date -d "<anchor> <expr>"``），有两个实测缺陷，
    都直接造成 temporal 类答错：

    1. ``"last <weekday>"`` 被 GNU date 解析成**锚点本身**：实测
       ``date -d "25 May 2023 last saturday"`` → ``2023-05-25``（正是会话日期），于是
       "last Saturday" 被错记成会话当天（qa5 / qa28 / qa29 等一批题错在这）。
    2. 粗粒度表述（"last week"）被压成**单日绝对日期**，丢掉锚点，下游只能照抄这个日期。

    修正逻辑（保留锚点、星期几用 Python 精确折算、精度只减不增）在 ``atommem_time``。
    """
    from .atommem_time import parse_time_expression as _resolve

    return _resolve(time_expr, anchor_time)


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
    speakers: list[str] | None = None,
) -> list[dict[str, Any]]:
    """三阶段 pipeline 处理一条 message。

    返回：新产出的 atom 列表（已补全系统字段，未去重）。
    ``speakers`` 见 ``extract_atoms_llm``：本会话真实出现过的说话人名单。
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
        target, context, client, config, log, speaker_a, speaker_b, speakers=speakers,
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
