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
from dataclasses import dataclass, field, replace
from typing import Any

from .. import llm
from ..io import memory_id, memory_role, memory_text, now_iso, salvage_json_records
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

def msg_positions(msgmem: list[dict[str, Any]]) -> dict[str, int]:
    """``msg_id -> 它在 msgmem 里的下标``。

    为什么要预建这张表：``recall_context`` 每次都要"找到 target 在哪"，不预建就是**每条 message
    扫一遍全表** —— 一个几千上万条 msgmem 的库里，一个会话几十条 message 就是几十万次查找，
    而且这段是**同步**跑的（在事件循环里）。建一次表 O(n)，之后每次 O(1)。
    """
    return {
        memory_id(record): index
        for index, record in enumerate(msgmem)
        if memory_id(record)
    }


def recall_context(
    target_id: str,
    msgmem: list[dict[str, Any]],
    atommem: list[dict[str, Any]],
    window: int = 5,
    positions: dict[str, int] | None = None,
) -> str:
    """召回与 target 相关的上下文（同会话前 N 条消息）。``positions`` 见 ``msg_positions``。"""
    # 找到 target 的位置（有预建表就用表，否则退化成扫一遍）
    if positions is not None:
        target_pos = positions.get(target_id, -1)
    else:
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
#
# **输入永远不超过模型上下文** —— 这是本阶段唯一的硬约束，所以整段围绕"预算"组织：
#
#   预算（max_input_tokens）
#     ├─ 扣掉 system prompt 与模板骨架（模型侧固定开销）
#     └─ 余下按比例分给「前文上下文」与「目标消息」
#          ├─ 上下文：按行从**最新**往回保留（消解指代靠的是最近的几轮）
#          └─ 目标消息：超预算就**切块**，逐块抽取后合并（块间留一点重叠，
#             免得一个事实正好被切在边界上、两边都抽不出来）
#
# 两条实测会烧掉整条 message 的失败路径，都在这里兜住：
#
#   · **输入超上下文**（400 ``maximum context length``）—— 重试解决不了。缩预算重切才是解，
#     所以识别出来（``llm.is_context_length_error``）后把预算砍半、重新切块再试。
#   · **输出被 max_tokens 截断**（``TruncatedCompletion``）—— 一条 message 的 atom 太多、
#     一次吐不完。先按对象边界**抢救**已吐出的部分，再把这一块切细重试（块小 → 一次要吐的
#     atom 少 → 放得下）。
#
# 两种情况都**只影响自己这一块**：别的块照跑，返回的 ``ok=False`` 让调用方知道"这条还没抽全"。

#: 整个 prompt（system + user）的 token 预算。默认值按 32k 上下文的模型留足余量。
DEFAULT_MAX_INPUT_TOKENS = 12000
#: 预算里分给「前文上下文」的比例（其余留给目标消息本身）。
CONTEXT_SHARE = 0.35
#: 相邻分块的重叠比例（按块预算折算）。
CHUNK_OVERLAP_RATIO = 0.1
#: 端点报错后预算每次砍半，最多砍这么多次。
MAX_SHRINK_ATTEMPTS = 3
#: 单块内容的 token 下限 —— 配置的预算连 prompt 骨架都装不下时，按这个跑并告警。
MIN_PART_TOKENS = 400
#: 缩预算的下限：一次抽取总得有几百 token 的原料，比这还小就别再切了。
MIN_SHRINK_TOKENS = 200
#: 预算里分给"目标消息"的最小额度（预算极小时兜底，避免算出 0 字的块）。
MIN_CONTENT_TOKENS = 64
#: 一条 message 最多切成多少块（防止"切了又报错、报了又切"烧成无限循环）。
MAX_UNITS_PER_MESSAGE = 64
#: 接续块的开场白（第一块之后不再重复上下文，只留这一句指明"还是同一条消息"）。
CONTINUATION_NOTE = "(continued -- this is the next part of the SAME message)"

#: 宽字符（CJK 等）范围：这些字一个≈一个 token，不能按"4 字符≈1 token"折算。（搬迁前的遗留，
#: 真正生效的那份是 ``llm._WIDE_RE`` —— 见下面 ``estimate_tokens`` 的别名。）
_WIDE_RE = re.compile(r"[⺀-鿿豈-﫿가-힯＀-￯]")


#: token 估算：**尺子只有一把** —— 与 summem 共用 ``llm.estimate_tokens``（那里的注释讲了
#: 为什么"宁高勿低"）。add 的两条路都要按它给 prompt 定预算，两边必须量得一样。
estimate_tokens = llm.estimate_tokens


def clip_text(text: str, budget_tokens: int, keep: str = "tail") -> str:
    """把一段文本压到 ``budget_tokens`` 以内。``keep="tail"`` 保留**结尾**（上下文用），
    ``keep="head"`` 保留开头。"""
    if not text or budget_tokens <= 0:
        return ""
    if estimate_tokens(text) <= budget_tokens:
        return text
    marker = "…(truncated)…"
    marker_tokens = estimate_tokens(marker)
    if budget_tokens <= marker_tokens:
        return ""
    allowed = budget_tokens - marker_tokens
    low, high = 0, len(text)
    while low < high:                       # 二分最大的、落在预算内的字符数
        middle = (low + high + 1) // 2
        piece = text[-middle:] if keep == "tail" else text[:middle]
        if estimate_tokens(piece) <= allowed:
            low = middle
        else:
            high = middle - 1
    piece = text[-low:] if keep == "tail" else text[:low]
    return f"{marker}{piece}" if keep == "tail" else f"{piece}{marker}"


def clip_context(context: str, budget_tokens: int) -> str:
    """按**行**从最新往回保留上下文（消解指代看的是最近的几轮，最早的可以丢）。

    行级而不是字符级：``msgmem`` 渲染出来的上下文一行就是一条消息，切成半行会让模型看到
    半句话 —— 那比没有上下文更糟。某一**行本身**就超预算时才退化成字符级截断（保留该行
    结尾，因为紧挨着目标消息）。
    """
    if not context or budget_tokens <= 0:
        return ""
    if estimate_tokens(context) <= budget_tokens:
        return context
    # 截断标记本身也占预算 —— 先扣掉它，否则"≤ budget"这句话就不成立。
    marker = "…(earlier context omitted)…"
    allowance = max(0, budget_tokens - estimate_tokens(marker))
    lines = context.splitlines()
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        cost = estimate_tokens(line) + 1
        if used + cost > allowance:
            if not kept:                    # 最后一行就超了 —— 只能字符级截它
                kept.append(clip_text(line, allowance, keep="tail"))
            break
        kept.append(line)
        used += cost
    kept.reverse()
    if len(kept) < len(lines):
        kept.insert(0, marker)
    return "\n".join(kept)


def split_message(content: str, budget_tokens: int, overlap_tokens: int = 0) -> list[str]:
    """把目标消息切成每块 ≤ ``budget_tokens`` 的若干块（按行切，行太长才按字符硬切）。

    切块是**为了放得下**，不是为了改写：每块仍会带着"这是同一条消息的第几部分"的说明，
    抽取规则一字不变；跨块重复抽出的事实由收尾的 ``_dedup_atoms`` 合并。
    """
    if budget_tokens <= 0 or estimate_tokens(content) <= budget_tokens:
        return [content]
    lines = content.splitlines(keepends=True) or [content]
    chunks: list[str] = []
    current: list[str] = []
    used = 0

    def flush() -> None:
        nonlocal current, used
        if current:
            chunks.append("".join(current))
            current, used = [], 0

    for line in lines:
        cost = estimate_tokens(line)
        if cost > budget_tokens:
            # 单行就超预算（长文档 / 一条超长消息）：先冲掉手上的块，再把这一行硬切。
            flush()
            chunks.extend(_hard_split(line, budget_tokens, overlap_tokens))
            continue
        if current and used + cost > budget_tokens:
            flush()
            if overlap_tokens > 0:
                tail = _overlap_tail(chunks[-1], overlap_tokens)
                if tail:
                    current.append(tail)
                    used += estimate_tokens(tail)
        current.append(line)
        used += cost
    flush()
    return [chunk for chunk in chunks if chunk.strip()] or [content]


def _max_chars_from(text: str, start: int, budget_tokens: int) -> int:
    """从 ``start`` 起，最多取多少字符才不超预算（二分，**量的是这一段的实际内容**）。

    为什么不能拿整段文本的开头估一个"每块多少字符"然后按固定步长切：混合语料里 ASCII 与
    CJK 的字符/token 比差 4 倍，按开头估出来的步长切到一段中文上就超预算了（那正是我们要
    防的 400）。逐块二分，代价可以忽略（只在走到硬切时发生）。
    """
    low, high = 0, len(text) - start
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[start:start + middle]) <= budget_tokens:
            low = middle
        else:
            high = middle - 1
    return low


def _max_chars_tail(text: str, budget_tokens: int) -> int:
    """最多取末尾多少字符才不超预算（与 ``_max_chars_from`` 对称）。"""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[len(text) - middle:]) <= budget_tokens:
            low = middle
        else:
            high = middle - 1
    return low


def _hard_split(text: str, budget_tokens: int, overlap_tokens: int) -> list[str]:
    """一行本身超预算时按字符硬切（每块的大小都现场量过，保证不超预算）。"""
    pieces: list[str] = []
    index = 0
    while index < len(text):
        size = _max_chars_from(text, index, budget_tokens)
        if size <= 0:                       # 预算小到连一个字符都放不下：切不动了
            break
        piece = text[index:index + size]
        pieces.append(piece)
        if index + size >= len(text):
            break
        overlap = 0
        if overlap_tokens > 0:
            overlap = min(_max_chars_tail(piece, overlap_tokens), size - 1)
        index += max(1, size - overlap)
    return pieces


def _overlap_tail(chunk: str, overlap_tokens: int) -> str:
    """取上一块的结尾（≤ ``overlap_tokens``）当下一块的开头，避免事实被切在边界上。"""
    if overlap_tokens <= 0 or not chunk:
        return ""
    lines = chunk.splitlines(keepends=True)
    taken: list[str] = []
    used = 0
    for line in reversed(lines):
        cost = estimate_tokens(line)
        if used + cost > overlap_tokens:
            break
        taken.append(line)
        used += cost
        if len(taken) >= 2:                 # 最多带两行，别把重叠本身变成负担
            break
    taken.reverse()
    return "".join(taken)


@dataclass
class _Unit:
    """一次 LLM 调用要处理的一份输入（上下文 + 内容 + 它自己的预算）。"""

    context: str
    content: str
    budget: int
    shrinks: int = 0


@dataclass
class MessageExtraction:
    """一条 message 的抽取结果。

    三种结局，调用方的处理完全不同（见 ``store._refresh_atoms``）：

    | 结局 | 字段 | 调用方该怎么做 |
    |---|---|---|
    | 抽成功 | ``ok=True`` | 记进"已抽过" |
    | 抽失败但**可能只是抖动** | ``ok=False`` | 记进 ``failed``，**稍后重试**（含端点恢复后的补漏） |
    | **重发也没用** | ``permanent=True`` | 记进 ``blocked``，**不再自动重试**（内容策略拦截 / 非法请求） |

    ``note`` 是人看的原因（进日志与记账），不参与判定。
    """

    atoms: list[dict[str, Any]] = field(default_factory=list)
    ok: bool = True
    permanent: bool = False
    note: str = ""
    calls: int = 0
    chunks: int = 0


def plan_units(context: str, content: str, budget_tokens: int, shrinks: int = 0) -> list[_Unit]:
    """把（上下文, 内容）切成不超过预算的若干份输入。``budget_tokens<=0`` = 不控制预算。"""
    if budget_tokens <= 0:
        return [_Unit(context=context, content=content, budget=0)]
    context_budget = int(budget_tokens * CONTEXT_SHARE)
    content_budget = max(MIN_CONTENT_TOKENS, budget_tokens - context_budget)
    clipped = clip_context(context, context_budget) if context else ""
    parts = split_message(
        content, content_budget, overlap_tokens=int(content_budget * CHUNK_OVERLAP_RATIO),
    )
    units: list[_Unit] = []
    for index, part in enumerate(parts):
        units.append(_Unit(
            context=clipped if index == 0 else CONTINUATION_NOTE,
            content=part, budget=budget_tokens, shrinks=shrinks,
        ))
    return units


async def _call_model(
    client: Any, config: dict[str, Any], messages: list[dict[str, Any]], gate: Any = None,
) -> str:
    """调一次模型；``gate`` 是**进程级**的并发闸门（None = 不限制）。

    闸门只包住这一**次**调用、不包住"等别的块" —— 否则一个 Add 的刷新会占着名额去等另一个
    名额，32 路并发会互相锁死。
    """
    if gate is None:
        return await llm.chat_completion(client=client, config=config, messages=messages)
    async with gate:
        return await llm.chat_completion(client=client, config=config, messages=messages)


async def extract_atoms_llm(
    target: dict[str, Any],
    context: str,
    client: Any,
    config: dict[str, Any],
    log: Logger,
    speaker_a: str = "speaker_a",
    speaker_b: str = "speaker_b",
    speakers: list[str] | None = None,
    gate: Any = None,
) -> MessageExtraction:
    """抽取一条 message 的 atom（**单条消息可能多次调用**：超预算就切块，报错就缩预算重切）。

    ``speakers`` 是**本次会话实际出现的说话人名单**（由 ``adapters.speakers_of`` 从已落盘的
    msgmem 推出），填进 prompt 的 ``{speakers}`` 槽。``speaker_a`` / ``speaker_b`` 只用来
    替换示例里的人名占位符 —— 名单超过两人（剧本、多人语料）时示例仍取前两个，**名单本身
    才是模型该照抄的**。
    """
    msg_id = memory_id(target)
    msg_time = (target.get("metadata") or {}).get("time", "")
    role = memory_role(target) or "unknown"
    content = memory_text(target)

    names = list(speakers) if speakers else [speaker_a, speaker_b]
    first, second = speaker_pair(names)
    system_prompt = (
        ATOM_EXTRACTION_SYSTEM
        .replace("{speakers}", speakers_phrase(names))
        .replace("{speaker_a}", first)
        .replace("{speaker_b}", second)
    )

    outcome = MessageExtraction()
    total_budget = int(config.get("max_input_tokens", DEFAULT_MAX_INPUT_TOKENS) or 0)
    if total_budget > 0:
        # 预算是给**整个 prompt** 的：先扣掉 system 与模板骨架这些模型侧的固定开销。
        skeleton = ATOM_EXTRACTION_USER.format(
            context="", msg_id=msg_id, msg_time=msg_time, role=role, content="",
        )
        overhead = estimate_tokens(system_prompt) + estimate_tokens(skeleton) + 32
        payload_budget = total_budget - overhead
        if payload_budget < MIN_PART_TOKENS:
            log.warn(
                f"{msg_id}: max_input_tokens={total_budget} 连 prompt 骨架都装不下"
                f"（固定开销约 {overhead} token），按最小块跑"
            )
            payload_budget = MIN_PART_TOKENS
    else:
        payload_budget = 0                  # 0 = 关闭预算控制（与旧行为一致）

    units = plan_units(context, content, payload_budget)
    outcome.chunks = len(units)
    if len(units) > 1:
        log.info(
            f"{msg_id}: 输入约 {estimate_tokens(context) + estimate_tokens(content)} token，"
            f"切成 {len(units)} 块抽取（预算 {payload_budget or '不限'} token）"
        )

    while units:
        if outcome.calls >= MAX_UNITS_PER_MESSAGE:
            outcome.ok = False
            log.warn(f"{msg_id}: 切块 / 缩预算尝试超过 {MAX_UNITS_PER_MESSAGE} 次，放弃剩余部分")
            break
        unit = units.pop(0)
        user_prompt = ATOM_EXTRACTION_USER.format(
            context=unit.context, msg_id=msg_id, msg_time=msg_time,
            role=role, content=unit.content,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            raw_text = await _call_model(client, config, messages, gate)
            outcome.calls += 1
            outcome.atoms.extend(parse_atom_array(raw_text, log, msg_id))
        except llm.TruncatedCompletion as exc:
            # 输出放不下（一条消息的 atom 太多）：先抢救已吐出的完整记录，再切细重试。
            outcome.calls += 1
            salvaged, dropped = salvage_json_records(getattr(exc, "content", "") or "")
            if salvaged:
                outcome.atoms.extend(item for item in salvaged if isinstance(item, dict))
                log.warn(
                    f"{msg_id}: 输出被 max_tokens 截断，抢救出 {len(salvaged)} 条"
                    f"（丢 {dropped} 个残片），切细重试"
                )
            retry = _shrink(unit, log, msg_id, reason="输出被截断")
            if retry is None:
                outcome.ok = False
            else:
                units = retry + units
        except Exception as exc:  # noqa: BLE001 - 单块失败不该让整条 message 没有产出
            outcome.calls += 1
            if llm.is_context_length_error(exc) and unit.shrinks < MAX_SHRINK_ATTEMPTS:
                log.warn(f"{msg_id}: 输入超上下文（{llm.describe_error(exc)}），缩预算重试")
                retry = _shrink(unit, log, msg_id, reason="输入超上下文")
                if retry is None:
                    outcome.ok = False
                else:
                    units = retry + units
            elif llm.is_content_filter_error(exc) and unit.context.strip():
                # **内容策略拦截**：缩输入救不回来（拦的是内容不是长度），但触发词常常在**前文
                # 上下文**里。所以去掉上下文再试一次（只试一次 —— 去掉后 unit.context 空了，
                # 再被拦就直接走下面的 permanent 分支）。
                log.warn(f"{msg_id}: 被内容策略拦截，去掉前文上下文重试一次")
                units = [replace(unit, context="")] + units
            elif llm.is_permanent_prompt_error(exc):
                # **重发同一个 prompt 也不会有不同结果**（内容策略 / 非法请求 / 认证失败）：
                # 再试只是白烧调用，而且会永远卡在账上（补漏每轮都会重来一遍）。放弃这条
                # message，把原因原样交出去让调用方记账。
                outcome.permanent = True
                outcome.ok = False
                outcome.note = f"{type(exc).__name__}: {exc}"[:200]
                log.warn(f"{msg_id}: 端点不接受这个 prompt（{type(exc).__name__}），放弃重试")
                break
            else:
                # 剩下的都当**抖动**（超时 / 建连失败 / 5xx）：记账待重试。
                outcome.ok = False
                log.warn(f"{msg_id}: 抽取失败 {type(exc).__name__}: {exc}")

    return outcome


def _shrink(unit: _Unit, log: Logger, msg_id: str, *, reason: str) -> list[_Unit] | None:
    """把一块的预算砍半、重新切细。切不动了（已到下限 / 砍无可砍）返回 None。

    下限是 ``MIN_SHRINK_TOKENS`` 而不是 ``MIN_PART_TOKENS``：后者是"配置给的预算还能不能用"的
    门槛（400），拿它当缩预算的底线会让配置偏小的情况**一次都缩不动**（800 → 400 直接放弃），
    而"端点说超了"恰恰要靠缩才能救回来。
    """
    if unit.shrinks >= MAX_SHRINK_ATTEMPTS:
        log.warn(f"{msg_id}: {reason}，已缩 {unit.shrinks} 次仍不行，放弃这一块")
        return None
    smaller = unit.budget // 2 if unit.budget > 0 else 0
    if unit.budget > 0 and smaller < MIN_SHRINK_TOKENS:
        log.warn(f"{msg_id}: {reason}，但预算已到下限（{unit.budget} token），放弃这一块")
        return None
    if smaller <= 0:                        # 预算控制关着（0）—— 无从缩起
        return None
    return plan_units(unit.context, unit.content, smaller, shrinks=unit.shrinks + 1)


def parse_atom_array(raw_text: str, log: Logger, msg_id: str) -> list[dict[str, Any]]:
    """把模型回复解析成 atom 数组（容忍 ``` 包裹、外围文字、以及被截断的数组）。"""
    cleaned = re.sub(r"```json\s*", "", raw_text or "")
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    if not cleaned:
        return []
    try:
        atoms = json.loads(cleaned)
    except json.JSONDecodeError:
        # 整份解不出（多半被截断）：按对象边界抢救，保住前面完整的那些。
        salvaged, dropped = salvage_json_records(cleaned)
        if salvaged:
            log.warn(f"{msg_id}: JSON 解析失败，按对象边界抢救出 {len(salvaged)} 条（丢 {dropped} 残片）")
            return salvaged
        log.warn(f"{msg_id}: JSON 解析失败")
        return []
    if isinstance(atoms, dict):
        atoms = [atoms]
    if not isinstance(atoms, list):
        log.warn(f"{msg_id}: LLM 返回不是数组，忽略")
        return []
    return [item for item in atoms if isinstance(item, dict)]



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
    gate: Any = None,
    positions: dict[str, int] | None = None,
) -> MessageExtraction:
    """三阶段 pipeline 处理一条 message。

    返回 ``MessageExtraction``（已补全系统字段、未去重）；``ok=False`` 表示至少一块没抽成，
    调用方应当把这条 message 当作"还没处理过"（见 ``MessageExtraction``）。

    ``gate`` 是**进程级**的模型并发闸门 —— 它只包住每一次 ``chat_completion``，
    所以一份 prompt 被切成 N 块也只是排 N 次队，不会占着名额等自己。
    ``positions`` 是 ``msg_positions(msgmem)`` 预建的位置表（批量处理时建一次，别每条重建）。
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
        positions=positions,
    )

    # Stage 2: 抽取（可能多次调用：切块 / 缩预算，见 extract_atoms_llm）
    extraction = await extract_atoms_llm(
        target, context, client, config, log, speaker_a, speaker_b,
        speakers=speakers, gate=gate,
    )
    raw_atoms = extraction.atoms

    if not raw_atoms:
        log.info(f"{msg_id}: 未抽取到 atom" + ("" if extraction.ok else "（抽取失败）"))
        return extraction

    # Stage 3: 后处理
    existing_by_id = {memory_id(a): a for a in existing_atoms if memory_id(a)}
    processed = postprocess_atoms(
        raw_atoms, msg_id, anchor_time, default_role, existing_by_id
    )

    log.info(
        f"{msg_id}: 产出 {len(processed)} 条 atom"
        + (f"（{extraction.calls} 次调用 / {extraction.chunks} 块）" if extraction.chunks > 1 else "")
    )
    return MessageExtraction(
        atoms=processed, ok=extraction.ok, permanent=extraction.permanent,
        note=extraction.note, calls=extraction.calls, chunks=extraction.chunks,
    )
