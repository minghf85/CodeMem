"""EvoMem 动作空间的解析与执行。

设计见 ``docs/evomem_plan.md``。模型每次输出一个动作块：

    <ADD>[atommem, ...]</ADD>
    <UPDATE>[atommem, ...]</UPDATE>
    <DELETE>[atommem, ...]</DELETE>
    <NOOP></NOOP>

约定（解析与执行都遵守）：

- 无论 ADD / UPDATE / DELETE，输出都是**完整的 atommem 列表**，每个元素必须带
  ``memory`` 和 ``metadata``，这样格式统一、好训练也好解析。
- **UPDATE / DELETE 必须回填 ``id``**：没有 id 就无法定位要改哪一条。plan 里说
  "DELETE 输出原始的要删除的 atommem 列表，不用 id 是为了防误删"，但意图表达的载体
  仍然是 id —— 我们要求模型把被删条目的完整内容一并输出，人眼可直接核对内容是否一致
  （见 ``verify_payload_matches``），id 只是机器定位用的索引。
- **ADD 的 id 由系统分配**（``{anchor_atom_id}_{n}``），模型输出里的 id 一律忽略。
- ``source`` / ``changelog`` 也由系统维护：UPDATE 走 ``apply_update`` 时旧值进
  changelog、source 并集；DELETE 走 ``apply_delete`` 时删掉的条目进 changelog。

本模块是**纯 CPU** 的：不打模型、不读盘、不碰网络。解析要稳健 —— 模型可能加
markdown 围栏、在标签前后写解释、漏掉闭合标签、把 JSON 写成裸数组或损坏的 JSON。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from . import atommem

# 动作名 -> 是否带载荷（NOOP 无载荷）
ACTIONS = ("ADD", "UPDATE", "DELETE", "NOOP")
PAYLOAD_ACTIONS = ("ADD", "UPDATE", "DELETE")

MAX_TAG_VALUE_LEN = 64
MAX_SOURCE_ITEMS = 32
MAX_CHANGELOG_ITEMS = 32

# 开场白/解释里引用标签名时不要被当成真实动作块。真实动作块必须像
# <ADD>[ ... ]</ADD> 或 <ADD>{ ... }</ADD>：紧跟 '[' 或 '{'。用**前瞻**而不是
# 捕获，否则会把开括号吃进 match 里，载荷就少了一个 '['（这曾导致合法输出被判为
# 非法 JSON）。用交替式 `(?=\[|\{)` 而不是字符组 `[[{]`，后者会触发嵌套集合告警。
_TAG_BLOCK_RE = re.compile(
    r"<\s*(ADD|UPDATE|DELETE|NOOP)\s*>\s*(?=\[|\{)", re.IGNORECASE
)

# NOOP 没有载荷，模型常写 <NOOP></NOOP> / <NOOP> / <NOOP/> / <NOOP>无变化</NOOP>。
_NOOP_RE = re.compile(r"<\s*NOOP\s*/?\s*>", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class EvomemAction:
    """一次解析出来的动作。``raw`` 保留模型原文，便于 trace 与出错复盘。"""

    kind: str
    payload: list[dict[str, Any]] = field(default_factory=list)
    raw: str = ""
    supplied: bool = False  # 模型是否显式写了这个标签（而非兜底推断）
    errors: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        return self.kind == "NOOP"

    def describe(self) -> str:
        detail = f"payload={len(self.payload)}"
        if self.repairs:
            detail += f" repairs={self.repairs}"
        if self.errors:
            detail += f" errors={self.errors}"
        return f"{self.kind}({detail})"


@dataclass
class ApplyResult:
    """apply_action 的结果：新的记忆列表 + 发生了什么。"""

    memories: list[dict[str, Any]]
    added_ids: list[str] = field(default_factory=list)
    updated_ids: list[str] = field(default_factory=list)
    deleted_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"+{len(self.added_ids)} ~{len(self.updated_ids)} "
            f"-{len(self.deleted_ids)} err={len(self.errors)}"
        )


class ActionParseError(ValueError):
    """模型输出里完全没有可识别的动作块。"""


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def parse_action(text: str) -> EvomemAction:
    """把模型输出解析成 EvomemAction。

    稳健性策略（按优先级）：

    1. 先找 NOOP（``<NOOP>`` / ``<NOOP></NOOP>`` / ``<NOOP/>``）。它没有载荷，最不
       容易出错，也最常出现在"无需改动"的回复里。
    2. 找到第一个形如 ``<ADD>[`` / ``<ADD>{`` 的载荷标签块，取载荷并解析。正常路径。
    3. 标签存在但载荷不是合法 JSON：依次尝试区间提取、括号修复、``_repair_json``。
       JSON 彻底解析不了时，保留为 ``errors`` 而不抛异常 —— 上层据此判定这一轮没有
       产生任何可执行动作。
    4. 完全没有标签：若正文里出现裸 JSON，按 ``[{"action": "ADD", ...}]`` 形状解析；
       再不行整体当 NOOP（比抛异常更安全：坏输出不该改记忆库）。

    返回的 payload 已经过 ``normalize_action_item`` 规范化，非法条目被丢弃并记入
    ``errors``。**任何解析失败路径都不会产出可执行动作**，保证坏输出不会改记忆库。
    """
    raw = text or ""
    stripped = raw.strip()

    # 1) NOOP 优先：无载荷，且常与说明文字共存
    noop = _NOOP_RE.search(stripped)
    payload_match = _TAG_BLOCK_RE.search(stripped)
    if noop is not None and (payload_match is None or noop.start() < payload_match.start()):
        return EvomemAction(kind="NOOP", raw=raw, supplied=True)

    # 2) 有载荷的动作
    if payload_match is not None:
        kind = payload_match.group(1).upper()
        if kind == "NOOP":  # 理论上到不了，_NOOP_RE 已覆盖
            return EvomemAction(kind="NOOP", raw=raw, supplied=True)
        body, repairs = _extract_payload_text(stripped, payload_match)
        raw_items, errors = _parse_payload(body, kind)
        items, item_errors = _normalize_items(raw_items)
        return EvomemAction(
            kind=kind,
            payload=items,
            raw=raw,
            supplied=True,
            errors=errors + item_errors,
            repairs=repairs,
        )

    # 3) 没有标签：尝试裸 JSON（含 [{"action": "ADD", ...}, ...] 形态）
    inferred = _parse_bare_json(stripped)
    if inferred is not None:
        bare_kind, payload, errors = inferred
        return EvomemAction(
            kind=bare_kind, payload=payload, raw=raw, supplied=False, errors=errors
        )

    # 兜底：当作 NOOP。坏输出不改记忆库。
    reason = "no action tag found" + (f": {stripped[:120]!r}" if stripped else " (empty output)")
    return EvomemAction(kind="NOOP", raw=raw, supplied=False, errors=[reason])


def _extract_payload_text(text: str, match: re.Match[str]) -> tuple[str, list[str]]:
    """从标签块里取出载荷文本。标签漏闭合时截到下一个标签之前。"""
    kind = match.group(1).upper()
    repairs: list[str] = []

    # 正常情形：闭合标签存在
    close = re.search(rf"</\s*{kind}\s*>", text[match.end():], re.IGNORECASE)
    if close is not None:
        body = text[match.end(): match.end() + close.start()]
        return body.strip(), repairs

    # 漏掉闭合标签：截到下一个动作标签或文本结尾
    next_tag = _TAG_BLOCK_RE.search(text, match.end())
    end = next_tag.start() if next_tag is not None else len(text)
    repairs.append(f"missing </{kind}> (truncated payload)")
    return text[match.end(): end].strip(), repairs


def _parse_payload(body: str, kind: str) -> tuple[list[dict[str, Any]], list[str]]:
    """把载荷文本解析成 dict 列表。返回 (items, errors)。

    解析顺序：整体 -> 取最外层 JSON 区间 -> ``_repair_brackets``。每一步都用
    ``_parse_any``（内含 atommem 的 ``_repair_json``）。只在**降级**时记一条 error，
    便于统计模型输出质量，但条目本身照常可用。
    """
    errors: list[str] = []
    cleaned = _strip_fences(body)
    if not cleaned:
        return [], errors

    parsed = atommem._parse_any(cleaned)
    if parsed is None:
        span = _extract_json_span(cleaned)
        if span is not None:
            parsed = atommem._parse_any(span)
            if parsed is not None:
                errors.append(f"{kind}: payload required JSON span extraction")
    if parsed is None:
        # 模型偶尔把开括号漏掉（`<ADD>{"memory": ...}]</ADD>` 或 `...}}]`），
        # _repair_json 不管这种"整体少一个容器"的情况，这里补一次。
        for fixed in _repair_brackets(cleaned):
            parsed = atommem._parse_any(fixed)
            if parsed is not None:
                errors.append(f"{kind}: payload had unbalanced brackets (auto-repaired)")
                break
    if parsed is None:
        errors.append(f"{kind}: payload is not valid JSON: {cleaned[:160]!r}")
        return [], errors
    if isinstance(parsed, dict):
        # {"atoms": [...]} / {"memory": ...}（单条）都接受，按可能性依次尝试
        for key in ("atoms", "memories", "items", kind.lower(), "actions"):
            value = parsed.get(key)
            if isinstance(value, list):
                parsed = value
                break
        else:
            parsed = [parsed]
    if not isinstance(parsed, list):
        errors.append(f"{kind}: payload is not a list (got {type(parsed).__name__})")
        return [], errors

    items = [item for item in parsed if isinstance(item, dict)]
    if len(items) != len(parsed):
        errors.append(f"{kind}: dropped {len(parsed) - len(items)} non-object entr(ies)")
    return items, errors


def _strip_fences(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    return re.sub(r"\s*```\s*$", "", text).strip()


def _extract_json_span(text: str) -> str | None:
    """取最外层 JSON 区间，用"最长的合法区间"兜底。

    对 ``[{"a": ...}]`` 这种用 ``find("[") + rfind("]")`` 会切出合法区间；但反过来
    用 ``find("{") + rfind("}")`` 会丢掉外层数组。所以两种括号都试，优先返回整体
    能解析的区间，否则返回最长的那个（信息最多）。
    """
    candidates: list[str] = []
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if 0 <= start < end:
            candidates.append(text[start: end + 1])
    if not candidates:
        return None
    for candidate in candidates:
        if atommem._parse_any(candidate) is not None:
            return candidate
    return max(candidates, key=len)


def _repair_brackets(text: str) -> list[str]:
    """为"容器括号不平衡"的载荷生成候选修法。

    主要指模型漏掉开括号（``{"memory": ...}}]``）或多写一个闭合括号
    （``[{"memory": ...}}]]``）。``_repair_json`` 处理不了这两种，因为整体结构本身
    是坏的。返回若干候选，调用方逐个尝试直到解析成功。
    """
    candidates: list[str] = []
    stripped = text.strip()
    if not stripped:
        return candidates

    # 1) 整体补一个开括号（缺 '['）
    if stripped[0] in "{[":
        candidates.append("[" + stripped)

    # 2) 去掉结尾多余的闭合括号，直到方括号/花括号平衡
    for cut in range(1, min(4, len(stripped))):
        candidate = stripped[:-cut].rstrip()
        if not candidate:
            continue
        if candidate.count("[") - candidate.count("]") == 0 and \
           candidate.count("{") - candidate.count("}") == 0:
            candidates.append(candidate)
            break

    # 3) 结尾补上缺失的闭合括号
    tail = stripped
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        missing = tail.count(open_ch) - tail.count(close_ch)
        if missing > 0:
            tail += close_ch * missing
    if tail != stripped:
        candidates.append(tail)

    return candidates


def _parse_bare_json(
    text: str,
) -> tuple[str, list[dict[str, Any]], list[str]] | None:
    """没有标签时的兜底：解析 [{"action": "ADD", "memory": ...}, ...]。"""
    cleaned = _strip_fences(text)
    if not cleaned:
        return None
    parsed = atommem._parse_any(cleaned)
    if parsed is None:
        span = _extract_json_span(cleaned)
        parsed = atommem._parse_any(span) if span else None
    if parsed is None:
        return None

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not parsed:
        return None

    actions: list[str] = []
    for item in parsed:
        if not isinstance(item, dict):
            return None
        kind = item.get("action") or item.get("op") or item.get("type")
        if not isinstance(kind, str) or kind.upper() not in ACTIONS:
            return None
        actions.append(kind.upper())

    kind = actions[0]
    errors = [f"no <{kind}> tag; inferred from bare JSON"]
    if len(set(actions)) > 1:
        errors.append(f"mixed actions in bare JSON {sorted(set(actions))}; keeping {kind}")
    items = [
        item for item, action in zip(parsed, actions)
        if action == kind
    ]
    normalized, item_errors = _normalize_items(items)
    return kind, normalized, errors + item_errors


def _normalize_items(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """逐条规范化：丢弃没有 memory 的条目，补齐缺失的 metadata 子字段。"""
    normalized: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, item in enumerate(items):
        record = normalize_action_item(item)
        if record is None:
            errors.append(f"item {index}: missing 'memory', dropped")
            continue
        normalized.append(record)
    return normalized, errors


def normalize_action_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """把模型输出的一条 atommem 规范化成统一的 memory item。

    只做形状修正（缺字段补默认值、tag/source 转字符串列表、changelog 统一成列表），
    **不编造内容**：ADD 的 id 留空由系统分配，UPDATE/DELETE 的 id 原样保留（可能是空，
    空 id 由 ``apply_update`` / ``apply_delete`` 报错拒绝）。
    """
    memory = item.get("memory")
    if not isinstance(memory, str) or not memory.strip():
        return None

    meta_in = item.get("metadata")
    if not isinstance(meta_in, dict):
        meta_in = {}

    mem_type = meta_in.get("type")
    if mem_type not in ("inner", "outer"):
        mem_type = "outer"

    tags: list[str] = []
    for tag in _as_list(meta_in.get("tag")):
        text = _tag_text(tag)
        if text:
            tags.append(text[:MAX_TAG_VALUE_LEN])

    source = [
        text[:MAX_TAG_VALUE_LEN]
        for text in (_as_text(s) for s in _as_list(meta_in.get("source")))
        if text
    ]

    time_value = meta_in.get("time")
    if not isinstance(time_value, str):
        time_value = "" if time_value is None else json.dumps(time_value, ensure_ascii=False)

    changelog: list[dict[str, str]] = []
    for entry in _as_list(meta_in.get("changelog"))[:MAX_CHANGELOG_ITEMS]:
        if isinstance(entry, dict):
            changelog.append(
                {
                    "time": _as_text(entry.get("time")) or "",
                    "content": _as_text(entry.get("content")) or "",
                }
            )
        elif isinstance(entry, str) and entry.strip():
            changelog.append({"time": "", "content": entry.strip()})

    record_id = meta_in.get("id")
    if not isinstance(record_id, str):
        record_id = ""

    return {
        "memory": memory.strip(),
        "metadata": {
            "id": record_id.strip(),
            "type": mem_type,
            "time": time_value.strip() if isinstance(time_value, str) else "",
            "tag": tags,
            "source": source[:MAX_SOURCE_ITEMS],
            "changelog": changelog,
        },
    }


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def _tag_text(tag: Any) -> str | None:
    if isinstance(tag, str):
        return tag.strip() or None
    if isinstance(tag, dict):
        key, value = tag.get("key"), tag.get("value")
        if isinstance(key, str) and isinstance(value, str) and key.strip():
            return f"{key.strip()}:{value.strip()}"
    return None


def _as_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if value is None:
        return None
    return str(value)


# ---------------------------------------------------------------------------
# 执行（纯 CPU）
# ---------------------------------------------------------------------------

def current_active(
    memories: list[dict[str, Any]], active: set[str] | None = None
) -> list[dict[str, Any]]:
    """只保留仍然存活的记忆（active 为 None 时全部视为存活）。"""
    if active is None:
        return list(memories)
    return [m for m in memories if _memory_id(m) in active]


def apply_action(
    action: EvomemAction,
    current: list[dict[str, Any]],
    active: set[str],
    anchor_atom_id: str,
) -> ApplyResult:
    """把动作作用到记忆库上，返回新状态。``current`` 不会被就地修改。"""
    memories = [dict(m) for m in current]
    if action.kind == "NOOP":
        return ApplyResult(memories=memories)
    if action.kind == "ADD":
        return apply_add(action.payload, memories, active, anchor_atom_id)
    if action.kind == "UPDATE":
        return apply_update(action.payload, memories, active)
    if action.kind == "DELETE":
        return apply_delete(action.payload, memories, active)
    return ApplyResult(memories=memories, errors=[f"unknown action {action.kind}"])


def _memory_id(memory: dict[str, Any]) -> str:
    value = (memory.get("metadata") or {}).get("id")
    return value if isinstance(value, str) else ""


def next_atom_number(memories: list[dict[str, Any]], anchor_atom_id: str) -> int:
    """找出 ``{anchor}_n`` 已用的最大 n，返回下一个可用序号。

    ``anchor_atom_id`` 是一个 **msgmem id**（如 ``session_1_1``）；新原子 id 形如
    ``session_1_1_{n}``，与 atommem 既有的 ``session_{s}_{m}_{n}`` 约定一致。
    即使该消息下还没有原子也从 1 开始。
    """
    prefix = f"{anchor_atom_id}_"
    used = [0]
    for memory in memories:
        memory_id = _memory_id(memory)
        if memory_id.startswith(prefix):
            suffix = memory_id[len(prefix):]
            # 只要 ``{anchor}_{n}`` 形态；``{anchor}_{n}_{k}`` 这种嵌套 id 不计入，
            # 否则会把别人的序号算进来。
            if suffix.isdigit():
                used.append(int(suffix))
    return max(used) + 1


def apply_add(
    payload: list[dict[str, Any]],
    current: list[dict[str, Any]],
    active: set[str],
    anchor_atom_id: str,
) -> ApplyResult:
    """追加新记忆。id 由系统按 ``{anchor_atom_id}_{n}`` 分配，模型给的 id 忽略。

    重复添加同一段文本时跳过（同一轮模型偶尔把同一条 ADD 写两遍）。
    """
    result = ApplyResult(memories=list(current))
    counter = next_atom_number(result.memories, anchor_atom_id)
    seen = {m.get("memory", "").strip() for m in result.memories}
    created_at = _now()

    for item in payload:
        memory_text = item["memory"].strip()
        if memory_text in seen:
            result.errors.append(f"ADD duplicate memory skipped: {memory_text[:60]!r}")
            continue
        meta = item["metadata"]
        record = {
            "memory": memory_text,
            "metadata": {
                "id": f"{anchor_atom_id}_{counter}",
                "type": meta.get("type") or "outer",
                "time": meta.get("time") or "",
                "tag": list(meta.get("tag") or []),
                "source": list(meta.get("source") or []),
                "changelog": [{"time": created_at, "content": "created"}],
            },
        }
        counter += 1
        seen.add(memory_text)
        result.memories.append(record)
        result.added_ids.append(record["metadata"]["id"])
        active.add(record["metadata"]["id"])
    return result


def apply_update(
    payload: list[dict[str, Any]],
    current: list[dict[str, Any]],
    active: set[str],
) -> ApplyResult:
    """就地替换已有记忆（保持列表位置）。

    - 目标 id 必须存在且存活，否则记错误并跳过（不改任何东西）。
    - ``memory`` 与旧值相同时算空操作。
    - ``changelog`` 追加旧 ``memory`` 与旧 ``time``，保留历史。
    - ``source`` 取新旧并集，provenance 不会因为改写而丢失。

    **不变量：UPDATE 绝不改 id**。evomem 的游标顺序由 id 派生（``atom_key``），id 一变
    顺序就失效 —— 游标会跳过或重访该原子。所以下面显式断言新记录沿用原 id；将来若有人
    "优化"成重新分配 id，会在这里直接失败而不是静默产生乱序。
    """
    result = ApplyResult(memories=list(current))
    positions = {_memory_id(m): i for i, m in enumerate(result.memories) if _memory_id(m)}

    for item in payload:
        target_id = item["metadata"].get("id") or ""
        meta = item["metadata"]
        new_memory = item["memory"].strip()
        new_changelog = meta.get("changelog") or []

        index = positions.get(target_id)
        if not target_id or index is None:
            result.errors.append(f"UPDATE: unknown id {target_id!r}, skipped")
            continue

        old = result.memories[index]
        if target_id not in active:
            result.errors.append(f"UPDATE: id {target_id!r} already deleted, skipped")
            continue

        old_memory = old.get("memory", "")
        old_meta = old.get("metadata") or {}
        if new_memory == old_memory.strip() and not _changed_scalar_fields(old_meta, meta):
            result.errors.append(f"UPDATE: id {target_id!r} unchanged, skipped")
            continue

        changelog = list(old_meta.get("changelog") or [])
        changelog.append({"time": _now(), "content": old_memory})

        old_time = old_meta.get("time") or ""
        new_time = meta.get("time") or ""
        if old_time and old_time != new_time:
            changelog.append({"time": _now(), "content": f"time: {old_time}"})

        if new_changelog:
            # 模型自己带了 changelog：并到系统生成的历史之前，避免丢信息
            changelog = list(new_changelog) + changelog

        source = _merge_unique(old_meta.get("source"), meta.get("source"))

        result.memories[index] = {
            "memory": new_memory,
            "metadata": {
                "id": target_id,
                "type": meta.get("type") or old_meta.get("type") or "outer",
                "time": new_time or old_time,
                "tag": list(meta.get("tag") or old_meta.get("tag") or []),
                "source": source,
                "changelog": changelog[-MAX_CHANGELOG_ITEMS:],
            },
        }
        # 不变量：改写不得改 id（游标顺序由 id 派生）。见函数 docstring。
        if _memory_id(result.memories[index]) != target_id:
            raise AssertionError(
                f"apply_update 改变了 id（{target_id!r} -> "
                f"{_memory_id(result.memories[index])!r}）；游标顺序会因此失效"
            )
        result.updated_ids.append(target_id)
    return result


def apply_delete(
    payload: list[dict[str, Any]],
    current: list[dict[str, Any]],
    active: set[str],
) -> ApplyResult:
    """删除记忆：id 必须存在且存活，``active`` 中移除。

    列表里的记录**保留**，只在 ``active`` 中摘掉（软删除）。这样 provenance /
    credit 事后归因仍然能查到被删的内容，也便于把删除动作写进轨迹。
    """
    result = ApplyResult(memories=list(current))
    known = {_memory_id(m) for m in result.memories if _memory_id(m)}

    for item in payload:
        target_id = item["metadata"].get("id") or ""
        if not target_id or target_id not in known:
            result.errors.append(f"DELETE: unknown id {target_id!r}, skipped")
            continue
        if target_id not in active:
            result.errors.append(f"DELETE: id {target_id!r} already deleted, skipped")
            continue
        active.discard(target_id)
        result.deleted_ids.append(target_id)
    return result


def verify_payload_matches(
    payload: list[dict[str, Any]], current: list[dict[str, Any]], action: str
) -> list[str]:
    """校验模型回填的内容是否与库中现存条目一致（防内容与 id 对不上）。

    只做"内容是否对得上"的检查，不阻断执行 —— 返回警告列表供 trace 记录。
    跳过没有 id 的条目（ADD 正常不带 id）。
    """
    by_id = {_memory_id(m): m for m in current if _memory_id(m)}
    warnings: list[str] = []
    for item in payload:
        target_id = item["metadata"].get("id") or ""
        if not target_id:
            continue
        existing = by_id.get(target_id)
        if existing is None:
            continue  # 未知 id 由 apply_* 报错
        if existing.get("memory", "").strip() != item["memory"].strip():
            warnings.append(
                f"{action}: id {target_id!r} supplied content differs from stored "
                f"(stored={existing.get('memory', '')[:60]!r})"
            )
    return warnings


def _changed_scalar_fields(old_meta: dict[str, Any], new_meta: dict[str, Any]) -> bool:
    """检查 metadata 字段是否有变化（包括标量和列表字段）。

    标量字段（type, time）：直接比较值
    列表字段（tag, source）：用集合比较（忽略顺序）
        - 只有当新值非空且与旧值不同时，才算有变化
        - 空列表表示"未提供新值"（因为 apply_update 使用合并而非替换）

    返回 True 表示有任何字段发生了变化。
    """
    # 检查标量字段
    for key in ("type", "time"):
        if new_meta.get(key) and new_meta.get(key) != old_meta.get(key):
            return True

    # 检查列表字段（tag、source）- 用集合比较忽略顺序
    # 只有当新值非空（表示"有意提供新值"）且与旧值不同时，才算变化
    for key in ("tag", "source"):
        new_val = new_meta.get(key) or []
        if new_val:  # 只检查非空的新值
            old_val = set(old_meta.get(key) or [])
            if set(new_val) != old_val:
                return True

    return False


def _merge_unique(left: Any, right: Any) -> list[str]:
    merged: list[str] = []
    for value in list(_as_list(left)) + list(_as_list(right)):
        text = _as_text(value)
        if text and text not in merged:
            merged.append(text)
    return merged[:MAX_SOURCE_ITEMS]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
