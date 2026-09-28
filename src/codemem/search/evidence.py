"""``evidence.jsonl`` 的校验与规范化 —— harness 侧的**唯一**产物检查点。

agent 是用 bash/jq/write 自由写文件的，所以产物可能不合法（缺字段、id 重复、没有 source）。
本模块在每条 QA 收尾时把文件读一遍，逐行判定：

    valid    通过全部不变量 → 原样交付
    repaired 只缺**系统自有字段**（changelog）→ 补全（``changelog=[{time, content:"created"}]``）
    rejected 缺必填字段 / 类型不对 / id 重复 / source 为空 → 丢出，并在轨迹里记账

**为什么不替 agent 修语义问题**：把"source 为空"补成编造的 id、把重复 id 改个名字，都会
掩盖真实的失败模式，让评测数字好看而 agent 没学会。"只补系统自有字段"这条界线是刻意的。

不变量（见 docs/evomem_plan_v3.md §7）：

1. 每行是合法 JSON 对象；
2. ``memory`` 是非空字符串；
3. ``metadata`` 是对象，``metadata.id`` 非空字符串，且**文件内唯一**；
4. ``metadata.type`` ∈ {inner, outer, raw}；
5. ``metadata.tag`` 是非空数组且至少一条 ``speaker:``；
6. ``metadata.source`` 是**非空**数组 —— 反幻觉护栏。从 inputs/atommem.jsonl 拷来的行天然
   满足；只有 agent 自己合成的记录受约束，而那恰好是最需要约束的地方。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..io import read_jsonl_report_text

VALID_TYPES = ("inner", "outer", "raw")
REQUIRED_META = ("id", "type", "time", "tag", "source")


@dataclass
class EvidenceReport:
    """一次校验的完整结果。``valid`` 是最终交付给评测的列表。"""

    valid: list[dict[str, Any]] = field(default_factory=list)
    repaired: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    parse_errors: list[dict[str, Any]] = field(default_factory=list)
    # **解析出的记录数**，不是原始行数：一行一个数组时会展开成多条，美化 JSON 兜底时
    # 整份文档算一批。"交付了多少条记录"才是这个字段要回答的问题。
    total_lines: int = 0
    empty_lines: int = 0
    multi_record_lines: int = 0
    whole_file_fallback: bool = False
    # 被 max_tokens 截断后按对象边界抢救回来的（尾部残片已丢）
    salvaged: bool = False
    dropped_fragments: int = 0
    path: str = ""

    @property
    def delivered(self) -> list[dict[str, Any]]:
        """交付列表：合法 + 补全过的（补全后的版本以 ``repaired`` 为准）。"""
        return self.valid

    @property
    def counts(self) -> dict[str, int]:
        return {
            "lines": self.total_lines,
            "valid": len(self.valid),
            "repaired": len(self.repaired),
            "rejected": len(self.rejected),
            "parse_errors": len(self.parse_errors),
            "empty_lines": self.empty_lines,
            "multi_record_lines": self.multi_record_lines,
            "whole_file_fallback": self.whole_file_fallback,
            "salvaged": self.salvaged,
            "dropped_fragments": self.dropped_fragments,
        }

    def summary(self) -> str:
        parts = [f"行 {self.total_lines}", f"交付 {len(self.valid)}"]
        if self.repaired:
            parts.append(f"补全 {len(self.repaired)}")
        if self.rejected:
            parts.append(f"拒绝 {len(self.rejected)}")
        if self.parse_errors:
            parts.append(f"解析失败 {len(self.parse_errors)}")
        if self.empty_lines:
            parts.append(f"空行 {self.empty_lines}")
        if self.multi_record_lines:
            parts.append(f"一行多条 {self.multi_record_lines}")
        if self.whole_file_fallback:
            parts.append("整份 JSON 兜底解析")
        if self.salvaged:
            parts.append(f"截断抢救（丢弃 {self.dropped_fragments} 个残片）")
        return " / ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            **self.counts,
            "rejected_reasons": [
                {"line": item.get("line"), "reason": item.get("reason")} for item in self.rejected
            ],
            "parse_error_lines": [item.get("line") for item in self.parse_errors],
            "delivered_ids": [_id_of(m) for m in self.valid],
        }


def _id_of(memory: dict[str, Any]) -> str:
    value = (memory.get("metadata") or {}).get("id")
    return value if isinstance(value, str) else ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(memory: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """校验单条记录。返回 ``(规范化后的记录, 拒绝原因)``；原因为 None 表示通过。

    只在**系统自有字段**上做规范化：``changelog`` 缺失或非法 → 补 ``created``；
    ``metadata`` 里多余的键原样保留（不删，避免抹掉 agent 的意图）。
    """
    if not isinstance(memory, dict):
        return {}, f"not a JSON object ({type(memory).__name__})"

    text = memory.get("memory")
    if not isinstance(text, str) or not text.strip():
        return {}, "memory must be a non-empty string"

    meta = memory.get("metadata")
    if not isinstance(meta, dict):
        return {}, "metadata must be an object"

    for key in REQUIRED_META:
        if key not in meta:
            return {}, f"metadata.{key} is missing"

    memory_id = meta.get("id")
    if not isinstance(memory_id, str) or not memory_id.strip():
        return {}, "metadata.id must be a non-empty string"

    if meta.get("type") not in VALID_TYPES:
        return {}, f"metadata.type must be one of {list(VALID_TYPES)}, got {meta.get('type')!r}"

    tags = meta.get("tag")
    if not isinstance(tags, list) or not tags:
        return {}, "metadata.tag must be a non-empty array"
    if not any(isinstance(t, str) and t.startswith("speaker:") for t in tags):
        return {}, "metadata.tag must include at least one 'speaker:<name>'"

    source = meta.get("source")
    if not isinstance(source, list) or not source:
        return {}, "metadata.source must be a non-empty array (provenance is required)"
    bad = [s for s in source if not isinstance(s, str) or not s.strip()]
    if bad:
        return {}, f"metadata.source contains non-string/empty entries: {bad[:3]}"

    # 只有系统自有字段会被补全。
    changelog = meta.get("changelog")
    if not isinstance(changelog, list) or not changelog:
        meta = dict(meta)
        meta["changelog"] = [{"time": _now(), "content": "created"}]
        memory = {**memory, "metadata": meta}
        return memory, "REPAIRED"

    return memory, None


@dataclass
class EvidenceState:
    """``evidence.jsonl`` 的形态统计 —— agent loop 的**进展判定**依据。

    与 ``validate_file`` 的区别：这里只做**廉价**统计（行数 / 唯一 id / 重复行 / 指纹），
    不做字段校验。agent 每一步都会读一次，所以不能太重。

    **进展判定用 ``fingerprint``，不是行数、也不是字节数。** 这是实测踩出来的：
    模型会用 ``>>`` 反复追加同样的行（QA2 写了 130 行只有 11 个唯一 id），行数在涨、
    字节数也在涨，但一个事实都没多。所以"进展"的定义必须是**唯一记录集合发生了变化**：

    - 追加一条重复记录 → 唯一集合不变 → **不算进展**（正是要掐掉的空转）；
    - 新增一条记录 / 改写已有正文 / 删掉一条 → 唯一集合变了 → 算进展。

    只看 id 个数也会漏：原地改写一条记录，id 数不变但内容变了，那是真进展。所以指纹取
    "按 id 排序后的 (id, memory) 对的哈希"。

    解析走 ``_iter_records``（与 ``validate_file`` **同一套**），所以「一行是数组」
    「整份文件是美化 JSON」这些形态两边行为一致 —— 这两个解析器曾经各写一份，
    结果是 agent 认为文件是坏的、validator 认为文件是好的。
    """

    lines: int = 0
    bytes: int = 0
    unique_ids: int = 0
    duplicate_lines: int = 0
    parse_errors: int = 0
    fingerprint: str = ""

    def signature(self) -> str:
        return self.fingerprint

    def summary(self) -> str:
        parts = [f"{self.lines} lines", f"{self.unique_ids} unique ids", f"{self.bytes}B"]
        if self.duplicate_lines:
            parts.append(f"{self.duplicate_lines} duplicate lines")
        if self.parse_errors:
            parts.append(f"{self.parse_errors} unparseable lines")
        return ", ".join(parts)


@dataclass
class ParsedFile:
    """``_iter_records`` 的结果：记录 + 形态统计。

    ``fallback_used`` 表示"逐行读坏了、改用整份 JSON 文档解析成功" —— 此时文件是**美化
    多行 JSON**，必须重写才能让下游按行读；``multi_record_lines`` 表示有某一行装了多条记录
    （通常是 ``jq -s`` 的紧凑输出），同样需要展开成逐行。
    """

    records: list[dict[str, Any]] = field(default_factory=list)
    bad_lines: int = 0
    empty_lines: int = 0
    multi_record_lines: int = 0
    fallback_used: bool = False
    # 被 max_tokens 截断、按对象边界抢救回来的：尾部残片已丢，值得在日志里说一声
    salvaged: bool = False
    dropped_fragments: int = 0

    @property
    def needs_rewrite(self) -> bool:
        """文件形态是否偏离"一行一条"的约定（偏离就必须规范化写回）。"""
        return bool(self.fallback_used or self.multi_record_lines)


def _iter_records(text: str) -> ParsedFile:
    """解析文本里的全部记录。**委托给 ``io.parse_jsonl_text``，不自己实现。**

    这里曾经有第三份独立的解析逻辑（``validate_file``、``read_evidence_state``、
    ``io.read_jsonl`` 各一份），结果是同一份文件三处判断不一致。实测踩到的正是这个：
    模型一次 ``write`` 写出 39 条记录、末尾的 ``]`` 被 ``max_tokens`` 截掉，
    ``io`` 能按对象边界抢救出记录，而这里的旧实现因为"坏行数必须 >= 2"只看到
    "1 行、解析失败"，于是**把全部证据判成空**。

    所以解析只有一处实现（``io``），本模块只把它翻译成 ``ParsedFile`` 的形态统计。
    """
    from ..io import parse_jsonl_text

    result = parse_jsonl_text(text)
    return ParsedFile(
        records=result.records,
        bad_lines=result.bad_lines,
        empty_lines=result.empty_lines,
        multi_record_lines=result.multi_record_lines,
        # 非 linewise 就说明文件形态偏离"一行一条"，必须规范化写回；
        # 抢救模式还要让调用方知道尾部残片被丢了。
        fallback_used=result.mode != "linewise",
        salvaged=result.mode == "salvaged",
        dropped_fragments=result.dropped_fragments,
    )


def read_evidence_state(path: Path) -> EvidenceState:
    """廉价地统计产物形态（agent 每步调用，不做字段校验）。"""
    try:
        data = path.read_bytes()
    except OSError:
        return EvidenceState()
    if not data:
        return EvidenceState()
    text = data.decode("utf-8", errors="replace")
    parsed = _iter_records(text)
    state = EvidenceState(bytes=len(data), parse_errors=parsed.bad_lines)
    seen: set[str] = set()
    unique_records: list[tuple[str, str]] = []
    for record in parsed.records:
        state.lines += 1
        memory_id = (record.get("metadata") or {}).get("id")
        if not isinstance(memory_id, str) or not memory_id:
            continue
        if memory_id in seen:
            state.duplicate_lines += 1
            continue
        seen.add(memory_id)
        body = record.get("memory")
        unique_records.append((memory_id, body if isinstance(body, str) else ""))
    state.unique_ids = len(seen)
    if unique_records:
        digest = hashlib.sha256()
        for memory_id, body in sorted(unique_records):
            digest.update(memory_id.encode())
            digest.update(b"\x00")
            digest.update(body.encode())
            digest.update(b"\x01")
        state.fingerprint = digest.hexdigest()[:16]
    return state


def validate_file(path: Path) -> EvidenceReport:
    """逐行校验产物文件。文件不存在 → 空报告（不是错误：agent 可能什么都没写）。

    解析走 ``_iter_records``（与 ``read_evidence_state`` **同一套**），所以对"文件坏没坏"
    的判断两边永远一致。**整体 JSON 兜底**是必要的容错 —— ``jq -s`` 不加 ``-c`` 时输出是
    **多行美化**的，逐行读会得到一堆"解析失败"碎行，而 ``write_normalized`` 随后会把它们
    全部丢掉，**等于把证据删了**。实测的 dedupe 命令就踩到了这个坑。
    """
    report = EvidenceReport(path=str(path))
    if not path.exists():
        return report

    text = path.read_text(encoding="utf-8", errors="replace")
    parsed = _iter_records(text)
    records = parsed.records
    report.empty_lines = parsed.empty_lines
    report.multi_record_lines = parsed.multi_record_lines
    report.whole_file_fallback = parsed.fallback_used
    report.salvaged = parsed.salvaged
    report.dropped_fragments = parsed.dropped_fragments
    report.total_lines = len(records)
    if parsed.bad_lines:
        report.parse_errors = [
            {"line": index + 1, "error": "line is not valid JSON on its own"}
            for index in range(min(parsed.bad_lines, 5))
        ]

    seen_ids: dict[str, int] = {}
    for position, record in enumerate(records, start=1):
        memory, reason = normalize(record)
        if reason == "REPAIRED":
            memory_id = _id_of(memory)
            if memory_id in seen_ids:
                report.rejected.append({
                    "line": position, "id": memory_id,
                    "reason": f"duplicate metadata.id {memory_id!r}",
                })
                continue
            seen_ids[memory_id] = position
            report.repaired.append(memory)
            report.valid.append(memory)
            continue
        if reason is not None:
            report.rejected.append({
                "line": position,
                "id": _id_of(record),
                "reason": reason,
            })
            continue
        memory_id = _id_of(memory)
        if memory_id in seen_ids:
            report.rejected.append({
                "line": position, "id": memory_id,
                "reason": f"duplicate metadata.id {memory_id!r}",
            })
            continue
        seen_ids[memory_id] = position
        report.valid.append(memory)

    return report


def validate_file(path: Path) -> EvidenceReport:
    """逐行校验产物文件。文件不存在 → 空报告（不是错误：agent 可能什么都没写）。

    解析走 ``_iter_records``（与 ``read_evidence_state`` **同一套**），所以对"文件坏没坏"
    的判断两边永远一致。**整体 JSON 兜底**是必要的容错 —— ``jq -s`` 不加 ``-c`` 时输出是
    **多行美化**的，逐行读会得到一堆"解析失败"碎行，而 ``write_normalized`` 随后会把它们
    全部丢掉，**等于把证据删了**。实测的 dedupe 命令就踩到了这个坑。
    """
    report = EvidenceReport(path=str(path))
    if not path.exists():
        return report

    text = path.read_text(encoding="utf-8", errors="replace")
    parsed = _iter_records(text)
    records = parsed.records
    report.empty_lines = parsed.empty_lines
    report.multi_record_lines = parsed.multi_record_lines
    report.whole_file_fallback = parsed.fallback_used
    report.salvaged = parsed.salvaged
    report.dropped_fragments = parsed.dropped_fragments
    report.total_lines = len(records)
    if parsed.bad_lines:
        report.parse_errors = [
            {"line": index + 1, "error": "line is not valid JSON on its own"}
            for index in range(min(parsed.bad_lines, 5))
        ]

    seen_ids: dict[str, int] = {}
    for position, record in enumerate(records, start=1):
        memory, reason = normalize(record)
        if reason == "REPAIRED":
            memory_id = _id_of(memory)
            if memory_id in seen_ids:
                report.rejected.append({
                    "line": position, "id": memory_id,
                    "reason": f"duplicate metadata.id {memory_id!r}",
                })
                continue
            seen_ids[memory_id] = position
            report.repaired.append(memory)
            report.valid.append(memory)
            continue
        if reason is not None:
            report.rejected.append({
                "line": position,
                "id": _id_of(record),
                "reason": reason,
            })
            continue
        memory_id = _id_of(memory)
        if memory_id in seen_ids:
            report.rejected.append({
                "line": position, "id": memory_id,
                "reason": f"duplicate metadata.id {memory_id!r}",
            })
            continue
        seen_ids[memory_id] = position
        report.valid.append(memory)

    return report


def _try_whole_file(text: str) -> list[Any] | None:
    """把整个文件当一个 JSON 文档解析。返回记录列表，失败返回 None。"""
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return None


def write_normalized(path: Path, report: EvidenceReport) -> None:
    """把校验后的记录规范化写回（去空行、补 changelog、剔掉被拒的行）。

    只在**校验发现问题**时才写回，且保留原文件为 ``evidence.jsonl.raw`` —— 排错时能看到
    agent 到底写了什么。
    """
    had_problems = bool(
        report.rejected or report.parse_errors or report.empty_lines or report.multi_record_lines
    )
    # 整体 JSON 兜底命中时**必须**重写：此时 valid 是从整份文档里解析出来的，
    # 原文件是"多行美化 JSON"，不重写的话下游按行读又会全部读坏。
    if report.whole_file_fallback and report.valid:
        had_problems = True
    if not had_problems:
        return
    backup = path.with_suffix(path.suffix + ".raw")
    backup.write_text(path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    with path.open("w", encoding="utf-8") as handle:
        for memory in report.valid:
            handle.write(json.dumps(memory, ensure_ascii=False) + "\n")
