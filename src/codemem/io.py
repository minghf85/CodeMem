"""跨步骤共享的基础设施：JSONL 读写、路径解析、时间格式化。

**为什么单独一层**：add / search / answer / eval 四个步骤都要读 jsonl、都要把
``{speaker_a}_{speaker_b}`` 解析成 ``data/`` 下的目录、都要把 memory item 的
``metadata.time`` 取出来。这些东西放在任何一个步骤里都会让另外三个反向依赖它 ——
所以提到顶层，让步骤之间只通过数据文件耦合（这正是 add→search→answer 链路的意图）。

依赖方向（严格单向，不允许出现反向 import）：

    io / llm / log / dataset          <- 无内部依赖
        ^
    add / search / answer / eval      <- 只依赖上面四个
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读 jsonl。**坏的/空的行直接跳过**，不抛异常 —— 产物经常是模型写的，得容错。

    需要知道"坏了几行"时用 ``read_jsonl_report``。
    """
    records, _bad, _empty = read_jsonl_report(path)
    return records


@dataclass
class ParseResult:
    """一次产物解析的结果。

    ``mode`` 说明文本是**怎么**解析出来的 —— 调用方需要它来决定要不要规范化写回：

        linewise  正常：一行一条（或一行一个紧凑数组）。原样可用。
        document  一行都解不出，整份文件是合法 JSON（``jq -s`` 的美化输出）。**必须重写**，
                  否则下游按行读又会读坏。
        salvaged  整份也不是合法 JSON，按对象边界抢救出完整的记录（被截断的数组）。
                  **必须重写**，且调用方应当知道丢了尾部。
    """

    records: list[dict[str, Any]] = field(default_factory=list)
    bad_lines: int = 0
    empty_lines: int = 0
    dropped_fragments: int = 0
    mode: str = "linewise"

    @property
    def needs_rewrite(self) -> bool:
        """文件形态是否偏离"一行一条"的约定（偏离就必须规范化写回）。"""
        return self.mode != "linewise" or self.multi_record_lines > 0

    @property
    def multi_record_lines(self) -> int:
        """有多少行装了多条记录（``jq -s`` 的紧凑数组）；这类也要展开成逐行。"""
        return self._multi

    _multi: int = 0

    def as_tuple(self) -> tuple[list[dict[str, Any]], int, int]:
        return self.records, self.bad_lines, self.empty_lines


def read_jsonl_report(path: Path) -> tuple[list[dict[str, Any]], int, int]:
    """读 jsonl 并返回 ``(记录, 坏行数, 空行数)``。文件不存在时返回空。"""
    if not path.exists():
        return [], 0, 0
    return read_jsonl_report_text(path.read_text(encoding="utf-8", errors="replace"))


def read_jsonl_report_text(text: str) -> tuple[list[dict[str, Any]], int, int]:
    """兼容入口：只要 ``(记录, 坏行数, 空行数)``。需要知道解析方式时用 ``parse_jsonl_text``。"""
    return parse_jsonl_text(text).as_tuple()


def parse_jsonl_text(text: str) -> ParseResult:
    """从**文本**解析记录 —— **整个项目唯一的产物解析路径**。

    ``io.read_jsonl``、``search/evidence.py`` 的校验与进展统计、``search/agent.py`` 的观测渲染
    全都走这里。曾经有三份独立实现，导致同一个文件三处判断不一致（实测后果：一次被截断的
    ``write`` 在一个地方能救回 38 条记录，在另一个地方被判成空文件）。

    三层容错，按代价从低到高：

    1. **linewise**：一行一个对象，或一行一个数组（``jq -s`` 的紧凑输出）。
    2. **document**：一行都解不出、但整份文件是合法 JSON（``jq -s`` 不加 ``-c`` 的美化输出）。
    3. **salvaged**：整份也不是合法 JSON —— 多半是被 ``max_tokens`` 截断的数组
       （实测：39 条记录、末尾 ``]`` 被切掉）。按对象边界切开，把完整的捞回来，
       只丢最后那个残片。不抢救的话前面几十条完好证据会一起消失。
    """
    result = ParseResult()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            result.empty_lines += 1
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            result.bad_lines += 1
            continue
        if isinstance(value, list):
            items = [item for item in value if isinstance(item, dict)]
            if len(value) > 1:
                result._multi += 1
            result.records.extend(items)
        elif isinstance(value, dict):
            result.records.append(value)
        else:
            result.bad_lines += 1

    # 逐行读出了东西（或本来就没有坏行）→ 就是 linewise
    if result.records or not result.bad_lines:
        return result

    # 备份逐行阶段的计数，供下面两种兜底失败时原样返回
    linewise = ParseResult(
        records=list(result.records), bad_lines=result.bad_lines,
        empty_lines=result.empty_lines, _multi=result._multi,
    )

    # 整份文档解析 → 先原样试，再试**语法修复**（模型常漏/多打括号与逗号；
    # 修复只动标点，不改字段内容）
    whole = _parse_whole_document(text)
    if whole is None:
        whole = _parse_whole_document(repair_json(text))
    if whole is not None:
        items = [item for item in whole if isinstance(item, dict)]
        return ParseResult(
            records=items, empty_lines=result.empty_lines, mode="document",
            _multi=1 if len(items) > 1 else 0,
        )

    salvaged, dropped = salvage_json_records(text)
    if not salvaged:
        # 按对象边界切不出来 → 多半是**括号没配平**（实测：metadata 少一个 `}`）。
        # 先修复再切。
        salvaged, dropped = salvage_json_records(repair_json(text))
    if salvaged:
        return ParseResult(
            records=salvaged, empty_lines=result.empty_lines, mode="salvaged",
            dropped_fragments=dropped, _multi=1 if len(salvaged) > 1 else 0,
        )
    return linewise


def salvage_json_records(text: str) -> tuple[list[dict[str, Any]], int]:
    """从**损坏的 JSON 数组**里逐条抢救记录。返回 ``(记录, 丢弃的残片数)``。

    两种损坏形态都要能救（都实测遇到过）：

    1. **被截断**：最后一条写到一半，``]`` 和尾巴没了 → 丢掉最后那条残片，前面的全保住。
    2. **括号漏写**：某条记录少一个 ``}``（实测：``metadata`` 的闭括号没了），
       于是 ``...}]}, {...`` 让整份文件变成非法 JSON → **全部记录一起消失**。

    做法：从 ``[`` 之后找第一个 ``{`` 作为候选起点，然后**按边界逐个推进** ——

    - 先从最近的 ``}`` 往后扫，找**未经修补就能解析**的片段（正常情况一次命中）；
    - 都失败才允许补 1~2 个闭括号再试（对应"漏写 ``}``"）；
    - 认下一条后，从它的结尾继续找下一个 ``{``，所以**永远不会跑进嵌套对象里**。

    先试"无需修补"是关键：``{"a":{"b":1},"c":2}`` 这种记录，若一上来就允许补括号，
    会误把 ``{"a":{"b":1}}`` 当成完整记录而丢掉 ``"c":2``。
    """
    records: list[dict[str, Any]] = []
    dropped = 0

    cursor = text.find("[")
    cursor = 0 if cursor < 0 else cursor + 1

    while True:
        start = text.find("{", cursor)
        if start < 0:
            break

        found: tuple[dict[str, Any], int] | None = None
        # 第一遍：完全不修补（正常边界）
        for extra in ("", "}", "}}"):
            pos = text.find("}", start)
            while pos >= 0:
                chunk = text[start:pos + 1] + extra
                try:
                    value = json.loads(chunk)
                except json.JSONDecodeError:
                    pos = text.find("}", pos + 1)
                    continue
                if isinstance(value, dict):
                    found = (value, pos)
                    break
                pos = text.find("}", pos + 1)
            if found is not None:
                break

        if found is None:
            dropped += 1          # 尾部残片（截断）或彻底无法解析的部分
            break

        value, pos = found
        records.append(value)
        cursor = pos + 1

    return records, dropped


def _parse_whole_document(text: str) -> list[Any] | None:
    """把整份文件当一个 JSON 文档解析；失败返回 None（由调用方决定要不要抢救）。"""
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


def write_jsonl(path: Path, items: Iterable[dict[str, Any]]) -> None:
    """整文件重写。先写临时文件再 ``os.replace``，避免中断时留下半个文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile

    descriptor, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def append_jsonl(path: Path, items: Iterable[dict[str, Any]]) -> None:
    """追加写（断点续跑用）。每行独立，中断只会丢最后一行。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def ensure_jsonl(path: Path) -> None:
    """确保文件存在（可能为空）。断点续跑时用来"占位"。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

def resolve_dir(target: str | Path, root: Path | None = None) -> Path | None:
    """把一个"目录名或路径"解析成实际目录。

    接受三种写法：``Caroline_Melanie``（相对 ``data/``）、``data/Caroline_Melanie``、
    或绝对路径。找不到返回 ``None``（调用方决定是跳过还是报错）。
    """
    base = root or DATA_DIR
    path = Path(target)
    candidates = [path] if path.is_absolute() else [base / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def default_config_path(name: str) -> Path:
    """``default_config_path("search")`` -> ``configs/search.yaml``。"""
    return CONFIG_DIR / f"{name}.yaml"


# ---------------------------------------------------------------------------
# session 记录（session_template.jsonl 的格式）
# ---------------------------------------------------------------------------

def msg_id(record: dict[str, Any]) -> str:
    """取 ``msg_id``。（旧的 ``metadata.id`` 形态也兼容，便于读历史产物。）"""
    value = record.get("msg_id")
    if isinstance(value, str) and value:
        return value
    return memory_id(record)


def msg_role(record: dict[str, Any]) -> str:
    """发言者。原始消息是说话人名字；两层 summary 的产物是 ``"summary"``。"""
    value = record.get("role")
    return value if isinstance(value, str) else ""


def msg_content(record: dict[str, Any]) -> str:
    """正文。兼容旧的 ``memory`` 字段。"""
    for key in ("content", "memory", "text"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def msg_time(record: dict[str, Any]) -> str:
    """时间。**保留原始形式**（如 ``"1:56 pm on 8 May, 2023"``），不做归一化。

    为什么不在这一步归一化：原始形式里的信息量比一个被猜出来的 ISO 串更可靠，
    而且猜错会把错误固化进下游。需要绝对日期的地方（相对时间推理）用
    确定性工具（bash 的 ``date -d``）现场折算 —— 那里能看见折算过程。
    """
    value = record.get("time")
    if isinstance(value, str):
        return value
    return memory_time(record)


def msg_kind(record: dict[str, Any]) -> str:
    """时间种类：``point`` / ``range`` / ``approx``（消解阶段写入；原始消息为空）。"""
    value = record.get("time_kind")
    return value if isinstance(value, str) else ""


def source_content(record: dict[str, Any]) -> str:
    """原始措辞（消解前）。未消解过的记录为空 —— 此时 ``content`` 就是原话。"""
    value = record.get("source_content")
    return value if isinstance(value, str) else ""


def source_time(record: dict[str, Any]) -> str:
    """原始时间词（会话时间戳）。未消解过的记录为空。"""
    value = record.get("source_time")
    return value if isinstance(value, str) else ""


def make_session_record(
    msg_id_value: str, role: str, content: str, time: str = ""
) -> dict[str, Any]:
    """建一条 session_template 记录。字段顺序固定，便于 diff 与人工阅读。"""
    return {"msg_id": msg_id_value, "role": role, "time": time, "content": content}


# ---------------------------------------------------------------------------
# memory item（旧格式，仅为读历史产物保留）
# ---------------------------------------------------------------------------

def memory_id(item: dict[str, Any]) -> str:
    """取 ``metadata.id``；缺失或非字符串返回空串（绝不返回 None，调用方少一层判断）。"""
    value = (item.get("metadata") or {}).get("id")
    return value if isinstance(value, str) else ""


def memory_text(item: dict[str, Any]) -> str:
    """取用于嵌入/展示的正文，兼容 memory item 与 message dict 两种形态。"""
    for key in ("memory", "text"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return ""


def memory_speaker(item: dict[str, Any]) -> str:
    """从 tag 里抽 ``speaker:<name>``；没有则 ``"unknown"``。"""
    tags = (item.get("metadata") or {}).get("tag") or []
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("speaker:"):
            return tag.split(":", 1)[1]
    return "unknown"


def memory_time(item: dict[str, Any]) -> str:
    value = (item.get("metadata") or {}).get("time")
    return value if isinstance(value, str) else ""


def dedupe_by_id(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 id 去重，保留首次出现。**没有 id 的条目一律保留** —— 它们无法判定重复。"""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        item_id = memory_id(item)
        if item_id and item_id in seen:
            continue
        if item_id:
            seen.add(item_id)
        result.append(item)
    return result


def format_time(value: Any) -> str:
    """把 time 字段规整成字符串（模型偶尔会给出非字符串）。空值返回空串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def now_iso() -> str:
    """带时区的 ISO 时间戳（changelog 的 time 字段用）。"""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# JSON 修复
# ---------------------------------------------------------------------------

def repair_json(text: str) -> str:
    """修小模型输出的常见 JSON 语法错误：缺失逗号、多余逗号、尾随逗号、裸控制字符。

    只做**语法**层面的修补，不改语义 —— 修完仍要过 ``json.loads`` 才算数，所以不会把
    一个语义已经错掉的对象"修"成看起来合法的样子。
    """
    text = re.sub(r"}\s*\{", "},{", text)          # 对象之间漏逗号
    text = re.sub(r"\]\s*\[", "],[", text)          # 数组之间漏逗号
    text = re.sub(r"}\s*\]\s*\]}\s*$", "}}]}", text)  # 多写了一层数组闭合
    text = re.sub(r"\]\s*\]\s*}", "]}", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)      # 尾随逗号
    text = _balance_brackets(text)
    return "".join(ch if ch >= " " or ch in "\n\t" else " " for ch in text)


def _balance_brackets(text: str) -> str:
    """补上**漏掉的**括号（不改动任何字符串内容）。

    实测形态：一条记录里 ``metadata`` 少打一个 ``}``，于是数组在下一个记录处
    ``...}]}, {...`` 变成非法 JSON。这类"括号没配平"是模型长输出的常见错误，
    而它会让**整份文件**解析失败 —— 不做修复就丢掉全部记录。

    做法：扫一遍统计括号栈，遇到 ``}`` 而栈顶是 ``[``（或反之）时**先补一个对应的闭括号**。
    只在真正错位处补，正确的地方一律不动，所以对合法 JSON 是恒等变换。
    """
    out: list[str] = []
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            out.append(char)
            continue
        if char in "[{":
            stack.append(char)
            out.append(char)
            continue
        if char in "]}":
            want = "{" if char == "}" else "["
            # 栈顶与当前闭括号不匹配 → 先把栈里缺的那些补上
            while stack and stack[-1] != want:
                out.append("}" if stack.pop() == "{" else "]")
            if stack:
                stack.pop()
            out.append(char)
            continue
        out.append(char)
    # 结尾还欠的括号一并补上
    while stack:
        out.append("}" if stack.pop() == "{" else "]")
    return "".join(out)
