"""add · 原子记忆的时间处理：相对时间表达 → 带锚点/绝对时间的 metadata.time。

**单一事实源**：``parse_time_expression`` 把 agent 写的相对时间表达（"last week"、"yesterday"、
"5 years ago"）结合消息自己的会话时间戳（锚点）折算成最终落盘的 ``metadata.time``。被
``atommem_agent_v2``（并行 add agent 的收尾）与 ``atommem``（旧串行 agent）共用。

**核心口径（与 search / judge 一致）**：相对时间表达**本身就是完整答案**，只要能带上锚点，
就**原样保留**，不折算成绝对日期：
- 粗粒度（"last week" / "last year" / "N ago"）→ 保留 ``"<原表达> (anchor: <锚点日期>)"``
- 精确时点（"yesterday" / "today" / "tomorrow"）→ 用 ``date -d`` 算出绝对日期
- 只有年份的（"last year"）→ 精度只减不增，给年份

这样下游（search 的 evidence、answer、judge）拿到的都是一致的、无歧义的时间表述。
"""

from __future__ import annotations

import re
import subprocess

# 时间表达归一化：口语 → date -d 能理解的相对偏移。
_EXPR_OFFSETS = (
    ("yesterday", "-1 day"),
    ("today", "+0 day"),
    ("tomorrow", "+1 day"),
    ("last week", "-1 week"),
    ("last month", "-1 month"),
    ("last year", "-1 year"),
    ("next week", "+1 week"),
    ("next month", "+1 month"),
    ("next year", "+1 year"),
)

# 精确时点（可安全折算成绝对日期）：这三种 + "last/next <weekday>"（星期几是精确的）。
_POINT_PATTERNS = (r"yesterday", r"today", r"tomorrow")

# 精确星期几：折算成绝对日期（"last Friday" / "next Monday"；缩写 "Fri" 也认）。
_WEEKDAY_PATTERN = re.compile(
    r"\b(last|next|this)\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b",
    re.IGNORECASE,
)

_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

# 粗粒度相对表达（保留原表达 + 锚点，不折算）。星期几**不在此列**（见上）。
_COARSE_PATTERNS = (
    r"last\s+(week|month|year|weekend|night|summer|winter|spring|autumn|fall)",
    r"next\s+(week|month|year|weekend|summer|winter|spring|autumn|fall)",
    r"this\s+(week|month|year|weekend)",
    r"\d+\s+(day|week|month|year)s?\s+ago",
    r"for\s+\d+\s+(day|week|month|year)s?",
    r"recently|soon|future|past|a few|couple of",
)


def _run_date(args: list[str]) -> str | None:
    """跑一次 ``date``（LC_ALL=C 保证英文月份），成功返回 stdout，失败返回 None。"""
    try:
        result = subprocess.run(
            ["date", *args], capture_output=True, text=True, timeout=5, env={"LC_ALL": "C"},
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001 - date 不可用/超时都不该炸整条链
        pass
    return None


def _extract_date_from_anchor(anchor_time: str) -> str | None:
    """从锚点时间串里取出 date 命令认识的日期部分。

    "1:56 pm on 8 May, 2023" → "8 May 2023"；"8 May 2023" → "8 May 2023"；ISO 原样。
    取不到返回 None。
    """
    match = re.search(r"on\s+(.+)$", anchor_time)
    if match:
        return match.group(1).strip().replace(",", "")
    match = re.search(r"(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})", anchor_time)
    if match:
        return match.group(1).replace(",", "")
    match = re.search(r"(\d{4}-\d{2}-\d{2})", anchor_time)
    if match:
        return match.group(1)
    return None


def _normalize_expr(expr: str) -> str:
    """把口语化表述转成 ``date -d`` 能理解的格式（yesterday → -1 day 等）。"""
    out = expr
    for word, offset in _EXPR_OFFSETS:
        out = out.replace(word, offset)
    out = re.sub(r"(\d+)\s+(day|week|month|year)s?\s+ago", r"-\1 \2", out)
    return out


def _parse_anchor_date(anchor_date: str) -> "date | None":
    """把 "8 May 2023" / "2023-05-08" 解析成 ``date``（取不到返回 None）。"""
    from datetime import date

    text = anchor_date.strip()
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    match = re.search(r"(\d{1,2})\s+([A-Za-z]{3,})\.?,?\s+(\d{4})", text)
    if match and match.group(2)[:3].lower() in _MONTHS:
        try:
            return date(int(match.group(3)), _MONTHS[match.group(2)[:3].lower()], int(match.group(1)))
        except ValueError:
            return None
    return None


def _resolve_weekday(direction: str, weekday: str, anchor_date: str) -> str | None:
    """把 "last Friday" / "next Monday" 折算成相对锚点的绝对日期（ISO）。

    "last" = 严格早于锚点的最近那个星期几；"next" = 严格晚于；"this" 同 "next"（对最近的那个）。
    与 ``eval/judge.resolve_relative_date`` 同一套算法，保证两边口径一致。
    """
    from datetime import timedelta

    base = _parse_anchor_date(anchor_date)
    if base is None:
        return None
    key = weekday[:3].lower()
    if key not in _WEEKDAYS:
        return None
    target = _WEEKDAYS[key]
    if direction.lower() == "last":
        delta = (base.weekday() - target) % 7 or 7
        return (base - timedelta(days=delta)).isoformat()
    # next / this：锚点之后最近的该星期几（严格晚于）。
    delta = (target - base.weekday()) % 7 or 7
    return (base + timedelta(days=delta)).isoformat()


def parse_time_expression(time_expr: str, anchor_time: str) -> str:
    """把相对时间表达 + 锚点时间 → 最终落盘的 ``metadata.time``。

    口径见模块文档：**粗粒度保留原表达 + 锚点，只有精确时点才折算成绝对日期**。

    返回：
    - 带锚点的相对表述（``"last week (anchor: 9 June 2023)"``）← 粗粒度
    - ISO 日期（``"2023-06-08"``）← 精确时点（yesterday/today/tomorrow）
    - 年份（``"2022"``）← 只有年份的粗粒度
    - 区间（``"2023-05-01/2023-05-07"``）
    - 空字符串（无时间表达）
    """
    if not time_expr or not time_expr.strip():
        return ""
    expr = time_expr.strip().lower()
    if not anchor_time or not anchor_time.strip():
        return time_expr  # 没有锚点只能保留原表述

    anchor_date = _extract_date_from_anchor(anchor_time)
    if not anchor_date:
        return f"{time_expr} (anchor: {anchor_time})"

    # 精确时点：可以安全地折算成绝对日期。
    if any(re.search(p, expr) for p in _POINT_PATTERNS):
        got = _run_date(["-d", f"{anchor_date} {_normalize_expr(expr)}", "+%Y-%m-%d"])
        if got:
            return got

    # 精确星期几（"last Friday" / "next Monday"）：星期几是精确的，折算成绝对日期。
    # 用 Python 算（与 eval/judge 的 resolve_relative_date 同款逻辑），不用 GNU date 的
    # "last friday" —— 后者不可靠、且不认缩写（"Last Fri"）。
    weekday_hit = _WEEKDAY_PATTERN.search(expr)
    if weekday_hit:
        got = _resolve_weekday(weekday_hit.group(1), weekday_hit.group(2), anchor_date)
        if got:
            return got

    # 粗粒度：`<原表达> (anchor: <锚点日期>)`，不折算（保留完整答案）。
    if any(re.search(p, expr) for p in _COARSE_PATTERNS):
        # 纯 "last year" / "N years ago" 给年份（精度只减不增），其余保留原表达。
        if re.fullmatch(r"last\s+year", expr) or re.fullmatch(r"\d+\s+years?\s+ago", expr):
            match = re.search(r"(\d{4})", anchor_date)
            if match:
                year = int(match.group(1)) - 1
                return str(year)
        return f"{time_expr} (anchor: {anchor_date})"

    # 已含绝对年份的表达：原样返回。
    if re.search(r"\d{4}", expr):
        return time_expr

    # 其余：保留原表达 + 锚点。
    return f"{time_expr} (anchor: {anchor_date})"


# memory 正文里禁止出现的裸相对时间词（时间信息应进 metadata.time）。
_FORBIDDEN_TIME_WORDS = (
    r"\byesterday\b", r"\btoday\b", r"\btomorrow\b",
    r"\bago\b", r"\blast\b", r"\bnext\b", r"\bpast\b",
    r"\brecent\b", r"\brecently\b", r"\bsoon\b",
    r"\bfuture\b", r"\bnow\b", r"\bcurrently\b", r"\bpresently\b",
    r"\bbefore\b", r"\bafter\b", r"\bsince\b", r"\buntil\b", r"\bduring\b",
    r"\d+\s+(day|week|month|year)s?(\s+ago)?",
)


def has_time_word(memory: str) -> bool:
    """memory 正文是否含裸相对时间词（供校验/日志用）。"""
    return any(re.search(p, memory, re.IGNORECASE) for p in _FORBIDDEN_TIME_WORDS)


# 与 has_time_word 语义相反，保留旧名以兼容既有调用点。
def validate_memory_no_time(memory: str) -> bool:
    """memory 正文不含裸相对时间词则返回 True。"""
    return not has_time_word(memory)
