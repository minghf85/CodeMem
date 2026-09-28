"""``timecalc`` —— 相对时间/区间推理的辅助工具（agent 通过 bash 调）。

为什么要单独做一个命令而不是让模型自己算：LoCoMo 里大量问题是"什么时候"，
而证据里给的是相对表述（"about a month ago"、"two weeks later"、"since I was a kid"）。
模型做日期算术**不可靠**（实测会把 2023-05-08 减一个月算成 2023-04-08 也可能算成别的），
而这类错误会直接变成错误证据。所以把算术交给确定性工具，模型只负责决定**要算什么**。

    timecalc shift 2023-06-17 -1month      -> 2023-05-17
    timecalc shift 2023-05-08 +2week       -> 2023-05-22
    timecalc diff 2023-05-08 2023-06-17    -> 40 days
    timecalc diff 2023-06-17 2023-05-08    -> -40 days
    timecalc range 2023-05-01 2023-05-31   -> 2023-05-01/2023-05-31
    timecalc info 2023-05-08T13:56:00      -> 2023-05-08 (Monday), 2023-W19

输出**纯文本单行**（不是 JSON）—— 它就是给人/模型看的算术结果，直接嵌进记忆正文。

只用 stdlib：``datetime`` + ``calendar``。月份加减按"同日、越界取月末"处理
（``2023-01-31 +1month`` → ``2023-02-28``），这与人类表述"一个月后"的直觉一致。
"""

from __future__ import annotations

import calendar
import re
import sys
from datetime import date, datetime, timedelta

# 支持的偏移单位（全写与缩写）
UNITS: dict[str, str] = {
    "day": "day", "days": "day", "d": "day",
    "week": "week", "weeks": "week", "w": "week",
    "month": "month", "months": "month", "mo": "month",
    "year": "year", "years": "year", "y": "year",
    "hour": "hour", "hours": "hour", "h": "hour",
    "minute": "minute", "minutes": "minute", "min": "minute", "m": "minute",
}

_OFFSET_RE = re.compile(r"^([+-]?)(\d+(?:\.\d+)?)\s*([a-zA-Z]+)$")
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?")


class TimeCalcError(ValueError):
    """入参无法解析。调用方把它渲染成一条错误观测，模型自己改。"""


# LoCoMo 的原始会话时间形如 ``"1:56 pm on 8 May, 2023"``。会话记录里的 time 字段
# **保留这个原始形式**（见 io.msg_time 的说明），所以 timecalc 必须能直接吃它 ——
# 否则 agent 拿到锚点却算不了，相对时间推理就断了。
_LOCOMO_RE = re.compile(
    r"^(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s*(\d{4})",
    re.IGNORECASE,
)
_MONTH_NAMES = {
    name.lower(): index
    for index, name in enumerate(
        ["January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"], 1
    )
}


def parse_datetime(text: str) -> datetime:
    """解析日期。接受三种写法：

    - ISO-8601：``2023-05-08`` / ``2023-05-08T13:56[:00]``（带时区按字面本地时间处理）
    - **LoCoMo 原始形式**：``1:56 pm on 8 May, 2023``（会话记录里的 ``time`` 就是这个）
    - 只给年月日或只有月份：``May 2023``（日取 1）

    支持原始形式是必需的：``msg_time`` 刻意不做归一化，所以锚点常常就是这个串。
    """
    value = text.strip().replace("Z", "").strip()
    if not value:
        raise TimeCalcError("cannot parse an empty date")

    locomo = _LOCOMO_RE.match(value)
    if locomo:
        hour = int(locomo.group(1)) % 12
        if locomo.group(3).lower() == "pm":
            hour += 12
        month = _MONTH_NAMES.get(locomo.group(5).lower())
        if month is None:
            raise TimeCalcError(f"unknown month in {text!r}")
        try:
            return datetime(
                int(locomo.group(6)), month, int(locomo.group(4)),
                hour, int(locomo.group(2)),
            )
        except ValueError as exc:
            raise TimeCalcError(f"invalid date {text!r}: {exc}") from exc

    match = _DATE_RE.match(value)
    if not match:
        # 兜底：``8 May 2023`` 这类缺时分的形式
        alt = re.match(r"^(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})$", value)
        if alt and alt.group(2).lower() in _MONTH_NAMES:
            try:
                return datetime(int(alt.group(3)), _MONTH_NAMES[alt.group(2).lower()],
                                int(alt.group(1)))
            except ValueError as exc:
                raise TimeCalcError(f"invalid date {text!r}: {exc}") from exc
        raise TimeCalcError(
            f"cannot parse date {text!r}: expected YYYY-MM-DD, YYYY-MM-DDThh:mm[:ss], "
            f"or '1:56 pm on 8 May, 2023'"
        )
    year, month, day = (int(match.group(i)) for i in (1, 2, 3))
    hour = int(match.group(4) or 0)
    minute = int(match.group(5) or 0)
    second = int(match.group(6) or 0)
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError as exc:
        raise TimeCalcError(f"invalid date {text!r}: {exc}") from exc


def parse_offset(text: str) -> tuple[float, str]:
    """``-1month`` / ``+2 weeks`` / ``3day`` → ``(-1, 'month')``。"""
    match = _OFFSET_RE.match(text.strip())
    if not match:
        raise TimeCalcError(
            f"cannot parse offset {text!r}: expected e.g. -1month, +2week, 3day"
        )
    sign, amount, unit = match.group(1), float(match.group(2)), match.group(3).lower()
    if unit not in UNITS:
        raise TimeCalcError(
            f"unknown unit {unit!r}: use one of {sorted(set(UNITS.values()))}"
        )
    return (-amount if sign == "-" else amount), UNITS[unit]


def add_months(moment: datetime, months: int) -> datetime:
    """加 N 个月，日按"越界取当月最后一天"处理（2023-01-31 +1month → 2023-02-28）。"""
    total = moment.month - 1 + months
    year = moment.year + total // 12
    month = total % 12 + 1
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def shift(moment: datetime, amount: float, unit: str) -> datetime:
    if unit == "month":
        # 非整数月按天数近似（罕见；模型应该用整数月）
        whole = int(amount)
        extra_days = (amount - whole) * 30.44
        result = add_months(moment, whole)
        return result + timedelta(days=extra_days)
    if unit == "year":
        whole = int(amount)
        extra_days = (amount - whole) * 365.25
        result = add_months(moment, whole * 12)
        return result + timedelta(days=extra_days)
    factor = {"day": 1, "week": 7, "hour": 1 / 24, "minute": 1 / 1440}[unit]
    return moment + timedelta(days=amount * factor)


def format_moment(moment: datetime) -> str:
    """有具体时分秒就带上，否则只给日期（避免凭空造出 00:00:00 这种假精度）。"""
    if (moment.hour, moment.minute, moment.second) == (0, 0, 0):
        return moment.strftime("%Y-%m-%d")
    if moment.second == 0:
        return moment.strftime("%Y-%m-%dT%H:%M")
    return moment.strftime("%Y-%m-%dT%H:%M:%S")


def format_delta(delta: timedelta) -> str:
    """把差值说成人能读的形式：天数为主，必要时给"约几个月/几年"。"""
    days = delta.total_seconds() / 86400
    parts = [f"{days:+.0f} days" if days >= 0 else f"{days:.0f} days"]
    absolute = abs(days)
    if absolute >= 27:
        parts.append(f"~{absolute / 30.44:.1f} months")
    if absolute >= 300:
        parts.append(f"~{absolute / 365.25:.1f} years")
    return ", ".join(parts)


def run(argv: list[str]) -> tuple[int, str]:
    """返回 ``(退出码, 输出文本)``。纯函数，方便自测。"""
    if not argv:
        return 2, USAGE
    command, rest = argv[0].lower(), argv[1:]

    try:
        if command == "shift":
            if len(rest) < 2:
                return 2, f"usage: timecalc shift <date> <offset>   e.g. shift 2023-06-17 -1month"
            base = parse_datetime(rest[0])
            # 允许多个偏移叠加：shift 2023-06-17 -1month +3day
            result = base
            for token in rest[1:]:
                amount, unit = parse_offset(token)
                result = shift(result, amount, unit)
            return 0, format_moment(result)

        if command == "diff":
            if len(rest) < 2:
                return 2, "usage: timecalc diff <date_a> <date_b>   -> days from a to b"
            left, right = parse_datetime(rest[0]), parse_datetime(rest[1])
            delta = right - left
            return 0, f"{format_delta(delta)}  ({format_moment(left)} -> {format_moment(right)})"

        if command == "range":
            if len(rest) < 2:
                return 2, "usage: timecalc range <start> <end>   -> ISO interval"
            return 0, f"{format_moment(parse_datetime(rest[0]))}/{format_moment(parse_datetime(rest[1]))}"

        if command == "info":
            if not rest:
                return 2, "usage: timecalc info <date>"
            moment = parse_datetime(rest[0])
            return 0, (
                f"{format_moment(moment)} ({moment.strftime('%A')}), "
                f"{moment.strftime('%Y-W%W')}, iso_week={moment.isocalendar()[1]}, "
                f"day_of_year={moment.timetuple().tm_yday}"
            )

        if command in ("-h", "--help", "help"):
            return 0, USAGE
        return 2, f"unknown command {command!r}\n\n{USAGE}"

    except TimeCalcError as exc:
        return 1, f"error: {exc}"


USAGE = """timecalc -- deterministic date arithmetic for relative-time reasoning

  timecalc shift <date> <offset> [...]   move a date; offsets like -1month, +2week, 3day
                                         (multiple offsets apply left to right)
  timecalc diff <date_a> <date_b>        signed difference from a to b, in days/months
  timecalc range <start> <end>           ISO interval "start/end"
  timecalc info <date>                   weekday, ISO week, day of year

Dates accept YYYY-MM-DD or YYYY-MM-DDThh:mm[:ss].
Examples:
  timecalc shift 2023-06-17 -1month          -> 2023-05-17
  timecalc shift 2023-05-08 +2week           -> 2023-05-22
  timecalc diff 2023-05-08 2023-06-17        -> +40 days, ~1.3 months  (...)
"""


def main(argv: list[str] | None = None) -> int:
    code, output = run(sys.argv[1:] if argv is None else argv)
    print(output, file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
