"""LLM-as-a-judge utilities for Locomo answer equivalence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .. import llm
from ..io import PROJECT_ROOT
from ..prompts import JUDGE_PROMPT

CONFIG_FILE = PROJECT_ROOT / "configs" / "eval.yaml"


def load_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """读 ``configs/eval.yaml`` 的 judge 段并叠加到默认调用参数上。"""
    import yaml

    merged = dict(llm.DEFAULT_CALL_CONFIG)
    if CONFIG_FILE.exists():
        loaded = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            merged.update(loaded.get("judge") or loaded)
    if config:
        merged.update(config)
    return merged

CONFIG_FILE = PROJECT_ROOT / "configs" / "judge.yaml"


# ---------------------------------------------------------------------------
# 相对时间的确定性折算
# ---------------------------------------------------------------------------

_MONTHS = {m.lower(): i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}


def _parse_date(text: str) -> "date | None":
    """从一句回答里抽出日期。支持 ISO / day-first / month-first / 只有年月。"""
    import re
    from datetime import date

    value = str(text or "")
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", value)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    match = re.search(r"(\d{1,2})\s+([A-Za-z]{3,})\.?,?\s+(\d{4})", value)
    if match and match.group(2)[:3].lower() in _MONTHS:
        try:
            return date(int(match.group(3)), _MONTHS[match.group(2)[:3].lower()], int(match.group(1)))
        except ValueError:
            return None
    match = re.search(r"([A-Za-z]{3,})\.?\s+(\d{1,2}),?\s+(\d{4})", value)
    if match and match.group(1)[:3].lower() in _MONTHS:
        try:
            return date(int(match.group(3)), _MONTHS[match.group(1)[:3].lower()], int(match.group(2)))
        except ValueError:
            return None
    match = re.search(r"([A-Za-z]{3,})\.?\s+(\d{4})", value)
    if match and match.group(1)[:3].lower() in _MONTHS:
        try:
            return date(int(match.group(2)), _MONTHS[match.group(1)[:3].lower()], 1)
        except ValueError:
            return None
    return None


def resolve_relative_date(reference: Any) -> "date | None":
    """把参考里 ``the week/weekend/<weekday> before X`` 折算成绝对日期。

    **为什么要在这里算**：judge 是 8B 模型，"13 Sep 2023 是周三，之前的周末是 9-10 号"
    这种推算它做不可靠 —— 实测同一条判 3 次都判错，而正确答案是确定的。
    日期算术属于**确定性计算**，不该交给语言模型（本项目在 search 阶段已经就这一点
    做过同样的取舍：相对时间交给确定性代码，而不是模型）。

    支持的形式（大小写不敏感）：
        the <weekday> before X      -> X 之前最近的那个星期几（严格早于 X）
        the weekend before X        -> X 之前最近的周六
        N weekends before X         -> X 之前第 N 个周六
        the week before X           -> X - 7 天
        N weeks before X            -> X - 7N 天

    **两个实测踩过的坑**（后果都是把对的答案判成错的，比不折算更糟）：

    1. 原正则把 ``Friday`` 匹配成了单位 ``day`` —— ``(day|week|weekend)s?`` 命中了
       "Fri**day**" 的后缀，于是 "The Friday before 15 July 2023" 被算成 X-1 天（7 月 8 日），
       而正确答案是 7 月 14 日。实测这一条直接把若干答对的 QA 判成 INCORRECT。
       修法：**星期几先匹配**，并用词首边界锚住。
    2. ``two weekends before X`` 里的 "two" 被丢掉了，只算了"最近的那个周六"。
       修法：weekend 分支也要乘数量。
    """
    import re
    from datetime import timedelta

    text = str(reference or "")

    # ---- 星期几分支：必须先试，否则 "Friday" 的 "day" 会被下面的单位正则吃掉 ----
    match = re.search(
        r"(?:the\s+)?(?:(a|one|two|three|four|five|six|\d+)\s+)?"
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b"
        r"\s+before\s+(.+)",
        text, re.IGNORECASE,
    )
    if match:
        base = _parse_date(match.group(3))
        if base is None:
            return None
        target = _WEEKDAYS[match.group(2).lower()]
        delta = (base.weekday() - target) % 7 or 7
        base = base - timedelta(days=delta)
        # "two Fridays before X" 这种也支持：再往前推整周
        count = _quantity(match.group(1))
        return base - timedelta(days=7 * (count - 1))

    # ---- 单位分支（day / week / weekend）----
    match = re.search(
        r"(?:the\s+)?(?:(a|one|two|three|four|five|six|\d+)\s+)?"
        r"\b(day|week|weekend)s?\b\s+before\s+(.+)",
        text, re.IGNORECASE,
    )
    if not match:
        return None
    base = _parse_date(match.group(3))
    if base is None:
        return None

    unit, count = match.group(2).lower(), _quantity(match.group(1))
    if unit == "weekend":
        # X 之前第 N 个周六。"before" 是**严格早于**：X 本身是周六时先退一周。
        first = base - timedelta(days=(base.weekday() - 5) % 7 or 7)
        return first - timedelta(days=7 * (count - 1))
    if unit == "day":
        return base - timedelta(days=count)
    return base - timedelta(days=7 * count)


_COUNT_WORDS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _quantity(word: Any) -> int:
    """把 "two" / "3" / None 统一成整数（None 与无法识别的一律当 1）。"""
    text = str(word or "a").strip().lower()
    if text.isdigit():
        return int(text)
    return _COUNT_WORDS.get(text, 1)


def temporal_hint(reference: Any, candidate: Any) -> str:
    """给 judge 的一行**已算好的**时间对照，让它只做比较、不做推算。

    预测能解析成日期、且与折算结果相差在容忍范围内（周末允许 ±1 天）时给出明确的
    "SAME DATE" 结论；否则只给出两个日期让 judge 自己判断。返回空串表示不适用
    （不是时间型参考，或有一边解析不出日期）—— 那时保持原行为，不干扰 judge。
    """
    expected = resolve_relative_date(reference)
    if expected is None:
        return ""
    actual = _parse_date(candidate)
    if actual is None:
        return (
            f"\n[PRECOMPUTED] The reference resolves to {expected.isoformat()}. "
            f"The predicted answer does not state an absolute date; judge it on its own wording."
        )
    # 周末允许周六/周日两种写法（相差 1 天）；其它形式要求精确到天
    tolerance = 1 if "weekend" in str(reference).lower() else 0
    if abs((actual - expected).days) <= tolerance:
        return (
            f"\n[PRECOMPUTED] Reference \"{reference}\" resolves to {expected.isoformat()}; "
            f"the prediction is {actual.isoformat()}. These are the SAME date -- treat the "
            f"temporal criterion as satisfied. Do not re-derive the arithmetic."
        )
    return (
        f"\n[PRECOMPUTED] Reference \"{reference}\" resolves to {expected.isoformat()}; "
        f"the prediction is {actual.isoformat()}. These differ by "
        f"{abs((actual - expected).days)} day(s)."
    )


def build_judge_messages(
    question: str, reference: str, candidate: Any, unsupported: bool = False
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a strict answer evaluator. Follow the instructions carefully. "
                "Respond with ONLY a JSON object, no markdown fences, no extra text."
            ),
        },
        {
            "role": "user",
            "content": (
                JUDGE_PROMPT.format(
                    question=question, reference=reference,
                    prediction=render_prediction(candidate, unsupported),
                )
                + temporal_hint(reference, candidate)
            ),
        },
    ]


def render_prediction(candidate: Any, unsupported: bool = False) -> str:
    """Render a prediction for the judge.

    Answer prompts now return structured JSON, but the judge is defined over an answer string.
    Passing a dict to str.format would render Python repr, so serialize dicts to JSON and turn
    a null/unsupported answer into an explicit placeholder.
    """
    if candidate is None:
        return "(no answer provided)"
    if isinstance(candidate, (dict, list)):
        candidate = json.dumps(candidate, ensure_ascii=False)
    text = str(candidate).strip()
    if not text:
        text = "(no answer provided)"
    if unsupported:
        text = f"{text} (model marked this answer as unsupported by the memories)"
    return text


def _normalize_label(label: Any) -> str | None:
    if not isinstance(label, str):
        return None
    label = label.strip().upper()
    if label == "WRONG":
        return "INCORRECT"
    return label if label in {"CORRECT", "INCORRECT"} else None


def parse_judge_output(text: str) -> dict[str, Any]:
    """Parse judge output.

    Primary format is JSON: {"label": "CORRECT"|"INCORRECT", "reason": "..."}.
    XML-style tags and loose CORRECT/INCORRECT mentions are still accepted as fallbacks.
    """
    # 1) JSON object (current prompt contract)
    try:
        parsed = json.loads(text.strip())
    except (json.JSONDecodeError, AttributeError):
        parsed = None
    if isinstance(parsed, dict):
        label = _normalize_label(parsed.get("label", parsed.get("judgement", parsed.get("judgment"))))
        if label:
            reason = parsed.get("reason", parsed.get("explanation", ""))
            return {"label": label, "reason": reason if isinstance(reason, str) else str(reason)}

    # 2) Legacy XML-style tags
    judgement_match = re.search(r'<judgement>\s*(CORRECT|INCORRECT)\s*</judgement>', text, re.IGNORECASE)
    if judgement_match:
        reason_match = re.search(r'<reason>\s*(.+?)\s*</reason>', text, re.DOTALL | re.IGNORECASE)
        return {
            "label": judgement_match.group(1).upper(),
            "reason": reason_match.group(1).strip() if reason_match else "",
        }

    # 3) Fallback: loose textual mention
    text_upper = text.upper()
    if "INCORRECT" in text_upper:
        return {"label": "INCORRECT", "reason": text.strip()}
    if "CORRECT" in text_upper:
        return {"label": "CORRECT", "reason": text.strip()}

    raise ValueError(f"Could not parse judge output: {text[:200]}")


def judge_answer(
    question: str,
    reference: str,
    candidate: Any,
    unsupported: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = load_config(config)
    raw = llm.complete_sync(merged, build_judge_messages(question, reference, candidate, unsupported))
    return parse_judge_output(raw)


async def judge_answer_async(
    client: Any,
    question: str,
    reference: Any,
    candidate: Any,
    unsupported: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = load_config(config)
    raw = await llm.chat_completion(
        client, merged, build_judge_messages(question, str(reference), candidate, unsupported)
    )
    return parse_judge_output(raw)
