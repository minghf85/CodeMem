"""LLM-as-a-judge utilities for Locomo answer equivalence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .eval_utils import DEFAULT_LLM_CONFIG, complete_sync, complete_with_client, load_yaml_config
from .prompts import JUDGE_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "configs" / "judge.yaml"


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
            "content": JUDGE_PROMPT.format(
                question=question, reference=reference, prediction=render_prediction(candidate, unsupported)
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
    merged = dict(DEFAULT_LLM_CONFIG)
    merged.update(load_yaml_config(CONFIG_FILE))
    if config:
        merged.update(config)
    raw = complete_sync(build_judge_messages(question, reference, candidate, unsupported), merged)
    return parse_judge_output(raw)


async def judge_answer_async(
    client: Any,
    question: str,
    reference: Any,
    candidate: Any,
    unsupported: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(DEFAULT_LLM_CONFIG)
    merged.update(load_yaml_config(CONFIG_FILE))
    if config:
        merged.update(config)
    raw = await complete_with_client(
        client,
        build_judge_messages(question, str(reference), candidate, unsupported),
        merged,
    )
    return parse_judge_output(raw)
