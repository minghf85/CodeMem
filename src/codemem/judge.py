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


def build_judge_messages(question: str, reference: str, candidate: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a strict answer evaluator. Follow the instructions carefully."},
        {"role": "user", "content": JUDGE_PROMPT.format(question=question, reference=reference, prediction=candidate)},
    ]


def parse_judge_output(text: str) -> dict[str, Any]:
    """Parse judge output with improved extraction for <judgement> and <reason> tags."""
    # Try to extract judgement and reason from XML-style tags
    judgement_match = re.search(r'<judgement>\s*(CORRECT|INCORRECT)\s*</judgement>', text, re.IGNORECASE)
    reason_match = re.search(r'<reason>\s*(.+?)\s*</reason>', text, re.DOTALL | re.IGNORECASE)

    if judgement_match:
        label = judgement_match.group(1).upper()
        reason = reason_match.group(1).strip() if reason_match else ""
        return {"label": label, "reason": reason}

    # Fallback: try to find CORRECT or INCORRECT in the text
    text_upper = text.upper()
    if "INCORRECT" in text_upper:
        return {"label": "INCORRECT", "reason": text.strip()}
    elif "CORRECT" in text_upper:
        return {"label": "CORRECT", "reason": text.strip()}

    # Last resort: try JSON parsing (for backward compatibility)
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict) and "label" in parsed:
            label = parsed["label"].upper()
            if label in {"CORRECT", "INCORRECT", "WRONG"}:
                # Normalize WRONG to INCORRECT
                if label == "WRONG":
                    label = "INCORRECT"
                return {"label": label, "reason": parsed.get("reason", "")}
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not parse judge output: {text[:200]}")


def judge_answer(question: str, reference: str, candidate: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_LLM_CONFIG)
    merged.update(load_yaml_config(CONFIG_FILE))
    if config:
        merged.update(config)
    raw = complete_sync(build_judge_messages(question, reference, candidate), merged)
    return parse_judge_output(raw)


async def judge_answer_async(
    client: Any,
    question: str,
    reference: Any,
    candidate: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(DEFAULT_LLM_CONFIG)
    merged.update(load_yaml_config(CONFIG_FILE))
    if config:
        merged.update(config)
    raw = await complete_with_client(
        client,
        build_judge_messages(question, str(reference), candidate),
        merged,
    )
    return parse_judge_output(raw)
