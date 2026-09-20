"""Metrics for answer quality and evidence retrieval."""

from __future__ import annotations

import re
import json
from collections import Counter
from typing import Any, Iterable

LOCOMO_CATEGORY_NAMES = {
	1: "multi_hop",
	2: "temporal",
	3: "open_domain",
	4: "single_hop",
	5: "adversarial_skip",
}


def normalize_text(text: Any) -> str:
	if isinstance(text, dict):
		# 解析后的答案对象：只对 answer 字段评分，不把 reasoning 等元数据算进去。
		if "answer" in text:
			text = text["answer"]
		else:
			text = json.dumps(text, ensure_ascii=False, sort_keys=True)
	elif isinstance(text, (list, tuple)):
		text = json.dumps(text, ensure_ascii=False, sort_keys=True)
	elif text is None:
		text = ""
	else:
		text = str(text)
	return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)).strip()


def exact_match(reference: Any, candidate: Any) -> float:
	return float(normalize_text(reference) == normalize_text(candidate))


def token_f1(reference: Any, candidate: Any) -> float:
	reference_tokens = normalize_text(reference).split()
	candidate_tokens = normalize_text(candidate).split()
	if not reference_tokens or not candidate_tokens:
		return float(reference_tokens == candidate_tokens)
	overlap = sum((Counter(reference_tokens) & Counter(candidate_tokens)).values())
	precision = overlap / len(candidate_tokens)
	recall = overlap / len(reference_tokens)
	return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evidence_recall(expected_ids: Iterable[str], retrieved_ids: Iterable[str]) -> float:
	expected = set(expected_ids)
	return 1.0 if not expected else len(expected & set(retrieved_ids)) / len(expected)


def summarize(results: list[dict[str, Any]]) -> dict[str, float]:
	scored = [item for item in results if not item.get("score_skipped", False)]
	if not scored:
		return {"count": 0.0, "llm_judge_accuracy": 0.0, "exact_match": 0.0, "token_f1": 0.0, "evidence_recall": 0.0}
	return {
		"count": float(len(scored)),
		"llm_judge_accuracy": sum(item.get("judge", {}).get("label") == "CORRECT" for item in scored) / len(scored),
		"exact_match": sum(item.get("exact_match", 0.0) for item in scored) / len(scored),
		"token_f1": sum(item.get("token_f1", 0.0) for item in scored) / len(scored),
		"evidence_recall": sum(item.get("evidence_recall", 0.0) for item in scored) / len(scored),
	}


def summarize_by_category(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
	"""Summarize scored categories; category 5 is reported but excluded from scoring."""
	by_category: dict[str, dict[str, float]] = {}
	for category, name in LOCOMO_CATEGORY_NAMES.items():
		items = [item for item in results if item.get("category") == category]
		scored = [item for item in items if not item.get("score_skipped", False)]
		by_category[name] = {
			"category": float(category),
			"count": float(len(items)),
			"scored_count": float(len(scored)),
			"llm_judge_accuracy": (
				sum(item.get("judge", {}).get("label") == "CORRECT" for item in scored) / len(scored)
				if scored else 0.0
			),
			"exact_match": sum(item.get("exact_match", 0.0) for item in scored) / len(scored) if scored else 0.0,
			"token_f1": sum(item.get("token_f1", 0.0) for item in scored) / len(scored) if scored else 0.0,
			"evidence_recall": sum(item.get("evidence_recall", 0.0) for item in scored) / len(scored) if scored else 0.0,
		}
	return by_category
