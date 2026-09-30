"""把新 judge 规则应用到已存的结果上，看哪些判分会翻转 —— 换 judge 前先量一下影响面。

用法：python scripts/judge_impact.py <run_dir>
      （对比同一份 answers 在"旧判据"与"新判据"下的得分）

现实中没法重跑旧 judge，所以这里做一件更实用的事：**扫描已有结果，挑出"按新规则应该翻转"
的条目**，让人工核对规则改动是否朝正确方向走。判定用一个纯规则近似（不是模型），
只覆盖三条明确的规则：空答案必错、集合型超集算对、日期等价。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codemem.eval import judge as J  # noqa: E402

STOP = {"the", "and", "with", "for", "from", "her", "his", "their", "was", "were", "has",
        "have", "that", "this", "during", "into", "about", "like", "some", "not", "she",
        "they", "its", "been", "also", "does", "did"}


def tokens(text) -> set[str]:
    return {t for t in re.findall(r"[a-z]{3,}", str(text or "").lower()) if t not in STOP}


def is_empty(candidate) -> bool:
    c = str(candidate or "").strip().lower()
    return c in ("", "none", "null") or c.startswith("null (")


def is_set_question(reference) -> bool:
    r = str(reference or "")
    return "," in r and len(r) < 200


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    if path.is_dir():
        path = path / "eval_eval.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    flips: list[tuple] = []
    for row in rows:
        label = row.get("judge_label")
        cand, ref = row.get("candidate"), row.get("reference")
        note = None
        # 规则 1：空答案必错（旧 judge 会放水）
        if label == "CORRECT" and is_empty(cand):
            note = "空答案被旧 judge 判对 -> 新规则应为 INCORRECT"
        # 规则 2：集合型超集算对（新 judge 一度从严）
        elif label != "CORRECT" and is_set_question(ref):
            rt, ct = tokens(ref), tokens(cand)
            if rt and rt <= ct:
                note = "预测是参考的超集却被判错 -> 新规则应为 CORRECT"
        # 规则 3：日期等价
        elif label != "CORRECT":
            expected = J.resolve_relative_date(ref)
            actual = J._parse_date(cand)
            if expected and actual and expected == actual:
                note = "日期与参考折算后相同却被判错 -> 新规则应为 CORRECT"
        if note:
            flips.append((row.get("qa_index"), row.get("category"), row.get("question"), note))

    print(f"{path}")
    print(f"  共 {len(rows)} 条，按新规则应翻转 {len(flips)} 条：")
    for idx, cat, question, note in flips:
        print(f"    [{idx}] cat{cat} {str(question)[:52]}")
        print(f"          {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
