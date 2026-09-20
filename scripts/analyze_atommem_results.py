"""Analyze atommem vs msgmem retrieval comparison results."""

import json
import sys
from pathlib import Path
from collections import defaultdict

def load_results(jsonl_path: Path) -> list[dict]:
    results = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def analyze_improvements(results: list[dict]) -> dict:
    """Analyze where atommem improves over msgmem."""
    improvements = {
        "recall_improved": 0,
        "recall_same": 0,
        "recall_worse": 0,
        "answer_improved": 0,
        "answer_same": 0,
        "answer_worse": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "only_atom_correct": 0,
        "only_msg_correct": 0,
    }

    examples = defaultdict(list)

    for r in results:
        if r.get("category") == 5:  # Skip adversarial
            continue

        # Recall comparison
        msg_recall = r.get("msgmem_evidence_recall", 0)
        atom_recall = r.get("atommem_evidence_recall", 0)

        if atom_recall > msg_recall:
            improvements["recall_improved"] += 1
        elif atom_recall == msg_recall:
            improvements["recall_same"] += 1
        else:
            improvements["recall_worse"] += 1

        # Answer comparison
        msg_correct = r.get("msgmem_judge", {}).get("label") == "CORRECT"
        atom_correct = r.get("atommem_judge", {}).get("label") == "CORRECT"

        if atom_correct and msg_correct:
            improvements["both_correct"] += 1
        elif not atom_correct and not msg_correct:
            improvements["both_wrong"] += 1
        elif atom_correct and not msg_correct:
            improvements["only_atom_correct"] += 1
            improvements["answer_improved"] += 1
            examples["atom_wins"].append(r)
        elif msg_correct and not atom_correct:
            improvements["only_msg_correct"] += 1
            improvements["answer_worse"] += 1
            examples["msg_wins"].append(r)

        if atom_correct == msg_correct:
            improvements["answer_same"] += 1

    return improvements, examples

def print_examples(examples: dict, max_examples: int = 3):
    """Print interesting examples."""
    print("\n" + "="*80)
    print("EXAMPLES WHERE ATOMMEM WINS")
    print("="*80)

    for i, r in enumerate(examples.get("atom_wins", [])[:max_examples], 1):
        print(f"\n--- Example {i} ---")
        print(f"Question: {r['question']}")
        print(f"Reference: {r['reference']}")
        print(f"Category: {r.get('category')}")
        print(f"\nEvidence Recall:")
        print(f"  msgmem:  {r.get('msgmem_evidence_recall', 0):.2f}")
        print(f"  atommem: {r.get('atommem_evidence_recall', 0):.2f}")
        print(f"\nPredictions:")
        print(f"  msgmem:  {r.get('msgmem_prediction', '')[:100]}")
        print(f"  atommem: {r.get('atommem_prediction', '')[:100]}")
        print(f"\nJudge Reasons:")
        print(f"  msgmem:  {r.get('msgmem_judge', {}).get('reason', '')[:150]}")
        print(f"  atommem: {r.get('atommem_judge', {}).get('reason', '')[:150]}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_atommem_results.py <eval_atommem.jsonl>")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)

    results = load_results(results_path)
    improvements, examples = analyze_improvements(results)

    total = len([r for r in results if r.get("category") != 5])

    print("="*80)
    print(f"ATOMMEM VS MSGMEM COMPARISON ({total} samples)")
    print("="*80)

    print("\n📊 RECALL COMPARISON:")
    print(f"  Improved:  {improvements['recall_improved']:3d} ({improvements['recall_improved']/total*100:5.1f}%)")
    print(f"  Same:      {improvements['recall_same']:3d} ({improvements['recall_same']/total*100:5.1f}%)")
    print(f"  Worse:     {improvements['recall_worse']:3d} ({improvements['recall_worse']/total*100:5.1f}%)")

    print("\n✅ ANSWER ACCURACY COMPARISON:")
    print(f"  Both correct:        {improvements['both_correct']:3d} ({improvements['both_correct']/total*100:5.1f}%)")
    print(f"  Only atom correct:   {improvements['only_atom_correct']:3d} ({improvements['only_atom_correct']/total*100:5.1f}%)")
    print(f"  Only msgmem correct: {improvements['only_msg_correct']:3d} ({improvements['only_msg_correct']/total*100:5.1f}%)")
    print(f"  Both wrong:          {improvements['both_wrong']:3d} ({improvements['both_wrong']/total*100:5.1f}%)")

    net_improvement = improvements['only_atom_correct'] - improvements['only_msg_correct']
    print(f"\n🎯 NET IMPROVEMENT: {net_improvement:+d} questions ({net_improvement/total*100:+.1f}%)")

    print_examples(examples)

if __name__ == "__main__":
    main()
