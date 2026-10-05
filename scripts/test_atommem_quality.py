#!/usr/bin/env python3
"""测试 atommem 产出质量：检查是否有违规的时间词、多句、空 memory 等。

用法：
    python scripts/test_atommem_quality.py data/Caroline_Melanie/atommem.jsonl
    python scripts/test_atommem_quality.py data/*/atommem.jsonl
"""

import json
import re
import sys
from pathlib import Path


def check_time_words(memory: str) -> list[str]:
    """检查 memory 是否含有禁止的时间词。"""
    forbidden = [
        r'\byesterday\b', r'\btoday\b', r'\btomorrow\b',
        r'\bago\b', r'\blast\b', r'\bnext\b', r'\bpast\b',
        r'\brecent\b', r'\brecently\b', r'\bsoon\b',
        r'\bfuture\b', r'\bnow\b', r'\bcurrently\b', r'\bpresently\b',
        r'\bbefore\b', r'\bafter\b', r'\bsince\b', r'\buntil\b', r'\bduring\b',
        r'\d+\s+(day|week|month|year)s?(\s+ago)?',
    ]
    found = []
    for pattern in forbidden:
        matches = re.finditer(pattern, memory, re.IGNORECASE)
        for match in matches:
            found.append(match.group(0))
    return found


def check_multi_sentence(memory: str) -> bool:
    """检查是否是多句（超过一个句号/问号/感叹号）。"""
    # 简单检查：数句号/问号/感叹号
    sentence_ends = re.findall(r'[.!?]', memory)
    return len(sentence_ends) > 1


def check_atom_quality(atom: dict) -> dict:
    """检查一条 atom 的质量，返回问题列表。"""
    issues = []
    memory = atom.get("memory", "")

    if not memory or not memory.strip():
        issues.append("empty_memory")

    # 检查时间词
    time_words = check_time_words(memory)
    if time_words:
        issues.append(f"time_words: {', '.join(set(time_words))}")

    # 检查多句
    if check_multi_sentence(memory):
        issues.append("multi_sentence")

    # 检查 metadata
    meta = atom.get("metadata", {})
    if not meta.get("id"):
        issues.append("no_id")
    if not meta.get("source"):
        issues.append("no_source")
    if meta.get("type") != "atom":
        issues.append(f"wrong_type: {meta.get('type')}")

    return {
        "id": meta.get("id", "unknown"),
        "memory": memory,
        "issues": issues,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_atommem_quality.py <atommem.jsonl> [...]")
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        paths.extend(Path(".").glob(arg))

    if not paths:
        print("No files found")
        sys.exit(1)

    total_atoms = 0
    total_issues = 0
    issue_counts = {}

    for path in paths:
        print(f"\n=== {path} ===")
        atoms = []

        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    atoms.append(json.loads(line))
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

        if not atoms:
            print("  (empty file)")
            continue

        total_atoms += len(atoms)
        file_issues = 0

        for atom in atoms:
            result = check_atom_quality(atom)
            if result["issues"]:
                file_issues += 1
                total_issues += 1
                for issue in result["issues"]:
                    issue_type = issue.split(":")[0]
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

                # 打印前 5 个问题
                if file_issues <= 5:
                    print(f"  [{result['id']}] {result['memory'][:60]}...")
                    print(f"    Issues: {', '.join(result['issues'])}")

        if file_issues == 0:
            print(f"  ✓ All {len(atoms)} atoms passed")
        else:
            print(f"  ✗ {file_issues}/{len(atoms)} atoms have issues")

    # 总结
    print("\n" + "=" * 60)
    print(f"Total atoms: {total_atoms}")
    print(f"Total with issues: {total_issues} ({100*total_issues/total_atoms:.1f}%)")

    if issue_counts:
        print("\nIssue breakdown:")
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {issue_type}: {count}")
    else:
        print("\n✓ No issues found!")


if __name__ == "__main__":
    main()
