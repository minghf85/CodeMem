#!/usr/bin/env python
"""只读体检：``data/api_store`` 里哪些 user / 会话**欠着 atom**（消息在、原子事实没抽出来）。

用途是两件事，都是"先看再动"：

1. **补漏前**：确认到底缺了什么 —— 是某几条 message 没抽成，还是某个会话的总结是空的。
2. **补漏后**：确认真的补齐了（`atom_state.json` 的 ``failed`` 清空、缺的 msg_id 没了）。

    python scripts/check_api_store.py                 # 全部 user
    python scripts/check_api_store.py --user <user_id>  # 只看一个
    python scripts/check_api_store.py --limit 5        # 每个 user 最多列 5 个欠账会话

**它只读，不写任何文件，也不会调模型** —— 所以服务在跑的时候也随时可以跑（产物都是原子整文件
替换，读到的一定是替换前或替换后的完整版本，不会是半个）。

判据故意跟补漏脚本（``codemem.api.store.UserStore.sessions_needing_repair``）一致：记账里
**显式写了** ``failed`` 的会话以记账为准；记账是**旧格式**（没有 ``failed`` 键，早先版本写的）
就按"覆盖名单 vs 语料"的差集判 —— 后者正是升级后第一次体检要看的那些。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codemem.add.memory import msg_session_number  # noqa: E402
from codemem.api.store import UserStore, iter_user_ids  # noqa: E402
from codemem.io import memory_id, read_jsonl  # noqa: E402

DEFAULT_STORE = PROJECT_ROOT / "data" / "api_store"


def inspect_user(store: UserStore) -> dict:
    """算出一个 user 的欠账清单（纯 CPU、只读）。

    判据**必须与补漏完全一致**（``UserStore.sessions_needing_repair``），否则会出现"体检说干净、
    补漏还在干活"这种最费解的状态。所以：

    - 记账是**新格式**（有 ``failed`` 键）→ 以它为准（它才是"抽过没抽成"的事实）；
    - 记账是**旧格式**（早先版本写的，没有该键）→ 拿 ``msgmem`` 与记账里的覆盖名单做差集。

    **不拿 ``atommem`` 反推覆盖**：一条 message 完全可以合法地产出 **0 条** atom（寒暄、纯提问），
    那种消息在 ``atommem`` 里当然找不到 —— 用 atommem 反推会把它们全报成"缺"，把体检变成噪音。

    三种理由对应三种处置：``记账说失败`` / ``旧记账（按差集推）`` / ``总结未完成`` → 补漏会自动重试；
    ``被端点拒绝`` / ``总结被端点拒绝`` → **不自动重试**（重发同一个 prompt 还是同样结果），
    换一个不拦内容的端点后用 ``--repair --retry-blocked`` 重来。
    """
    records = read_jsonl(store.msgmem_path)
    atoms = read_jsonl(store.atommem_path)
    summaries = {
        memory_id(record)
        for record in read_jsonl(store.summem_path)
        if str(record.get("memory") or "").strip()
    }
    state = store._load_atom_state()  # noqa: SLF001 - 与补漏用同一份判据

    by_session: dict[int, list[str]] = {}
    for record in records:
        index = msg_session_number(memory_id(record))
        if index is not None:
            by_session.setdefault(index, []).append(memory_id(record))

    missing_sessions: list[tuple[int, list[str], str]] = []
    for index in sorted(by_session):
        entry = state.get(str(index)) or {}
        ids = by_session[index]
        gap: list[str] = []
        reason = ""
        if "failed" in entry:                       # 新格式：记账说了算
            gap = [str(x) for x in (entry.get("failed") or [])]
            reason = "记账说失败" if gap else ""
            if not gap and entry.get("blocked"):
                gap = [str(x) for x in entry["blocked"]]
                reason = "被端点拒绝（不自动重试）"
        else:                                       # 旧格式（早先版本写的，没有 failed 键）
            covered = {str(x) for x in (entry.get("atoms") or [])}
            gap = [mid for mid in ids if mid not in covered]
            reason = "旧记账（按差集推）" if gap else ""
            if not gap and f"session_{index}_summary" not in summaries:
                reason = "旧记账：总结为空"          # 老版本失败时写的是空总结，记账里看不出来
        if not gap and entry.get("summary_failed"):
            reason = "总结未完成"
        elif not gap and entry.get("summary_blocked"):
            reason = "总结被端点拒绝（不自动重试）"
        if gap or reason:
            missing_sessions.append((index, gap, reason))

    return {
        "user_id": store.user_id,
        "dir": store.root,
        "messages": len(records),
        "atoms": len(atoms),
        "sessions": len(by_session),
        "summaries": len(summaries),
        "missing": missing_sessions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/check_api_store.py",
        description="只读体检：哪些 user / 会话欠着 atom（不写文件、不调模型）",
    )
    parser.add_argument("--store", default=str(DEFAULT_STORE), help="store 根目录")
    parser.add_argument("--user", default="", help="只看这一个 user_id")
    parser.add_argument("--limit", type=int, default=10, help="每个 user 最多列几个欠账会话")
    args = parser.parse_args()

    root = Path(args.store)
    if not root.exists():
        print(f"找不到 {root}", file=sys.stderr)
        return 1

    user_ids = [args.user] if args.user else iter_user_ids(root)
    if not user_ids:
        print(f"{root} 下没有 user", file=sys.stderr)
        return 1

    totals = Counter()
    dirty_users: list[dict] = []
    for user_id in user_ids:
        store = UserStore(user_id, root)
        if not store.exists():
            if args.user:
                print(f"{user_id}: 没有落盘语料（{store.root}）", file=sys.stderr)
            continue
        info = inspect_user(store)
        totals["users"] += 1
        totals["messages"] += info["messages"]
        totals["atoms"] += info["atoms"]
        if not info["missing"]:
            continue
        dirty_users.append(info)
        totals["dirty_users"] += 1
        totals["sessions"] += len(info["missing"])
        totals["messages_missing"] += sum(len(gap) for _, gap, _ in info["missing"])

    for info in dirty_users:
        print(f"\n{info['user_id']}")
        print(f"  {info['dir']}")
        print(f"  {info['messages']} 条 message / {info['atoms']} 条 atom / "
              f"{info['sessions']} 个会话 / {info['summaries']} 条总结")
        for index, gap, reason in info["missing"][:max(0, args.limit)]:
            detail = f"缺 {len(gap)} 条: {', '.join(gap[:6])}" if gap else "总结为空"
            print(f"    session_{index:<5} {reason:<16} {detail}"
                  + ("  …" if len(gap) > 6 else ""))
        if len(info["missing"]) > args.limit:
            print(f"    …（还有 {len(info['missing']) - args.limit} 个欠账会话）")

    print("\n" + "=" * 60)
    if not dirty_users:
        print(f"干净：{totals['users']} 个 user / {totals['messages']} 条 message / "
              f"{totals['atoms']} 条 atom，没有欠账")
        return 0
    print(f"{totals['dirty_users']}/{totals['users']} 个 user 有欠账："
          f"{totals['sessions']} 个会话 / {totals['messages_missing']} 条 message 缺 atom")
    print("补漏方式（**服务没在跑时**）：")
    print("    python -m codemem.api --repair                    # 重试 failed 的那些")
    print("    python -m codemem.api --repair --retry-blocked    # 连被端点拒绝的也重来"
          "（换了不拦内容的端点之后）")
    print("或者直接重启服务，启动后台任务会自动补（见 configs/api.yaml 的 repair_on_startup）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
