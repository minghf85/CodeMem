"""把已有的 ``data/api_store/{user}/`` 语料**按新适配层重跑一遍**。

为什么需要它：``msgmem`` 是**已经落盘**的产物。修好适配层（说话人从 ``user`` 变成真名、
摘掉 ``[beam_100k][session_1][D1:21][text]`` 这类记账标签）之后，**存量语料不会自己变好** ——
它们是用旧代码写的，``role`` 还是 ``user``、正文还带着标签。这个脚本拿 ``requests.jsonl``
（每次 ``/add`` 的**原始报文留痕**）当输入，重新走一遍 ``store`` 的写入路径。

两个阶段，可以分开跑：

::

    # ① 只重建 msgmem（纯 CPU、零模型成本、秒级）：先把"谁能被 grep 到"修对
    python scripts/rebuild_api_store.py --all --stage msgmem

    # ② 再重建 summem / atommem（要调模型，逐会话 + 逐消息）：把消解结果也修对
    python scripts/rebuild_api_store.py --all --stage memories

    # 先看要动什么、不动手
    python scripts/rebuild_api_store.py --all --dry-run

    # 只挑几个 user
    python scripts/rebuild_api_store.py --users u_0061ec94..._3803b510,u_00eada3e..._f54e556c

**为什么分两阶段**：``--stage msgmem`` 之后 agent 至少能 grep 到真名（这是最要命的那处），
而**不花一分钱模型成本**；``summem``/``atommem`` 才是贵的，可以等确认第一批没问题再跑。

**幂等与安全**：``--stage msgmem`` 只重写 ``msgmem.jsonl``，``meta.json`` 的 session 账本按
重放出来的结果重算（``applied_request_ids`` 一并重建，于是重放不会重复写）；原文件先备份成
``*.bak-<时间戳>``。**先 dry-run 看一眼条数对不对再真跑。**
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codemem import adapters, llm  # noqa: E402
from codemem.add.memory import write_memories  # noqa: E402
from codemem.api import store as store_mod  # noqa: E402
from codemem.api.models import normalize_content  # noqa: E402
from codemem.api.store import UserStore, _iso_from_ms, slug  # noqa: E402
from codemem.io import make_memory, now_iso, read_jsonl  # noqa: E402
from codemem.log import Logger  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = PROJECT_ROOT / "data" / "api_store"


def load_adds(path: Path) -> list[dict[str, Any]]:
    """按原始顺序取出该 user 的全部 ``/add`` 报文（跳掉损坏行）。"""
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("endpoint") == "add" and record.get("messages"):
                out.append(record)
    return out


def rebuild_msgmem(adds: list[dict[str, Any]]) -> tuple[list[dict], dict[str, Any], list[str]]:
    """把 /add 序列重放成 ``(msg 记录, 新 meta, 形态日志)``（纯 CPU）。

    会话号分配沿用 ``store`` 的规则：**按 session_id 首次出现的顺序**编号，同一 session 的
    多次 add 依次续号。于是重放出来的 id 与当初一致（``session_1_1`` 还是 ``session_1_1``），
    下游的 evidence source、summem 的 ``session_N_summary`` 全都不用改。
    """
    records: list[dict] = []
    sessions: dict[str, Any] = {}
    next_index = 1
    applied: list[str] = []
    notes: list[str] = []

    for add in adds:
        request_id = str(add.get("request_id") or "")
        session_id = str(add.get("session_id") or "")
        if request_id and request_id in applied:
            continue                      # 幂等：同一 request 只算一次

        entry = sessions.get(session_id)
        if not isinstance(entry, dict):
            entry = {"index": next_index, "count": 0}
            sessions[session_id] = entry
            next_index += 1
        session_index = int(entry["index"])
        start = int(entry["count"])

        messages = [
            {**message, "content": normalize_content(message.get("content"))}
            for message in add.get("messages") or []
        ]
        batch = adapters.adapt_messages(messages)
        notes.append(
            f"session_{session_index} x{len(batch.messages)} {batch.family}"
            + (f" speakers={'/'.join(batch.speakers)}" if batch.speakers else "")
        )
        for offset, message in enumerate(batch.messages, start=1):
            records.append(make_memory(
                memory=message.content,
                memory_type="msg",
                mem_id=f"session_{session_index}_{start + offset}",
                time=_iso_from_ms(message.timestamp),
                role=message.role,
                tag=[],
                source=[],
                changelog=[{"time": now_iso(), "content": "created"}],
            ))
        entry["count"] = start + len(batch.messages)
        if request_id:
            applied.append(request_id)

    meta = {
        "sessions": sessions,
        "next_index": next_index,
        "applied_request_ids": applied,
        "pending": {},
    }
    return records, meta, notes


def backup(path: Path, stamp: str) -> str:
    """把要覆盖的文件备份成 ``<name>.bak-<stamp>``（不存在就跳过）。"""
    if not path.exists():
        return ""
    target = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copy2(path, target)
    return target.name


async def rebuild_user(
    user_id: str,
    root: Path,
    *,
    stage: str,
    config: dict[str, Any],
    client: Any,
    log: Logger,
    dry_run: bool,
    allow_loss: bool = False,
) -> dict[str, Any]:
    """重建一个 user 的语料。``stage`` = ``msgmem``（纯 CPU）或 ``memories``（含 LLM 刷新）。"""
    store = UserStore(user_id, root)
    adds = load_adds(store.requests_path)
    if not adds:
        return {"user": user_id, "status": "SKIPPED", "reason": "requests.jsonl 里没有 add"}

    records, meta, notes = rebuild_msgmem(adds)
    before = len(read_jsonl(store.msgmem_path))
    result: dict[str, Any] = {
        "user": user_id, "status": "DRY" if dry_run else "OK",
        "adds": len(adds), "msgs_before": before, "msgs_after": len(records),
        "sessions": len(meta["sessions"]), "notes": notes,
    }

    # **重放的结果比现有语料少 = 会丢数据**，一律拒写。实测真有这种 user（某次 ``/add`` 的
    # ``requests.jsonl`` 留痕丢了 —— 留痕是尽力而为的，写失败不阻断请求），于是重放少了一整批。
    # 消息数只减不增是这一层唯一不该发生的事，所以它不是"警告一下继续"，而是**停下来**。
    if records and before > len(records) and not allow_loss:
        result["status"] = "REFUSED"
        result["reason"] = (
            f"重放得到 {len(records)} 条，现有 {before} 条 —— 少 {before - len(records)} 条，"
            "疑似 requests.jsonl 留痕不全；加 --allow-loss 才会覆盖（会丢那些消息）"
        )
        return result
    if dry_run:
        return result

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result["backup"] = backup(store.msgmem_path, stamp)
    store.root.mkdir(parents=True, exist_ok=True)
    write_memories(store.msgmem_path, records)
    store_mod._atomic_write_json(store.meta_path, {**store.meta(), **meta})
    # 重放改变了语料，"哪些 message 已经抽过"的记账（atom_state.json）跟着作废 ——
    # 留着它，下一次增量刷新就会跳过这些 message，重建出来的 atom 反而是旧的。
    store.atom_state_path.unlink(missing_ok=True)

    if stage == "memories":
        # 逐会话重刷 summem / atommem（复用 add 的写入路径）。
        # ``force=True``：跳过 ``atom_state.json`` 的增量记账，**整会话重抽** —— 重建的
        # 全部意义就是"换一套规则重算已抽过的消息"，增量在这里正好是反的。
        refreshed = 0
        for session_index in sorted(meta["sessions"].values(), key=lambda e: int(e["index"])):
            await store._refresh_memories(          # noqa: SLF001 - 就是它的写入路径
                session_index=int(session_index["index"]), config=config,
                client=client, log=log, force=True,
            )
            refreshed += 1
        result["sessions_refreshed"] = refreshed
    return result


async def async_main(args: argparse.Namespace) -> None:
    root = Path(args.store)
    if not root.exists():
        print(f"找不到 {root}", file=sys.stderr)
        return

    if args.users:
        targets = list(args.users)
    else:
        targets = store_mod.iter_user_ids(root)
        if args.only_with_adds:
            targets = [
                user_id for user_id in targets
                if load_adds(root / slug(user_id) / "requests.jsonl")
            ]

    if not targets:
        print("没有匹配的 user", file=sys.stderr)
        return

    config: dict[str, Any] = {}
    client = None
    if args.stage == "memories" and not args.dry_run:
        print("阶段 memories 需要模型端点（读 configs/api.yaml）；先确认端点已在跑。")
        from codemem.api.server import load_config
        config = load_config()
        client = llm.make_client(config.get("index") or {})
    log = Logger(level=args.log_level)

    try:
        for user_id in targets:
            try:
                result = await rebuild_user(
                    user_id, root, stage=args.stage, config=config, client=client,
                    log=log, dry_run=args.dry_run, allow_loss=args.allow_loss,
                )
            except Exception as exc:  # noqa: BLE001 - 单个 user 失败不影响其它
                print(f"  [FAIL] {user_id[:32]}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            if result["status"] == "SKIPPED":
                print(f"  [skip] {user_id[:32]}: {result['reason']}")
                continue
            if result["status"] == "REFUSED":
                print(f"  [拒绝] {user_id[:32]}: {result['reason']}", file=sys.stderr)
                continue
            flag = "[dry] " if args.dry_run else "     "
            print(
                f"  {flag}{user_id[:32]}  adds={result['adds']:3d}"
                f"  会话={result['sessions']:2d}"
                f"  msg {result['msgs_before']} -> {result['msgs_after']}"
            )
            if args.verbose:
                for note in result["notes"]:
                    print(f"          {note}")
    finally:
        if client is not None:
            await client.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按新适配层重跑 api_store 存量语料")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="全部 user")
    group.add_argument("--users", help="逗号分隔的 user_id 列表")
    parser.add_argument("--store", default=str(DEFAULT_STORE), help="api_store 根目录")
    parser.add_argument("--stage", choices=("msgmem", "memories"), default="msgmem",
                        help="msgmem=只重建原始消息层（纯 CPU）；memories=连 summem/atommem 一起刷")
    parser.add_argument("--dry-run", action="store_true", help="只报会写多少条，不动文件")
    parser.add_argument("--allow-loss", action="store_true",
                        help="即使重放条数少于现有条数也覆盖（会丢消息；默认拒写）")
    parser.add_argument("--only-with-adds", action="store_true", help="跳过没有 add 的 user")
    parser.add_argument("--verbose", action="store_true", help="每条 add 的形态与说话人")
    parser.add_argument("--log-level", default="warn")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.users:
        args.users = [part.strip() for part in args.users.split(",") if part.strip()]
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
