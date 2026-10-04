"""按 ``user_id`` 隔离的持久化存储：追加消息、request_id 去重、生成三层记忆。

**一个 user 一个目录**，结构对齐离线管线（add 的三层记忆文件）：

    data/api_store/{slug(user_id)}/
      meta.json                 session_id -> 会话号、每会话消息数、已应用的 request_id
      msgmem.jsonl              原始消息（type=msg；会话时间戳是锚点）
      summem.jsonl              每个会话一份总结（type=summary）
      atommem.jsonl             逐 message 抽出的原子事实（type=atom）

**为什么沿用 ``session_1_3`` 这套 id 约定**：search agent 的 system prompt、evidence 的
``source`` 校验、`date -d` 时间锚点示例全部建立在这套约定上。存储层照抄它，整条 search
管线就**一行都不用改**。

**幂等**是协议硬要求：Add 重试时 ``request_id`` 不变，同一条不能写入两次。所以每次 Add 先查
``meta.applied_request_ids`` —— 命中就直接返回成功，连语料都不碰。

**写锁**：同一 user 的并发 Add 会交错读改写同一份语料，故按 user_id 分桶一把 ``asyncio.Lock``。
不同 user 互不影响（各自的目录）。
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..add import atommem as atommem_mod
from ..add import summem as summem_mod
from ..add.memory import (
    ATOMMEM_NAME,
    MSGMEM_NAME,
    SUMMEM_NAME,
    group_by_session,
    write_memories,
)
from ..io import DATA_DIR, make_memory, now_iso, read_jsonl
from ..search import toolcfg
from .models import normalize_content

META_NAME = "meta.json"

_SLUG_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

#: 同一 user 的写锁。进程内有效（API 是单进程多协程）；不同 user 各一把。
_user_locks: dict[str, "asyncio.Lock"] = {}


def _lock_for(user_id: str) -> "asyncio.Lock":
    import asyncio

    lock = _user_locks.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        _user_locks[user_id] = lock
    return lock


def slug(user_id: str) -> str:
    """把 ``user_id`` 变成一个安全的目录名。

    ``user_id`` 里常见 ``:`` 等路径不安全字符（如 ``eval:<run_id>:locomo:conv-0``），一律替换成
    ``_``；再拼一段短 hash，避免不同的 user_id 归一化后撞同一个目录（例如 ``a:b`` 与 ``a/b``）。
    """
    cleaned = _SLUG_SAFE.sub("_", user_id).strip("._-") or "user"
    digest = hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:80]}_{digest}"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _iso_from_ms(milliseconds: Any) -> str:
    """Unix **毫秒** → ISO-8601 UTC（``2023-05-08T13:56:00Z``）。缺失/非法返回空串。

    ISO 是 ``date -d`` 能无歧义解析的形式，正好对上 agent 的时间算术。
    """
    if milliseconds is None:
        return ""
    try:
        moment = datetime.fromtimestamp(float(milliseconds) / 1000.0, tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return ""
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Any) -> None:
    """原子写一个 JSON 对象（meta.json 用）。走与 jsonl 相同的临时文件 + os.replace。"""
    from ..io import write_jsonl

    write_jsonl(path, [payload])


def _read_meta(path: Path, user_id: str) -> dict[str, Any]:
    """读 meta.json；缺失/损坏时给出干净的空结构（不抛 —— 一个新 user 本就该是空的）。"""
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            raw = None
        if isinstance(raw, dict):
            raw.setdefault("user_id", user_id)
            raw.setdefault("sessions", {})
            raw.setdefault("next_index", 1)
            raw.setdefault("applied_request_ids", [])
            return raw
    return {"user_id": user_id, "sessions": {}, "next_index": 1, "applied_request_ids": []}


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

class UserStore:
    """一个 ``user_id`` 的存储句柄。无状态（每次从磁盘读），并发由 ``_lock_for`` 串行化写入。"""

    def __init__(self, user_id: str, root: Path | None = None) -> None:
        self.user_id = user_id
        self.root = (root or (DATA_DIR / "api_store")) / slug(user_id)

    # -- 路径 ---------------------------------------------------------------

    @property
    def meta_path(self) -> Path:
        return self.root / META_NAME

    @property
    def msgmem_path(self) -> Path:
        return self.root / MSGMEM_NAME

    @property
    def summem_path(self) -> Path:
        return self.root / SUMMEM_NAME

    @property
    def atommem_path(self) -> Path:
        return self.root / ATOMMEM_NAME

    def exists(self) -> bool:
        """该 user 是否已经有落盘语料（Search 据此决定是"没记忆"还是"去检索"）。"""
        return self.msgmem_path.exists() and self.msgmem_path.stat().st_size > 0

    # -- 读 -----------------------------------------------------------------

    def meta(self) -> dict[str, Any]:
        return _read_meta(self.meta_path, self.user_id)

    def sessions(self) -> list[dict[str, Any]]:
        """全部 msg 记忆（按会话号、消息序号排序）。"""
        return read_jsonl(self.msgmem_path)

    def session_records(self, session_index: int) -> list[dict[str, Any]]:
        """某个会话的全部 msg 记忆。"""
        return [
            r for r in self.sessions()
            if atommem_mod.msg_session_number((r.get("metadata") or {}).get("id") or "")
            == session_index
        ]

    # -- 写 -----------------------------------------------------------------

    async def append(
        self,
        *,
        request_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        config: dict[str, Any],
        client: Any,
        log: Any | None = None,
    ) -> dict[str, Any]:
        """把一个 Add 请求写成三层记忆，返回小结 dict。

        ``messages`` 是已归一化的 ``[{"role", "content", "timestamp"}]``。步骤：

        1. 取该 user 的写锁；
        2. **幂等**：``request_id`` 已应用 → 直接返回（不碰语料）；
        3. 分配 ``id`` / 转时间，把新消息**以原始措辞**追加到 ``msgmem.jsonl``；
        4. **在返回前**重算该会话的 ``summem``（一次 LLM 总结）与该会话的 ``atommem``
           （逐 message 的 add agent，只处理本会话）—— 兑现"持久化后立即可检索"；
        5. 原子写回语料 + meta。

        只重算**本会话**：其它会话的记忆原样保留（增量），所以重复 Add 不会重跑其它会话的抽取。
        """
        lock = _lock_for(self.user_id)
        async with lock:
            self.root.mkdir(parents=True, exist_ok=True)
            meta = self.meta()

            if request_id in meta["applied_request_ids"]:
                if log is not None:
                    log.info(f"Add 幂等命中：request_id={request_id!r} 已应用，跳过写入")
                return {"written": 0, "duplicate": True, "session_index": None}

            # ---- 该 session 在本地语料里的会话号 ----
            sessions_map: dict[str, Any] = meta["sessions"]
            entry = sessions_map.get(session_id)
            if not isinstance(entry, dict):
                entry = {"index": int(meta.get("next_index", 1)), "count": 0}
                sessions_map[session_id] = entry
                meta["next_index"] = int(entry["index"]) + 1
            session_index = int(entry["index"])

            # ---- 追加消息到 msgmem.jsonl（序号接在已有条数之后）----
            records = read_jsonl(self.msgmem_path)
            start = int(entry.get("count", 0))
            for offset, message in enumerate(messages, start=1):
                content = normalize_content(message.get("content"))
                records.append(make_memory(
                    memory=content,
                    memory_type="msg",
                    mem_id=f"session_{session_index}_{start + offset}",
                    time=_iso_from_ms(message.get("timestamp")),
                    role=str(message.get("role") or ""),
                    tag=[],
                    source=[],
                    changelog=[{"time": now_iso(), "content": "created"}],
                ))
            entry["count"] = start + len(messages)
            write_memories(self.msgmem_path, records)

            # ---- 重算本会话的 summem 与 atommem ----
            refreshed = await self._refresh_memories(
                session_index=session_index, config=config, client=client, log=log,
            )

            # ---- meta：记下这个 request_id（幂等键）----
            meta["applied_request_ids"] = [*meta["applied_request_ids"], request_id]
            _atomic_write_json(self.meta_path, meta)

            if log is not None:
                log.info(
                    f"Add 写入 {len(messages)} 条 -> session {session_index}"
                    f"（共 {entry['count']} 条）记忆刷新={'成功' if refreshed else '未变'}"
                )
            return {"written": len(messages), "duplicate": False, "session_index": session_index}

    async def _refresh_memories(
        self,
        *,
        session_index: int,
        config: dict[str, Any],
        client: Any,
        log: Any | None,
    ) -> bool:
        """重算该会话的 summem（一次 LLM）与 atommem（逐 message agent）。失败不抛。

        见 offline 的 ``add/summem.py`` 与 ``add/atommem.py``。
        """
        speaker_a, speaker_b = "user", "assistant"
        try:
            # --- summem：替换该会话那一条 ---
            sess_records = self.session_records(session_index)
            if sess_records:
                summary = await summem_mod.summarize_session(
                    session_index=session_index,
                    session_time=str((sess_records[0].get("metadata") or {}).get("time") or ""),
                    records=sess_records,
                    client=client,
                    config=config,
                    speaker_a=speaker_a,
                    speaker_b=speaker_b,
                )
                existing = read_jsonl(self.summem_path)
                summary_id = f"session_{session_index}_summary"
                existing = [
                    r for r in existing if (r.get("metadata") or {}).get("id") != summary_id
                ]
                write_memories(self.summem_path, existing + [summary])
            # --- atommem：只跑本会话的 message（agent）---
            await self._refresh_atoms(
                session_index=session_index, config=config, client=client, log=log,
            )
        except Exception as exc:  # noqa: BLE001 - 记忆刷新失败不该让 Add 失败
            if log is not None:
                log.warn(f"session {session_index} 记忆刷新失败：{type(exc).__name__}: {exc}")
            return False
        return True

    async def _refresh_atoms(
        self, *, session_index: int, config: dict[str, Any], client: Any, log: Any | None,
    ) -> None:
        """对本会话的每条 message 跑 add agent，重建本会话的 atom（就地写 atommem.jsonl）。"""
        from .. import llm
        from ..search import agent as agent_mod  # noqa: F401  (供 monkeypatch 的完整性)
        from ..search.runner import ModelRunner

        # Add 侧**必须**用 add_tool.json（search 的 ``tool_config`` 是给检索 agent 的，别混用）。
        catalog = toolcfg.load_catalog(
            config.get("add_tool_config") or "configs/add_tool.json"
        )
        catalog.validate(["read", "edit", "bash"])

        msg_records = self.session_records(session_index)
        summaries_by_session = group_by_session(read_jsonl(self.summem_path))
        summaries = {n: rs[0] for n, rs in summaries_by_session.items() if rs}

        atom_cfg = dict(config)
        # api.yaml 用 ``add_max_steps`` 表达 add agent 的步数；离线 add 用 ``max_steps``。
        if "add_max_steps" in atom_cfg and "max_steps" not in atom_cfg:
            atom_cfg["max_steps"] = atom_cfg["add_max_steps"]
        # 工具配置：Add 侧**必须**用 add_tool.json（与 search 的 tool.json 不同）。
        atom_cfg["tool_config"] = config.get("add_tool_config") or "configs/add_tool.json"
        # 模型参数：api.yaml 的 ``index`` 段给 add 侧（summem / atommem）用 → 映射成 ``atommem`` 段。
        if "index" in config and "atommem" not in atom_cfg:
            atom_cfg["atommem"] = config["index"]
        runner = atommem_mod._Runner(
            client, atom_cfg, log or llm_quiet_logger(), self.root / ".atommem.model_calls.jsonl"
        )
        # 本会话的 atom 先清掉，再逐条重建（增量：只动本会话）。
        existing = read_jsonl(self.atommem_path)
        my_ids = {(r.get("metadata") or {}).get("id") for r in msg_records}
        kept = [r for r in existing if not (
            set((r.get("metadata") or {}).get("source") or []) & my_ids
        )]
        # 关键：先把"保留的其它会话 atom"落盘，agent 才能看到一个干净的基线，
        # 且 run_message 用 before_lines 能正确隔离出它**本会话**新增的行。
        write_memories(self.atommem_path, kept)
        for position, target in enumerate(msg_records):
            prior = [
                r for r in msg_records[:position]
                if atommem_mod.msg_session_number((r.get("metadata") or {}).get("id") or "")
                == session_index
            ]
            try:
                await atommem_mod.run_message(
                    target=target, prior=prior, summaries=summaries, workspace=self.root,
                    atommem_path=self.atommem_path, config=atom_cfg, catalog=catalog,
                    runner=runner, log=log or llm_quiet_logger(),
                    speaker_a="user", speaker_b="assistant",
                )
            except Exception as exc:  # noqa: BLE001
                if log is not None:
                    log.warn(f"atom 抽取失败 {target.get('metadata', {}).get('id')}：{exc}")
        runner.model.close()


def llm_quiet_logger():
    from ..log import Logger

    return Logger(level="warn")


# ---------------------------------------------------------------------------
# 排错用
# ---------------------------------------------------------------------------

def debug_dump(store: UserStore, stream: Any | None = None) -> None:
    """把该 user 的语料条数打到 stderr（排错用，不影响协议）。"""
    stream = stream or sys.stderr
    print(
        f"[store] user={store.user_id} root={store.root} "
        f"msg={len(store.sessions())}",
        file=stream,
    )
