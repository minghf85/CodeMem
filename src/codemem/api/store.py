"""按 ``user_id`` 隔离的持久化存储：追加消息、request_id 去重、物化语料、消解。

**一个 user 一个目录**，结构对齐既有管线：

    data/api_store/{slug(user_id)}/
      meta.json                 session_id -> 会话号、每会话消息数、已应用的 request_id
      sessions/session_{n}.jsonl  一次会话一个文件（消解后上下文无关；原话在 source_*）

**为什么沿用 ``session_1_3`` 这套约定**：search agent 的 system prompt、evidence 的
``source`` 校验（只认 ``msg_id``）、`date -d` 时间锚点示例全部建立在这套约定上。
存储层照抄它，整条 search 管线就**一行都不用改**。

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

from ..add import resolve as resolve_mod
from ..add.resolve import list_session_files
from ..add.session import SESSIONS_DIR, session_filename
from ..io import (
    DATA_DIR,
    make_session_record,
    msg_time,
    read_jsonl,
    write_jsonl,
)
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
    def sessions_dir(self) -> Path:
        return self.root / SESSIONS_DIR

    def session_path(self, session_index: int) -> Path:
        return self.sessions_dir / session_filename(session_index)

    def exists(self) -> bool:
        """该 user 是否已经有落盘语料（Search 据此决定是"没记忆"还是"去检索"）。"""
        return self.sessions_dir.is_dir() and any(self.sessions_dir.glob("session_*.jsonl"))

    # -- 读 -----------------------------------------------------------------

    def meta(self) -> dict[str, Any]:
        return _read_meta(self.meta_path, self.user_id)

    def sessions(self) -> list[dict[str, Any]]:
        """全部消息（合并 sessions/ 下的全部会话文件，按会话号、消息序号排序）。"""
        records: list[dict[str, Any]] = []
        for _, path in list_session_files(self.root):
            records.extend(read_jsonl(path))
        return records

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
        """把一个 Add 请求写进语料，返回小结 dict。

        ``messages`` 是已归一化的 ``[{"role", "content", "timestamp"}]``。步骤：

        1. 取该 user 的写锁；
        2. **幂等**：``request_id`` 已应用 → 直接返回（不碰语料）；
        3. 分配 ``msg_id``、转时间，把新消息**以原始措辞**追加到该 session 文件；
        4. **在返回前**消解该 session 文件（时间锚定 + 指代消解，LLM）——
           兑现"持久化后立即可检索"；
        5. 原子写回语料 + meta。

        消解读取每条记录的**原始视图**（``source_content``/``source_time`` 优先），所以
        "已消解的旧消息 + 新追加的原始消息"能一起重新消解，幂等。
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

            # ---- 追加消息（沿用同一个会话号，序号接在已有条数之后）----
            # 新消息写成**原始格式**（content=原话、time=会话时间戳）；该文件的既有记录
            # 可能已消解（带 source_content/source_time），resolve 会各取各的原始视图。
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            path = self.session_path(session_index)
            records = read_jsonl(path)
            start = int(entry.get("count", 0))
            for offset, message in enumerate(messages, start=1):
                content = normalize_content(message.get("content"))
                records.append(
                    make_session_record(
                        msg_id_value=f"session_{session_index}_{start + offset}",
                        role=str(message.get("role") or ""),
                        content=content,
                        time=_iso_from_ms(message.get("timestamp")),
                    )
                )
            entry["count"] = start + len(messages)
            write_jsonl(path, records)

            # ---- 消解该 session（其余 session 原样保留）----
            refreshed = await self._resolve_session(
                session_index=session_index,
                config=config,
                client=client,
                log=log,
            )

            # ---- meta：记下这个 request_id（幂等键）----
            meta["applied_request_ids"] = [*meta["applied_request_ids"], request_id]
            _atomic_write_json(self.meta_path, meta)

            if log is not None:
                log.info(
                    f"Add 写入 {len(messages)} 条 -> session {session_index}"
                    f"（共 {entry['count']} 条）resolve={'成功' if refreshed else '未变'}"
                )
            return {"written": len(messages), "duplicate": False, "session_index": session_index}

    async def _resolve_session(
        self,
        *,
        session_index: int,
        config: dict[str, Any],
        client: Any,
        log: Any | None,
    ) -> bool:
        """消解 ``session_index`` 的会话文件（就地覆盖）。失败不抛 —— 消解不出来不该让 Add 失败。

        见 ``add.resolve.refresh_session``。输入取原始视图，所以旧消息不会被重复消解。
        """
        try:
            return await resolve_mod.refresh_session(
                self.root,
                session_index,
                _index_call_config(config),
                client,
                speaker_a="user",
                speaker_b="assistant",
            )
        except Exception as exc:  # noqa: BLE001 - 消解失败不该让 Add 整体失败
            if log is not None:
                log.warn(f"session {session_index} 消解失败：{type(exc).__name__}: {exc}")
            return False


def _index_call_config(config: dict[str, Any]) -> dict[str, Any]:
    """从 api 配置里取出 index 抽取调用参数（``index`` 段 + 顶层重试参数）。

    与 ``search.runner.gen_call_config`` 同一套做法：模型参数在子段里，重试参数在顶层，
    ``llm.chat_completion`` 需要的是合起来的一份。
    """
    merged: dict[str, Any] = {}
    section = config.get("index") or config.get("summary") or {}
    if isinstance(section, dict):
        merged.update(section)
    for key in ("timeout", "max_retries", "backoff_base", "rate_limit_backoff",
                "max_backoff", "backoff_jitter"):
        if key in config:
            merged[key] = config[key]
    merged.setdefault("temperature", 0.2)
    merged.setdefault("max_tokens", 4096)
    return merged


# ---------------------------------------------------------------------------
# 排错用
# ---------------------------------------------------------------------------

def debug_dump(store: UserStore, stream: Any | None = None) -> None:
    """把该 user 的语料条数打到 stderr（排错用，不影响协议）。"""
    stream = stream or sys.stderr
    print(
        f"[store] user={store.user_id} root={store.root} "
        f"sessions={len(store.sessions())}",
        file=stream,
    )
