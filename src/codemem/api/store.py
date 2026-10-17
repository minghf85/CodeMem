"""按 ``user_id`` 隔离的持久化存储：追加消息、request_id 去重、生成三层记忆。

**一个 user 一个目录**，结构对齐离线管线（add 的三层记忆文件）：

    data/api_store/{slug(user_id)}/
      meta.json                 session_id -> 会话号、每会话消息数、已应用的 request_id、pending（写前日志）
      msgmem.jsonl              原始消息（type=msg；会话时间戳是锚点）
      summem.jsonl              每个会话一份总结（type=summary）
      atommem.jsonl             逐 message 抽出的原子事实（type=atom）

**为什么沿用 ``session_1_3`` 这套 id 约定**：search agent 的 system prompt、evidence 的
``source`` 校验、`date -d` 时间锚点示例全部建立在这套约定上。存储层照抄它，整条 search
管线就**一行都不用改**。

**幂等**是协议硬要求：Add 重试时 ``request_id`` 不变，同一条不能写入两次。所以每次 Add 先查
``meta.applied_request_ids`` —— 命中就直接返回成功，连语料都不碰。

**崩溃一致 / 中断续跑**：Add 不是"一次性写完"，而是**写前日志（WAL）**—— 先把意图
（`request_id` + 待写 msg 记录）记进 ``meta.pending``，再落 ``msgmem``、再刷新记忆，最后把
``request_id`` 从 ``pending`` 移进 ``applied_request_ids`` 落定。任意时刻被 kill（SIGKILL/OOM/
重启），重启后都能凭 ``pending`` 把这次 Add **补完**（``recover`` / ``recover_all``，也由
``append`` 顶部懒执行），且重发同一 ``request_id`` 不会重复写消息。

**锁的作用域 = "不调模型的那些行"**（这是本模块唯一重要的并发纪律）：

    ┌─ 语料写锁（每 user 一把，纯 CPU）──────────────────────────────┐
    │  meta.json：幂等键 / 会话号 / WAL / 计数                       │
    │  msgmem.jsonl：追加消息（原子整文件重写）                      │
    │  requests.jsonl：请求留痕                                      │
    │  summem / atommem 的"读—改—写"落盘（合并、去重后写回）         │
    └────────────────────────────────────────────────────────────────┘
    ┌─ 刷新锁（每 user 每 session 一把，**可以跨模型调用**）──────────┐
    │  一个会话的三层记忆重建：总结一次 + 逐 message 抽 atom         │
    └────────────────────────────────────────────────────────────────┘

为什么要把刷新挪出写锁：Add 的耗时 99% 在模型调用上（一次总结 + 每条 message 一次抽取）。
旧实现把整段刷新放在写锁里，于是**同一个 user 的并发 Add 全部退化成串行** —— 32 路并发进来
只有 1 路在跑，另外 31 路在锁上排长队，`/search` 的请求留痕也被这把锁堵住（它只写一行）。
现在写锁里只剩微秒级的 CPU 段，模型调用在锁外并发跑，总并发由进程级的模型闸门统一约束。

不同 user 互不影响（各自的目录）。``asyncio.to_thread`` 把整文件读写与去重（CPU）挪出事件
循环 —— 否则一个 user 的一次全量重写会卡住**所有**请求的响应（单进程多协程里，事件循环是
共享的）。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import adapters
from ..add import summem as summem_mod
from ..add.memory import (
    ATOMMEM_NAME,
    MSGMEM_NAME,
    SUMMEM_NAME,
    msg_session_number,
    write_memories,
)
from ..io import DATA_DIR, make_memory, memory_id, now_iso, read_jsonl
from .models import normalize_content

META_NAME = "meta.json"
#: 抽取覆盖记录：``{session_index: {"summarized": N, "atoms": [msg_id, ...]}}``。
#: 让"重刷一个会话"只处理**还没抽过**的 message（见 ``UserStore._refresh_memories``）。
ATOM_STATE_NAME = "atom_state.json"

_SLUG_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

#: 同一 user 的**语料写锁**。进程内有效（API 是单进程多协程）；不同 user 各一把。
#: **纪律：持有它的时候一次模型都不许调**（那会把整个 user 的请求串起来，见模块开头）。
_user_locks: dict[str, "asyncio.Lock"] = {}

#: 同一 (user, session) 的**刷新锁**：一个会话的记忆重建（总结 + 逐 message 抽取）串行化。
#: 它**可以**跨模型调用持有 —— 同一会话的两次 Add 本来就该共用一次抽取，而不是各抽一遍。
_session_locks: dict[tuple[str, int], "asyncio.Lock"] = {}

#: 本进程**正在处理**的 ``request_id``。看到 ``meta.pending`` 里有它时不要当"被中断的 Add"去
#: 恢复 —— 它只是还在跑刷新（写下 WAL 到落定之间一直躺在 pending 里，见 ``append`` 的第 4~7 步）。
#: 不加这一层判断，同时进来的 N 个 Add 会互相"恢复"，把同一批 message 抽 N 遍。
_in_flight: set[str] = set()


def _lock_for(user_id: str) -> "asyncio.Lock":
    import asyncio

    lock = _user_locks.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        _user_locks[user_id] = lock
    return lock


def _session_lock_for(user_id: str, session_index: int) -> "asyncio.Lock":
    import asyncio

    key = (user_id, int(session_index))
    lock = _session_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[key] = lock
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
            raw.setdefault("pending", {})
            return raw
    return {"user_id": user_id, "sessions": {}, "next_index": 1,
            "applied_request_ids": [], "pending": {}}


def _add_config(config: dict[str, Any]) -> dict[str, Any]:
    """把 API 配置里的 add 侧（``index`` 段 + 重试/并发）合成 add 模块能**直接**用的配置。

    ``summem``/``atommem`` 的 ``gen_call_config`` 与 pipeline 的 ``llm.chat_completion``
    都读**顶层** ``model``/``base_url``/...，而 API 把它们放在 ``index`` 段 —— 这里扁平化成
    一份模型配置，两个模块都用它，避免"读完发现模型配置不在顶层、悄悄退回本地模型"。
    """
    merged = dict(config.get("index") or {})
    for key in ("timeout", "connect_timeout", "max_retries", "backoff_base",
                "rate_limit_backoff", "max_backoff", "backoff_jitter"):
        if key in config:
            merged.setdefault(key, config[key])
    merged.setdefault("concurrency", config.get("add_concurrency", 8))
    # 抽取失败**就地重试**的轮数、退避与时间上界（服务侧的健壮性旋钮，见 ``_refresh_atoms``）。
    for key, default in (("atom_retry_rounds", 2), ("atom_retry_backoff", 2.0),
                         ("atom_retry_budget", 30.0)):
        merged.setdefault(key, config.get(key, default))
    if config.get("add_max_steps") is not None:
        merged.setdefault("max_steps", config["add_max_steps"])
    return merged


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

    @property
    def requests_path(self) -> Path:
        """原始请求留痕文件：每次 ``/add`` / ``/search`` 一行（见 ``append_request``）。"""
        return self.root / "requests.jsonl"

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
            if msg_session_number((r.get("metadata") or {}).get("id") or "")
            == session_index
        ]

    # -- 写 -----------------------------------------------------------------

    async def append_request(self, record: dict[str, Any]) -> None:
        """把一条原始 API 请求追加到 ``requests.jsonl``（排错/分析用）。

        ``server.py`` 在每次 ``/add`` / ``/search`` 后调用 —— 存的是**原始报文**（add 的
        messages / search 的 query），不是推导后的产物，所以事后能回答"这条请求/这条 QA 到底
        说了什么"。留痕是**尽力而为**：写失败绝不影响请求本身。记录自动补 ``time``。
        """
        entry = {"time": now_iso(), **record}
        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        async with _lock_for(self.user_id):
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                with self.requests_path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
            except OSError:
                pass    # 留痕失败不阻断请求

    async def append(
        self,
        *,
        request_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        config: dict[str, Any],
        client: Any,
        log: Any | None = None,
        gate: Any = None,
    ) -> dict[str, Any]:
        """把一个 Add 请求写成三层记忆，返回小结 dict。

        ``messages`` 是已归一化的 ``[{"role", "content", "timestamp"}]``。**崩溃一致**：先把
        "意图"写进 ``meta.pending``（写前日志，WAL），再落 ``msgmem``、再刷新记忆、最后落定。
        任意时刻被 kill，重启后都能凭 ``pending`` 把这次 Add 补完（见 ``recover``），且重发同一
        ``request_id`` 不会重复写。

        步骤（**锁内只做 CPU，模型调用一律在锁外** —— 见模块开头的并发纪律）：

        1. **锁内**：取语料写锁；补完遗留 pending 的**消息部分**（``_adopt_pending``）；
        2. **锁内**：幂等 —— ``request_id`` 已应用 / 已待续跑 → 不写任何东西、直接返回，
           但**遗留 pending 的刷新照做**（重试一条卡住的 Add 就该把它推完）；
        3. **锁内**：分配会话号与 msg_id，构建本批 msg 记录（过形态适配层）；
        4. **锁内**：WAL —— ``{session_id, session_index, start, records}`` 进 ``meta.pending``，
           **预留 id 空间**，原子写 meta（在碰 msgmem 之前）；
        5. **锁内**：追加 msg 记录到 ``msgmem.jsonl``（原子整文件重写）；
        6. **锁外**：重建本会话的 ``summem`` / ``atommem``（模型调用；与其他 Add、其他 session
           并发；总并发由 ``gate`` 约束）；
        7. **锁内**：成功才把 ``request_id`` 落定（pending → applied）；失败则保留 pending，
           下次启动 / 下次同 user 请求时自动重试刷新。
        """
        lock = _lock_for(self.user_id)
        backlog: list[tuple[str, int]] = []
        duplicate: dict[str, Any] | None = None

        # ---------------- 锁内（纯 CPU）----------------
        async with lock:
            self.root.mkdir(parents=True, exist_ok=True)
            meta = self.meta()

            # 先补完上一个被中断的 Add 的**消息部分**（幂等；pending 为空时零成本）。
            # 它们的**记忆刷新**放到锁外（那才是耗时的部分）。
            backlog = await self._adopt_pending(meta, log=log)

            if request_id in meta["applied_request_ids"]:
                if log is not None:
                    log.info(f"Add 幂等命中：request_id={request_id!r} 已应用，跳过写入")
                duplicate = {"written": 0, "duplicate": True, "session_index": None}
            elif request_id in meta["pending"]:
                # 上一次同 id 的 Add 已落盘消息、只是记忆刷新没跑完（见 _adopt_pending）。
                if log is not None:
                    log.info(f"Add 续跑命中：request_id={request_id!r} 待补刷新，消息已落盘")
                duplicate = {"written": 0, "duplicate": True, "pending": True,
                             "session_index": None}
            else:
                # ---- 该 session 在本地语料里的会话号 ----
                sessions_map: dict[str, Any] = meta["sessions"]
                entry = sessions_map.get(session_id)
                if not isinstance(entry, dict):
                    entry = {"index": int(meta.get("next_index", 1)), "count": 0}
                    sessions_map[session_id] = entry
                    meta["next_index"] = int(entry["index"]) + 1
                session_index = int(entry["index"])
                start = int(entry.get("count", 0))

                # ---- 构建本批 msg 记录（msg_id 接在已预留条数之后）----
                #
                # **先过形态适配层**（``codemem/adapters.py``）：平台把"谁说的""什么时候"编码在
                # 报文的各个角落（真名在 content 前缀里、时间在 ``[Time: ...]`` 标签里、三元组糊在
                # JSON 里），直接落 msgmem 会让 atommem 全变成"user said ..."，而查询问的是
                # "Caroline …"，grep 直接落空。适配层只做**无损换位**：把信息搬进 ``role`` /
                # ``time``，摘掉平台记账标签，正文一字不改。
                #
                # ``normalize_content`` 在这里再兜一次（``server`` 已经做过多模态降级）：适配层
                # 只认字符串，而 ``append`` 也可能被别的调用方直接喂 ``ContentPart[]``。
                batch = adapters.adapt_messages([
                    {**message, "content": normalize_content(message.get("content"))}
                    for message in messages
                ])
                records = [
                    make_memory(
                        memory=message.content,
                        memory_type="msg",
                        mem_id=f"session_{session_index}_{start + offset}",
                        time=_iso_from_ms(message.timestamp),
                        role=message.role,
                        tag=[],
                        source=[],
                        changelog=[{"time": now_iso(), "content": "created"}],
                    )
                    for offset, message in enumerate(batch.messages, start=1)
                ]

                # ---- WAL：先记意图并预留 id 空间，原子写 meta（在碰 msgmem 之前）----
                meta["pending"][request_id] = {
                    "session_id": session_id,
                    "session_index": session_index,
                    "start": start,
                    "records": records,
                    "created": now_iso(),
                }
                entry["count"] = start + len(records)
                _atomic_write_json(self.meta_path, meta)

                # ---- 落 msgmem（原子整文件重写；读改写整个过程挪到线程里）----
                def _append_msgs() -> None:
                    existing = read_jsonl(self.msgmem_path)
                    write_memories(self.msgmem_path, existing + records)

                await asyncio.to_thread(_append_msgs)
                total_count = int(entry["count"])

        # ---------------- 锁外（模型调用，可并发）----------------
        # 幂等 / 续跑命中：本请求一个字都不写，但**遗留的 pending 照样补完** —— 客户端重试一条
        # 卡住的 Add，这一次调用就该把它推完（否则它要等到下一次新请求或重启才动）。
        if duplicate is not None:
            await self._settle(backlog, config=config, client=client, log=log, gate=gate)
            return duplicate

        # 本请求的刷新排在前面（它的响应在等它）；遗留 pending 顺带补完。
        _in_flight.add(request_id)
        try:
            refreshed = await self._refresh_memories(
                session_index=session_index, config=config, client=client, log=log, gate=gate,
            )
            if refreshed:
                await self._finalize(request_id, log=log)
            await self._settle(backlog, config=config, client=client, log=log, gate=gate)
        finally:
            _in_flight.discard(request_id)

        if log is not None:
            log.info(
                f"Add 写入 {len(records)} 条 -> session {session_index}"
                f"（共 {total_count} 条）形态={batch.family}"
                f"{' 说话人=' + '/'.join(batch.speakers) if batch.speakers else ''}"
                f" 记忆刷新={'成功' if refreshed else '待续跑'}"
            )
        return {
            "written": len(records),
            "duplicate": False,
            "session_index": session_index,
            "refreshed": refreshed,
        }

    async def _finalize(self, request_id: str, *, log: Any | None = None) -> bool:
        """把 ``request_id`` 从 pending 移进 applied（幂等；落定前重读 meta）。返回是否落定。"""
        async with _lock_for(self.user_id):
            meta = self.meta()
            if request_id not in (meta.get("pending") or {}):
                return False
            meta["pending"].pop(request_id, None)
            meta["applied_request_ids"] = [*meta["applied_request_ids"], request_id]
            _atomic_write_json(self.meta_path, meta)
        if log is not None:
            log.debug(f"Add 落定：request_id={request_id!r}")
        return True

    async def _settle(
        self,
        entries: list[tuple[str, int]],
        *,
        config: dict[str, Any],
        client: Any,
        log: Any | None,
        gate: Any = None,
    ) -> int:
        """补完 ``_adopt_pending`` 挑出来的遗留 pending（锁外刷新 + 锁内落定）。返回落定条数。

        单条失败只 warn、不中断 —— 它留在 pending 里，下次启动 / 下次同 user 请求再试。
        """
        settled = 0
        for request_id, session_index in entries:
            try:
                refreshed = await self._refresh_memories(
                    session_index=session_index, config=config, client=client,
                    log=log, gate=gate,
                )
            except Exception as exc:  # noqa: BLE001 - 单条恢复失败不该影响其它
                refreshed = False
                if log is not None:
                    log.warn(f"恢复 pending 失败 {request_id!r}：{type(exc).__name__}: {exc}")
            if refreshed and await self._finalize(request_id, log=log):
                settled += 1
        if settled and log is not None:
            log.info(f"续跑完成：{settled}/{len(entries)} 个被中断的 Add 已落定")
        return settled

    # -- 中断续跑（恢复）---------------------------------------------------

    async def _adopt_pending(
        self, meta: dict[str, Any], *, log: Any | None,
    ) -> list[tuple[str, int]]:
        """把 ``meta.pending`` 里的记录补进 msgmem（**纯 CPU**），返回待刷新清单。

        崩溃点都吃得下：``msgmem`` 已写则按 ``memory_id`` 跳过补写；刷新是"整会话重建"故可重入。
        **只做不需要模型的那一半** —— 刷新本身由调用方在锁外做（``_settle``），否则一个遗留的
        pending 会让这个 user 之后的每个请求都在锁上排队（旧实现就是这样）。
        """
        entries: list[tuple[str, int]] = []
        pending = meta.get("pending") or {}
        if not pending:
            return entries

        for request_id in list(pending):
            if request_id in _in_flight:
                # 本进程正在跑的 Add：它只是**还没落定**，不是被中断了。恢复它会重复刷一遍。
                continue
            entry = pending.get(request_id)
            if not isinstance(entry, dict):
                pending.pop(request_id, None)
                continue
            session_id = str(entry.get("session_id") or "")
            session_index = int(entry.get("session_index") or 0)
            start = int(entry.get("start") or 0)
            records = [r for r in (entry.get("records") or []) if isinstance(r, dict)]

            def _fill() -> None:
                existing = read_jsonl(self.msgmem_path)
                have = {memory_id(r) for r in existing}
                missing = [r for r in records if memory_id(r) not in have]
                if missing:
                    write_memories(self.msgmem_path, existing + missing)

            await asyncio.to_thread(_fill)

            # 补回（可能缺失/损坏的）会话条目；count 取最大（预留语义，避免 msg_id 撞车）
            sessions_map: dict[str, Any] = meta["sessions"]
            item = sessions_map.get(session_id)
            if not isinstance(item, dict):
                item = {"index": session_index, "count": 0}
                sessions_map[session_id] = item
            item["count"] = max(int(item.get("count", 0)), start + len(records))
            entries.append((request_id, session_index))

        if entries:
            _atomic_write_json(self.meta_path, meta)
            if log is not None:
                log.info(f"发现 {len(entries)} 个被中断的 Add，消息已就位，待补记忆刷新")
        return entries

    async def recover(
        self,
        *,
        config: dict[str, Any],
        client: Any,
        log: Any | None = None,
        gate: Any = None,
    ) -> int:
        """补完该 user 遗留的 pending Add（幂等、可重入）。无 pending 时零成本。返回落定条数。"""
        async with _lock_for(self.user_id):
            meta = self.meta()
            backlog = await self._adopt_pending(meta, log=log)
        return await self._settle(backlog, config=config, client=client, log=log, gate=gate)

    # -- 记忆刷新（锁外跑模型 / 锁内落盘）-----------------------------------

    async def _refresh_memories(
        self,
        *,
        session_index: int,
        config: dict[str, Any],
        client: Any,
        log: Any | None,
        gate: Any = None,
        force: bool = False,
    ) -> bool:
        """重算该会话的 summem（一次 LLM）与 atommem（逐 message 的 **pipeline**）。失败不抛。

        summem 见 ``add/summem.py``；atommem 见 ``add/atommem_pipeline.py``（与离线
        ``add --mode pipeline`` 同一条路径）。

        三件事让它能扛住并发：

        1. **模型调用在语料写锁之外**，同一会话由 ``_session_lock_for`` 串行（同会话的两次 Add
           共用一次重建，而不是各抽一遍）；
        2. **增量**：只有**还没抽过**的 message 会重新走 LLM，覆盖记录在 ``atom_state.json``
           （``force=True`` 则整会话重来，供 ``rebuild_api_store`` 用）。Add 是追加式的，
           一条 message 的抽取只依赖它**之前**的原文（``recall_context`` 只往前看），所以早抽过的
           结论不会因为后面追加了消息而变；
        3. **模型并发由 ``gate`` 管**：闸门只包住单次调用（见 ``atommem_pipeline._call_model``），
           所以"一份 prompt 被切成 N 块"只是排 N 次队，不会占着名额等自己。

        **说话人名单从已落盘的 msgmem 里取**（``adapters.speakers_of``），不使用硬编码的
        ``user``/``assistant``：这份语料里真名在正文前缀里（``Christina Baker:``），适配层已经
        把它搬进了 ``role``，这里照读即可。名单取自落盘数据而不是内存变量，所以**恢复
        （``recover``）重跑时算出来的名单与首次完全一致**。
        """
        add_cfg = _add_config(config)
        try:
            async with _session_lock_for(self.user_id, session_index):
                sess_records = await asyncio.to_thread(self.session_records, session_index)
                if not sess_records:
                    return True
                speakers = adapters.speakers_of(sess_records)
                state = await asyncio.to_thread(self._load_atom_state)
                key = str(session_index)
                entry = dict(state.get(key) or {})

                # --- summem：该会话的消息数没变过就不重跑（一次 LLM）---
                if force or int(entry.get("summarized", -1)) != len(sess_records):
                    summary = await self._summarize_session(
                        session_index=session_index, sess_records=sess_records,
                        speakers=speakers, client=client, config=add_cfg, log=log, gate=gate,
                    )
                    # 只有**写得出字**才落盘：一次抖动不该把上一版（虽然过时但可用）的总结冲成空。
                    if summary and str(summary.get("memory") or "").strip():
                        await self._write_summary(session_index, summary)
                    self._record_summary(
                        session_index, entry, summary,
                        record_count=len(sess_records), log=log,
                    )

                # --- atommem：只跑还没抽过的 message（pipeline）---
                covered = set() if force else {str(x) for x in (entry.get("atoms") or [])}
                done, failed, blocked = await self._refresh_atoms(
                    session_index=session_index, sess_records=sess_records,
                    covered=covered, config=add_cfg, client=client, log=log, gate=gate,
                )
                # 记账按**每条 message 自己的成败**：抽成的照记；被端点拒绝的（内容策略等）也记进
                # 覆盖名单 —— 它们**不该**被自动重试（重发同一个 prompt 还是同样结果），但得留痕
                # 可见（`blocked`），换模型后可以用 `--repair --retry-blocked` 重来。失败（抖动）
                # 的那几条留白，只重试它们。
                entry["atoms"] = sorted(covered | done | set(blocked))
                entry["failed"] = sorted(failed)
                entry["blocked"] = dict(sorted(blocked.items()))
                await self._update_atom_state(session_index, entry)
        except Exception as exc:  # noqa: BLE001 - 记忆刷新失败不该让 Add 失败
            if log is not None:
                log.warn(f"session {session_index} 记忆刷新失败：{type(exc).__name__}: {exc}")
            return False
        return True

    def _record_summary(
        self, session_index: int, entry: dict[str, Any], summary: dict[str, Any] | None,
        *, record_count: int, log: Any | None,
    ) -> None:
        """把一次总结的**进度**记进 ``entry``（写盘由调用方做，见 ``_refresh_memories``）。

        为什么不能只看"memory 是不是空的"：``summarize_session`` 现在会在 ``metadata.error`` 里
        写失败类别，而且**部分成功**（长会话分段总结时前几段成了、后面某段没成）是常态 ——
        那时 memory 非空但总结并不完整，直接记"做完了"会把缺的那一段永久盖住。

        | 情况 | 记什么 | 下次还重试吗 |
        |---|---|---|
        | 成了 | ``summarized`` = 消息数 | 消息数变了才重跑 |
        | 失败（抖动 / 只成了几段） | ``summary_failed`` | **会**（补漏 / 下次刷新） |
        | 端点不接受这个 prompt | ``summary_blocked`` + ``summarized`` | **不会**（换模型后 `--repair --retry-blocked`） |

        被端点拒绝的那种**照样把 ``summarized`` 推上去**：否则它会在**每次**同会话的 Add 上
        再试一遍（重发同一份 prompt 还是同样结果），而 `summary_blocked` 已经把这个事实记下来了。
        """
        complete = bool(summary) and bool(str(summary.get("memory") or "").strip())
        kind = str(((summary or {}).get("metadata") or {}).get("error") or "")
        if complete and not kind:
            entry["summarized"] = record_count
            entry.pop("summary_failed", None)
            entry.pop("summary_blocked", None)
            return
        if kind == "permanent":
            entry["summary_blocked"] = True
            entry["summarized"] = record_count
            entry.pop("summary_failed", None)
        else:
            entry["summary_failed"] = True
            entry.pop("summary_blocked", None)
        if log is not None:
            log.warn(
                f"session {session_index} 总结未完成"
                + ("（端点不接受该 prompt，不再自动重试）" if kind == "permanent" else "（待重试）")
                + f"：{kind or 'empty'}"
                + ("；保留原总结" if not complete else "；只有部分段落")
            )

    async def _summarize_session(
        self, *, session_index: int, sess_records: list[dict[str, Any]], speakers: list[str],
        client: Any, config: dict[str, Any], log: Any | None, gate: Any = None,
    ) -> dict[str, Any] | None:
        """跑一次会话总结（模型调用）。``gate`` 只包住这一次调用，不套整段刷新。"""
        session_time = str((sess_records[0].get("metadata") or {}).get("time") or "")
        call = summem_mod.summarize_session(
            session_index=session_index, session_time=session_time, records=sess_records,
            client=client, config=config, speakers=speakers,
        )
        if gate is None:
            return await call
        async with gate:
            return await call

    async def _write_summary(self, session_index: int, summary: dict[str, Any]) -> None:
        """把该会话的总结替换进 summem.jsonl（锁内纯 CPU）。"""
        def _write() -> None:
            existing = read_jsonl(self.summem_path)
            summary_id = f"session_{session_index}_summary"
            existing = [
                r for r in existing if (r.get("metadata") or {}).get("id") != summary_id
            ]
            write_memories(self.summem_path, existing + [summary])

        async with _lock_for(self.user_id):
            await asyncio.to_thread(_write)

    # -- 抽取覆盖记录（增量刷新的依据）-------------------------------------

    @property
    def atom_state_path(self) -> Path:
        """``atom_state.json``：每个会话"总结到第几条 / 哪些 message 已抽过"（见 ``_refresh_memories``）。"""
        return self.root / ATOM_STATE_NAME

    def _load_atom_state(self) -> dict[str, Any]:
        """读覆盖记录。缺失/损坏 → 空 dict（**等于"全都得重抽"**，是最安全的一侧）。"""
        if not self.atom_state_path.exists():
            return {}
        try:
            raw = json.loads(self.atom_state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _save_atom_state(self, state: dict[str, Any]) -> None:
        _atomic_write_json(self.atom_state_path, state)

    def sessions_needing_repair(self, include_blocked: bool = False) -> list[int]:
        """还欠着活的会话号：抽取失败过（``failed``）、总结为空（``summary_failed``）等。

        ``include_blocked=True`` 时把**被端点拒绝**的也算进来（``blocked`` / ``summary_blocked``）——
        那些默认**不**自动重试（重发同一个 prompt 还是同样结果），只有换了一个不拦内容的端点、
        明确要求 `--retry-blocked` 时才重来。

        **只读记账就能判定的**（记账里显式写了 ``failed`` 的会话）直接判；**没写 ``failed`` 键的**
        条目是更早版本留下的（那时还不记失败），只能拿"覆盖名单 vs 语料"的差集兜一次 —— 但**只在
        该会话确实刷新过**（状态里有条目）时才做这个差集。

        为什么要卡这一条：差集分不清"抽失败了"和"这台机器上还没轮到抽"（旧版语料根本没有记账
        文件），后者一旦被当成前者，启动补漏就会把整库重抽一遍。而这个差集也正是**升级场景**
        需要的：服务在旧版本下跑出过失败，记账里没有 ``failed``，重启后只有它能发现那些欠账。
        差集需要读一遍 ``msgmem``，但只在真有旧格式条目时读。
        """
        state = self._load_atom_state()
        due: set[int] = set()
        legacy: list[int] = []
        for key, entry in state.items():
            if not isinstance(entry, dict):
                continue
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            if "failed" in entry:                       # 新格式：记账说了算
                if (entry.get("failed") or entry.get("summary_failed")
                        or (include_blocked
                            and (entry.get("blocked") or entry.get("summary_blocked")))):
                    due.add(index)
            else:                                       # 旧格式：不知道当时成没成
                legacy.append(index)

        if legacy:
            # 旧记账还漏了一件事：**总结成没成**。老版本不记 `summary_failed`，失败时写的是一条
            # 空总结，光看记账分不出来 —— 所以按"那条 summary 记录在不在、有没有字"再判一次。
            # （总结只有成功才会写出字来，所以"空/缺"就是"没成"，不会误报。）
            summaries = {
                memory_id(record): str(record.get("memory") or "").strip()
                for record in read_jsonl(self.summem_path)
            }
            for index in legacy:
                if not summaries.get(f"session_{index}_summary"):
                    due.add(index)

            covered = {
                index: {str(x) for x in (state[str(index)].get("atoms") or [])}
                for index in legacy
            }
            for record in read_jsonl(self.msgmem_path):
                index = msg_session_number(memory_id(record))
                if index is None or index not in covered:
                    continue
                if memory_id(record) not in covered[index]:
                    due.add(index)
                    covered.pop(index)                  # 这个会话已确认欠账，不必再逐条看
        return sorted(due)

    async def repair(
        self,
        *,
        config: dict[str, Any],
        client: Any,
        log: Any | None = None,
        gate: Any = None,
        include_blocked: bool = False,
    ) -> int:
        """把这个 user "欠着"的会话重刷一遍。返回**修好了**的会话数（仍失败的留在记账里）。

        ``_refresh_memories`` 只对"整个会话炸了"返回 False，单条 message 的失败要靠重读记账判断，
        所以这里刷完再看一眼 ``failed`` / ``summary_failed``（``include_blocked`` 时还有 ``blocked``）
        还在不在。

        ``include_blocked=True`` 会先把"被端点拒绝"的记录**从覆盖名单里摘掉**再重刷 —— 换了一个
        不拦内容的端点之后，那几条才有机会重新抽出来。
        """
        pending = await asyncio.to_thread(self.sessions_needing_repair, include_blocked)
        fixed = 0
        for session_index in pending:
            try:
                if include_blocked:
                    await self._clear_blocked(session_index)
                await self._refresh_memories(
                    session_index=session_index, config=config, client=client,
                    log=log, gate=gate,
                )
            except Exception as exc:  # noqa: BLE001 - 单个会话不该拖垮整轮补漏
                if log is not None:
                    log.warn(f"补漏 session {session_index} 失败：{type(exc).__name__}: {exc}")
                continue
            entry = self._load_atom_state().get(str(session_index)) or {}
            still = entry.get("failed") or entry.get("summary_failed")
            if include_blocked:
                still = still or entry.get("blocked") or entry.get("summary_blocked")
            if not still:
                fixed += 1
        return fixed

    async def _clear_blocked(self, session_index: int) -> None:
        """把该会话"被端点拒绝"的记账清掉（连同覆盖名单里的那几条）。

        只在 ``repair(include_blocked=True)``（CLI 的 ``--repair --retry-blocked``）时用：**换了一个
        不拦内容的端点**之后再重来一遍。平时它们**不**动 —— 重发同一个 prompt 还是同样结果。
        """
        def _clear() -> None:
            state = self._load_atom_state()
            entry = state.get(str(session_index))
            if not isinstance(entry, dict):
                return
            rejected = {str(mid) for mid in (entry.get("blocked") or {})}
            if rejected:
                entry["atoms"] = [
                    mid for mid in (entry.get("atoms") or []) if str(mid) not in rejected
                ]
            entry.pop("blocked", None)
            if entry.pop("summary_blocked", None):
                # 重新总结的前提是"消息数变了才重跑"，被拒的那次把 summarized 推上去了 ——
                # 不把它抹掉，"重来一遍"就会被这条规则挡住、总结永远补不上。
                entry.pop("summarized", None)
            self._save_atom_state(state)

        async with _lock_for(self.user_id):
            await asyncio.to_thread(_clear)

    async def _update_atom_state(self, session_index: int, entry: dict[str, Any]) -> None:
        """把这个会话的进度**并进** ``atom_state.json``（锁内读—改—写）。

        必须"并"而不是"整份覆盖"：各会话各自刷新、并发跑，整份覆盖会把别的会话刚写进去的进度
        冲掉 —— 被冲掉的会话下次重抽一遍（实测：8 路并发 Add 的抽取调用从 8 次涨到 14 次）。
        """
        def _update() -> None:
            state = self._load_atom_state()
            state[str(session_index)] = entry
            self._save_atom_state(state)

        async with _lock_for(self.user_id):
            await asyncio.to_thread(_update)

    # -- atommem 刷新 -------------------------------------------------------

    async def _refresh_atoms(
        self, *, session_index: int, sess_records: list[dict[str, Any]], covered: set[str],
        config: dict[str, Any], client: Any, log: Any | None, gate: Any = None,
    ) -> tuple[set[str], bool]:
        """对本会话**还没抽过**的 message 跑 atommem pipeline，重建本会话的 atom。

        返回 ``(本次抽成功的 message id 集合, 仍然失败的 message id 集合, 被端点拒绝的 message)`` ——
        前者之外的两类都会记账：失败的进 ``failed``（下次刷新 / 启动补漏重试），被拒绝的进
        ``blocked``（**不再自动重试**，见 ``MessageExtraction`` 的三种结局）。

        与离线 ``add --mode pipeline`` 同一条路径（``atommem_pipeline.run_message_pipeline``）：
        各 message 的抽取**互不依赖**（召回只读 msgmem、id 按各 source 独立分配、去重是收尾一次
        全局 pass），所以并发抽取后**一次性**去重落盘。本会话之外的 atom 原样保留（增量）。

        **三处并发/一致性的处理**：

        - 并发由 ``gate`` 管，且它是**每次模型调用**的闸门（不是套在整条 pipeline 外面的那种）——
          套在外面会和 pipeline 内层的闸门自我嵌套：任务占着一个名额、再去等另一个名额，
          几十路并发时会僵住。调用方没给闸门时（脚本 / 测试直接调 store）这里**补一个本地的**，
          作用域退化成"这一条会话"，但同样按调用排队；
        - 重抽的 message 的旧 atom 先**摘掉**再合并（``redo``），所以"刷新失败后重试"或
          ``force=True`` 都不会留下两份同源 atom（id 是从 ``_1`` 重新编号的，不摘会撞 id）；
        - **读—改—写整个落盘过程在语料写锁内、且在线程里**（去重是 O(n²) 的 CPU 活，
          放在事件循环里会卡住所有请求的响应）。
        """
        from ..add.atommem_pipeline import msg_positions, run_message_pipeline

        todo = [r for r in sess_records if memory_id(r) not in covered]
        if not todo:
            return set(), set(), {}

        if gate is None:
            gate = asyncio.Semaphore(max(1, int(config.get("concurrency", 8) or 8)))
        # 一次读全 + 建一次位置表：别让每条 message 各扫一遍 msgmem（库里上万条时那是几十万次
        # 查找，而且这段是同步跑的）。
        msgmem = await asyncio.to_thread(read_jsonl, self.msgmem_path)
        positions = await asyncio.to_thread(msg_positions, msgmem)
        session_speakers = adapters.speakers_of(sess_records)
        redo = {memory_id(r) for r in todo}

        async def extract(target: dict[str, Any]):
            node_log = (log or llm_quiet_logger()).bind(memory_id(target))
            try:
                return await run_message_pipeline(
                    target=target, msgmem=msgmem, existing_atoms=[], client=client,
                    config=config, log=node_log, speakers=session_speakers, gate=gate,
                    positions=positions,
                )
            except Exception as exc:  # noqa: BLE001 - 单条失败不该让整个 add 失败
                node_log.warn(f"atom 抽取失败：{type(exc).__name__}: {exc}")
                return None

        # **失败就地重试**（`llm.py` 的退避重试之下的第二层）。为什么要有第二层：那些退避重试
        # 对付的是"单次调用"的抖动，而这里更常见的失败是**建连超时**（远程端点在并发新建连接时
        # 丢 SYN，`llm.py` 的注释里记着实测）—— 一次批量抽取里十几条一起撞上，退避 2s/4s 不够。
        #
        # 但**重试必须有时间上界**：一次 Add 的响应在等这次刷新，而一次彻底失败的调用本身就要
        # 花掉 `connect_timeout × max_retries + 退避`（默认 ≈36s）。所以预算按**整个刷新**算
        # （``atom_retry_budget``，默认 30s，含第一次尝试）：慢失败直接落到``failed`` 交给补漏，
        # 而不是把这个请求磨到客户端超时；只有快失败（瞬时抖动）才配得上就地重试。
        rounds = max(0, int(config.get("atom_retry_rounds", 2) or 0))
        backoff = float(config.get("atom_retry_backoff", 2.0) or 0.0)
        budget = float(config.get("atom_retry_budget", 30.0) or 0.0)
        started = time.monotonic()
        pending = list(todo)
        # ``msg_id -> 该 message 的 atom``：**只留一份**。id 是 ``{msg_id}_{k}``、每次尝试都从
        # ``_1`` 开始编号，两次尝试的产出一起留下就会撞 id。
        best: dict[str, list[dict[str, Any]]] = {}
        done: set[str] = set()
        # ``msg_id -> 原因``：端点**不接受这个 prompt**（内容策略 / 非法请求），重发也一样。
        # 它不进重试轮次，调用方把它记成 blocked（不再自动重试，但留痕可见）。
        blocked: dict[str, str] = {}
        failed: list[dict[str, Any]] = []

        for attempt in range(rounds + 1):
            results = await asyncio.gather(*(extract(t) for t in pending))
            failed = []
            for target, result in zip(pending, results):
                msg_id = memory_id(target)
                if result is None:                      # 整个任务抛了（工具层/连接层的意外）
                    failed.append(target)
                    continue
                current = best.get(msg_id) or []
                # 成功的尝试优先（它是完整的）；都没成功时留产出更多的那次
                if result.ok or len(result.atoms) > len(current):
                    best[msg_id] = result.atoms
                if result.ok:
                    done.add(msg_id)
                elif result.permanent:
                    blocked[msg_id] = result.note or "permanent"
                else:
                    failed.append(target)
            if not failed or attempt >= rounds:
                break
            delay = backoff * (2 ** attempt)
            elapsed = time.monotonic() - started
            if budget > 0 and elapsed + delay > budget:
                if log is not None:
                    log.warn(
                        f"session {session_index}: {len(failed)}/{len(todo)} 条抽取失败，"
                        f"已耗时 {elapsed:.0f}s、超出重试预算 {budget:.0f}s —— 留给补漏重试"
                    )
                break
            if log is not None:
                log.warn(
                    f"session {session_index}: {len(failed)}/{len(todo)} 条抽取失败，"
                    f"{delay:.0f}s 后重试（第 {attempt + 1}/{rounds} 轮）"
                )
            if delay > 0:
                await asyncio.sleep(delay)
            pending = failed

        atoms = [atom for group in best.values() for atom in group]
        if best:

            def _merge() -> None:
                from ..add.atommem import merge_atoms

                existing = read_jsonl(self.atommem_path)
                # 保留"本会话之外"以及**本次没重抽的** atom（按 source 剔除要重建的那些）。
                kept = [
                    a for a in existing
                    if not (set((a.get("metadata") or {}).get("source") or []) & redo)
                ]
                # **只把新抽出来的并进去**（``merge_atoms``）—— 别用 ``_dedup_atoms(kept + atoms)``：
                # 那会把整个库两两重比一遍（O(n²)），而每次 Add 都要跑一次，库大了就是"卡住"。
                write_memories(self.atommem_path, merge_atoms(kept, atoms))

            async with _lock_for(self.user_id):
                await asyncio.to_thread(_merge)

        failed_ids = {memory_id(target) for target in failed}
        if failed_ids and log is not None:
            log.warn(
                f"session {session_index}: {len(failed_ids)}/{len(todo)} 条 message 抽取未成"
                f"（已记账，下次刷新 / 启动补漏会重试）：{', '.join(sorted(failed_ids)[:5])}"
            )
        if blocked and log is not None:
            log.warn(
                f"session {session_index}: {len(blocked)} 条 message 端点不接受其 prompt"
                f"（内容策略 / 非法请求，不再自动重试；换了模型可用 --repair --retry-blocked 重来）："
                f"{', '.join(sorted(blocked)[:5])}"
            )
        return done, failed_ids, blocked



def llm_quiet_logger():
    from ..log import Logger

    return Logger(level="warn")


# ---------------------------------------------------------------------------
# 启动自愈：把 store 根下所有 user 遗留的 pending Add 补完
# ---------------------------------------------------------------------------

def iter_user_ids(root: Path) -> list[str]:
    """扫描 ``root`` 下所有 user 目录，取 ``meta.json`` 里的 ``user_id``（缺 meta 的目录跳过）。

    目录名是 ``slug(user_id)``，故读回 ``user_id`` 才能重建 ``UserStore``（它能算出同一个目录）。
    """
    user_ids: list[str] = []
    if not root.exists():
        return user_ids
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        meta_path = child / META_NAME
        if not meta_path.exists():
            continue
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        user_id = raw.get("user_id") if isinstance(raw, dict) else None
        if isinstance(user_id, str) and user_id:
            user_ids.append(user_id)
    return user_ids


async def recover_all(
    root: Path,
    *,
    config: dict[str, Any],
    client: Any,
    log: Any | None = None,
    gate: Any = None,
) -> int:
    """启动自愈：补完 ``root`` 下每个 user 遗留的 pending Add，返回落定总数。

    ``pending`` 常态为空 —— 此时只是读一遍各 user 的小 ``meta.json``，几乎零成本；只有真被中断
    过的 Add 才会触发 LLM 重跑刷新。单个 user 出错只 warn，不影响其它。

    各 user 之间**并行**（每个 user 有自己的写锁与目录，互不相干）；模型调用共享 ``gate``。
    """
    user_ids = iter_user_ids(root)
    if not user_ids:
        return 0
    results = await asyncio.gather(*(
        _recover_user(user_id, root, config=config, client=client, log=log, gate=gate)
        for user_id in user_ids
    ))
    total = sum(results)
    if total and log is not None:
        log.info(f"启动续跑：共落定 {total} 个被中断的 Add")
    return total


async def _recover_user(
    user_id: str,
    root: Path,
    *,
    config: dict[str, Any],
    client: Any,
    log: Any | None,
    gate: Any = None,
) -> int:
    store = UserStore(user_id, root)
    try:
        return await store.recover(config=config, client=client, log=log, gate=gate)
    except Exception as exc:  # noqa: BLE001 - 单个 user 出错不影响其它
        if log is not None:
            log.warn(f"恢复 user {user_id!r} 失败：{type(exc).__name__}: {exc}")
        return 0


async def repair_all(
    root: Path,
    *,
    config: dict[str, Any],
    client: Any,
    log: Any | None = None,
    gate: Any = None,
    include_blocked: bool = False,
) -> dict[str, int]:
    """**补漏**：把记账里"抽取失败 / 总结为空"的会话重刷一遍。返回统计。

    它和 ``recover_all`` 各管一件事，别混：

    | | 管什么 | 触发条件 |
    |---|---|---|
    | ``recover_all`` | 被中断的 Add（消息可能都没落全） | ``meta.pending`` 里有条目 |
    | ``repair_all`` | 消息都在、只是**抽取那次没成** | ``atom_state.json`` 里有 ``failed`` |

    为什么必须有后者：一条 message 抽取失败（端点建连超时之类）时，Add 请求本身**仍然是成功的**
    —— 消息已可检索、只是少了几条 atom，所以它不会留在 ``pending`` 里，重启的续跑也捞不到它。
    没有补漏，那些 atom 就要等到"同会话的下一次 Add"才补得上，而语料可能再也不会有 Add 了。

    ``include_blocked=True`` 额外重来那些**被端点拒绝**的（内容策略 / 非法请求）—— 默认不动它们，
    因为重发同一个 prompt 还是同样结果；换了端点才值得重来。

    代价：常态下每个 user 只读一个小 ``atom_state.json``，**没有欠账就一个模型调用都不发**。
    各 user 之间并行（各自目录与写锁）；模型调用共享 ``gate``。
    """
    stats = {"users": 0, "sessions": 0, "fixed": 0, "remaining": 0}
    user_ids = iter_user_ids(root)
    if not user_ids:
        return stats
    stats["users"] = len(user_ids)
    results = await asyncio.gather(*(
        _repair_user(user_id, root, config=config, client=client, log=log, gate=gate,
                     include_blocked=include_blocked)
        for user_id in user_ids
    ))
    for due, fixed in results:
        stats["sessions"] += due
        stats["fixed"] += fixed
    stats["remaining"] = stats["sessions"] - stats["fixed"]
    if stats["sessions"] and log is not None:
        log.info(
            f"补漏完成：{stats['fixed']}/{stats['sessions']} 个会话已补全"
            + (f"，{stats['remaining']} 个仍欠账（端点恢复后重启或再跑 --repair）"
               if stats["remaining"] else "")
        )
    return stats


async def _repair_user(
    user_id: str,
    root: Path,
    *,
    config: dict[str, Any],
    client: Any,
    log: Any | None,
    gate: Any = None,
    include_blocked: bool = False,
) -> tuple[int, int]:
    """返回 ``(欠账的会话数, 修好的会话数)``。单个 user 出错只 warn。"""
    store = UserStore(user_id, root)
    try:
        due = await asyncio.to_thread(store.sessions_needing_repair, include_blocked)
        if not due:
            return 0, 0
        return len(due), await store.repair(
            config=config, client=client, log=log, gate=gate, include_blocked=include_blocked,
        )
    except Exception as exc:  # noqa: BLE001 - 单个 user 出错不影响其它
        if log is not None:
            log.warn(f"补漏 user {user_id!r} 失败：{type(exc).__name__}: {exc}")
        return 0, 0
