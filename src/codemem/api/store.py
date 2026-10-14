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

**写锁**：同一 user 的并发 Add 会交错读改写同一份语料，故按 user_id 分桶一把 ``asyncio.Lock``。
不同 user 互不影响（各自的目录）。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
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
    ) -> dict[str, Any]:
        """把一个 Add 请求写成三层记忆，返回小结 dict。

        ``messages`` 是已归一化的 ``[{"role", "content", "timestamp"}]``。**崩溃一致**：先把
        "意图"写进 ``meta.pending``（写前日志，WAL），再落 ``msgmem``、再刷新记忆、最后落定。
        任意时刻被 kill，重启后都能凭 ``pending`` 把这次 Add 补完（见 ``recover``），且重发同一
        ``request_id`` 不会重复写。

        步骤：

        1. 取该 user 的写锁；**先补完遗留的 pending**（``_recover_locked``）；
        2. **幂等**：``request_id`` 已应用 / 已待续跑 → 直接返回（不重复写）；
        3. 分配会话号与 msg_id，构建本批 msg 记录；
        4. **WAL**：把 ``{session_id, session_index, start, records}`` 写进 ``meta.pending`` 并
           **预留 id 空间**（``sessions[..].count += len``），原子写 meta —— **在碰 msgmem 之前**；
        5. 追加 msg 记录到 ``msgmem.jsonl``（原子整文件重写）；
        6. 重算本会话的 ``summem`` 与 ``atommem``；成功则把 ``request_id`` 落定（pending → applied）；
           刷新失败则**保留 pending**，下次启动 / 下次同 user 请求时自动重试刷新。
        """
        lock = _lock_for(self.user_id)
        async with lock:
            self.root.mkdir(parents=True, exist_ok=True)
            meta = self.meta()

            # ---- 先补完上一个被中断的 Add（幂等；pending 为空时零成本）----
            await self._recover_locked(meta, config=config, client=client, log=log)

            if request_id in meta["applied_request_ids"]:
                if log is not None:
                    log.info(f"Add 幂等命中：request_id={request_id!r} 已应用，跳过写入")
                return {"written": 0, "duplicate": True, "session_index": None}
            if request_id in meta["pending"]:
                # 上一次同 id 的 Add 已落盘消息、只是记忆刷新没跑完（见 _recover_locked）。消息此时
                # 已由上面的恢复落盘，这里不再重复写；刷新留待下次续跑。
                if log is not None:
                    log.info(f"Add 续跑命中：request_id={request_id!r} 待补刷新，消息已落盘")
                return {"written": 0, "duplicate": True, "pending": True, "session_index": None}

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

            # ---- 落 msgmem（原子整文件重写）----
            existing = read_jsonl(self.msgmem_path)
            write_memories(self.msgmem_path, existing + records)

            # ---- 刷新本会话的 summem / atommem；成功才落定 ----
            refreshed = await self._refresh_memories(
                session_index=session_index, config=config, client=client, log=log,
            )
            if refreshed:
                meta["pending"].pop(request_id, None)
                meta["applied_request_ids"] = [*meta["applied_request_ids"], request_id]
                _atomic_write_json(self.meta_path, meta)

            if log is not None:
                log.info(
                    f"Add 写入 {len(records)} 条 -> session {session_index}"
                    f"（共 {entry['count']} 条）形态={batch.family}"
                    f"{' 说话人=' + '/'.join(batch.speakers) if batch.speakers else ''}"
                    f" 记忆刷新={'成功' if refreshed else '待续跑'}"
                )
            return {
                "written": len(records),
                "duplicate": False,
                "session_index": session_index,
                "refreshed": refreshed,
            }

    # -- 中断续跑（恢复）---------------------------------------------------

    async def _apply_pending(
        self,
        meta: dict[str, Any],
        request_id: str,
        config: dict[str, Any],
        client: Any,
        log: Any | None,
    ) -> bool:
        """把一个 pending 的 Add 补完（幂等）。True=已落定，False=刷新未成（留待下次）。

        崩溃点都吃得下：``msgmem`` 已写则按 ``memory_id`` 跳过补写；刷新是"整会话重建"故可重入。
        只在本方法**成功**时改 ``meta``（弹 pending、记 applied）；失败时不动 meta，保持 pending。
        """
        entry = meta.get("pending", {}).get(request_id)
        if not isinstance(entry, dict):
            meta.get("pending", {}).pop(request_id, None)
            return False

        session_id = str(entry.get("session_id") or "")
        session_index = int(entry.get("session_index") or 0)
        start = int(entry.get("start") or 0)
        records = [r for r in (entry.get("records") or []) if isinstance(r, dict)]

        # 1. 确保 msgmem 含这些记录（按 id 去重 —— 崩在 msgmem 写之后时不重复补）
        existing = read_jsonl(self.msgmem_path)
        have = {memory_id(r) for r in existing}
        missing = [r for r in records if memory_id(r) not in have]
        if missing:
            write_memories(self.msgmem_path, existing + missing)

        # 2. 补回（可能缺失/损坏的）会话条目；count 取最大（预留语义，避免 msg_id 撞车）
        sessions_map: dict[str, Any] = meta["sessions"]
        item = sessions_map.get(session_id)
        if not isinstance(item, dict):
            item = {"index": session_index, "count": 0}
            sessions_map[session_id] = item
        item["count"] = max(int(item.get("count", 0)), start + len(records))

        # 3. 刷新本会话记忆；失败则保留 pending，下次再续
        refreshed = await self._refresh_memories(
            session_index=session_index, config=config, client=client, log=log,
        )
        if not refreshed:
            return False

        # 4. 落定：pending -> applied（成功后才动 meta）
        meta["pending"].pop(request_id, None)
        meta["applied_request_ids"] = [*meta["applied_request_ids"], request_id]
        return True

    async def _recover_locked(
        self,
        meta: dict[str, Any],
        *,
        config: dict[str, Any],
        client: Any,
        log: Any | None,
    ) -> int:
        """补完 ``meta.pending`` 里的全部 Add，返回落定条数。单条失败只 warn、不中断。

        每个 entry 每次调用只处理一遍，故刷新持续失败时不会在此处空转（下次启动 / 下次同 user
        请求再重试一次）。有落定才写一次 meta。
        """
        pending = meta.get("pending") or {}
        if not pending:
            return 0
        finalized = 0
        for request_id in list(pending):
            try:
                if await self._apply_pending(meta, request_id, config, client, log):
                    finalized += 1
            except Exception as exc:  # noqa: BLE001 - 单条恢复失败不该影响其它
                if log is not None:
                    log.warn(f"恢复 pending 失败 {request_id!r}：{type(exc).__name__}: {exc}")
        if finalized:
            _atomic_write_json(self.meta_path, meta)
            if log is not None:
                log.info(f"续跑完成：{finalized}/{len(pending)} 个被中断的 Add 已落定")
        return finalized

    async def recover(
        self,
        *,
        config: dict[str, Any],
        client: Any,
        log: Any | None = None,
    ) -> int:
        """补完该 user 遗留的 pending Add（幂等、可重入）。无 pending 时零成本。返回落定条数。"""
        lock = _lock_for(self.user_id)
        async with lock:
            meta = self.meta()
            return await self._recover_locked(meta, config=config, client=client, log=log)

    async def _refresh_memories(
        self,
        *,
        session_index: int,
        config: dict[str, Any],
        client: Any,
        log: Any | None,
    ) -> bool:
        """重算该会话的 summem（一次 LLM）与 atommem（逐 message 的 **pipeline**）。失败不抛。

        summem 见 ``add/summem.py``；atommem 见 ``add/atommem_pipeline.py``（与离线
        ``add --mode pipeline`` 同一条路径）。

        **说话人名单从已落盘的 msgmem 里取**（``adapters.speakers_of``），不使用硬编码的
        ``user``/``assistant``：这份语料里真名在正文前缀里（``Christina Baker:``），适配层已经
        把它搬进了 ``role``，这里照读即可。名单取自落盘数据而不是内存变量，所以**恢复
        （``recover``）重跑时算出来的名单与首次完全一致**。
        """
        add_cfg = _add_config(config)
        try:
            # --- summem：替换该会话那一条 ---
            sess_records = self.session_records(session_index)
            if sess_records:
                speakers = adapters.speakers_of(sess_records)
                summary = await summem_mod.summarize_session(
                    session_index=session_index,
                    session_time=str((sess_records[0].get("metadata") or {}).get("time") or ""),
                    records=sess_records,
                    client=client,
                    config=add_cfg,
                    speakers=speakers,
                )
                existing = read_jsonl(self.summem_path)
                summary_id = f"session_{session_index}_summary"
                existing = [
                    r for r in existing if (r.get("metadata") or {}).get("id") != summary_id
                ]
                write_memories(self.summem_path, existing + [summary])
            # --- atommem：只跑本会话的 message（pipeline）---
            await self._refresh_atoms(
                session_index=session_index, config=add_cfg, client=client, log=log,
            )
        except Exception as exc:  # noqa: BLE001 - 记忆刷新失败不该让 Add 失败
            if log is not None:
                log.warn(f"session {session_index} 记忆刷新失败：{type(exc).__name__}: {exc}")
            return False
        return True

    async def _refresh_atoms(
        self, *, session_index: int, config: dict[str, Any], client: Any, log: Any | None,
    ) -> None:
        """对本会话的每条 message 跑 **atommem pipeline**，重建本会话的 atom。

        与离线 ``add --mode pipeline`` 同一条路径（``atommem_pipeline.run_message_pipeline``）：
        本会话各 message 的抽取**互不依赖**（召回只读 msgmem、id 按各 source 独立分配、去重是
        收尾一次全局 pass），所以并发抽取后**一次性**去重落盘。本会话之外的 atom 原样保留
        （增量：只重建本会话）。
        """
        from ..add.atommem import _dedup_atoms
        from ..add.atommem_pipeline import run_message_pipeline

        msg_records = self.session_records(session_index)
        if not msg_records:
            return
        msgmem = read_jsonl(self.msgmem_path)
        session_speakers = adapters.speakers_of(msg_records)
        my_ids = {memory_id(r) for r in msg_records}
        existing = read_jsonl(self.atommem_path)
        # 保留"本会话之外"的 atom（其 source 与本会话的 message id 不相交）。
        kept = [
            a for a in existing
            if not (set((a.get("metadata") or {}).get("source") or []) & my_ids)
        ]

        gate = asyncio.Semaphore(max(1, int(config.get("concurrency", 8))))

        async def extract(target: dict[str, Any]) -> list[dict[str, Any]]:
            async with gate:
                try:
                    return await run_message_pipeline(
                        target=target, msgmem=msgmem, existing_atoms=[], client=client,
                        config=config, log=log or llm_quiet_logger(),
                        speakers=session_speakers,
                    )
                except Exception as exc:  # noqa: BLE001 - 单条失败不该让整个 add 失败
                    if log is not None:
                        log.warn(
                            f"atom 抽取失败 {memory_id(target)}：{type(exc).__name__}: {exc}"
                        )
                    return []

        groups = await asyncio.gather(*(extract(t) for t in msg_records))
        merged = _dedup_atoms(kept + [a for group in groups for a in group])
        write_memories(self.atommem_path, merged)


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
) -> int:
    """启动自愈：补完 ``root`` 下每个 user 遗留的 pending Add，返回落定总数。

    ``pending`` 常态为空 —— 此时只是读一遍各 user 的小 ``meta.json``，几乎零成本；只有真被中断
    过的 Add 才会触发 LLM 重跑刷新。单个 user 出错只 warn，不影响其它。
    """
    total = 0
    for user_id in iter_user_ids(root):
        store = UserStore(user_id, root)
        try:
            total += await store.recover(config=config, client=client, log=log)
        except Exception as exc:  # noqa: BLE001 - 单个 user 出错不影响其它
            if log is not None:
                log.warn(f"恢复 user {user_id!r} 失败：{type(exc).__name__}: {exc}")
    if total and log is not None:
        log.info(f"启动续跑：共落定 {total} 个被中断的 Add")
    return total
