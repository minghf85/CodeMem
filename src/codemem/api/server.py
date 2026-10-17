"""FastAPI 封装：``POST /add`` 与 ``POST /search``（挑战赛协议）。

端点契约见 README「封装 API」一节。四条贯穿全局的约定：

- **Add 幂等**：``request_id`` 是幂等键。重试时同一个 id 不能再写一次 —— 由
  ``store.UserStore.append`` 里的 applied_request_ids 保证。
- **Add 可中断续跑**：写前日志（``meta.pending``）保证**任意时刻被 kill 都不会丢/重**——
  启动时（lifespan）与每次同 user 请求都会把被中断的 Add 补完（``store.recover_all``）。
- **Search 的 ``data`` 永不为缺失**：无记忆 / 无证据一律返回 ``{"data": []}``。
- **id 原样回显**：Add 的响应把 ``request_id`` / ``user_id`` / ``session_id`` 原样送回。

启动：

    python -m codemem.api --port 8000
    # 或
    uvicorn codemem.api.server:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .. import llm
from ..io import PROJECT_ROOT
from ..log import Logger
from . import retriever
from . import store as store_mod
from .models import AddRequest, AddResponse, SearchRequest, SearchResponse, normalize_content
from .store import UserStore

CONFIG_FILE = PROJECT_ROOT / "configs" / "api.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    # 离线配置来源：API 的模型与行为**直接复用**它们（见 load_config / _merge_offline_configs）
    "search_config": "configs/search.yaml",
    "add_config": "configs/add.yaml",
    "tool_config": "configs/tool.json",
    # API 自有：存储 / 并发 / 检索默认
    "store_dir": "data/api_store",
    "concurrency": 32,
    "add_concurrency": 32,
    "top_k": 100,
    "log": {"level": "info", "output": "data/api_runs"},
    # 下面这些会被离线配置覆盖（这里给缺省，离线文件缺失时兜底）。
    "generator": {"base_url": "http://127.0.0.1:30000/v1", "api_key": "sglang", "model": "local",
                  "temperature": 0.2, "max_tokens": 4096, "enable_thinking": False},
    "index": {"base_url": "http://127.0.0.1:30000/v1", "api_key": "sglang", "model": "local",
              "temperature": 0.2, "max_tokens": 4096, "enable_thinking": False},
    "add_max_steps": 8,
    "timeout": 300,
    "connect_timeout": 10.0,
    "max_retries": 3,
    "backoff_base": 2.0,
    "rate_limit_backoff": 10.0,
    "max_backoff": 60.0,
    "backoff_jitter": 0.2,
}

#: search.yaml 里属于 search agent（API 的 Search 走 agent loop）的键 —— 提到 API 顶层。
_SEARCH_TOP_LEVEL_KEYS = (
    "max_steps", "no_progress_patience", "max_verify_rounds", "observation_max_chars",
    "write_nudge_after", "evidence_soft_limit", "context_max_chars", "context_keep_recent",
    "bash_timeout", "repeat_limit", "max_parse_retries", "tool_failure_notice_after",
    "timeout", "connect_timeout", "max_retries",
    "backoff_base", "rate_limit_backoff", "max_backoff", "backoff_jitter",
)
#: add.yaml 里属于 add 侧（summem + atommem pipeline）模型/行为的键 —— 放进 API 的 ``index`` 段。
_ADD_INDEX_KEYS = (
    "model", "base_url", "api_key", "temperature", "max_tokens", "enable_thinking",
    "context_window", "timeout", "connect_timeout", "max_retries", "backoff_base",
    "rate_limit_backoff", "max_backoff", "backoff_jitter",
)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """读 ``configs/api.yaml``，再把离线 ``search.yaml`` / ``add.yaml`` 并进来（缺省值兜底）。

    **``api.yaml`` 显式写了哪个键，就由它说了算** —— 离线配置只填它没写的那些。否则
    "把 API 的并发调到 32"这种话就永远说不出口：``search.yaml`` 里同名的 ``concurrency``
    会在合并时把它覆盖回去（那个键在离线是"每条 QA 一个 agent loop 的 QA 级并发"，
    与"API 同时接多少条 query"是两件事，只是恰好同名）。
    """
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    explicit: set[str] = set()
    target = Path(path) if path else CONFIG_FILE
    if target.exists():
        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                explicit.add(key)
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                    config[key].update(value)
                else:
                    config[key] = value
    _merge_offline_configs(config, explicit)
    return config


def _merge_offline_configs(config: dict[str, Any], explicit: set[str] | None = None) -> None:
    """把离线 ``search.yaml`` / ``add.yaml`` 的模型与行为并进 API 配置（就地修改）。

    只在离线文件**确实给了**该键、且 ``api.yaml`` **没显式写**时才覆盖 —— 文件缺失 / 少字段时
    保留 api.yaml 与默认值，不会因为读不到文件就把 API 打回"本地模型"。
    """
    from ..add import atommem as atommem_mod
    from ..search import runner as search_runner

    explicit = explicit or set()

    search_path = _resolve_path(config.get("search_config") or "configs/search.yaml")
    if search_path.exists():
        search_cfg = search_runner.load_config(search_path)
        if isinstance(search_cfg.get("generator"), dict):
            config["generator"] = dict(search_cfg["generator"])
        for key in _SEARCH_TOP_LEVEL_KEYS:
            if key in search_cfg:
                config[key] = search_cfg[key]
        if search_cfg.get("concurrency") is not None and "concurrency" not in explicit:
            config["concurrency"] = search_cfg["concurrency"]

    add_path = _resolve_path(config.get("add_config") or "configs/add.yaml")
    if add_path.exists():
        add_cfg = atommem_mod.load_config(add_path)
        index = dict(config.get("index") or {})
        for key in _ADD_INDEX_KEYS:
            if add_cfg.get(key) is not None:
                index[key] = add_cfg[key]
        config["index"] = index
        if add_cfg.get("max_steps") is not None:
            config["add_max_steps"] = add_cfg["max_steps"]
        # ``max_input_tokens``（atommem 抽取的 prompt 预算）不属于模型调用参数，单独搬过来，
        # 否则 API 侧的 add 会用默认值跑 —— 而"输入超上下文"正是它要防的。
        if add_cfg.get("max_input_tokens") is not None:
            index["max_input_tokens"] = add_cfg["max_input_tokens"]
        if add_cfg.get("context_window") is not None:
            index["context_window"] = add_cfg["context_window"]
        if add_cfg.get("concurrency") is not None and "add_concurrency" not in explicit:
            config["add_concurrency"] = add_cfg["concurrency"]


# ---------------------------------------------------------------------------
# app 状态
# ---------------------------------------------------------------------------

class ApiState:
    """进程级共享状态：配置、日志、模型客户端、并发闸门。

    客户端**长驻复用**（连接池 / keep-alive）；并发用 asyncio.Semaphore —— API 是单进程多协程，
    三个信号量分工明确：

    | 闸门 | 管什么 |
    |---|---|
    | ``search_sem`` | 同时最多跑多少条 Search（每条一个 agent loop） |
    | ``add_sem`` | 同时最多接多少个 Add 请求（一次 summem + 该会话逐 message 的抽取） |
    | ``add_gate`` | **进程级**的 add 侧**模型调用**上限 —— Add 内部还会按 message 并发（一次 Add 可能几十条 message、每条一次调用），只看 ``add_sem`` 会把真实并发放大几十倍 |

    ``add_gate`` 是唯一能真正约束住"打到模型端点的并发请求数"的那个：它按**每次
    chat completion** 计数，被 ``Add``→summem / atommem 整条链路共用（见
    ``atommem_pipeline._call_model``）。
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.log = Logger.from_config(config, prefix="api")
        self.client = llm.make_client(_client_config(config))
        self.add_client = llm.make_client(_client_config(config, section="index"))
        self.store_root = _resolve_dir(config.get("store_dir", "data/api_store"))
        self.tool_config = _resolve_path(config.get("tool_config", "configs/tool.json"))
        search_limit = max(1, int(config.get("concurrency", 32)))
        add_limit = max(1, int(config.get("add_concurrency", 32)))
        # 模型调用闸门的额度**可以**单独配（`model_concurrency`）：Add 的并发上限 = "同时接几个
        # 请求"，而真正决定"同时开几条到端点的连接"的是这里。端点在建连阶段会丢 SYN 时，就把
        # 它调小（如 16），而不必牺牲"接得住 32 个并发 Add"。
        model_limit = max(1, int(config.get("model_concurrency") or add_limit))
        # 补漏（启动后台任务 / --repair）**单独一把小闸门**：它跑得久、又是后台活，绝不能把
        # 32 个模型额度全占了 —— 否则服务一起来，前几分钟所有 Add/Search 都在等它的名额，
        # 看起来就是"卡住了"。
        repair_limit = max(1, int(config.get("repair_concurrency") or max(1, model_limit // 4)))
        self.search_sem = asyncio.Semaphore(search_limit)
        self.add_sem = asyncio.Semaphore(add_limit)
        self.add_gate = asyncio.Semaphore(model_limit)
        self.repair_gate = asyncio.Semaphore(repair_limit)

    async def aclose(self) -> None:
        await self.client.close()
        await self.add_client.close()


async def _repair_in_background(state: "ApiState") -> None:
    """启动后的**补漏**（后台任务）：把上次抽取失败的会话重刷一遍。

    为什么是后台而非阻塞启动：补漏要调模型（可能几分钟），而 /health 与服务本身不该等它。
    取消/异常都只记日志 —— 它绝不能影响服务运行（`repair_all` 内部已经逐 user 兜异常）。

    **用它自己的小闸门**（``repair_gate``）而不是 ``add_gate``：后台活不该占满模型额度，
    否则服务刚起来那几分钟，前台的 Add/Search 都在排队等它，看起来就是"卡住了"。
    """
    try:
        await store_mod.repair_all(
            state.store_root, config=state.config, client=state.add_client,
            log=state.log, gate=state.repair_gate,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        state.log.warn(f"启动补漏失败：{type(exc).__name__}: {exc}")


def _resolve_dir(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_path(value: str | Path) -> Path:
    return _resolve_dir(value)


def _client_config(config: dict[str, Any], section: str = "generator") -> dict[str, Any]:
    """把某个模型子段 + 顶层重试参数合成 ``llm.make_client`` 能吃的配置。"""
    merged: dict[str, Any] = {}
    sub = config.get(section) or {}
    if isinstance(sub, dict):
        merged.update(sub)
    for key in ("timeout", "max_retries", "backoff_base", "rate_limit_backoff",
                "max_backoff", "backoff_jitter"):
        if key in config:
            merged[key] = config[key]
    return merged


# ---------------------------------------------------------------------------
# app 工厂
# ---------------------------------------------------------------------------

def create_app(config: dict[str, Any] | None = None) -> FastAPI:
    """建 ``FastAPI`` app。测试用 ``create_app(cfg)``，生产用模块级 ``app``。"""
    cfg = config or load_config()
    # 状态**立即**建好（而不是等 startup 钩子）—— 这样 app.state.api 在任何时刻都可用，
    # 测试与工具无需进入 lifespan 上下文就能读到它。lifespan 只负责关闭客户端。
    state = ApiState(cfg)

    @asynccontextmanager
    async def lifespan(app_: FastAPI):
        """启动自愈（续跑 + 补漏）；关闭时先停掉补漏任务，再清理模型客户端。"""
        # ⓪ 线程池按负载调大：整文件读写 / 去重 / 语料派生结果的构建都挪到了线程里
        # （`asyncio.to_thread`），默认池只有 min(32, CPU+4) 个工人 —— 32 路并发时一排队，
        # **所有**请求都会跟着慢下来（线程池是全局共享的）。
        workers = int(cfg.get("thread_workers") or max(8, 2 * max(
            int(cfg.get("concurrency", 32)), int(cfg.get("add_concurrency", 32)))))
        try:
            asyncio.get_running_loop().set_default_executor(
                ThreadPoolExecutor(max_workers=min(256, workers), thread_name_prefix="codemem")
            )
        except Exception as exc:  # noqa: BLE001 - 调不动就算了，绝不该挡住启动
            state.log.warn(f"线程池调整失败（用默认的）：{type(exc).__name__}: {exc}")

        # ① 续跑：pending 常态为空 → 只读一遍各 user 的小 meta.json；只有真被 kill 过的 Add
        # 才会触发 LLM 重跑刷新。整段兜住异常，恢复失败绝不能挡住启动。
        try:
            await store_mod.recover_all(
                state.store_root, config=state.config, client=state.add_client,
                log=state.log, gate=state.add_gate,
            )
        except Exception as exc:  # noqa: BLE001
            state.log.warn(f"启动续跑失败：{type(exc).__name__}: {exc}")

        # ② 补漏：把"消息都在、只是那次抽取失败"的会话重刷一遍（`recover_all` 捞不到它们 ——
        # 那条 Add 是**成功**的，不在 pending 里）。常态下一个模型调用都不发。
        # **放后台**：它可能等模型等几分钟，不该压着启动、也不该让 /health 迟迟不返回。
        repair_task: asyncio.Task | None = None
        if bool(state.config.get("repair_on_startup", True)):
            repair_task = asyncio.create_task(_repair_in_background(state))
        try:
            yield
        finally:
            # **收尾要有上界**：Ctrl-C / SIGTERM 之后如果这里无限等，进程就永远退不掉
            # （实测：按了二十次 Ctrl-C 还在跑）。补漏是后台活，等不到就直接放它去死。
            if repair_task is not None:
                repair_task.cancel()
                try:
                    await asyncio.wait_for(repair_task, timeout=5.0)
                except asyncio.CancelledError:
                    pass
                except asyncio.TimeoutError:
                    state.log.warn("补漏任务 5 秒内没停下（线程里的活取消不掉），直接退出")
                except Exception as exc:  # noqa: BLE001
                    state.log.warn(f"补漏任务退出时报错：{type(exc).__name__}: {exc}")
            await state.aclose()

    app = FastAPI(title="CodeMem API", version="0.1.0", lifespan=lifespan)
    app.state.api = state

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/add")
    async def add(request: Request) -> JSONResponse:
        """接收一批消息并持久化；返回前确保它们**立即可检索**。

        幂等由 store 保证；这里只做协议层的解析/校验与错误兜底。
        """
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse({"detail": "request body must be JSON"}, status_code=400)
        try:
            parsed = AddRequest.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 - pydantic 的校验错误统一成 422
            return JSONResponse({"detail": str(exc)}, status_code=422)

        store = UserStore(parsed.user_id, state.store_root)
        messages = [
            {"role": m.role, "content": normalize_content(m.content), "timestamp": m.timestamp}
            for m in parsed.messages
        ]
        # 原始请求留痕（排错/分析）：存归一化后的 messages（原话 + role + 毫秒时间戳）。
        request_record = {
            "endpoint": "add",
            "user_id": parsed.user_id,
            "request_id": parsed.request_id,
            "session_id": parsed.session_id,
            "messages": messages,
        }
        async with state.add_sem:
            try:
                summary = await store.append(
                    request_id=parsed.request_id,
                    session_id=parsed.session_id,
                    messages=messages,
                    config=state.config,
                    client=state.add_client,
                    log=state.log,
                    # 进程级的**模型调用**闸门：一次 Add 会按 message 并发抽取，只有它能把
                    # "打到端点的并发请求数"真正压在 add_concurrency 以内。
                    gate=state.add_gate,
                )
            except Exception as exc:  # noqa: BLE001 - 不让异常穿透成 500 堆栈
                state.log.error(f"Add 失败：{type(exc).__name__}: {exc}")
                await store.append_request(
                    {**request_record, "result": {"success": False,
                                                  "error": f"{type(exc).__name__}: {exc}"}}
                )
                return JSONResponse({"detail": "add failed", "success": False}, status_code=500)

        await store.append_request({**request_record, "result": {"success": True, **summary}})
        return JSONResponse(
            AddResponse(
                success=True,
                request_id=parsed.request_id,
                user_id=parsed.user_id,
                session_id=parsed.session_id,
            ).model_dump()
        )

    @app.post("/search")
    async def search(request: Request) -> JSONResponse:
        """按 query 检索该 user 的记忆，返回排好序的 ``data[]``（空则 ``[]``）。"""
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse({"detail": "request body must be JSON"}, status_code=400)
        try:
            parsed = SearchRequest.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"detail": str(exc)}, status_code=422)

        query = normalize_content(parsed.query)
        top_k = parsed.top_k if parsed.top_k is not None else int(state.config.get("top_k", 100))
        store = UserStore(parsed.user_id, state.store_root)
        raw_query = payload.get("query")     # 原始 query（可能是 str 或 ContentPart[]，供留痕）

        log = state.log.bind(f"search:{store.user_id}")
        data: list[Any] = []
        run_dir: Any = None
        async with state.search_sem:
            try:
                data, run_dir = await retriever.search(
                    store=store,
                    query=query,
                    config=state.config,
                    client=state.client,
                    tool_config=state.tool_config,
                    log=log,
                    top_k=top_k,
                    options=parsed.options,
                    raw_query=raw_query,
                )
            except Exception as exc:  # noqa: BLE001 - 单次检索失败返回空结果，不 500
                log.error(f"Search 失败：{type(exc).__name__}: {exc}")
                data = []

        # 原始请求留痕：search 的 query 本身（"连问题是什么都不知道"就是缺它）。无记忆时也记。
        await store.append_request({
            "endpoint": "search",
            "user_id": parsed.user_id,
            "query": query,
            "query_raw": raw_query,
            "options": parsed.options,
            "top_k": top_k,
            "result": {
                "data_count": len(data),
                "run_dir": str(run_dir) if run_dir else "",
            },
        })
        return JSONResponse(SearchResponse(data=data).model_dump())

    return app


# 模块级 app（``uvicorn codemem.api.server:app`` 用）
app = create_app()
