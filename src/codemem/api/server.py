"""FastAPI 封装：``POST /add`` 与 ``POST /search``（挑战赛协议）。

端点契约见 README「封装 API」一节。三条贯穿全局的约定：

- **Add 幂等**：``request_id`` 是幂等键。重试时同一个 id 不能再写一次 —— 由
  ``store.UserStore.append`` 里的 applied_request_ids 保证。
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
from .models import AddRequest, AddResponse, SearchRequest, SearchResponse, normalize_content
from .store import UserStore

CONFIG_FILE = PROJECT_ROOT / "configs" / "api.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "generator": {"base_url": "http://127.0.0.1:30000/v1", "api_key": "sglang", "model": "local",
                  "temperature": 0.2, "max_tokens": 4096, "enable_thinking": False},
    "summary": {"base_url": "http://127.0.0.1:30000/v1", "api_key": "sglang", "model": "local",
                "temperature": 0.2, "max_tokens": 4096, "enable_thinking": False},
    "store_dir": "data/api_store",
    "tool_config": "configs/tool.json",
    "concurrency": 8,
    "add_concurrency": 8,
    "top_k": 100,
    "timeout": 300,
    "max_retries": 3,
    "backoff_base": 2.0,
    "rate_limit_backoff": 10.0,
    "max_backoff": 60.0,
    "backoff_jitter": 0.2,
    "log": {"level": "info", "output": "data/api_runs"},
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """读 ``configs/api.yaml``，缺省值兜底（与其它步骤同形）。"""
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    target = Path(path) if path else CONFIG_FILE
    if target.exists():
        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                    config[key].update(value)
                else:
                    config[key] = value
    return config


# ---------------------------------------------------------------------------
# app 状态
# ---------------------------------------------------------------------------

class ApiState:
    """进程级共享状态：配置、日志、模型客户端、并发闸门。

    客户端**长驻复用**（连接池 / keep-alive）；并发用 asyncio.Semaphore —— API 是单进程多协程，
    一个信号量即可对 agent loop 与 summary 调用统一限流。
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.log = Logger.from_config(config, prefix="api")
        self.client = llm.make_client(_client_config(config))
        self.add_client = llm.make_client(_client_config(config, section="summary"))
        self.store_root = _resolve_dir(config.get("store_dir", "data/api_store"))
        self.tool_config = _resolve_path(config.get("tool_config", "configs/tool.json"))
        self.search_sem = asyncio.Semaphore(max(1, int(config.get("concurrency", 8))))
        self.add_sem = asyncio.Semaphore(max(1, int(config.get("add_concurrency", 8))))

    async def aclose(self) -> None:
        await self.client.close()
        await self.add_client.close()


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
        """进程关闭时清理模型客户端（连接池要关干净）。"""
        try:
            yield
        finally:
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
        async with state.add_sem:
            try:
                await store.append(
                    request_id=parsed.request_id,
                    session_id=parsed.session_id,
                    messages=messages,
                    config=state.config,
                    client=state.add_client,
                    log=state.log,
                )
            except Exception as exc:  # noqa: BLE001 - 不让异常穿透成 500 堆栈
                state.log.error(f"Add 失败：{type(exc).__name__}: {exc}")
                return JSONResponse({"detail": "add failed", "success": False}, status_code=500)

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

        log = state.log.bind(f"search:{store.user_id}")
        async with state.search_sem:
            try:
                data = await retriever.search(
                    store=store,
                    query=query,
                    config=state.config,
                    client=state.client,
                    tool_config=state.tool_config,
                    log=log,
                    top_k=top_k,
                )
            except Exception as exc:  # noqa: BLE001 - 单次检索失败返回空结果，不 500
                log.error(f"Search 失败：{type(exc).__name__}: {exc}")
                data = []

        return JSONResponse(SearchResponse(data=data).model_dump())

    return app


# 模块级 app（``uvicorn codemem.api.server:app`` 用）
app = create_app()
