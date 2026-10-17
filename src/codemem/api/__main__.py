"""``python -m codemem.api`` —— 启动 uvicorn 承载封装 API。

    python -m codemem.api                       # 默认 0.0.0.0:8000
    python -m codemem.api --port 9000
    python -m codemem.api --config configs/api.yaml --log-level debug
    python -m codemem.api --recover             # 只跑一次启动自愈（补完被中断的 Add）后退出
    python -m codemem.api --repair              # 只跑一次补漏（重抽上次失败的会话）后退出

启动时（lifespan）会自动做这两件事：续跑**阻塞**做（快，且不补完就不该对外服务），
补漏**放后台**做（可能要调几分钟模型，见 `server._repair_in_background`）。关服务时补漏任务
先被取消、再关模型客户端。
"""

from __future__ import annotations

import argparse

from .server import create_app, load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.api",
        description="启动 CodeMem 封装 API（Add / Search）",
    )
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    parser.add_argument("--config", default="", help="配置（默认 configs/api.yaml）")
    parser.add_argument("--log-level", default="", help="覆盖日志级别（debug/info/warn/error）")
    parser.add_argument(
        "--recover", action="store_true",
        help="不启服务，只跑一次启动自愈（补完被中断的 Add）后退出。"
             "**请在服务没在跑时用** —— 它用自己的锁，和正在服务的进程同库双写会互相覆盖",
    )
    parser.add_argument(
        "--repair", action="store_true",
        help="不启服务，只跑一次补漏（重抽上次失败的会话）后退出；可与 --recover 同用。"
             "同样请在服务没在跑时用（重启服务也会自动做这件事）",
    )
    parser.add_argument(
        "--retry-blocked", action="store_true",
        help="配合 --repair：连端点不接受其 prompt 的那些（内容策略 / 非法请求）也重来一遍。"
             "默认不动它们 —— 重发同一个 prompt 还是同样结果；换了不拦内容的端点后再用这个开关",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config or None)
    if args.log_level:
        section = dict(config.get("log") or {})
        section["level"] = args.log_level
        config["log"] = section

    if args.recover or args.repair:
        import asyncio

        from . import store as store_mod
        from .server import ApiState

        async def _run() -> tuple[int, dict[str, int]]:
            state = ApiState(config)
            try:
                settled = 0
                stats: dict[str, int] = {}
                if args.recover:
                    settled = await store_mod.recover_all(
                        state.store_root, config=state.config,
                        client=state.add_client, log=state.log, gate=state.add_gate,
                    )
                if args.repair:
                    stats = await store_mod.repair_all(
                        state.store_root, config=state.config,
                        client=state.add_client, log=state.log, gate=state.repair_gate,
                        include_blocked=args.retry_blocked,
                    )
                return settled, stats
            finally:
                await state.aclose()

        settled, stats = asyncio.run(_run())
        if args.recover:
            print(f"续跑完成：落定 {settled} 个被中断的 Add")
        if args.repair:
            print(
                f"补漏完成：{stats.get('fixed', 0)}/{stats.get('sessions', 0)} 个会话已补全"
                + (f"，{stats.get('remaining', 0)} 个仍欠账（端点恢复后再跑一次）"
                   if stats.get("remaining") else "")
            )
        return

    # uvloop（装了就用）：32 路并发 + 大量子进程（agent 的 grep）时，事件循环本身不再是瓶颈。
    # 没装就静默跳过 —— 它是可选的加速件，不是依赖，**绝不能拦住启动**（故连异常类型也不挑）。
    try:
        import uvloop

        uvloop.install()
    except Exception:  # noqa: BLE001
        pass

    import uvicorn

    app = create_app(config)
    server_opts = config.get("server")
    if not isinstance(server_opts, dict):
        server_opts = {}
    uvicorn.run(
        app, host=args.host, port=args.port, log_level="warning",
        # 队列上限（configs/api.yaml 的 server.limit_concurrency）：0/缺省 = 不设限，
        # 超限的请求直接被拒（503）而不是排队等一个要跑几十秒的 agent loop。
        limit_concurrency=int(server_opts.get("limit_concurrency", 0) or 0) or None,
        timeout_keep_alive=float(server_opts.get("timeout_keep_alive", 75) or 75),
        # **收尾的上界**：Ctrl-C / SIGTERM 之后，等在飞的请求最多这么久，到点就强退。
        # 0/缺省 = uvicorn 默认（**无限等**）—— 那就是"按了二十次 Ctrl-C 还在跑"的原因：
        # 一个卡在模型调用里的请求能把优雅关闭拖住任意久。
        timeout_graceful_shutdown=int(server_opts.get("timeout_graceful_shutdown", 0) or 0) or None,
    )


if __name__ == "__main__":
    main()
