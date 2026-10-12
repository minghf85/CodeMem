"""``python -m codemem.api`` —— 启动 uvicorn 承载封装 API。

    python -m codemem.api                       # 默认 0.0.0.0:8000
    python -m codemem.api --port 9000
    python -m codemem.api --config configs/api.yaml --log-level debug
    python -m codemem.api --recover             # 只跑一次启动自愈（补完被中断的 Add）后退出
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
        help="不启服务，只跑一次启动自愈（补完被中断的 Add）后退出",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config or None)
    if args.log_level:
        section = dict(config.get("log") or {})
        section["level"] = args.log_level
        config["log"] = section

    if args.recover:
        import asyncio

        from . import store as store_mod
        from .server import ApiState

        async def _run() -> int:
            state = ApiState(config)
            try:
                return await store_mod.recover_all(
                    state.store_root, config=state.config,
                    client=state.add_client, log=state.log,
                )
            finally:
                await state.aclose()

        total = asyncio.run(_run())
        print(f"续跑完成：落定 {total} 个被中断的 Add")
        return

    import uvicorn

    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
