"""``python -m codemem.api`` —— 启动 uvicorn 承载封装 API。

    python -m codemem.api                       # 默认 0.0.0.0:8000
    python -m codemem.api --port 9000
    python -m codemem.api --config configs/api.yaml --log-level debug
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config or None)
    if args.log_level:
        section = dict(config.get("log") or {})
        section["level"] = args.log_level
        config["log"] = section

    import uvicorn

    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
