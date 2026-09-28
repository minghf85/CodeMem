"""eval 步骤入口：``python -m codemem.eval``（见 ``runner.py``）。"""

from __future__ import annotations

from . import runner


def main(argv: list[str] | None = None) -> None:
    runner.main(argv)


if __name__ == "__main__":
    main()
