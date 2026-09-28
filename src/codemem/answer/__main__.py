"""answer 步骤入口：``python -m codemem.answer``（见 ``answer.py``）。"""

from __future__ import annotations

from . import answer


def main(argv: list[str] | None = None) -> None:
    answer.main(argv)


if __name__ == "__main__":
    main()
