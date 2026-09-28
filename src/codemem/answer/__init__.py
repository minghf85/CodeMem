"""answer 步骤：question + evidence.jsonl → 答案（只用证据，不检索）。

入口：``python -m codemem.answer``（见 ``__main__.py`` / ``answer.py``）。
"""

from . import answer

__all__ = ["answer"]
