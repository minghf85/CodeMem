"""eval 步骤：判断回答是否正确并计算指标（输入是 answer 的产物）。

入口：``python -m codemem.eval``（见 ``__main__.py`` / ``runner.py``）。

    runner.py    评分编排：逐条打分（CPU 指标 + judge）→ 汇总 → 失败归类
    judge.py     LLM-as-a-judge：判断候选答案与参考答案是否语义等价
    metrics.py   纯 CPU 指标：exact match / token F1 / evidence recall
"""

from . import judge, metrics, runner

__all__ = ["judge", "metrics", "runner"]
