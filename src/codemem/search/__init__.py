"""search 步骤：question + 两层语料 → evidence.jsonl（能回答该问题的记忆列表）。

入口：``python -m codemem.search``（见 ``__main__.py`` / ``runner.py``）。

模块分工：

    runner.py      编排：建 workspace → 逐 QA 并发跑 agent → 校验 → 落 trajectory
    agent.py       agent loop：工具调用 / 错误重试 / 重复检测 / 上下文压缩 / 无进展检测
    tools.py       read / edit / bash（原子写 + 路径约束 + 截断保护；write 已删，edit 含 append）
    toolcfg.py     configs/tool.json 的加载与 prompt 注入渲染 + 响应解析
    verifier.py    独立判定"这份 evidence 能否回答该问题"（保守默认 + missing[]）
    evidence.py    evidence.jsonl 的解析/校验/规范化 + 进展指纹
    prune.py       `--prune-runs`：清理历史运行目录

**检索靠 grep**：agent 的工具箱就是 read/edit/bash，检索是 `grep` 一条命令。
没有向量索引、没有 embedding 服务、没有外部检索 CLI —— 语料是 JSONL，一行一条记录，
所以 grep 直接可用、行号稳定，可以 `grep -n` 定位再用 `read offset/limit` 精读。
"""

from . import (
    agent,
    evidence,
    prune,
    runner,
    toolcfg,
    tools,
    verifier,
)

__all__ = [
    "agent", "evidence", "prune", "runner", "toolcfg", "tools", "verifier",
]
