"""search 步骤：question + 两层语料 → evidence.jsonl（能回答该问题的记忆列表）。

入口：``python -m codemem.search``（见 ``__main__.py`` / ``runner.py``）。

模块分工：

    runner.py      编排：建 workspace → 逐 QA 并发跑 agent → 校验 → 落 trajectory
    agent.py       agent loop：工具调用 / 错误重试 / 重复检测 / 上下文压缩 / 无进展检测
    tools.py       read / write / edit / bash（原子写 + 路径约束 + 截断保护）
    toolcfg.py     configs/tool.json 的加载与 prompt 注入渲染 + 响应解析
    verifier.py    独立判定"这份 evidence 能否回答该问题"（保守默认 + missing[]）
    evidence.py    evidence.jsonl 的解析/校验/规范化 + 进展指纹
    searchctl.py   `search` 终端命令（关键词检索 CLI，--embed 走语义）+ 建索引入口
    searchmem.py   混合检索：dense + BM25 + tag 三路，RRF 融合（唯一检索实现）
    index.py       检索索引：dense 向量一次性落盘 / 加载 / 打分
    embedder.py    嵌入（分批 + 单批重试）
    timecalc.py    `timecalc` 终端命令（确定性日期算术）
"""

# 刻意**不**在这里 import ``searchctl`` 与 ``timecalc``：它们既是模块、又是
# ``python -m codemem.search.searchctl`` 的入口。包 ``__init__`` 先把它们导进来，
# runpy 再执行同一个模块时就会发一条 RuntimeWarning: "found in sys.modules after import
# of package ... but prior to execution"。
#
# 这条警告本身无害，但**会污染 agent 的观测**：bash 把 stderr 并进 stdout，于是每次
# ``search`` 的每一行输出前面都多一行警告，模型看到的是噪音（实测它照样能读，但白白
# 吃掉上下文，也让观测的"第一行"不再是数据）。需要这两个模块时显式 import 即可
# （``from codemem.search import searchctl`` 仍然可用）。
from . import (
    agent,
    embedder,
    evidence,
    index,
    runner,
    searchmem,
    toolcfg,
    tools,
    verifier,
)

__all__ = [
    "agent", "embedder", "evidence", "index", "runner",
    "searchmem", "toolcfg", "tools", "verifier",
]
