"""search 步骤：question + 两层语料 → evidence.jsonl。

对每条 QA，一个 agent 在私有工作目录里用 ``read``/``write``/``edit``/``bash`` 自主探索，
把**能回答该问题**的记忆列表写进 ``evidence.jsonl``。写完由 verifier 独立判定够不够，
不够就把缺口喂回去再跑一轮。

**检索就是 grep** —— 没有向量索引、没有 embedding 服务、没有检索 CLI，语料是两个 JSONL
文件，一行一条记录。时间算术用 bash 的 ``date -d``。

语料是**两层**：``session_summaries.jsonl``（粗筛：定位到哪次会话）与 ``sessions.jsonl``
（原始消息：精确措辞与时间锚点）。

产物（``data/search_runs/{experiment}_{ts}/``）::

    {dir}/inputs/            只读输入（硬链的两层语料）
    {dir}/qa_{idx}/evidence.jsonl   ← 唯一产物
    {dir}/{dir}.qa_trajectories.jsonl
    {dir}/{dir}.model_calls.jsonl
    summary.json

用法::

    python -m codemem.search --sample Caroline_Melanie --dry-run   # 先看 prompt
    python -m codemem.search --sample Caroline_Melanie --max-qa 3  # 小范围试跑
    python -m codemem.search                                       # 全部目录
"""

from __future__ import annotations

import sys

from . import runner


def main(argv: list[str] | None = None) -> None:
    runner.main(argv)


if __name__ == "__main__":
    main()
