"""search 步骤：question + 两层语料 → evidence.jsonl。

对每条 QA，一个 agent 在私有工作目录里用 ``read``/``write``/``edit``/``bash``
（bash 的核心是 ``jq`` 与两条终端命令 ``search`` / ``timecalc``）自主探索，
把**能回答该问题**的记忆列表写进 ``evidence.jsonl``。写完由 verifier 独立判定够不够，
不够就把缺口喂回去再跑一轮。

语料是**两层**：``session_summaries.jsonl``（粗筛：定位到哪次会话）与 ``sessions.jsonl``
（原始消息：精确措辞与时间锚点）。检索默认走**关键词**，``search --embed`` 才是语义。

产物（``data/search_runs/{experiment}_{ts}/``）::

    {dir}/inputs/            只读输入（硬链的两层语料）+ 预建索引 + search 包装
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
