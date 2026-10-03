"""search 步骤：question + 会话语料 → evidence.jsonl。

对每条 QA，一个 agent 在私有工作目录里用 ``read``/``edit``/``bash`` 自主探索，
把**能回答该问题**的记忆列表写进 ``evidence.jsonl``。写完由 verifier 独立判定够不够，
不够就把缺口喂回去再跑一轮。

**检索就是 grep** —— 没有向量索引、没有 embedding 服务、没有检索 CLI，语料是 JSONL
文件，一行一条记录。时间算术用 bash 的 ``date -d``。

语料是一个 ``sessions/`` 目录：**每个会话一个文件**，内容已**消解**成上下文无关的形式
（时间是绝对时间或带绝对锚点的相对时间，指代已换回原始含义；原话保留在 ``source_content`` /
``source_time``）。于是 grep 命中的就是能回答问题的原料，同一实体跨会话都写成同一个真名。

产物（``data/search_runs/{experiment}_{ts}/``）::

    {dir}/inputs/sessions/   只读输入（硬链的会话语料，每会话一个文件）
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
