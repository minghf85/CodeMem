"""search 步骤：question + 三层记忆 → evidence.jsonl。

对每条 QA，一个 agent 在私有工作目录里用 ``read``/``edit``/``bash`` 自主探索，
把**能回答该问题**的记忆列表写进 ``evidence.jsonl``。写完由 verifier 独立判定够不够，
不够就把缺口喂回去再跑一轮。

**检索就是 grep** —— 没有向量索引、没有 embedding 服务、没有检索 CLI，记忆是 JSONL
文件，一行一条记录。时间算术用 bash 的 ``date -d``。

语料是 add 阶段产出的**三个记忆文件**（同一格式、不同 ``type``）：``atommem.jsonl``
（提炼过的原子事实，自足、指代消解、时间锚定 —— 首选）、``msgmem.jsonl``（原话，补具体
答案与溯源）、``summem.jsonl``（会话总结，定位"哪次会话谈过这件事"）。三者都可 grep。

产物（``data/search_runs/{experiment}_{ts}/``）::

    {dir}/inputs/{msgmem,summem,atommem}.jsonl   只读输入（硬链的三个记忆文件）
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
