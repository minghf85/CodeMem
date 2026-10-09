"""add 步骤：原始对话 → msgmem → summem → atommem（三层记忆）。

入口：``python -m codemem.add``（见 ``__main__.py``）。

    msgmem.py    原始对话 → msgmem.jsonl（type=msg 的原始消息，纯 CPU，会话时间戳是锚点）
    summem.py    逐 session 一次 LLM → summem.jsonl（type=summary 的会话总结）
    atommem.py   逐 message 的 **pipeline** 抽取 → atommem.jsonl（type=atom 的原子事实）
                 （抽取主体在 atommem_pipeline.py；逐消息 agent 变体在 add-agent 分支）
    memory.py    三个文件的公共读/写/分组工具

**为什么是"三层记忆 + pipeline 逐消息抽取"**。原子记忆（atommem）是检索的主语料：一条 atom
就是一个自足、指代消解、时间锚定的事实，grep 命中的就是能回答问题的原料，同一实体跨会话
都写成同一个真名（grep 即跨会话）。msgmem 保留原话（补具体答案与溯源），summem 提供会话
粒度的路标。三条记忆链格式统一（``memory_template.jsonl``），search 用同一套 grep/read 检索。

**atommem 的核心原理**：逐个 message 处理。后面的 message 的信息可能受前面内容影响、可能
跨 session 影响，所以处理一条 message 时，agent 需要看到前面**相关的**信息才能把指代消解对、
把隐含事实补全 —— 于是它是一个 agent loop（默认 ``max_step=8``），而不是单次调用。

时间的三条硬约束（贯穿三层）：**自足**（绝对时间或带绝对锚点的相对时间，禁止无锚点相对词）、
**精度只减不增**（源说"去年"就只到年）、**指代消解成真名/真实地点**。
"""

from . import atommem, memory, msgmem, summem

__all__ = ["atommem", "memory", "msgmem", "summem"]
