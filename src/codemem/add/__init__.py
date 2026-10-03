"""add 步骤：原始对话 → session 文件 → 消解后的、上下文无关的记忆。

入口：``python -m codemem.add``（见 ``__main__.py``）。

    session.py   原始对话 → sessions/session_{n}.jsonl（原始措辞，纯 CPU）
    resolve.py   逐消息消解 时间 + 指代 → 就地改写 sessions/*.jsonl（上下文无关）

**为什么是"消解"而不是"摘要"或"索引/图"**。摘要是一段浓缩成品，实测会诱导 search agent
不检索直接凭印象作答（152 条 QA 里 150 条一次 grep 都没做）。而 `index.jsonl`（三元组）/
`nodes+edges`（图）都是为"跨会话定位"额外造的一层结构。这一版换个更直接的做法：**把原始
消息本身改写成自足的** —— 时间锚定成绝对时间或带绝对锚点的相对时间，指代消解成原始含义。
于是关键词检索命中的就是能回答问题的原料，同一实体跨会话都写成同一个真名（grep 即跨会话），
通读时每个 session 文件自足、不会"读到后面忘了前面"。

时间的三条硬约束：**自足**（禁止无锚点相对词）、**精度只减不增**（源说"去年"就只到年）、
**区分点/段/大概**（``time_kind`` = point / range / approx）。原话保留在 ``source_content`` /
``source_time``（溯源 + re-resolve 幂等的底）。

（旧的 ``atommem``、``dpo``、``session summary``、顶层 ``speakers summary``、``index.jsonl``、
``nodes.jsonl``/``edges.jsonl`` 均已删除 —— 前两者见 commit b1b5283 可取回。）
"""

from . import resolve, session

__all__ = ["resolve", "session"]
