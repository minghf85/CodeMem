"""add 步骤：原始对话 → session 记录 → session summary。

入口：``python -m codemem.add``（见 ``__main__.py``）。

    session.py   原始对话 → sessions.jsonl（session_template 格式，纯 CPU）
    summary.py   每个 session → 一份 session summary

**为什么是 session summary 而不是原子记忆**：原子把会话结构打碎了 —— 孤立事实无法回答
"他们什么时候认识的""这段关系怎么发展的"。session summary 保留"这次谈了什么、什么变了"，
并给检索一个由粗到细的入口（先定位到哪次会话，再进原文）。

（旧的 ``atommem``（原子抽取）、``dpo``（为抽取模型造偏好数据）、以及顶层
``speakers summary`` 均已删除 —— 前两者见 commit b1b5283 可取回。）
"""

from . import session, summary

__all__ = ["session", "summary"]
