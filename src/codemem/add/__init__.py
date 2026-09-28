"""add 步骤：原始对话 → session 记录 → 两层 summary。

入口：``python -m codemem.add``（见 ``__main__.py``）。

    session.py   原始对话 → sessions.jsonl（session_template 格式，纯 CPU）
    summary.py   两层 summary：session summary → speakers summary

**为什么用两层 summary 而不是原子记忆**：原子把会话结构打碎了 —— 孤立事实无法回答
"他们什么时候认识的""这段关系怎么发展的"。session summary 保留"这次谈了什么、什么变了"；
speakers summary 再串成整段关系的轨迹。层级也给检索一个由粗到细的入口。

（旧的 ``atommem``（原子抽取）与 ``dpo``（为抽取模型造偏好数据）已删除 ——
见 commit b1b5283 可取回。）
"""

from . import session, summary

__all__ = ["session", "summary"]
