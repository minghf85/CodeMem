"""CodeMem：面向代码/对话记忆的记忆系统。

四个核心步骤（严格单向依赖，步骤之间只通过**数据文件**耦合）：

    add     一段对话记忆 → sessions/session_N.jsonl（消解后、上下文无关的会话文件）
    search  question + sessions/ 语料 → evidence.jsonl（能回答该问题的记忆列表）
    answer  question + evidence.jsonl → 答案
    eval    答案正确性 + 各项指标

共享层（被上面四步依赖，自身不依赖任何步骤）：

    io         JSONL 读写、路径解析、memory item 访问器
    llm        chat completion：重试/退避/截断检测
    log        分级日志（终端 + 落盘）
    dataset    LoCoMo 数据集模型：sample/QA/evidence 映射

各步骤入口：

    python -m codemem.add       python -m codemem.search
    python -m codemem.answer    python -m codemem.eval
"""

__all__ = ["dataset", "io", "llm", "log"]
