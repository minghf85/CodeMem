"""封装 API：把 add / search 两半管线包成一个多用户、可持久化、幂等的 HTTP 服务。

外部评测按固定协议打两类请求：

    Add     request_id + user_id + session_id + messages[]  -> {success, ...}
            语义是"消息已持久化且**立即可检索**"；重试时 request_id 不变，必须幂等。
    Search  query (+ options) + user_id + top_k             -> {"data": [{id, content, score, created_at}]}

模块分工：

    models.py     协议模型（pydantic）+ ContentPart 归一化
    store.py      每 user_id 一个目录：追加消息、request_id 去重、物化两层语料、刷新跨会话索引
    retriever.py  复用 search 的 code-agent 管线：query -> evidence.jsonl -> data[]
    server.py     FastAPI app：POST /add、POST /search、GET /health
    __main__.py   python -m codemem.api 启动 uvicorn

设计上**不改动任何现有 search/add 代码**：存储刻意沿用 `session_template` 与 `session_1_3` 这套
msg_id 约定，于是 agent 的 prompt、evidence 的 `source` 校验、`date -d` 时间锚点全部照常工作。
"""
