# CodeMem

**面向对话记忆的原子记忆与证据检索系统。** 给定一段两人对话记忆与一个问题，CodeMem 会检索出
**一份能直接回答该问题的记忆列表** —— 不是原始消息的子集，而是把相对时间折算成绝对日期、
消解指代、把散落事实合并后的**推导型证据**。

> 📄 **完整技术报告见 [`docs/core.md`](docs/core.md)** —— 架构、设计依据、每一次实测踩坑与
> 修正、量化评测结论都在那里。本 README 只做快速上手。

---

## 它解决什么

记忆问答的难点不在"存下来"，而在"检索到能回答问题的东西"。传统向量 RAG 有两个结构性缺陷：

- **排序不可解释**：一条"她要去游泳了"与"她参加过哪些活动"的向量距离并不近（实测排名 101），
  把 `-k` 调到 100 也救不回来 —— 相关度天梯本身就不指向它。
- **检索到的不是答案**：语料里说的是 `yesterday`、`she`、`that lake sunrise`，下游拿到这些
  片段仍然无法作答。

CodeMem 的答案是让一个 **code agent** 自己检索与推导：它用 `grep` 在 JSONL 语料上多跳检索、
用 `date -d` 做时间算术，然后**写出一份能直接回答问题的证据**（`evidence.jsonl`），由一个
独立的 verifier 判定"够不够"，不够就把缺口喂回去再跑一轮。

**零外部依赖**：没有向量索引、没有 embedding 服务、没有检索 CLI。整条链路只在调模型时联网。

---

## 架构

四个步骤，依赖方向严格单向，步骤之间**只通过数据文件耦合**（不互相 import）：

```
add ──► sessions.jsonl ──► session_summaries.jsonl
                             │
                             ▼
search ──────────────► evidence.jsonl      （question + 记忆库 → 能回答该问题的记忆列表）
                             │
                             ▼
answer ──────────────► answers.jsonl       （question + evidence → 答案）
                             │
                             ▼
eval ────────────────► eval.jsonl          （判断对错 + 各项指标）
```

| 步骤 | 做什么 | 入口 |
|---|---|---|
| **add** | 一段对话记忆 → 原始消息 + 每个 session 一份 summary | `python -m codemem.add` |
| **search** | question + 两层语料 → `evidence.jsonl`（搜索 agent：grep 多跳检索 + 时间推导） | `python -m codemem.search` |
| **answer** | question + `evidence.jsonl` → 答案（只用证据，不检索） | `python -m codemem.answer` |
| **eval** | 答案正确性 + 指标（judge / token F1 / evidence recall / 失败归类） | `python -m codemem.eval` |
| **api** | Add / Search 的 HTTP 封装（多用户隔离、request_id 幂等） | `python -m codemem.api` |

---

## 快速开始

```bash
pip install openai pyyaml tqdm fastapi uvicorn      # Python 3.12+，另需可执行的 jq
```

### 1. 构建记忆（add）

```bash
python -m codemem.add                       # 全部目录
python -m codemem.add Caroline_Melanie      # 指定目录
```

产出两层语料（`data/{speaker_a}_{speaker_b}/`）：`sessions.jsonl`（原始消息）与
`session_summaries.jsonl`（每个 session 一份概览，供检索由粗到细定位）。

### 2. 检索证据（search）

```bash
python -m codemem.search --sample Caroline_Melanie --dry-run       # 先看渲染后的 prompt，不调模型
python -m codemem.search --sample Caroline_Melanie --max-qa 3 --max-steps 8   # 小范围试跑
python -m codemem.search --experiment search                        # 全部
```

每条 QA 的产物是 `evidence.jsonl` —— 一份**能直接回答问题**的记忆列表。

### 3. 作答与评测（answer / eval）

```bash
python -m codemem.answer                    # 回答最近一次 search 的全部 QA
python -m codemem.eval                      # 判分 + 指标（含 LLM judge）
python -m codemem.eval --no-judge           # 只算 CPU 指标（不调模型）
```

### 一条命令跑通

```bash
python scripts/run_eval.py --quick          # search → answer → eval，打印摘要表
```

### 启动 API 服务

```bash
python -m codemem.api                       # 默认 0.0.0.0:8000
```

---

## 目录结构

```
configs/            add / search / answer / eval / api 的配置 + tool.json（agent 上下文的单一事实源）
data/               原始数据 + 各步骤的产物（语料、证据、运行轨迹）
docs/core.md        技术报告（完整设计依据与实测结论）
src/codemem/
  io.py llm.py log.py dataset.py    共享层
  add/ search/ answer/ eval/        四个步骤
  api/                              Add / Search 的 HTTP 封装
scripts/            run_eval.py（端到端）、test_search.py / test_api.py（纯 CPU 自测）
```

---

## 自测

纯 CPU、零模型成本、零网络（假模型 + 真工具）：

```bash
python scripts/test_search.py       # 全链路断言（含两项静态检查）
python scripts/test_api.py          # 封装 API 自测：幂等 / 多用户隔离 / msg_id 与时间 / data 映射 / 多模态
```

---

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/core.md`](docs/core.md) | **技术报告**：架构、搜索 agent 设计、时间推理、评测结论、全部实测踩坑 |
| 本文件 | 快速上手与项目导航 |

---

## 提交说明与运行指南

本节面向评测方，说明已部署服务、容量与限制，并披露仓库、作者、技术报告与方法改动。

### 已部署的 Add / Search 服务

封装 API 的实现在 `src/codemem/api/`（FastAPI + uvicorn），把本仓库的 add/search 两半管线包成
一个**多用户、按 `user_id` 隔离、按 `request_id` 幂等**的 HTTP 服务。

| 端点 | 请求 | 响应 |
|---|---|---|
| `POST /add` | `{request_id, messages:[{role, content, timestamp?}], user_id, session_id}` | `{success, request_id, user_id, session_id}` |
| `POST /search` | `{query, options?, user_id, top_k?}` | `{data:[{id, content, score, created_at}]}` |
| `GET /health` | —— | `{status:"ok"}` |

```bash
python -m codemem.api --host 0.0.0.0 --port 8000 --config configs/api.yaml
```

三条契约保证：**Add 幂等**（同一 `request_id` 重试不重复写入）；**Add 返回即"立即可检索"**
（返回前已同步刷新 session summary）；**Search 的 `data` 永不为缺失**（无记忆或无证据一律
返回 `{"data": []}`）。详细设计与存储布局见 [`docs/core.md` §4](docs/core.md)。

### 容量与运行限制

- **运行环境**：单进程 Python 3.12 + FastAPI/uvicorn；语料为本地 JSONL（无外部数据库、
  无向量索引、无 embedding 服务）。模型调用通过 OpenAI 兼容端点（`configs/api.yaml` 的
  `generator` / `summary` 段）。
- **并发**：Search 并发上限 `concurrency`（默认 8）、Add 并发上限 `add_concurrency`（默认 8），
  由进程内 `asyncio.Semaphore` 限流。**每条 Add 触发一次 summary 模型调用、每条 Search 触发
  一次（或多次）agent 模型调用** —— 容量瓶颈在模型端点，请据此配置并发。
- **单条 Search 预算**：`max_steps`（默认 30 步）、`max_verify_rounds`（默认 2 轮），
  可用 `configs/api.yaml` 调整。单条 query 的墙钟时间取决于模型吞吐。
- **数据隔离**：按 `user_id` 一目录（`data/api_store/{slug(user_id)}/`），互不可见；
  `top_k` 默认 100，用于截断返回条数（正式外部评测固定 100）。
- **多模态限制**：`content` 支持 `str` 或有序 `ContentPart[]`。数组按原顺序归一化为**一个
  文本串**，文本 part 原样保留、图片 part 降级为 `[image: <url>]` 文本标记。
  ⚠️ **本系统不做图像理解** —— 多模态赛道只保留图片引用；文本 / 代码赛道是完整支持路径。
- **已知运行限制**：Search 依赖模型的工具调用一致性；弱模型偶发不守 JSON 协议（回显 prompt
  文本而非工具调用）。生产评测建议使用稳定的指令模型。

### 披露

- **公开仓库**：<https://github.com/minghf85/CodeMem>
- **原始方法作者**：**minghf85**（本方法即由作者本人设计并实现）。
- **技术报告**：[`docs/core.md`](docs/core.md)。该文件即本方法的技术报告，完整记录架构、
  设计依据、实验过程与评测结论。
- **方法改动**：**无**。仓库已公开于 <https://github.com/minghf85/CodeMem>，本方法为原创方法，
  未基于他人方法做改动。
