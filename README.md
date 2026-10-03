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

CodeMem 的答案是让一个 **code agent 自己检索与推导**：给它一份会话清单和 `grep`/`read` 两个
工具，**由它自己决定**这题该关键词检索（`grep` 一个词、拿新词再 `grep`）还是通篇读（把那几个
会话文件从头读到尾）—— 用 `date -d` 做时间算术，把结论**逐条写进** `evidence.jsonl`。跑完由一个
独立的 verifier 判定"够不够"，不够就把缺口喂回去再跑一轮，最后产出一份**能直接回答问题的
证据**。**没有流程图：从哪找、找多少、什么时候停，都是模型自己的判断。**

**零外部依赖**：没有向量索引、没有 embedding 服务、没有检索 CLI。整条链路只在调模型时联网。

---

## 架构

四个步骤，依赖方向严格单向，步骤之间**只通过数据文件耦合**（不互相 import）：

```
add ──► sessions/session_N.jsonl   （原始措辞 → 消解后上下文无关）
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
| **add** | 一段对话记忆 → 消解后的会话文件（时间锚定 + 指代消解，上下文无关） | `python -m codemem.add` |
| **search** | question + `sessions/` 语料 → `evidence.jsonl`（朴素 agent loop：检索 → 取证 → 验证） | `python -m codemem.search` |
| **answer** | question + `evidence.jsonl` → 答案（只用证据，不检索） | `python -m codemem.answer` |
| **eval** | 答案正确性 + 指标（judge / token F1 / evidence recall / 失败归类） | `python -m codemem.eval` |
| **api** | Add / Search 的 HTTP 封装（多用户隔离、request_id 幂等） | `python -m codemem.api` |

---

## 快速开始

```bash
pip install openai pyyaml tqdm fastapi uvicorn      # Python 3.12+（grep / date 用系统自带命令）
```

### 1. 构建记忆（add）

```bash
python -m codemem.add                       # 全部目录
python -m codemem.add Caroline_Melanie      # 指定目录
```

产出 `data/{speaker_a}_{speaker_b}/sessions/session_N.jsonl`（**一次会话一个文件**）。add 的
第二步 `resolve` 把每条消息改写成**上下文无关**的形式：**时间**锚定成绝对时间（`7 May 2023`）
或带绝对锚点的相对时间（`the week before 9 June 2023`），**指代**消解成真名 —— 粒度严格照源话，
绝不擅自加精度；原话保留在 `source_content` / `source_time`。于是 grep 命中的**就是能回答问题的
原料**，同一实体跨会话都写成同一个真名。**没有摘要层，也没有额外的索引/图层** —— 消解后的消息
本身就是导航。

### 2. 检索证据（search）

```bash
python -m codemem.search --sample Caroline_Melanie --dry-run       # 先看渲染后的 prompt，不调模型
python -m codemem.search --sample Caroline_Melanie --max-qa 3 --max-steps 8   # 小范围试跑
python -m codemem.search --experiment search                        # 全部
```

每条 QA 的产物是 `evidence.jsonl` —— 一份**能直接回答问题**的记忆列表。search 走三步：
agent 自己选检索方式（关键词 grep / 通篇读会话），把结论逐条追加进 `evidence.jsonl`，
最后交给独立 verifier 判一次。

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
  prompts.py                        所有 prompt 常量（含 search 的证据标准 EVIDENCE_STANDARD）
  add/ search/ answer/ eval/        四个步骤（add 含 session / resolve 两步）
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
（返回前已同步消解该会话：时间锚定 + 指代消解）；
**Search 的 `data` 永不为缺失**（无记忆或无证据一律返回 `{"data": []}`）。详细设计与存储
布局见 [`docs/core.md` §4](docs/core.md)。
