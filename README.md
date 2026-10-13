# CodeMem

**面向对话记忆的原子记忆与证据检索系统。** 给定一段两人对话记忆与一个问题，CodeMem 会检索出
**一份能直接回答该问题的记忆列表** —— 不是原始消息的子集，而是把相对时间折算成绝对日期、
消解指代、把散落事实合并后的**推导型证据**。

> **想快速理解整套方法**：读完本文件即可。想深入某一步，看
> [`docs/add.md`](docs/add.md) / [`docs/search.md`](docs/search.md)；系统设计与契约在
> [`docs/core.md`](docs/core.md)；每一轮的实测数字、失败归因与运行手册在
> [`docs/log.md`](docs/log.md)。

---

## 方法：四步，单向依赖

```
add ────► msgmem.jsonl / summem.jsonl / atommem.jsonl   一段对话 → 三层记忆
                             │
                             ▼
search ─────────────────► evidence.jsonl   question + 三层记忆 → 能直接回答它的记忆列表
                             │
                             ▼
answer ─────────────────► answers.jsonl    question + evidence → 答案（不再检索）
                             │
                             ▼
eval ───────────────────► eval.jsonl       判对错 + 各项指标
```

| 步骤 | 一句话 | 入口 |
|---|---|---|
| **add** | 一段对话 → 三层记忆（原话 / 会话总结 / 原子事实） | `python -m codemem.add` |
| **search** | question + 三层记忆 → `evidence.jsonl`（**code agent 自己检索与推导**） | `python -m codemem.search` |
| **answer** | question + `evidence.jsonl` → 答案（**只用证据，不检索**） | `python -m codemem.answer` |
| **eval** | 答案正确性 + 指标（judge / token F1 / evidence recall / 失败归类） | `python -m codemem.eval` |
| **api** | Add / Search 的 HTTP 封装（多用户隔离、`request_id` 幂等、崩溃续跑） | `python -m codemem.api` |

**核心主张**：让一个 **code agent 自己检索与推导** —— 给它一份会话清单和 `grep`/`read`/`edit`
三个工具，**由它自己决定**这题该关键词检索还是通篇读、用 `date -d` 做时间算术，把结论逐条写进
`evidence.jsonl`；最后交给一个独立的 verifier 判"够不够"，不够就把缺口喂回去再跑一轮。
**没有流程图：从哪找、找多少、什么时候停，都是模型自己的判断。**

**零外部依赖**：没有向量索引、没有 embedding 服务、没有检索 CLI。四个步骤之间**只通过数据文件
耦合**（不互相 import），整条链路只在调模型时联网。

---

## 快速开始

```bash
pip install openai pyyaml tqdm fastapi uvicorn      # Python 3.12+（grep / date 用系统自带命令）
```

### 1. 构建记忆（add）

```bash
python -m codemem.add                       # 全部目录（msgmem → summem → atommem）
python -m codemem.add Caroline_Melanie      # 指定目录
python -m codemem.add --stage atommem       # 只跑某一层
```

产出 `data/{speaker_a}_{speaker_b}/` 下的**三层记忆文件**（都是 JSONL，一行一条）：

| 文件 | `type` | 内容 | 生成方式 |
|---|---|---|---|
| `msgmem.jsonl` | `msg` | 每条原始消息（原话），`time` 是会话时间戳（锚点） | 纯 CPU，由 dataset 直接转换 |
| `summem.jsonl` | `summary` | 每个会话一份总结 | LLM，逐 session 一次 |
| `atommem.jsonl` | `atom` | 逐 message 抽出的**原子事实**（≤20 词、自足、单一事实） | **pipeline**（单次 LLM 抽取 + 代码级消解） |

`msgmem` 保留原始措辞与原始时间，是时间/指代的锚点；`atommem` 才是消解后的产物 —— 指代消解成
真名、时间**保留成带锚点的相对表述**（`the week before 9 June 2023`），粒度严格照源话。

> ⚙️ **add 默认走 pipeline**（整目录并行：召回 → 单次 LLM 抽 `{memory, time_expr, role, tag}`
> → 代码级后处理）。419 条 message 从串行 ~45 分钟降到 ~1 分钟。完整说明见
> [`docs/add.md`](docs/add.md)。（逐 message 的 **agent** 变体另存于 `add-agent` 分支，主分支只保留 pipeline。）

### 2. 检索证据（search）

```bash
python -m codemem.search --sample Caroline_Melanie --dry-run       # 先看渲染后的 prompt，不调模型
python -m codemem.search --sample Caroline_Melanie --max-qa 3 --max-steps 8   # 小范围试跑
python -m codemem.search --experiment search                        # 全部
```

每条 QA 的产物是 `evidence.jsonl` —— 一份**能直接回答问题**的记忆列表。search 走三步：
agent 自己选检索方式（关键词 `grep` / 通篇读），在 `inputs/` 下的三层记忆里把结论逐条追加进
`evidence.jsonl`，最后交给独立 verifier 判一次。

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
python -m codemem.api --recover             # 只跑一次启动自愈（补完被中断的 Add）后退出
```

---

## 目录结构

```
configs/            add / search / answer / eval / api 的配置 + tool.json（search agent 上下文的事实源）
data/               原始数据 + 各步骤的产物（三层记忆、证据、运行轨迹）
docs/               文档：add.md / search.md（分步骤）· core.md（设计）· log.md（实验与手册）
src/codemem/
  io.py llm.py log.py dataset.py    共享层
  prompts.py                        所有 prompt 常量
  add/ search/ answer/ eval/        四个步骤
  api/                              Add / Search 的 HTTP 封装
scripts/
  run_eval.py                       一条命令跑通 search → answer → eval
  test_search.py test_api.py        纯 CPU 自测（假模型 + 真工具）
  eval_baseline.py eval_rag.py      历史对照链路：全上下文 baseline / embedding RAG
  eval_atommem.py analyze_atommem_results.py   同上，原子记忆检索对照
```

---

## 自测

纯 CPU、零模型成本、零网络（假模型 + 真工具）：

```bash
python scripts/test_search.py       # 全链路断言（含两项静态检查）
python scripts/test_api.py          # 封装 API 自测：幂等 / 多用户隔离 / msg_id 与时间 / data 映射 / 请求留痕 / 多模态
```

---

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/add.md`](docs/add.md) | **add 步骤**：三层记忆、pipeline（默认）、时间/指代消解 |
| [`docs/search.md`](docs/search.md) | **search 步骤**：code-agent 取证、`grep`/`date -d` 工具契约、verifier |
| [`docs/core.md`](docs/core.md) | **技术设计**：架构、每一步的设计依据、契约与不变量 |
| [`docs/log.md`](docs/log.md) | **实验日志与运行手册**：逐轮实测数字与失败归因、被取代过的设计、里程碑与待办 |
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

四条契约保证：**Add 幂等**（同一 `request_id` 重试不重复写入）；**Add 返回即"立即可检索"**
（返回前已同步重建该会话的 `summem` 与 `atommem`：会话总结 + 逐消息原子事实）；
**Add 可中断续跑**（写前日志保证任意时刻被 kill 都不丢不重，重启后服务自己把被中断的 Add
补完，见 `docs/core.md` §4）；**Search 的 `data` 永不为缺失**（无记忆或无证据一律返回
`{"data": []}`）。

**每个 `user_id` 的目录里都留痕了原始请求**：`requests.jsonl` 记下每一次 `/add`（原始 messages）
与 `/search`（原始 query），每次检索的运行目录里还有 `request.json`。方便事后复盘"这条请求/这条
QA 到底问了什么"。详细设计与存储布局见 [`docs/core.md` §4](docs/core.md)。
