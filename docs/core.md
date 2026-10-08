# CodeMem 技术报告

本文件是 CodeMem 的**技术报告**：完整记录系统的架构、各步骤的设计依据、每一次实测踩坑与
修正，以及量化的评测结论。它是本项目所有设计决策的单一事实源。

面向使用者与评测方的**快速入口**见 [`../README.md`](../README.md)。

---

## 0. 架构总览

整个架构围绕**四个步骤**组织，依赖方向严格单向，步骤之间**只通过数据文件耦合**（不互相 import）：

```
add ──► msgmem.jsonl / summem.jsonl / atommem.jsonl   （原始话 + 会话总结 + 原子事实）
                             │
                             ▼
search ──────────────► evidence.jsonl          （question + 三层记忆 → 能回答该问题的记忆列表）
                             │
                             ▼
answer ──────────────► answers.jsonl           （question + evidence → 答案）
                             │
                             ▼
eval ────────────────► eval.jsonl / summary.json   （判断对错 + 各项指标）
```

| 步骤 | 做什么 | 入口 |
|---|---|---|
| **add** | 一段对话记忆 → 三层记忆文件（`msg` / `summary` / `atom`） | `python -m codemem.add` |
| **search** | question + 三层记忆 → `evidence.jsonl`（朴素 agent loop：检索 → 取证 → 验证） | `python -m codemem.search` |
| **answer** | question + `evidence.jsonl` → 答案（只用证据，不检索） | `python -m codemem.answer` |
| **eval** | 答案正确性 + 指标（judge / token F1 / evidence recall / 失败归类） | `python -m codemem.eval` |

此外 `src/codemem/api/` 提供一层面向外部评测的 **HTTP 封装（Add / Search）**，把 add/search
两半管线包成一个多用户、幂等的服务（见 §4）。

### 目录结构

```
configs/
  add.yaml      # add 的模型参数（msgmem 阶段纯 CPU；summem 一次/session；atommem 逐消息 agent）
  search.yaml   # search 的 agent/检索/预算参数
  answer.yaml   # answer 的模型参数
  eval.yaml     # eval 的 judge 参数
  tool.json     # search 的工具描述/schema/prompt/示例/检索策略引导（检索 agent 上下文的事实源）
  add_tool.json # add agent 的工具描述/prompt/原子标准（逐消息抽 atommem 用）
  api.yaml      # 封装 API 的模型/存储/预算参数
data/
  correct_locomo10.json        # 原始对话数据（含 QA 与 evidence 标注）
  {speaker_a}_{speaker_b}/
    msgmem.jsonl               # add 产物：逐条原始消息（type=msg，会话时间戳是锚点）
    summem.jsonl               # add 产物：每会话一条总结（type=summary）
    atommem.jsonl              # add 产物：逐消息抽出的原子事实（type=atom）
  search_runs/{experiment}_{ts}/   # search/answer/eval 的运行目录
    {dir}/inputs/                   只读输入（硬链的三层记忆文件）
    {dir}/qa_{idx}/evidence.jsonl   search 的产物
    answers.jsonl                   answer 的产物（该次运行的全部回答）
    eval.jsonl / summary.json       eval 的产物
  eval_runs/                   # 三个 baseline 评测脚本的输出
  api_store/{slug(user_id)}/   # 封装 API 的多用户语料（见 §4）
src/codemem/
  io.py llm.py log.py dataset.py    # 共享层：四个步骤都依赖，自身不依赖任何步骤
  prompts.py                        # 所有 prompt 常量（search 的 EVIDENCE_STANDARD、add 的 ATOM_AGENT_STANDARD）
  add/    msgmem.py summem.py atommem.py memory.py __main__.py
  search/ runner.py agent.py tools.py toolcfg.py verifier.py evidence.py
          prune.py __main__.py
  answer/ answer.py __main__.py
  eval/   runner.py judge.py metrics.py legacy.py __main__.py
  api/    models.py store.py retriever.py server.py __main__.py
scripts/
  run_eval.py                        # 一条命令跑通 search → answer → eval 并打印摘要
  test_search.py                     # 纯 CPU 自测（假模型 + 真工具，零模型成本）
  test_api.py                        # 封装 API 的纯 CPU 自测（幂等 / 隔离 / data 映射）
  eval_baseline.py eval_rag.py eval_atommem.py analyze_atommem_results.py  # baseline 对照（见 §6）
  regenerate_rejected.py transfer_data_to_msswift_standard.py locomo_data_loader.py
```

**共享层（`codemem/` 顶层四个模块）** —— 放在任何一个步骤里都会造成反向依赖，所以提出来：

| 模块 | 职责 |
|---|---|
| `io.py` | JSONL 读写（含"整份美化 JSON"兜底）、路径解析、memory item 访问器、`repair_json` |
| `llm.py` | chat completion：退避/重试/截断检测/qwen3 关思考 |
| `log.py` | 分级日志（终端 + 落盘，支持 `bind` 加前缀） |
| `dataset.py` | LoCoMo 数据模型：sample/QA/evidence 映射（`D1:3` → `session_1_3` 的唯一实现） |
| `prompts.py` | 各步骤的 prompt 常量。search 的**证据标准**（`EVIDENCE_STANDARD`）与 add 的**原子标准**（`ATOM_AGENT_STANDARD`）都在这里 |

### 环境

```bash
pip install openai pyyaml tqdm fastapi uvicorn
```

Python 3.12+。`grep` / `date` 是系统自带命令，agent 直接通过 `bash` 用它们。

---

## 1. add — 把对话记忆添加进来（三层记忆）

```bash
python -m codemem.add                      # msgmem → summem → atommem，全部目录
python -m codemem.add Caroline_Melanie     # 只处理指定目录
python -m codemem.add --stage msgmem       # 只跑第一层（纯 CPU，不调模型）
python -m codemem.add --stage atommem --limit-messages 20   # 只跑前 20 条 message（调试）
```

> **atommem 默认走三阶段 pipeline**（单次 LLM 抽取 + 后处理），这是产出 1000+ 条、judge 最高
> 的那一版。另有并行 code agent（`--mode agent-v2`）与串行 agent（`--mode agent`），用于对比；
> agent 更灵活但 8B 在 loop 里**不稳定**（同一条消息多次跑产出 0/2/4 不等），故不作默认。

**三层，产出三种记忆文件**（都在 `data/{speaker_a}_{speaker_b}/` 下，JSONL 一行一条）：

| 文件 | `type` | 内容 | 生成方式 |
|---|---|---|---|
| `msgmem.jsonl` | `msg` | 每条原始消息（**原话**）；`time` 是会话时间戳，`role` 是说话人真名 | 纯 CPU，由 dataset 直接转换 |
| `summem.jsonl` | `summary` | 每个会话一份总结 | LLM，**逐 session 一次** |
| `atommem.jsonl` | `atom` | 逐 message 抽出的**原子事实**（≤20 词、自足、单一事实） | **并行 add agent 逐 message 处理**（每 message 看到此前全部内容） |

记录格式对齐 `memory_template.jsonl`：

```json
{"memory": "...", "metadata": {"id": "", "type": "msg|summary|atom", "time": "", "role": "",
                               "tag": [], "source": [], "changelog": [{"time": "", "content": ""}]}}
```

**msgmem 是锚点，atommem 是产物。** msgmem 保留原始措辞与原始会话时间戳 —— 它是消解指代与折算
时间的**依据**；atommem 才是消解后的结果（指代消解成真名、时间折算成绝对日期或带绝对锚点的相对
时间，精度只减不增）。summem 居中，是「哪次会话谈过什么」的路标。

三个刻意的取舍（不变）：

- **id 用 1-indexed**（`session_{s}_{i}`，与 LoCoMo 的 `dia_id`（`D1:1`）对齐）—— evidence 映射
  是恒等变换，不需要每处 ±1（那是 bug 的温床）。atom 的 id 是 `session_{s}_{i}_{k}`。
- **`role` 用真实人名**（`Caroline`/`Melanie`）而不是 `speaker1`/`speaker2` —— 指代消解（"she"
  是谁）依赖它，丢掉就等于让 agent 从对话里猜。
- **msgmem 的 `time` 保留原始会话时间戳**（如 `"1:56 pm on 8 May, 2023"`）—— 它是折算相对时间的
  **锚点**；atommem 的 `time` 是消解后的结果。

### 1.1 为什么是"三层记忆 + 并行 add agent"，而不是"单次抽取"

第一版（S14，串行 agent loop）虽然灵活，但 8B 模型在多步交互里把控不住质量：16% 的 atom 仍是
多句、时间表达不稳定（"5 years ago" / "last year" 会直接写进 memory）、tag key 会自造 100+ 个。
中间试过一版**三阶段 pipeline**（召回 → 单次 LLM → 后处理）：稳定、快，但召回是**启发式**的
（只给同会话前 N 条消息），跨会话指代（"the same lake as last time"）消解不了，集合型问题的
成员也收不全 —— 实测 judge 从 60.5% 掉下来。

现在的做法把两者的优点合起来：**保留 agent（能自己 grep 全部前文），但改成并行 + 代码级兜底**。

```
每条 message 一个独立 agent（彼此并行，不依赖对方产出）
  ├─ 上下文：这条 message **之前全部内容**（前面所有 session + 本 session 前面的 message）
  │          —— 不靠启发式窗口，agent 自己 grep msgmem / summem 决定读什么
  ├─ 工具：read / edit / bash（与 search 同一套 tools.py，catalog 渲染同一套 prompt）
  ├─ 产出：写进独立文件 {msg_id}_atoms.jsonl（parallel 无写冲突，最后合并去重）
  └─ prompt 里 4 条硬规则：memory 不含时间词 / 一事实一 atom / 指代消解 / tag 用固定 key

收尾（harness，纯 CPU）：
  ├─ 时间解析：agent 写的相对时间 → date -d 折算成绝对日期（毫秒级、永远准）
  ├─ 时间词剥离：memory 里残留的裸相对词（last year / 4 years ago）剥掉（见 §1.2）
  ├─ tag 收敛：_normalize_tags 把 key 收进固定 12 个
  └─ 合并去重：跨 message 同事实只留一条，source 取并集
```

**为什么并行比串行好**（这是这一版的关键）：

| | 串行 agent（旧） | 并行 agent（现在） |
|---|---|---|
| 看到的前文 | 前 N-1 条 message **已产出的 atom** + 原文 | 前文**全部原文**（msgmem/summem） |
| 依赖 | 第 N 条依赖前 N-1 条跑完（慢、易被前一条的错带偏） | 无依赖，**全部并行**（快、错误不传播） |
| 幂等 | 要按 source 增量覆盖，重跑一条要重算其后全部 | 每条独立，重跑一条只影响它自己 |
| agent 的活 | 既要抽取、又要查重合并、又要维护 id | 只管**抽干净的事实**，合并交给收尾 |

—— 关键是**并行是可行的**：消解指代所需的全部信息（前文原话）在 msgmem 里本来就齐，不必等
前一条的 atom 产出。旧版"后面的 message 要 grep 前面**已生成的 atom**"这个前提，其实可以换成
"grep 前文**原文**"，于是依赖消失、并行成立。

**代码级兜底是稳定性的来源**（8B 软约束不可靠，纪律必须下沉，见 §1.2）：

1. **时间**：agent 只负责把"这句话里有没有时间信息"写进 `time`，折算交给 `date -d` —— 
   8B 心算"13 Sep 是周几"实测不可靠，代码永远准。
2. **memory 无时间词**：prompt 明确禁止（`time` / `time_expr` 分家也是为此），残留由 §1.2 的
   剥离器兜底。
3. **tag key**：`_normalize_tags` 收敛，落盘后 key 恒 ≤ 12 个。
4. **工具调用顺畅**：`enable_thinking=false`（否则 8B 输出 `<think>` 不走工具）、prompt 里给
   完整 JSON 调用示例、缺 `path` 默认 `evidence.jsonl`、`grep "a|b"` 自动补 `-E` —— 这些都是
   实测"8B 会踩"的地方，逐条在 harness 里堵掉（见 §2.16）。

**实测（Caroline_Melanie 全量，412 条 atom）**：

| 指标 | 结果 |
|---|---|
| memory 含时间词 | 剥前后 7.5% → **1.0%**（剥离器兜底后） |
| time 字段解析成功 | **84%**（"last year"→2022、"last Saturday"→倒推日期全部正确） |
| 非标准时间格式 | **0 条**（全部 `YYYY-MM-DD` 或 ISO） |
| 多句 atom | 1 条 |
| 全量耗时 | ~10 分钟（并行；串行版本要数倍） |

对比中间那版 pipeline（同语料）：judge 60.5% vs pipeline 掉到明显更低 —— **agent 能自己找全
跨会话上下文**这件事是 pipeline 的启发式召回给不了的。

#### 消解的两件事（写进 add agent 的原子标准）

**① 时间：memory 完全不含时间词，时间信息全部在 metadata.time。**

pipeline 把时间处理分成两个阶段：

- **Stage 2（LLM）**：模型只需识别并提取 `time_expr`（原始表达如 "yesterday"），
  **禁止在 memory 里写任何时间词**。
- **Stage 3（后处理）**：代码用 `date -d` 把 `time_expr` + 锚点 → 绝对时间或带锚点的相对时间。

| 源话（消息 dated <anchor date>） | time_expr（模型提取） | metadata.time（代码解析） |
|---|---|---|
| "yesterday" | `"yesterday"` | `2023-05-07` （anchor: 8 May 2023） |
| "last week" | `"last week"` | `the week before 2023-05-08` |
| "last year" | `"last year"` | `2022` （精度只减不增） |
| "on <a date>"（本身绝对） | `"on 7 May 2023"` | `2023-05-07` |
| 无时间 | `""` | `""` |

**memory 字段的禁止词列表**（代码验证）：yesterday, today, tomorrow, ago, last, next, past, 
recent, recently, soon, future, now, currently, presently, before, after, since, until, during, 
N days/weeks/months/years。

**② 指代：换成原始含义。** `she`/`he`/`it`/`there`/`this`/`that` → 真名 / 真实地点 / 真实物件。
召回窗口（默认前 5 条同会话消息）提供上下文，模型据此消解。

**③ 一事实一 atom。** 一个 atom = **一个句子、一个主张**；**禁止多句 memory**。
`"<A> does <fact1>. <A> does <fact2>."` 是两条 atom，不是一条。

**④ tag key 纪律（prompt 声明 + harness 兜底）。** `tag` 用 `key:value`，key **只能从固定集合里选**，
不得自造：`speaker / topic / activity / action / event / state / sentiment / relation / entity /
possession / time / other`。由 `_normalize_tags` 在后处理时强制收敛：裸词/近义词按 `_TAG_ALIASES`
映射到规范 key，其余未知 key 一律写成 `other:<原值>`（值保留、只收敛 key）。

**⑤ 具体内容不得抽象化。** 可枚举的具名/带数字内容本身就是事实（`CAPTURE THE SPECIFICS`）——
不许换成一句笼统的情感或类别。

#### 并行 add agent 怎么跑（`src/codemem/add/atommem_agent_v2.py`）

```bash
# 每条 message 一个独立 agent，全部并行（受 concurrency 限流）
#   system prompt：catalog 渲染的工具描述 + 原子标准 + 4 条硬规则 + JSON 调用示例
#   user prompt：目标 message + 会话清单（agent 自己 grep 前文消解指代）
#   工具：read / edit / bash，产出写进独立文件 {msg_id}_atoms.jsonl
#   收尾（harness）：
#     1. 时间解析：agent 写的相对时间 → date -d 折算绝对日期
#     2. 时间词剥离：memory 残留的裸相对词剥掉（见下）
#     3. tag 收敛 + 补全 id/source/type/changelog
#     4. 合并去重：跨 message 同事实只留一条，source 取并集
```

**核心原理（用户原话）**：逐个 message 处理时，后面的 message 的信息可能受前面内容影响、可能
**跨 session 影响**，所以需要看到前面**相关的**信息才能做出优秀的消解。这一版**并行**成立，是因为
把"看前面已生成的 atom"换成了"看前面**原文**"（msgmem/summem 本来就齐全）—— 于是每条 message
互不依赖，全部并行。

- **并行**：每条 message 独立跑，写进各自的 `{msg_id}_atoms.jsonl`（无写冲突），最后合并。
  **串行 agent**（`--mode agent`）与**三阶段 pipeline**（`--mode pipeline`）保留用于对比。
- **agent 只填内存里的 `memory` + `metadata.{time, role, tag}`**；`id` / `source` / `type` /
  `changelog` 由系统在收尾时补上 —— 省了模型的力气，也保证这些字段永远正确。
- **时间解析在收尾**：`parse_time_expression` 用 `date -d` 把相对时间 + 锚点 → 绝对时间，
  精度只减不增（"last year" → 年份，不编月日）。
- **`time` 承载该事实的**全部**时间信息**：时点 → 绝对日期；时长/年龄/"N ago" → 起点或区间；区间 →
  `YYYY-MM-DD/YYYY-MM-DD` 或带锚点措辞；无时间 → `""`。
- **幂等**：重跑一条 message 时，先把它产出的 atom（`source` 指向它）去掉再重新生成 ——
  重复跑不会累积重复 atom。

**时间词剥离（harness 兜底，`_strip_relative_time`）**：8B 即使被 prompt 明确禁止，仍有约 7.5%
的 atom 把 "last year" / "4 years ago" 留在 `memory` 正文里（而 `time` 已解析好）。这里由代码
在收尾时把这些**裸相对时间措辞**从句子里删掉 —— 不是改写事实，是删掉已经进了 `time` 的冗余。
三条护栏保证不制造残句：删完太短（时间即主体）、删掉超过一半、或以悬空分词/虚词结尾
（"figurines bought"、"a lot happened since"）时一律**保留原文**。实测把违规率从 7.5% 压到 **1.0%**。

**工具调用稳定性**（实测"8B 会踩"的地方，逐条在 harness 堵掉）：

| 现象 | 处理 |
|---|---|
| 输出 `<think>` 而不调工具 | `enable_thinking=false`（qwen3 端点发 `chat_template_kwargs`） |
| 不知道 JSON 调用格式 | prompt 里给**完整调用示例**（bash / read / edit / done 各一） |
| `edit` 缺 `path` | 默认 `evidence.jsonl`（或 add 的 `{msg_id}_atoms.jsonl`） |
| `grep "a\|b"` 没加 `-E` | harness 自动补 `-E` 并说明修正 |
| 重复追加同行 | 重复检测：同工具同参数且文件未变 → 拒绝并给脱困提示 |
| 一直在读不动笔 | `write_nudge_after` 步后注入"该落笔了"提醒 |

**去重**（收尾）分两遍：① 精确去重（归一化后完全相同）；② 近义去重（token 集合 Jaccard ≥ 0.6，
或去情绪动词后共享核心实词 ≥3 个且共享率 ≥0.25）—— 8B 仍会用不同措辞重写同一事实（"values X" /
"appreciates X"），prompt 拦不全，所以代码兜底。

> 旧的 `run_message` agent loop 版本仍保留在 `atommem.py`，用 `--use-agent` 可切回去对比。
> 默认用 pipeline（`use_pipeline=True`）。

---

## 2. search — 生成 evidence.jsonl

**目标**：对每条 question，产出一份**真正能回答它的记忆列表**，落盘为 `evidence.jsonl`。

```bash
# **先 dry-run**：只建 inputs/ + 渲染 system prompt 打印出来，不调模型
python -m codemem.search --sample Caroline_Melanie --dry-run

# 调试首选：前 3 条 QA、每条最多 8 步，看完整数据流
python -m codemem.search --sample Caroline_Melanie --max-qa 3 --max-steps 8 --log-level debug

# 全部目录全部 QA
python -m codemem.search --experiment search
```

| 参数 | 说明 |
|---|---|
| `dirs` / `--sample NAME` | speaker 目录名或路径；缺省处理全部含 `msgmem.jsonl` 的目录 |
| `--config PATH` / `--tool-config PATH` | 配置（默认 `configs/search.yaml` / `configs/tool.json`） |
| `--qa 3,7,12` | **只跑指定的 QA 原始下标**（逗号/空格分隔）。优先于 `--max-qa` 与步数预算：点名了就一定跑完。配 `--list-qa` 查下标 |
| `--repeat N` | 每条 QA 重复跑 N 次（默认 1）。用来反复测同一条 QA 看结果稳不稳、定位偶发问题 |
| `--list-qa` | 只列出该目录的 QA 下标就退出，不跑演化 |
| `--tag NAME` | 给这次运行打标记，写进输出目录名与 summary（如 `--tag promptA`） |
| `--max-qa N` | **调试闸门**：每目录最多处理前 N 条 QA（0=全部） |
| `--max-steps N` / `--max-verify-rounds N` | 覆盖步数上限 / verifier 轮数 |
| `--concurrency N` | 覆盖 QA 级并发 |
| `--dry-run` | 只渲染 prompt 后打印，不调模型 |
| `--log-level` / `--log-output` | 覆盖 `log.level` / `log.output` |

**为什么不是 v2 那套流程**：v2 实测（5 QA / 164 次调用）动作分布 `UPDATE 146 / NOOP 18 /
ADD 0 / DELETE 0`，库净变化 `+0 ~81 -0` —— 只会原地改写、从不新增、81 次无效改写。三条根因：
① 检索只做一次，候选集第一步就定死（多跳问题死在这）；② 动作空间只有"单条原地精修"，没有
能力**创造**把两条记忆连起来的事实；③ 成功判据（"更自足"）不可检验。

**code agent 一次解决三条**：检索是 shell 命令，想调几次调几次（→①）；`edit` 能写出
全新记忆（→②）；产物是文件，可被独立验证（→③）。外加一个附带的正确性收益：agent 只见到
`jq` 吐出的**原始 JSONL 行**，从来没有"展示串"这个概念 —— v2 把
`[id] (time) Speaker | type |` 抄进正文的格式 bug 从根上不可能发生。

### 2.2 核心抽象：一个朴素的 agent loop

整个 search 就是**一个 loop**：给定问题和一份库，agent 反复「检索 → 读 → 把结论写进
`evidence.jsonl`」，直到它认为证据够了、回一句纯文本。跑完由独立 verifier 判一次"能否回答"，
不足则把缺口喂回去再跑一轮。**没有阶段、没有上下文块控制工具、没有分层概览。**

```
每条 QA 一个私有目录（QA 之间零共享 → QA 级并发；base 库只读）

  <run>/{dir}/inputs/msgmem.jsonl  summem.jsonl  atommem.jsonl   只读硬链（三层记忆）
  <run>/{dir}/qa_{idx}/                       agent 的工作目录（bash 的 cwd）
      inputs/                                 指向公共 inputs/ 的一个链接
      evidence.jsonl                          **唯一产物**（用 edit append 逐条追加）

  agent loop（read / edit / bash，≤ max_steps 步）
      system prompt：问题 + 会话清单 + 两种检索策略 + 证据标准
      agent 自己选策略、自己检索、自己把结论写进 evidence.jsonl
      ↓ 不再调用工具 = "我做完了"
  verifier（只看 question + evidence，看不到检索过程，**不看 reference**）
      ↓ 不足 → 把 missing[] 作为新一轮 user 消息喂回去，再跑一轮（≤ max_verify_rounds）
  校验 evidence.jsonl（§2.13）+ 校验 inputs/ 未被篡改
```

**为什么砍到这么简单**：之前试过更"结构化"的版本 —— 先读一层整体印象判答、再由 harness
逐个注入 session summary、用 `next(discard, add)` 让 agent 显式管上下文块。结果是**流程越复杂、
agent 越不检索**：那套版本里 150/152 条 QA 一次 grep 都没做。把流程拆掉、只留"检索工具 +
一份写笔记的地方"之后，agent 反而开始真去 grep、真去读会话（同一批 QA 重跑，之前编造
"Canada / 一幅马的画 / The Power of Now"的题，现在都从原文里读出了正确答案）。

这与项目的核心主张一致：**检索是 agent 的活，不是流程的活。** 我们该做的是把工具给好、
把"什么算合格证据"讲清楚、把答案的判据留给模型自己；而不是替它把"先看什么、再看什么、
什么时候丢弃"设计成一条流水线 —— 那既束缚它在不同题型上的灵活性，也在实测里直接抑制了
它检索的意愿。拟人的做法就是：**你知道这个问题，你知道有哪些材料，你自己决定怎么找、找多少。**

**开场只给"路标"，不给"成品"。** agent 拿到的是一份**会话清单**（`session_N (时间)`，
**不含任何事实内容**）—— 它知道这段对话被切成了哪几次会话、各在什么时候，于是能决定
"这题该 grep 什么词"或"这几次会话得通篇读"。**"路标"就是三层记忆文件本身**（见 §1.1）：
`atommem.jsonl` 的原子已经自足（时间绝对、指代消解），`summem.jsonl` 的会话总结给定位，
但**两者都只让 agent 自己 grep**，**不把总结正文塞进 prompt**。它**不会**在 prompt 里拿到任何
"浓缩成品"（整体印象、逐会话摘要）—— 因为实测那两种都会让 agent 跳过检索（见 §2.16）。

**两种检索策略集成在一个 loop 里，agent 运行中自己切换**（写进 system prompt 但只给判据、
不给流程）—— 因为一条 question 不一定一开始就能判定该用哪种，可能先试关键词、不行再转通读：

| 策略 | 判据 | 怎么做 |
|---|---|---|
| **① 关键词直达** | 问题里点了可以直接 grep 的：人名、地名、书名、物件、事件 | `grep -rin "词" inputs/`；读 `inputs/atommem.jsonl` 里的原子（提炼过的、首选），需要原话再 `inputs/msgmem.jsonl`。读到新名字就**再 grep 那个新词**（多跳） |
| **② 通读相关 session** | 答案是**集合**、或散落各处、或**根本不知道该搜什么词** | `grep -rln "词" inputs/` 定位相关会话（`summem.jsonl` 给会话号），再 `read` 那几段 `msgmem.jsonl` 从头读到尾、边读边收成员 |

什么时候用哪种、要不要切换，是模型的判断。它**不必一开始选定**：先用①，发现收不全（集合/散布）
就转②；②读到的成员名又能拿回①去 grep 它在别的会话的出现。我们只负责把两条路都摆在它面前。

**verifier 在最后判一次**：独立 auditor 只看 `(question, evidence)`，判"能否回答"；不足就把
`missing[]` 喂回去再跑一轮。它为什么必须独立：一个既能编辑又能宣布成功的 agent，最省力的
动作永远是宣布成功（v2 的 NOOP 泛滥就是这一偏差的温和版本）。**不给它参考答案**是因为用
reference 当闸门就是 oracle 泄漏（RL 会学会猜答案，推理时也拿不到答案键）—— 闸门检验的是
**可回答性**，正确性只在评测/奖励时用 judge 算。


### 2.3 检索就是 `grep`，时间算术就是 `date -d`

**这一版把检索工具砍到底**：没有向量索引、没有 embedding 服务、没有检索 CLI。语料是 JSONL
文件（一行一条记录），检索就是 **`grep`**，时间算术就是 **`date -d`**。

**为什么要删向量检索**：向量检索换来的能力是"换个说法也能搜到"，代价却是一个必须常驻的外部服务、
一次几十秒的索引构建（一个目录 17MB）、以及一个**无法解释的排序** —— 它为什么把这条排
101 名，你查不出来。而多跳问题靠的是**词面匹配**（第一跳找到的实体名，正是第二跳的检索词），
那正是 grep 的强项。删掉之后 agent 需要学的思维模型只剩一条：
**一切都是一条对 JSONL 文件的命令**。

```bash
grep -rin "husband" inputs/                          # ① 三个记忆文件里 husband 出现在哪（命中原子即相关）
grep -rln "husband" inputs/                          # ② 哪些行提到它（取定位，配 summem 判会话）
read inputs/atommem.jsonl                            # ③ 读提炼过的原子（首选原料）
read inputs/msgmem.jsonl                             # ④ 需要原话核对时读原始消息
date -d "9 Jun 2023 -5 year" +%Y                     # ⑤ 需要时算时间 -> 2018
```

**语料是三个记忆文件**（`inputs/msgmem.jsonl` / `inputs/summem.jsonl` / `inputs/atommem.jsonl`，
都是 JSONL 一行一条，消解后的原料在 atommem，原话在 msgmem）：

| 动作 | 命令 | 何时 |
|---|---|---|
| 全库关键词检索 | `grep -rin "词" inputs/` | 关键词直达（策略①）；命中即相关行 |
| 定位相关会话 | `grep -rln "词" inputs/` | 通读策略②的第一步：拿到该读哪几段（配 `summem.jsonl` 的会话号） |
| 读原子记忆 | `read inputs/atommem.jsonl` | 首选原料：已拆分、已消解、自足的原子 |
| 读原始消息 | `read inputs/msgmem.jsonl` | 需要原话/更早上下文时 |

**关键词特别明显的题可以直取**：问题里的词直接命中 `inputs/`（比如
"charity race"）就在那条原子上落证据。集合型/散布型问题（"她参加过哪些活动"）则相反 ——
先 `grep -rln` 看哪些行相关、`summem.jsonl` 定位会话，再**通读那几段 `msgmem.jsonl`、边读边收成员**：实测 LoCoMo
的标注证据里，1985 条 QA 中 1187 条只用 1 个 session，但有 330 条跨 2 个以上、最多跨 15 个。
用哪种、什么时候切换，由 agent 自己判断（判据见 §2.2 的策略表）。

**两个实测修掉的坑**（写在这里因为它们都是 grep 特有的）：

- **`grep` 退出码 1 = 没匹配，不是崩溃。** 工具层会附一句 `Command exited with code 1`，
  于是观测**非空**，只看"输出为空"判不出来。模型看到 "exited with code 1" 的第一反应往往是
  环境坏了，然后重试同一个命令（再被重复检测拒掉，白烧两步）。所以观测层专门翻译这句话。
- **`date` 的输出跟随 locale。** 中文环境下 `date -d "8 May 2023 -1 day" +"%d %b %Y"` 吐
  `07 5月 2023` —— 那是一条下游读不了的记录。harness 注入 `LC_ALL=C` 把它钉死成
  `07 May 2023`（已实测确认）。

### 2.4 时间推理：会话日期 ≠ 事件日期（实测 27% → 修）

这是目前最实的一处问题。第一次全量跑完，**temporal 类只有 10/37 = 27% 正确**，
而其它三类都在 61-77%。把 63 条 INCORRECT 全部逐条读完之后，根因是**三件事叠在一起**，
其中两件是**我们自己的 prompt 写错了**。

#### 主因：把会话时间当成了事件时间（27 条 temporal 错误里占 15 条）

看一条真实的失败：

| | |
|---|---|
| 源消息 | `session_3_1`（时间戳 **9 June 2023**）："I wanted to tell you about my school event **last week**." |
| 标注答案 | **The week before 9 June 2023** |
| 模型答 | **9 June 2023** ❌ |

模型完全没做偏移 —— 它把**会话发生的时间**当成了**事件发生的时间**。同样的形态反复出现
（`session_8_2` 15 July 答 "15 July"、`session_9_2` 17 July 答 "17 July"、`session_10_3`
20 July 答 "20 July"……），15 条错误里候选答案**恰好等于证据里那条消息自己的时间戳**。

应当输出的答案（"the week before 9 June 2023"）其实**不需要任何上下文**就能理解 ——
带锚点的相对表述本身就是完整答案，而模型把它抹平成了锚点。

#### 主因之二：prompt 明令禁止了正确答案的形式

`ANSWER_PROMPT` 里原本写着：

> NEVER leave a relative time expression in the final answer. Replace every relative time
> with the concrete date …

而 LoCoMo 的参考答案**本身就是相对锚点**（94/262 条是 `... before X` 形态）。
这条指令等于**命令模型不要给出正确答案的形式**，逼它去算 —— 算不准就退化成抄会话时间。
现在改成：**带锚点的表述是完整答案，可以直接用**；只有无锚点的相对词
（"next month"、"recently"）才是禁止的。

#### 主因之三：judge 自己的日期算术是错的（最隐蔽，也最该先修）

`judge.resolve_relative_date` 的正则 `(day|week|weekend)s?` 会命中 **"Fri**day**" 的后缀**，
于是 "The Friday before 15 July 2023" 被当成 "1 day before" 算成 **7 月 8 日**（正确是 7 月 14 日）。
"two weekends before X" 的数量词也被丢了（算成最近的那个周末，而不是再往前一周）。

更糟的是 prompt 里的 **worked example 也是错的**，直接教 judge 拒绝正确答案：

| 例子 | prompt 原本写的 | 正确值 |
|---|---|---|
| The Friday before 15 July 2023 | July 8（说 15 Jul 是周六） | **7/14**（15 Jul 是**周六**，之前的周五是 14） |
| The weekend before 17 July 2023 | 10-11 July | **7/15-16**（17 Jul 是**周一**） |

修完之后（13 个手工核对过的用例 + PRECOMPUTED 提示），judge 的折算全部正确。

**但要诚实说明它的影响面**：修好 judge 之后重算旧结果，27 条 temporal 错误里
**只有 1 条**会因为算术修正而转正。也就是说 **judge 的 bug 是真的，但它不是主因** ——
主因是答案本身确实错了。修 judge 的意义在于**让测量可信**，否则你没法判断 prompt 改动
到底有没有用。

#### 另外两种小形态

- **无锚点的相对词**（2 条）："next month" —— 源会话是 25 May 2023，正确答案 "June 2023"。
  现在明确禁止："next month" 单说毫无意义，必须写成 "June 2023（即 25 May 那次谈话的下个月）"。
- **unsupported**（3 条）：证据里其实有信息（如 `session_7_8` "This book I read last year"），
  但被写成了别的措辞，answer 阶段判成"证据不足"直接放弃。

#### 现在的规则（`tool.json` 的 `time_policy`）

1. **会话时间戳是"谈话发生的时间"，不是"事件发生的时间"** —— 它是用来**倒推**的锚点，
   把锚点本身当成事件日期是最常见、也最严重的错误。
2. **照录源话 + 锚点**：`Caroline said she gave a speech 'last week' (message dated 9 June
   2023) -- i.e. the week before 9 June 2023.` 引号里的原话 + 锚点日期一起写进记录，
   即使算术有偏差，锚点仍在。
3. **源用什么单位就答什么单位**：'last year' → 年份；'last Saturday' → 星期几。
4. **带锚点的短语就是答案**，可直接保留；可以附上算出的日期，但不许用猜的日期替换锚点。
5. **算术精确且锚点已知时才算出绝对日期**，用 `date -d`，不准心算。
6. **禁止无锚点的相对词**。

**verifier 也要同步**，否则它会反向逼 agent 编造：它在"够不够"这一关很容易把锚点表述判成
不完整（"这不是一个日期"）。所以它的 prompt 里明确写了 *Match the precision of the
evidence -- do NOT demand more*，外加一条"记录上的日期是**说这话的时间**"。

#### 源语料的表述往往比参考答案还粗

看几个真实例子：

| 源消息（括号是会话时间） | 标注参考答案 |
|---|---|
| "I painted that lake sunrise **last year**!"（8 May 2023） | **2022** |
| "a friend made it for my 18th birthday **ten years ago**"（27 Jun 2023） | **10 years ago** |
| "I ran a charity race **last Saturday**"（25 May 2023） | "The Saturday before 25 May 2023" |
| "my daughter's birthday **last night**"（14 Aug 2023） | **13 August** |
| "we went camping **two weekends ago**"（17 Jul 2023） | "two weekends before 17 July 2023" |

统计 LoCoMo 全部 **262 条 when 问题**：相对锚点表述（"... before X"）**94 条**、
绝对点或区间 143 条、周/月/年区间约 65 条、`N ago`/`since Y` 数十条 ——
**绝大多数答案根本不是"某年某月某日"**。

所以写进 prompt 的是四条规则（`tool.json` 的 `time_policy`）：

1. **先看源用了什么单位，就按那个单位答**，不要细化。源说 "last year" → 答案是年份。
2. **相对表述本身就是正确答案，别抹平它。** "the week before 9 June 2023" 精确且无歧义；
   把它压成一个猜出来的日期是**信息损失**。
3. **只在算术精确且锚点已知时才算出绝对日期**（"yesterday" + 该消息自己的时间），
   并且用 `date -d` 算，**不准心算**。
4. **区间是合法答案**（"June 2023"、"the week of 23 August 2023"）。

外加一条与检索强相关的推论：**evidence 记录必须自带锚点**。下游读者看不到会话，
所以 "she went there the week before" 这样的记录是没用的 —— 锚点日期必须写进正文。

### 2.4.1 一次"效果变差"的完整归因：58% → 47% 里有多少是真的

改完时间规则后重跑，分数**从 58% 掉到 47%**。逐条归因之后，结论是：
**这 11 个点里绝大部分是判分口径变了，不是答案变差了。**

做法：把两次运行（`search_20260929_185935` / `search_20260929_214956`）的逐条结果按
**同一套规则**重新判一遍，再比较。三件事同时发生了：

| 现象 | 影响 | 性质 |
|---|---|---|
| **旧 judge 把"空答案"判成 CORRECT** —— 实测 **11 条**（"The prediction correctly identifies that the answer is unsupported…" → 竟然判对） | 旧分数**虚高** | 旧 judge 的 bug，新 judge 修对了 |
| **新 judge 把"集合型问题多给成员"判成错** —— 集合型 67% → 26%，而参考答案只是**不完整的列表** | 新分数**虚低** | 新 judge 引入的过严，已修 |
| **temporal 真的变好了** 27.8% → 50.0% | 真实提升 | 时间规则的功劳 |

**统一判据后**（把两边都用同一套规则重判）：

```
                 旧 run      新 run
multi_hop      18/30 60% →  9/30 30%
temporal       10/36 28% → 18/36 50%     ← 真实提升
open_domain     5/13 38% →  5/13 38%
single_hop     43/69 62% → 39/69 57%
总体           76/148 51% → 71/148 48%
```

**剩下的 20 条"变差"里，14 条的答案覆盖率不低于旧答案**（措辞变了、实质等价，被模型判成
不同），只有 6 条是真的变差。也就是说：**净变化基本在噪声范围内，唯一确凿的移动是
temporal +22 个点。**

**为什么判分口径会漂移这么多**，根因是上一版把 judge 的 `reason` 挪到了第一位
（为了 CoT），副作用是模型从"直接比较"变成"先复述判据标题再比较" —— 点名判据的比例
**42% → 82%**，而且反射性地去点 `CORE FACTUAL ACCURACY`，在那条判据下"多给内容 = 加错误
主张 = INCORRECT"。这一条现在明确写了：**集合型问题上超集是 CORRECT**，且 **reason 里不要
复述判据标题**。

**教训**：judge 是被测量的一部分。它的 prompt 一改，**历史分数就不能直接比了**。
`scripts/judge_impact.py` 就是为此写的 —— 它拿新规则扫一遍已存的结果，列出"按新规则应该
翻转"的条目，改 judge 之前先跑它看一眼影响面。

### 2.4.2 时间处理的最优方案：保留相对表达，不折算（2026-10-06）

§2.4 的规则里留着一条**自相矛盾**的指导：一边说"带锚点的短语就是答案"（规则 4），一边又写
"**Compute when you can: lead with the computed date**"（旧规则 3）。8B 在矛盾下倾向于执行
"尽量折算"，于是把 "last week" 算成了**会话日期本身** —— 恰好踩中规则 1 明令禁止的错误。
全量跑后 temporal 只有 36%，但 evidence_recall 高达 83% —— **检索到了，是作答/判定把时间搞错了**。

#### 三个叠加的 bug（逐条实测确认）

**① search 把相对表达错误折算成会话日期。**

```
源消息 (session_3_1, 时间 9 June 2023): "my school event last week"
atommem (正确): time = "last week (anchor: 9 June 2023)"  ← pipeline 已保留
search 写的 evidence (错): "Caroline gave a speech at a school on 9 June 2023"  ← "last week" 被吞成锚点本身
参考答案: "The week before 9 June 2023"
```

**② judge 折算 candidate 时丢掉了修饰语，把逐字相同的答案判成不同日期。**

`judge._parse_date("the week before 6 July 2023")` 走的是**绝对日期**分支 —— 它抠出
"6 July 2023" 算成 `2023-07-06`，**完全无视 "the week before"**。而 reference 走
`resolve_relative_date` 算出 `2023-06-29`。于是：

```
reference: "The week before 6 July 2023"  → 2023-06-29
candidate: "the week before 6 July 2023"  → 2023-07-06   ← 逐字相同却算成差 7 天
judge: "These differ by 7 day(s)" → INCORRECT
```

逐字相同的正确答案被判错。**这是 judge 最隐蔽、也最该先修的 bug** —— 它污染测量，
让"答案其实对了"的题显示为错。

**③ answer 把锚点当成事件时间作答。**

#### 最优方案（用户定的口径）

> **相对时间表达本身就是一个完整、精确、充沛的答案。** "The week before 6 July 2023" 自带锚点
> 6 July，任何人（包括 judge）都能验证两个答案是否相同 —— **不需要、也不应该折算成绝对日期**。
> 折算是有损的：它丢掉"一周内哪天"的信息（"the week before X" 是一个**周区间**，不是一个点），
> 还会引入算错的风险。而"上周发生的"（模糊描述）= 一个周内发生，**不能具体到某一天**。
> **不要随意理解日期。**

落成三条规则：

| 表达 | 处理 | 理由 |
|---|---|---|
| `yesterday` / `today` / `tomorrow` | **折算成绝对日期**（`date -d`，单步精确） | 唯一"精确且平凡"的折算 |
| `last week` / `last Friday` / `the week before X` / `N years ago` | **保留原表达 + 锚点**，不折算 | 自带锚点即完整答案；折算是信息损失 |
| `a few weeks ago` / `recently` | 保留为**粗粒度窗口**，绝不细化成日期 | 源就是模糊的，细化 = 编造 |

**具体修改**：

1. **`tool.json` 的 `time_policy`**（search 侧）：删掉"compute when you can"，改成
   "**preserve, don't convert**" + "只有 yesterday/today/tomorrow 才折算" + "按源的单位作答，绝不细化"。
2. **`tool.json` 的 worked examples**：把"lead with the computed date"的例子改成
   "keep 'last week' + anchor" 的正确形态。
3. **`prompts.ANSWER_PROMPT`**（answer 侧）：加"keep the source's own time wording, anchored --
   do NOT flatten it into a date" + "read the unit the source used and answer at that unit"。
4. **`prompts.EVIDENCE_STANDARD`**：第 3 条改为 "TIME ANCHORED, KEPT AS WRITTEN"。
5. **`judge._parse_date`**：**先试相对表达**（复用 `resolve_relative_date` 的同一套逻辑），
   识别不到再回退到绝对日期解析 —— 保证 reference 与 candidate **口径完全一致**。
   把折算逻辑抽成 `_resolve_relative_expression` 供两个函数共用。

#### 实测（Caroline_Melanie 的 3 条 temporal QA）

| QA | 修改前 | 修改后 |
|---|---|---|
| qa8 | `9 June 2023`（吞成锚点）❌ | `the week before 9 June 2023` ✅ CORRECT |
| qa9 | `9 June 2023` ❌ | `the week before 9 June 2023` ✅ CORRECT |
| qa28 | `15 July 2023` ❌ | `the week before 15 July 2023` ⚠️（差在 "Friday" vs "week"，非时间方案问题）|

token_f1 从 ~0.5 升到 **0.944**。新生成的 evidence 正是期望的形态：

```
Caroline gave a speech at a school 'last week' (message dated 7:55 pm on 9 June, 2023)
  -- i.e. the week before 9 June 2023.
```

### 2.4.3 全量跑后的错因分析：最大瓶颈是"集合题检索不到"（2026-10-06）

时间方案落地后全量跑（`search_20261006_112146`）：**judge 55.4%**，temporal 从 36% 升到 **61.1%**
（时间方案生效），但 single_hop 62.1%、multi_hop 40.7%、open_domain 33.3%。

#### 按 recall 拆开 58 个错误，瓶颈一目了然

| 错误形态 | 条数 | 含义 |
|---|---|---|
| **recall=0（检索不到）** | **29** | evidence 完全没覆盖标注证据 —— 最大瓶颈 |
| unsupported（没答） | 13 | search 判"证据不足"直接放弃 |
| recall=1（检索到但答错） | 12 | 证据对，作答/判定错 |
| recall 部分 | 4 | 集合收不全 |

**29 条 recall=0 里绝大多数是"集合型问题"**（"what activities/pets/books/items does X have"、
"where has X camped"、"what types of pottery…"）—— 标注证据是一**组散点**，search 收漏一条就大量丢分。

#### 三个具体错因（逐条看轨迹确认）

**① 集合题，agent 只做关键词 grep，不读会话。**

qa18「Where has Melanie camped?」（真答案 `beach, mountains, forest`）的 12 步轨迹全是
`grep "camp"` / `grep "mountains"` / `grep "campfire"` —— **它从没用过 Strategy 2 的"读会话"**。
成员散落在多个会话，关键词法永远抓不全。

**② 具体内容被抽象化。**

qa127「What did Caroline make for a local church?」原子层**是对的**（`Caroline made a stained
glass window`），但 search 写成 `"Caroline created a painting"` —— **"stained glass window" 被抽象成了
"painting"**。qa112（`a sunset with a palm tree` → `a nature-themed artwork`）同理。
这正是 `ATOM_AGENT_STANDARD` 规则 13 禁止的，但 search 的 `EVIDENCE_STANDARD` 没有对应条款。

**③ answer 复述 prompt 模板占位符。**

qa18 的 answer 输出 `"the week before <session date>"`、qa71 输出 `"the year before session_7_8"`
—— 因为 `ANSWER_PROMPT` 的工作示例里写的是 `<session date>` / `<year-1>` 这种**模板标记**，
模型逐字照抄了，还把它当成了答案。

#### 修改

1. **`strategy`（策略选择）**：加一条判据 —— 集合/计数题**必须走 Strategy 2**，并警告
   "grep 人名会匹配几乎每一行，`head` 只看到噪声，不是成员"。
2. **`EVIDENCE_STANDARD` 第 8 条**：新增 `SPECIFICS PRESERVED` —— 保留具体内容（stained glass
   window / sunset with a palm tree / 标语原文），禁止抽象成类别。
3. **`verifier`**：加两条 —— 集合题要**具体成员**（不是类别）、答案要**保留具体细节**。
4. **`ANSWER_PROMPT` 示例**：把 `<session date>` / `<year-1>` 占位符换成**真实示意日期**
   （9 June 2023 / 2022），并明确"示例里的日期是示意值，必须替换成真实日期；绝不复制占位符、
   session id 或示例值"。
5. `system prompt` 上限 19500 → **19700**（上述规则是有意的、针对最大瓶颈的增长）。

#### 诚实记录：一次失败的方向

最初把规则写成"you MUST use Strategy 2"，结果 **充分率从 42% 掉到 0%** —— 8B 把它理解成
"要 grep 人名"，于是 `grep "Melanie" | head -20` 反复重试、卡住不写 evidence。
**加规则 ≠ 改行为**：抽象描述（"list the sessions, then read"）8B 会误读，必须给**具体命令形态**
和**明确的反例**（"grep 人名只会看到噪声"）。这是 add / search agent 反复验证过的教训。

#### 实测（20 条混合题，中等规模验证）

| 指标 | 全量（改前） | 20 条（改后） |
|---|---|---|
| judge_accuracy | 55.4% | **60.0%** |
| unsupported_rate | 10.0% | **0.0%** |
| token_f1 | 0.404 | **0.442** |
| sufficient_rate | 42% | **75%** |

（20 条样本小、判分条数少，只作方向参考；全量验证见下。）


**一句话总结**：相对时间**保留原样 + 附锚点**是答案，绝对日期只是"锚点 + 偏移"的另一写法；
judge 两边都要用同一套折算逻辑，否则会把逐字相同的答案判成不同。

### 2.5 输出 schema：`reasoning` 必须放在第一个字段

所有要求模型输出 JSON 的 prompt（answer / baseline / rag / judge / verifier）里，
**推理字段一律排在首位**：

```json
{"reasoning": "...先写推理...", "answer": "...", "unsupported": false}
```

理由：模型是自回归的，**按字段顺序生成**。`reasoning` 在前 = 先推理再作答（CoT）；
放在后面就变成"先给答案，再补一段解释" —— 那段解释是**事后编的**，它不会改变已经生成的答案。
judge 同理：`reason` 在前、`label` 在后，等于先做比较再下结论。

解析是按 key 取的，所以字段顺序对下游完全无影响；改的只是生成顺序。测试里有断言盯着
每个 prompt 的首字段。

### 2.6 上下文预算：工具描述不能吃掉 agent 的步数

system prompt 是**每条 QA 都完整注入一次**的（agent loop 的每次模型调用都带着它），所以它的
每一句都在跟检索结果抢上下文。实测现在 **~18770 字符**：

| 段落 | 字符 | 说明 |
|---|---|---|
| system body | ~4900 | 检索策略（STRATEGY）+ 三条 worked example（关键词检索 / 相对时间 / 通篇读） |
| EVIDENCE_STANDARD | 2035 | 一条证据要写法上合格（§2.8）—— 自足 / 指代 / 时间锚点 / 不编造精度 / 溯源 |
| TOOLS | ~1700 | 三个工具（read / edit / bash）的 description + schema |
| COMMANDS | 1485 | grep、date 的用法与示例 |
| WRITING_POLICY | 1981 | 证据是推导、不是抄的 |
| TIME_POLICY | 1942 | 时间精度规则（§2.4） |
| SCHEMA | 650 | evidence 记录字段 |
| WORKSPACE | 640 | 目录布局与只读规则 |
| PROTOCOL | 400 | 响应格式 |

**比"有块机制"那一版反而小了一点** —— 砍掉了整套上下文块流程描述与 `next` 工具的 schema。
"证据标准"那一段保留（它换来的是"什么算一条合格的证据"）。

> **2026-10-05 变更 C 使 prompt 增长。** 两种检索策略从散文改成**标准操作流程（SOP）+ 特例
> 处理**（§2.9），STRATEGY 从 ~2.6k 涨到 ~3.4k。这是用户明确要求的内容，是有意增长；
> 上限断言相应从 `< 18000` 抬到 `< 19500`，仍用于防止**非预期**膨胀。同时把 prompt 里
> 残留的数据集实体（真实人名/日期/书名）全部替换成占位符（`<person>` / `<date>` /
> `<key word>`），断言里新增一条**负向检查**：渲染后的 prompt 不得出现任何已知数据集实体。

`test_search.py` 里有一条断言把这个上限钉住（`< 19500 字符`）——**涨回去就等于白做这次简化**，
所以它该在 CI 里红，而不是等到某次跑完发现步数又不够用了。

**Verifier 同理**：它的 prompt 里没有 few-shot 示例，唯一的例外是**时间精度的两条说明**
（它很容易把锚点表述判成不完整，于是反向逼 agent 编造更精确的日期 —— 正是 §2.4 要避免的）。

### 2.7 集合型问题靠遍历，不靠相似度

有一类问题的答案是**一个集合**："Melanie 参加哪些活动？"、"Caroline 有哪些爱好？"。
旧版试过用向量相似度解决它 —— **结构性做不到**：相似度按与 query 的距离排序，而"她要去
游泳了"这条记录与"Melanie 参加什么活动"的距离并不近（实测稠密检索排名 **101**），
于是永远进不了 top-k，把 `-k` 调到 100 也救不回来，因为相关度天梯本身就不指向它。

旧版的第二个尝试是**按标签枚举**（`--tag activity`）。那个方向对，但**收益取决于标签词表
的质量**，而词表由模型即兴生成：同一个"活动"概念，四个标注答案分别落在 `action` /
`topic:activities` / `action:Swimming` / `topic:art` 四个不同的 key 上，还有 30 多个别的
key。**没有哪一个 `--tag` 能一次收齐** —— 那是 add 侧的问题，不是检索算法的问题。

这一版换了个更直接的办法：**通篇读**（策略 B）。扫一遍 session summaries 看哪几页相关，
然后**把那几次会话从头读到尾**、边读边收成员 —— 没有"相关度天梯"这回事，命中的就是概览/正文里
**字面**提到的东西，覆盖是可数的、可验证的。集合型问题往往要**对着读好几个会话**（同一个成员
在不同会话里反复出现），这正需要 agent 自己决定读哪几页、读多深。

### 2.8 证据是**推导**出来的，不是抄的（附"证据标准"）

这是 search 的核心语义，也是相对上一版最大的一处变化：

> **`evidence.jsonl` 不是库的子集，而是一份可以直接回答问题的记忆列表。** 下游只看到这个
> 文件 —— 看不到语料、看不到原始消息。所以所需的推理必须**在记录里完成**。

因此 agent **可以也应该改写正文**：

- **消解指代**：`she`/`he`/`there` → 真实姓名/地点；
- **消解时间**：`yesterday` 这类**算术精确且锚点已知**的表述 → 绝对日期（用 `date -d`，锚点是该会话
  记录里的时间戳 —— 每条原始消息都带着它，不必非去找会话首条）；
- **合并事实**：两条只有一起看才有意义的记录 → 一条陈述结论的记录；
- **推理**：时间题要算术、事件题要合并、**性格/属性题要把散落陈述汇总成一条**；
- **补一条语料里没有任何单条记录陈述过的结论**，`source` 列出推理所依据的全部 id。

实测产出（性格/属性推理）：

```
content: "Caroline is pursuing a career in counseling and mental health,
          driven by her interest in adoption and family planning."
source:  ["session_1_9", "session_2_8", "session_1_17"]
```

**为什么把"证据标准"单独抽成一段 prompt**：**"找到了证据"和"写出来的证据能用"是两件事。**
实测里最常见的失败不是没检索到，而是检索到了却写成一条下游读不懂的记录 —— 指代没消解
（"she went there"）、时间只有一个无锚点的相对词（"next month"）、或把两条只有合看才有意义的
记录原样抄成两条。这一类问题与**检索策略**无关，只与**记录的写法**有关。所以它被提成一份
**可核对的清单**（7 条：自足 / 指代消解 / 时间锚点 / 不编造精度 / 断言而非抄录 / source 真实 /
该合并就合并），正文放在 `codemem/prompts.py` 的 `EVIDENCE_STANDARD`，经 `tool.json` 的
`{{EVIDENCE_STANDARD}}` 槽位注入 system prompt（与 `writing_policy` / `time_policy` 并列，
`tool.json` 仍可覆盖措辞）。改了它就等于改了"什么算一条合格的证据"。

反幻觉护栏不变：**`source` 必须非空**，且必须是**本会话真读过的** id（不是记得或猜的），
`source` 为空的记录一律 reject。**唯一的例外**是"仅凭印象就答"那条路径 —— 那时模型只见过
印象、没见过原话，所以它的 `source` 指向印象里**标注给这条事实的** `session_N`（一个可回溯到
会话的 id），而不是某条具体消息 id。

### 2.9 code-agent loop 的几件事

上一版实测暴露了四个问题，这一版逐一修掉（每条都有日志证据）：

| 实测问题 | 做法 |
|---|---|
| **93% 的观测是空的**（`jq ... >> file` 重定向走了 stdout），模型看不见自己刚写了什么，于是盲目重试 | 每次**变更型**调用后由 harness 附一段**证据摘要**：行数 / 唯一 id 数 / 末尾几行 |
| **60/60 步没有一次完成信号**，12~24 步全烧在重复追加 | **重复检测**：同一工具+参数、**且文件状态未变**时直接拒绝并给出脱困提示（状态感知 —— 文件变了就允许重跑） |
| 空转被算作"有进展"（往文件里追加重复行） | 进展判据改成**唯一记录集合的指纹**，不是行数、也不是字节数；重复追加不算进展 |
| prompt 随历史无限涨（3.3k→4.5k tok / 12 步） | **上下文压缩**：超预算即把中段坍缩成摘要，保留 system + 最近 N 条 + **工具用过的 id 集合**（保 id 是关键，否则模型会忘了已收过哪些而重复追加） |
| **超长工具观测一直堆在上下文里**（一次 `read` 整个 msgmem 就是几千字符，跨轮不再需要却永远占着） | **遗忘**（§2.9.1）：按"轮数 + 长度"双阈值，把够旧且超长的观测定点替换成一行占位符 |

**逐条追加：`edit` 需要专门的 append 模式，不能靠"用最后一行当 oldText"去模拟。** 流程要求
"每看完一页就把理解**追加**到 `evidence.jsonl`"。最初的做法是让模型用精确替换 `edit` 自己造
`oldText`（拿已有的最后一行当锚点）。实测（8B）第一次跑就暴露三连击：**① 文件还不存在时
`edit` 直接报 `File not found`**（每个 QA 的第一次追加必然踩到）；**② 摘要把行截断到 160 字符**，
模型复不出那个精确的 `oldText`，要么填个近似串、要么留空（`edits[0].oldText must not be empty`）；
**③ `edit` 失败后它退回 `write` 整份覆盖**，把前面几个 session 的证据全冲掉 —— 实测"访问了
5 个 session、最后只剩 1 条记录"。

修法是把这个动作**升为一等参数**，而不是继续靠 prompt 绕：

- `edit` 增加 **`append` 模式**（`append="<文本>"`），**文件不存在时直接创建** —— 一次修掉
  ① 和 ②；追加本就是这套流程的一等操作，就不该让模型用替换去拼。
- **`write` 工具整个删掉** —— 没有"整份覆盖"这个入口，"一个 session 写错就冲掉前面全部证据"
  在机制上不可能发生（于是 ③ 也跟着消失）。要修正已有记录用 `edit` 的 replace 模式。
- 追加的防护：`append` 为空 → 拒绝（空写只可能是截断）；追加内容与文件**最后一行**逐字相同
  → 拒绝（模型偶尔把刚写过的又追加一遍）。
- 证据摘要仍把**末尾几行**给模型看（给到 200 字符），但它不再承担"要被逐字节复制"的职责。

改完后同一题：连续 `edit` 追加，5 条记录全部留住；用户报告里的 `File not found` 也不再出现。

外加**错误重试**与终止语义的收紧：

- **解析失败 ≠ 完成**。只有模型**明确回复纯文本**才算"我做完了"。以 `{` 开头却解析失败
  （通常是被 `max_tokens` 截断）会**注入纠正消息重试**（`max_parse_retries`），空回复同样
  重试。混在一起会让 verifier 对着一份半成品去判。实测 qa1 就靠这条救回了正确答案。
- **空 evidence 直接判不足**，且**不问模型** —— 最有用的反馈是"你跑了 12 步却什么都没写"
  这个事实本身（实测这是最常见的失败：一直在读、从不动笔）。verifier 会返回三条可执行指令。
- **文件完整性**：`write`/`edit` **原子写**（临时文件 + `os.replace`），避免截断输出**原地
  留下半个文件**；`write` 用空内容覆盖 `evidence.jsonl` 被**拒绝**；大幅缩短给出警告。
- **产物解析统一**：`validate_file` 与 agent 的进展统计共用同一个解析器 `_iter_records`
  （它们曾经各写一份，结果 agent 认为文件坏了、validator 认为文件是好的）。并容忍两种
  `jq` 自然产物：**一行一个数组**（`jq -s`）与**整份美化 JSON**（`jq -s` 不加 `-c`）——
  后者以前会让 `write_normalized` 把证据**全部删掉**。

### 2.9.1 遗忘系统：按"轮数 + 长度"双阈值定点清理（变更 D）

**与上下文压缩互补，各管一件事**：压缩是**按总量触发**、把**中段**整体坍缩（兜底）；
遗忘是**按观测个体**、**按年龄**把**够旧且够长**的单条观测替换成占位（定点）。

一次 QA 只要 `read` 过整个 `msgmem.jsonl`，那条观测就有几千字符；读过之后它**再也不会被用到**，
却一直躺在上下文里跟后续步骤抢预算。所以规则是：

- **超长**：单条工具观测的字符数 `> long_observation_chars` 才有资格被遗忘（短观测不动）。
- **够旧**：该观测出现后已经过了 `forget_after_turns` 个**回合**（一个回合 = 一次 assistant
  工具调用 + 它后面那条观测）。
- 替换成一行：`[observation forgotten after {N} turns -- it exceeded {M} chars. Re-run the command if you need it again.]`
  —— 模型知道"这条被忘了、需要就重跑"，不会误以为库里没有。
- **system、初始 user、最近 `forget_keep_recent_turns` 轮**一律不动（默认 2 轮）。

三个参数都在 `configs/search.yaml`（`search/runner.py` 的 `DEFAULT_CONFIG` 同款）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `forget_after_turns` | 4 | 观测超过这么多**回合**后才有资格被遗忘；**0 = 关闭** |
| `long_observation_chars` | 2000 | 长度超过此值的观测才算"超长"；**0 = 关闭** |
| `forget_keep_recent_turns` | 2 | 始终保留最近这么多轮 |

**两个阈值都 > 0 才启用**（任一为 0 即整机关闭）。调用点在 `run_agent_round` 主循环里、
**在 `compact_messages` 之前**（每轮模型调用前）—— 先做定点遗忘、再按总量兜底压缩。
add 阶段（`add/atommem.py`）也走同一个 `run_agent_round`，但步数少，默认关闭（两个阈值 0）。

实测（2026-10-05 小样本重跑 `--max-qa 3 --max-steps 8`）：qa1 / qa2 各命中一次 ——
`遗忘 1 条超长观测（>2000 字符、>4 回合），省下 7141 / 6898 字符`，无崩溃、证据正常产出。
`test_search.py` 里有独立单测覆盖：短观测不忘、新观测不忘、超长且旧的被替换成占位、system 保留。

### 2.10 存储与运行目录

旧版这一节是整条链路最重的一块：向量索引（一个目录 17MB）、共享缓存、按 `corpus_sha256`
判失效、跨运行复用、三个清理命令。**现在这些全都不存在了** —— 语料就是 `data/{dir}/` 下的三个 JSONL
记忆文件，硬链进运行目录（inode 共享，0 额外空间），agent 直接 grep。

| 内容 | 位置 | 说明 |
|---|---|---|
| `msgmem.jsonl` / `summem.jsonl` / `atommem.jsonl` | 每次运行的 `inputs/` | **硬链**（inode 共享） |
| evidence / trajectory / 日志 | 每次运行目录 | 每次一份（本来就该有） |

**每个 QA 的工作目录只放一个链接。** 早期实现给每个 `qa_{idx}/inputs/` **逐个**软链 5 个
文件，但那些东西在一个运行目录内对每个 QA 完全相同 —— 等于把同一组链接复制 152 遍。
每个目录项占一个 4K 块，实测全量跑浪费 **4.2MB**（目录 458 个 + 软链 608 个），
而全部证据加起来才 45KB。现在整目录一个链接，每 QA 从 **6 个条目降到 2 个**。

链接优先用**符号链接**：工具层的路径约束会 `resolve()`，解析后落在工作目录之外，所以 agent 写
`inputs/...` 会被直接拒绝。Windows 上普通账户建符号链接需要开发者模式/管理员权限
（`WinError 1314`），`link_dir_or_copy` 会退回**逐文件硬链** —— 硬链下 `bash >>` 会写穿到
源文件（实测确认过），而收尾的 sha256 校验能抓到并把该 QA 作废；`write`/`edit` 的原子替换
则只会悄悄换掉自己那份链接，源文件不受影响。

**轨迹里的观测要截断。** `{dir}.qa_trajectories.jsonl` 实测 **4.8MB**，其中
`rounds[].steps[]` 占 95%，而"原始工具观测"（`result_text`）一项就占 55%。观测是**唯一**
记录"模型当时看到什么"的地方（`model_calls.jsonl` 默认不写 messages），所以是**截断而非
删除** —— 默认每步保留 600 字符。要全文设 `trajectory_observation_chars: 0`。

**`model_calls.jsonl` 默认不写完整 `messages`**：那份 messages 是**整个不断增长的上下文**，
相邻两次记录 95% 重复 —— 实测单条 21KB、一个 QA（16 次调用）**800KB**、全量约 **1.6GB**。
默认只记模型的原始输出 + 上下文规模，降到 **8KB/QA**。要复现"模型当时看到什么"用
`--log-full-messages`。

**历史运行目录可一键清理**：

```bash
python -m codemem.search --prune-runs                      # 只列出
python -m codemem.search --prune-runs --yes                # 真正删除
python -m codemem.search --prune-runs --keep-last 3 --yes  # 保留最近 3 次
```

默认保留**含 `full` 的那次运行**（没有则保留最近一次）。实测清理 27 个调试跑次回收
**24.5MB**（`40M → 29M`）。

### 2.11 两条产物稳健性的实测坑

- **被 `max_tokens` 截断的数组必须抢救**。模型一次 `write` 写出 39 条记录、末尾的 `]`
  被截掉（7687 字节处断）。旧实现逐行读只看到"1 行、解析失败"，于是**把全部证据判成空** ——
  比"少一条"糟得多。现在按对象边界切开抢救，完整的收下、只丢最后一个残片，并在日志里
  明确写出"截断抢救（丢弃 N 个残片）"，**绝不静默**。
- **解析只有一处实现**。这个 bug 的根因是同一段解析逻辑存在三份（`io` / `evidence` 校验 /
  `evidence` 进展统计），对同一份文件三处判断不一致。现在全部走
  `io.parse_jsonl_text`，它用 `mode`（`linewise` / `document` / `salvaged`）明确说明
  文本是**怎么**解析出来的，而不是让调用方从"坏行数"去猜。

实测过的一条时间推理轨迹（qa0，"When did Caroline go to the LGBTQ support group?"，
reference 是 **7 May 2023**）：

```bash
grep -rin "support group" inputs/
# -> atommem.jsonl: "Caroline went to an LGBTQ support group on 7 May 2023." (source session_1_3)
#    msgmem.jsonl:  "I went to a LGBTQ support group yesterday" (time "1:56 pm on 8 May, 2023")
#                   —— 原话 + 锚点：atommem 里的 7 May 就是 add agent 由它算出的
date -d "8 May 2023 -1 day" +"%d %b %Y"                        # -> 07 May 2023（复核）
```

agent 拿到的 atommem 里**已经是消解后的原子**，它据此**写一条证据**：

```
content: "Caroline went to an LGBTQ support group on 7 May 2023."
source:  ["session_1_3"]
```

一条记录，直接回答"什么时候"，下游不需要再做任何解析。**注意这中间发生的事**：语料里的
`yesterday` 早在 add 阶段就被消解成绝对日期（写进 atommem）—— 这就是 §1.1 说的"add 侧消解"，
search 阶段只需照取。（`yesterday` 相对于该消息自己的时间，算术精确、锚点已知；如果源说的是
"last year"，atom 的 time 就该是年份而不是日期。）

### 2.12 用法

```bash
# **先 dry-run**：只建 inputs/ + 渲染 system prompt 打印出来，不调模型
python -m codemem.search --sample Caroline_Melanie --dry-run

# 调试首选：只跑前 3 条 QA、每条最多 8 步，看完整数据流
python -m codemem.search --sample Caroline_Melanie --max-qa 3 --max-steps 8 --log-level debug

# 全部目录全部 QA（注意成本：每条 QA ≈ steps + verifier 次调用）
python -m codemem.search --experiment search

# 纯 CPU 自测（假模型 + 真工具，零模型成本）
python scripts/test_search.py
```

| 参数 | 说明 |
|---|---|
| `dirs` / `--sample NAME` | speaker 目录名或路径；缺省处理全部含 `msgmem.jsonl` 的目录 |
| `--config PATH` | 检索配置（默认 `configs/search.yaml`） |
| `--tool-config PATH` | 工具配置（默认 `configs/tool.json`） |
| `--max-qa N` | **调试闸门**：每个目录最多处理前 N 条 QA（0=全部） |
| `--max-steps N` | 覆盖 `max_steps`（默认 30） |
| `--max-verify-rounds N` | 覆盖 `max_verify_rounds`（默认 2） |
| `--concurrency N` | 覆盖 QA 级并发（默认 4） |
| `--dry-run` | 只渲染 prompt 后打印，不调模型 |
| `--log-level` / `--log-output` | 覆盖 `log.level` / `log.output` |

### 2.13 不变量（CPU 侧强制，违反即拒绝该动作、不中断 QA）

| # | 不变量 | 动机 |
|---|---|---|
| 1 | `evidence.jsonl` 每行必须是 `{"content": 非空, "metadata.source": 非空数组}`；`score` / `changelog` 系统可补，**其余字段一律不替 agent 造** | `source` 非空是**反幻觉护栏**：它必须列出**真读过**的 msg_id |
| 2 | `edit` 路径必须落在 `qa_{idx}/` 内 | 只读输入不被误写（软链逃逸也会被拦） |
| 3 | 每 QA 收尾校验**整个会话语料树**的 sha256（逐文件）；不一致 → `tampered=true`，该 QA 作废 | bash 可以绕开 #2，检测成本极低 |
| 4 | 只补全**系统自有字段**（缺 `changelog` → 补 `created`；缺 `score` → 补 0.5），其余不合规的行 **reject 并记账** | 补全语义问题会掩盖失败模式 |
| 5 | 工具 / 解析 / 模型调用失败 = 一次 failed step，记账后继续 | 所有失败路径都退化为无害一步 |
| 6 | 连续 `no_progress_patience` 步**唯一记录指纹**没有变化 → **提前收尾**（判据是唯一记录集合，不是行数/字节数；见 §2.7） | 把空转变成机制而不是 prompt 请求 |
| 7 | 产物解析容忍两种 `jq` 自然输出：一行一个数组、整份美化 JSON（否则规范化会把证据全删） | `jq` 是 agent 的主要工具，它的自然输出必须能被接住 |
| 8 | `edit append` **缺 `path` 时默认 `evidence.jsonl`** | 8B 实测常忘填 path，旧行为是报错 → 它把整条调用重复 10+ 次、12 步烧光、0 条证据（一个显然的默认值不值得判错） |

### 2.14 输出与日志

`data/search_runs/{experiment}_{ts}/`：

- `{dir}/qa_{idx}/evidence.jsonl` — **唯一产物**（不合规时另存 `evidence.jsonl.raw`）
- `{dir}/inputs/` — 只读输入（硬链的三层记忆文件 `msgmem` / `summem` / `atommem`）
- `{dir}/{dir}.qa_trajectories.jsonl` — 每 QA 一行：`evidence_ids` / `steps` / `counts_by_tool` /
  `verifications` /
  `rounds`（每轮含每步的模型输出、工具调用、观测、evidence 行数变化）/
  `tampered` / `stop_reason` / `final_answer`
- `{dir}/{dir}.model_calls.jsonl` — 每次模型调用的原始输出（排错用；`--log-full-messages`
  才带完整 messages）
- `summary.json` — 配置、每目录摘要（`qa_done` / `evidence_total` / `steps_total` /
  `sufficient_rate` / `tampered` / `model_calls`）

日志分级（`configs/search.yaml` 的 `log` 段，命令行可覆盖），**每行都带 `[目录:qaN]` 前缀**
便于并发时区分来源：

- **`info`（默认）**——每个 QA 开跑（含 question 原文与 category）、每个工具调用一行结果
  （含 evidence 行数）、verifier 判定、QA 收尾汇总。
- **`debug`**——上面全部 + 完整数据流：渲染后的 system prompt、每一步的观测、模型的原始输出、
  verifier 的输入与输出、evidence.jsonl 全文、异常 traceback。

`stop_reason` 取值：`sufficient`（verifier 判足够）/ `verify_rounds_exhausted`（判了几轮仍不足）
/ `steps_exhausted`（总步数用尽，agent 还在 `next` 翻页）/ `no_progress(N)`（连续 N 步无进展
提前收尾）/ `error`。

> ⚠️ **成本**：每条 QA ≈ `检索步数 + verifier 次数` 次模型调用。system 段实测 **~15900 字符（约 4k token）且固定** ——
> 可命中 prefix cache；随对话历史与工具观测增长，每步再涨 ~1–2k（见 §2.9 的压缩机制）。
> **唯一的关键旋钮是 `max_steps`** —— 集合型问题（要读多个 session）比事实型贵得多，按最贵的给。
> 先用 `--max-qa 3 --max-steps 20` 试跑，`--dry-run` 可先看 prompt 全貌。
>
> ⚠️ **没有外部依赖**：没有嵌入服务、没有索引、没有检索 CLI。agent 用系统自带的 `grep`
> 检索、用 `date -d` 做算术。整条链路只在调模型时联网。

**原子性**：只写 `data/search_runs/` 下的文件，`inputs/` 是硬链且收尾校验 sha256，
**绝不修改** `data/{dir}/` 下的任何语料。

### 2.15 首次全量跑（Caroline_Melanie，152 QA）的实测结论

> **这组数字来自更早一次全量跑**（当时是"会话时间当事件时间"那一版 prompt），保留它是为了
> 记录下面三个诊断结论 —— 它们与语料层数、与后来的分阶段改动都无关，至今仍是当前设计的依据。
> **当前朴素 loop（§2.2）的实测见 §2.16**，这组数字是更早版本的，不能当作它的成绩。
>
> **两次完整运行的对比在 §2.4.1** —— 那里说明了为什么"58% → 47%"这个表面跌幅
> 绝大部分是判分口径变化，而不是答案变差（用统一判据重判后是 51% → 48%，
> 唯一确凿的移动是 temporal 28% → 50%）。
>
> 所以：**跨版本的分数不能直接比**，除非用同一套 judge 判据重判一遍
> （`scripts/judge_impact.py` 就是干这个的）。

```
search   152/152 QA，0 错误，verifier 判"足够" 64%，473 条 evidence
answer   147 条（5 条无 evidence 可答），证据不足 22 条
eval     judge_accuracy 55.1%   token_f1 0.310   evidence_recall 43.6%
```

| category | n | 准确率 | recall |
|---|---|---|---|
| single_hop | 68 | 66.2% | 41.2% |
| open_domain | 12 | 75.0% | 22.5% |
| multi_hop | 30 | 43.3% | 23.1% |
| temporal | 37 | 37.8% | **70.3%** |

**三个诊断结论**（都是查了数据才敢下的）：

**① `evidence_recall` 不是可靠的失败信号。** 74 条 recall=0 里有 **36 条答案是对的** ——
标注 evidence 只列 1-4 条消息，而同一事实往往有其他消息佐证。所以"recall 低"不等于
"检索差"；判断链路好坏必须看 judge 准确率，recall 只能当参考。

**② temporal 的 recall 70% 但准确率 38% —— 瓶颈已从检索转到作答与判定。** 其中一部分是
**judge 的测量缺陷**：参考用相对表述（"the week before 6 July 2023"）而预测给绝对日期
（`2023-06-29`，**两者在时间点上完全相等**），judge 却判错。8B 推算"13 Sep 2023 是周三、
之前的周末是 9-10 号"这类算术**不可靠**（实测同一条判 3 次都错）。

修法：**日期算术交给代码**（`judge.resolve_relative_date`），把算好的结果作为
`[PRECOMPUTED] ... SAME date` 提示塞给 judge，让它只做比较、不做推算 ——
与 search 阶段"相对时间交给 `date -d`"是同一个取舍。效果：temporal 37.8% → **43.2%**，
且 5 个已知误判里 4 个稳定转正。

**③ 每一条 QA 只写 3.2 条 evidence（均值），且 4-8 条是准确率甜点**：

| evidence 条数 | n | 准确率 |
|---|---|---|
| 0（空） | 8 | 0% |
| 1-3 | 99 | 54% |
| **4-8** | **27** | **81%** |
| 9+ | 13 | 46% |

塞太多会引入噪声（9+ 掉到 46%），塞太少则信息不足。`evidence_soft_limit: 40` 太宽松，
**调到 12 左右更贴合实测的最优区间**。

---

### 2.16 从 bad case 反推的一次大改：31.6% → 59.1%

上一节那张表是"印象判答 + 逐 session"版本的。这一节记录**怎么从 bad case 找到根因、
把 search 砍成一个朴素 loop** 的全过程 —— 这是本项目最实的一次方法改进，也是"范畴判断
优先于堆流程"的最好例证。

#### 先看数据：104 条 INCORRECT 是什么样的

拿旧版全量跑（152 QA）的 `eval_eval.jsonl` 逐条分类：

| 现象 | 条数 | 性质 |
|---|---|---|
| **一次 `grep` 都没做的 QA** | **150 / 152** | 检索被整个短路 |
| INCORRECT 里"证据在但答错"（= 编造） | **97 / 104** | 不是检索不到，是压根没去检索 |
| verifier 判 sufficient 的 / 其中错的 | 131 / **87** | verifier 被"看起来完整"的编造骗过 |
| 集合型问题 `recall=0` | 17 / 18 | 只给一两个成员就收工 |

#### 典型 bad case：0 步检索 + 编造 + 假来源

qa92「What country is Caroline's grandma from?」（参考 **Sweden**）的完整轨迹只有两步：

```
step1 edit: {"content": "Caroline's grandma is from Canada, as Melanie mentioned
             during their conversation...", "metadata":{"source":["session_13"]}}
step2 纯文本 = 完成
```

它**从没读过 session_13**（正确答案在 `session_4_3`），编了一个国家名、还配了个看似
可溯源的假 source，verifier 判 sufficient。同形态的还有：
- qa106「新鞋用来干什么」→ 编"pottery"（答案 running），证据里写着 "**likely required** the use of her new shoes"（臆测）；
- qa93「奶奶的礼物」→ 编"一幅马的画"（答案 necklace）；
- qa104「推荐的书」→ 编"The Power of Now"（答案 "Becoming Nicole"）；
- qa88「期待收养过程的什么」→ 把问题复述一遍当答案。

#### 根因：**顶层概览太像"成品"，把检索短路了**

这些 QA 的共同点是**开头读了一眼 speakers summary 就直接作答**。speakers summary 是
"整段关系的浓缩印象"，里面提过几乎所有主线（家庭、职业、兴趣）。agent 看到问题沾边，
就凭印象凑一条答案 —— 它**有把握**（因为印象里确实提过"家庭背景"），于是不去核对。
浓缩成品 = 一条"看起来够用"的捷径，而捷径一旦存在，模型就会走。

**这不是加规则能修的。** "检索要彻底"这类叮嘱在 8B 上反复失效（§2.9 已多次记录）。
能修的是**类别判断**：既然"给成品"会诱导跳步，那就**不给成品**。

#### 改法：删掉成品，砍成一个朴素 loop

1. **删掉 speakers summary 这一层**（add 回到两步）。开场只给一份**会话清单**
   （`session_N (时间)`，不含任何事实）—— 它是路标，不是答案。
2. **删掉 `next(discard, add)` 上下文块控制工具与整套 stage 机制**，回到**一个 agent loop**：
   `read` / `edit` / `bash` 三个工具 + 一份简短引导（这题该关键词检索还是通篇读，**由 agent
   自己选**）。
3. **唯一新增的引导**是一段 8 行的策略表（§2.2）：关键词检索 vs 通篇读，各给**判据 + 一句做法**，
   **不给流程**。加一句 `write from what you READ, never from what you assume` 针对编造。

#### 效果（同一目录、同一 judge、Caroline_Melanie）

| | 旧（印象判答 + 逐 session） | 新（朴素 loop） |
|---|---|---|
| **judge_accuracy** | 31.6% (48/152) | **59.1% (78/132)** |
| evidence_recall | — | 0.633 |
| unsupported_rate | — | 0.083 |
| token_f1 | — | 0.426 |
| single_hop | 32.9% | **63.5%** |
| temporal | 10.8% | **60.0%**（recall 0.80） |
| multi_hop | 34.4% | **57.7%** |

**检索密度**：旧版全量 **2 次 bash 调用**；新版单目录 **500+ 次**（每个 QA 平均 3-20 步
真检索）。**这就是全部差别**：同样是 8B 模型、同样的语料，只是把它从"看一眼印象就答"
逼回"必须真去 grep"。

#### 修掉的两个 harness 级 bug（与模型能力无关）

跑完新流程后仍有 23 条**零证据** QA 直接失败。逐条看轨迹，其中两个根因是**我们的工具**：

- **`grep "a|b"` 没加 `-E`**：基本正则里 `|` 是**字面竖线**，这条命令永远匹配不到东西、
  返回空，模型据此认定"库里没有"。实测一次全量 **95 次带 `|` 的 grep 有 63 次没加 `-E`**
  （qa2 连查 12 次全是这种空 grep）。**harness 现在自动补 `-E`** 并在观测里说明修正
  （`tools._fix_grep_alternation`）—— 这是"工具该替模型兜住的错"，不是 prompt 能解决的。
- **`edit` 缺 `path`**：8B 写证据时常常只给 `append`、忘了 `path`，旧行为是报
  `path must be a string, got NoneType`，然后它把**整条一模一样的调用重复 10 次**（每次
  被重复检测拒掉），12 步烧光、0 条证据。现在缺 `path` 默认 `evidence.jsonl`。

#### 仍未解决：open_domain（反事实题）从 77% 掉到 25%

诚实记录：这一类（"Caroline 会不会信教""会不会写书"）旧版反而更高（77%）。原因是旧版
凭印象作答时，对这类**没有单一事实、要拿全局印象做推理**的题恰好蒙对得多；新版逼它去
检索，而库里根本没有一条消息直接说"她信不信教"，它检索一圈后要么返回 `null`（证据不足）、
要么给个无锚点回答（qa77 答 "soon"）。**这类题的正确解法是"检索少量关键事实 + 让 answer
阶段从证据推理"，而不是在 search 阶段硬找"那句不存在的话"** —— 属于 §2.2 说的
"哪些判断该留给下游"的问题，待办。但代价是值得的：用它的 4 个点换 temporal +49 点、
single_hop +31 点、multi_hop +23 点。


### 2.17 又一次大改：引入跨会话索引，31.6%→59.1%→等待新数字

§2.16 把 search 从"印象判答"救回"必须真检索"（31.6%→59.1%）。但 59.1% 的跑次
（`search_20260930_173758`）逐条看 INCORRECT，**检索已经发生了**（141 条 QA 里 0 条零
bash），瓶颈换成了两处：

1. **时间推理仍不稳**：temporal 类 16 条答错、9 条找到了却**算错**；只有 7/37 条用了
   `date -d` —— 模型习惯把会话日期直接当事件日期写（正是 §2.4 反复强调、却仍会犯的错）。
2. **跨会话联系收不全**：cat1 multi_hop 平均要跨 **2.41 个 session**（最多 7 个），而 agent
   逐 session 读时"读到后面忘了前面"；只有 33/152 条 QA 读过定位层。散落在几次会话里的
   同一件事（"Caroline 为帮孩子做过的事"分散在 4 次会话）总是只收到一部分。

#### 改法：把定位层从"摘要"换成"索引"

**删掉 `session_summaries`，改成 `index.jsonl`**（设计见 §1.1）：逐会话抽 `(主体, 关系, 对象)`
三元组，合并同一关系在不同会话的出现，每条标注它出现在哪些 session/msg_id/源话时间词。

| | 摘要（已删） | 索引（当前） |
|---|---|---|
| 形态 | 一段浓缩成品 | 一条条 (s,r,o) + 来源标注 |
| agent 拿它干什么 | "哦大致讲了这些" → 常不读原文 | "这事在 session 2、8、13 都出现过" → 去读原文核对 |
| 跨会话连接 | 无（每会话独立） | **合并后有**（一条记录带多个 session 号） |

同时按 §2.16 的教训继续**做减法**：search agent 不再有 `next`，只剩 `read`/`edit`/`bash`；
引导由"关键词检索 vs 通篇读"两条改为"grep 索引 / grep 原文 / 读整段会话"三条（§2.2）；
system prompt 明确**核心目标**是产出符合证据标准的 `evidence.jsonl`。

（本节待新跑次数字回来补齐效果对比表。）

### 2.18 又一次大改：不做导航层，直接把消息消解成上下文无关

§2.17 的 `index.jsonl`（扁平三元组）与 §2.18（原）的 `nodes/edges` 图，都是为"跨会话定位"
**额外造一层导航**：agent 先 grep 指针，再回头读原文。这条路有效，但始终是**两层**——指针
会漏、会压缩、措辞与原文不一致，agent 得在"指针"和"原文"之间来回对。这一版**把导航层整个
去掉**，改成**把每条消息本身改写成自足的**（`add/resolve.py`）：

- **时间消解**：把相对时间锚定成**绝对时间**或**带绝对锚点的相对时间**，粒度严格照源话
  （`last year` → `2022`，不编月日），并标注 `time_kind` = point / range / approx。
- **指代消解**：`she`/`it`/`there` → 真名 / 真实地点，逐消息、喂前文消息作上下文。
- **原话保留**：消解结果进 `content`/`time`，原话进 `source_content`/`source_time`
  （溯源 + re-resolve 幂等的底）。

于是 grep 命中的**就是能回答问题的原料**，不需要指针，也不需要回头对原文；同一实体跨会话
都写成同一个真名，`grep -rin NAME inputs/sessions/` 一次命中它的全部会话——这正是图"边带
session 号"想做的事，但不需要单独一层。

**为什么值得再把图也砍掉**（三条，都是"少一层结构"的收益）：

1. **单一事实源。** 图是原文的**压缩**，压缩就有损；消解是原文的**改写**，无损（原话还在
   `source_*`）。agent 只需读一种东西。
2. **两种检索策略更自然。** 关键词直达（策略①）直接命中自足句；通读（策略②）时每个会话
   文件也自足——不会"读到后面忘了前面"。而图路线下，两种策略都要先经"指针"这一跳。
3. **增量更简单。** 图要"逐会话重抽 + 全局重新合并"；消解只需"重新消解那一个会话文件"
   （读原始视图，天然幂等），没有跨会话的合并态要维护。

代价说清楚：消解是 **O(消息数)** 次 LLM 调用（图是 O(会话数)×逐消息，两者同量级），且
**消解质量直接决定检索质量**——消解错了（把事件时间写成会话时间、把代词解成错的人）就会
固化进语料。所以 prompt 里把三条时间硬约束（自足 / 精度只减不增 / 区分点段大概）写成规则，
并对每条消息都给出**会话时间锚点**（`source_time`）。

（本节待新跑次数字回来补齐效果对比表。）



## 3. answer — 从证据作答

**输入是 search 的产物 `evidence.jsonl`**，不是整个记忆库 —— 这正是整条链路的意义：
search 已经把"回答这个问题所需的记忆"提炼成一份小列表（含推理结论、时间已解析），
answer 只需要基于它作答。

```bash
python -m codemem.answer                       # 回答最近一次 search 的全部 QA
python -m codemem.answer --limit 20            # 只回答前 20 条（调试）
python -m codemem.answer --sample Caroline_Melanie
python -m codemem.answer --evidence-dir data/search_runs/search_20260927_113054
```

| 参数 | 说明 |
|---|---|
| `--evidence-dir PATH` | search 的运行目录；缺省取最近一次 |
| `--sample NAME` / `--limit N` | 只处理指定目录 / 最多回答 N 条 |
| `--concurrency N` | 覆盖并发数（默认 8） |
| `--include-category5` | 包含 adversarial 类（默认跳过） |
| `--output PATH` | 结果输出路径（默认写到运行目录下的 `answers.jsonl`） |

**两个关键设计**：

- **只在证据内作答**。prompt 要求"仅用给定记忆，不得编造日期/数字/人名"；不足以回答时输出
  `{"answer": null, "unsupported": true}`，而不是猜。这个 `unsupported` 是 eval 区分
  **"答错"** 和 **"证据不足"** 的唯一依据。
- **`unsupported` 尊重模型的显式判断**，只在字段缺失时才由"答案是否为空"推断。模型说
  "证据够但我答不出"（`unsupported: false` + 空答案）是 **answer 步骤**的问题，不该被归成
  证据不足 —— 否则 eval 的失败归类会指向错误的方向。

---

## 4. 封装 API（Add / Search）

面向外部评测的 HTTP 封装：把 add/search 两半管线包成一个**多用户、按 `user_id` 隔离、
按 `request_id` 幂等**的服务。

```bash
python -m codemem.api                       # 默认 0.0.0.0:8000
python -m codemem.api --port 9000 --log-level debug
# 或
uvicorn codemem.api.server:app --port 8000
```

| 端点 | 请求 | 响应 |
|---|---|---|
| `POST /add` | `{request_id, messages:[{role, content, timestamp?}], user_id, session_id}` | `{success, request_id, user_id, session_id}` |
| `POST /search` | `{query, options?, user_id, top_k?}` | `{data:[{id, content, score, created_at}]}` |
| `GET /health` | —— | `{status:"ok"}` |

三条贯穿全局的约定：

- **Add 幂等**。`request_id` 是幂等键 —— 重试时同一个 `request_id` **不会再写一遍**
  （由 `meta.applied_request_ids` 记录）。响应的三个 id 原样回显。
- **Add 返回即"立即可检索"**。写入原始消息后，**在返回前**同步重建该会话的 `summem`（一次 LLM
  总结）与 `atommem`（逐 message 的 add agent），所以返回 `success:true` 时消息已消解、可直接检索。
- **Search 的 `data` 永不为缺失**。没有记忆（该 user 从没 Add 过）或没有证据，一律返回
  `{"data": []}`，绝不给 5xx 空响应。

**Search 复用 search 步骤的同一套编排**（`runner.run_phases`，与离线 search 完全一致）：
`query` 归一化成 question → 建运行目录（硬链三个记忆文件进 `inputs/`）→ 一个朴素 agent loop：
`grep` 检索、把**推导出的**证据用 `edit` 追加进 `evidence.jsonl` → verifier 判"够不够"
（不足则回灌再补一轮）→ 映射成协议 `data[]`（按 `metadata.score` 降序、`top_k` 截断）。
**不是朴素关键词检索** —— 协议要的是"能直接回答问题的记忆列表"，而 agent 会照取 atommem 的
绝对日期、消解指代、把散落的事实合并成一条。代价：每条 query 一次（或多次）LLM 调用。

**存储布局**（按 `user_id` 一目录，`data/api_store/{slug(user_id)}/`）：

```
meta.json                 session_id -> 会话号、每会话消息数、已应用的 request_id
msgmem.jsonl              逐条原始消息（type=msg；会话时间戳是锚点）
summem.jsonl              每会话一条总结（type=summary）
atommem.jsonl             逐消息抽出的原子事实（type=atom）
runs/{ts}/                每次 Search 的运行目录（inputs/ 硬链 + qa/evidence.jsonl，可审计）
```

**沿用离线 add 的三层记忆文件与 `session_1_3` 约定**，于是 search 的 prompt、evidence 的
`source` 校验、`date -d` 时间锚点全都照常工作。
`time` 由协议的 `timestamp`（Unix 毫秒）转成 ISO-8601 UTC（`date -d` 可无歧义解析）。

**多模态**：`content` 允许是 `str` 或有序 `ContentPart[]`；数组按原顺序归一化成**一个字符串**，
文本 part 原样保留，图片 part 降级成 `[image: <url>]` 文本标记（可 grep、可溯源、可审计）。
⚠️ **本系统不做图像理解** —— 多模态赛道只把图片当引用保留；文本 / 代码赛道是完整支持路径。

自测（纯 CPU、零模型成本、零网络 —— 假模型 + 真工具 / 真存储）：

```bash
python scripts/test_api.py     # 44 项断言：幂等 / 多 user 隔离 / msg_id 与时间 / data 映射 / 多模态
```

配置在 `configs/api.yaml`：`generator`（search 的 agent+verifier 模型）、`index`（Add 侧
summem/atommem 模型）、`add_tool_config`（Add agent 的工具配置）、`store_dir`、`concurrency`、
`top_k`、`add_max_steps` 与 agent 预算。

### 4.1 实现要点

- **`models.py`**：协议模型（pydantic）+ `ContentPart[]` → 文本的归一化（唯一实现，
  Add 与 Search 共用）。
- **`store.py`**：每 `user_id` 一个目录；`UserStore.append` 在**写锁**内完成"查幂等键 →
  分配 `msg_id` → 追加消息到 `msgmem.jsonl` → 重建该会话的 `summem`（一次 LLM）与 `atommem`
  （逐 message 的 add agent）→ 记 request_id"，三层记忆文件原子写回。
- **`retriever.py`**：直接调 `search.runner.run_phases`（与离线 search 同一套编排），
  复用现成的 `ModelRunner` / `build_shell_prefix` / `link_dir_or_copy` / `validate_file`。
  **不另写一套**：在线与离线的检索行为必须一致，否则"本地调好的 prompt 上线就不准了"。
- **`server.py`**：FastAPI app，协程级 `asyncio.Semaphore` 限流；`create_app(config)` 供测试注入。
- 一处实测踩坑：agent 的 cwd 是 `qa/`、prompt 里所有检索示例写的是 `inputs/...`（相对 cwd），
  所以 `qa/inputs` 必须**也**存在一份链接（与 `runner.run_qa` 一致），否则 grep 永远空。

---

## 5. eval — 判断对错与指标

```bash
python -m codemem.eval                    # 评最近一次 answer（含 LLM judge）
python -m codemem.eval --no-judge         # 只算 CPU 指标（不调模型）
python -m codemem.eval --answers path/to/answers.jsonl --experiment my_exp
```

指标分三组，各回答一个不同的问题：

| 组 | 指标 | 回答的问题 |
|---|---|---|
| **答案质量** | `judge_accuracy` / `exact_match` / `token_f1` | 答对了吗？ |
| **证据质量** | `evidence_recall` | search 找到的记忆覆盖了标注证据吗？ |
| **行为** | `unsupported_rate` / `evidence_count` | 是"证据不足"还是"答错了"？塞了多少条？ |

**失败归类**（`summary.json` 的 `failures`）是最有诊断价值的一项，直接指向下一步该修什么：

| 归类 | 含义 | 该修哪里 |
|---|---|---|
| `missing_evidence` | 证据不足（`unsupported`） | search 的检索/推理 —— 该找的记忆没找到 |
| `wrong_answer` | 证据在但答错 | answer 的 prompt，或 evidence 写歪了 |
| `judge_error` | judge 调了但判不出来 | judge |
| `unjudged` | **没跑 judge**（`--no-judge`） | —— 既不能说对也不能说错 |

> `--no-judge` 时 `judge_accuracy` 报 **`n/a`** 而不是 `0.0`：报 0 会被读成"全答错了"，那是误导。

`evidence_recall` 的比对基准是标注里的**消息** id（`session_1_3`），而 evidence 里存的可能是
**原子记忆** id（`session_1_3_1`）—— 所以打分时会同时取记录的 id 和它 `source` 里的消息 id，
二者任一命中即算覆盖。少了这一步，recall 会恒为 0。

judge 对 category 5（adversarial，故意不可回答）**只记录不评分**，与数据集口径一致。

---

## 6. Baseline 对照（三个独立评测脚本）

这三个脚本各自"检索 + 作答 + 评判"一条龙，**早于 answer/eval 两个步骤**。它们读 `msgmem.jsonl`
与 `atommem.jsonl` —— 这两个文件现在由 add 的三层记忆流水线生成（记录格式已对齐
`memory_template.jsonl`）。脚本是**独立的历史对照链路**（自带的检索/作答/评判口径与现管线不同），
代码保留在 `scripts/` 与 `eval/legacy.py`。新增能力请走四个步骤，不要往这里加。
新增能力请走四个步骤，不要往这里加。

### 6.1 Baseline 评测（完整上下文）

```bash
# 全上下文 baseline，使用所有 msgmem.jsonl 内容，实时显示 judge accuracy 和 ETA
python scripts/eval_baseline.py --experiment baseline_qwen3 --limit 10

# 中断后继续：指定同一个实验目录
python scripts/eval_baseline.py --experiment baseline_qwen3 --resume
```

### 6.2 RAG 评测（msgmem 检索）

```bash
# 通过 sglang /v1/embeddings API 做 embedding RAG，从 msgmem.jsonl 中检索最多 30 条记忆
python scripts/eval_rag.py --experiment rag_qwen3 --limit 10

# 中断后继续
python scripts/eval_rag.py --experiment rag_qwen3 --resume
```

### 6.3 AtomMem 评测（对比 msgmem 和 atommem）

**同时评测两种记忆类型的检索召回和答案准确率**：

```bash
# 快速测试（50个样本）
python scripts/eval_atommem.py --experiment atommem_test --limit 50

# 完整评测（1986个样本）
python scripts/eval_atommem.py --experiment atommem_full

# 断点续跑
python scripts/eval_atommem.py --experiment atommem_full --resume

# 分析结果
python scripts/analyze_atommem_results.py data/eval_runs/atommem_test_*/eval_atommem.jsonl
```

**测试结果（50个样本）**：

| 指标 | msgmem | atommem | 提升 |
|------|--------|---------|------|
| Evidence Recall | 28.0% | **85.1%** | **+204%** |
| Judge Accuracy | 16.0% | **48.0%** | **+200%** |
| Token F1 | 0.089 | 0.169 | +90% |

- **68%** 的问题召回率提升（atommem > msgmem）
- **0%** 的问题召回率下降（atommem 从不比 msgmem 差）
- **净答案准确率提升 +32%**（+16 个问题）

逐题明细可用 `scripts/analyze_atommem_results.py` 分析某次运行的 `eval_atommem.jsonl`。

### 6.4 输出文件

每次运行创建 `data/eval_runs/{experiment}_{YYYYMMDD_HHMMSS}/`，目录包含对应的
`eval_baseline.jsonl`、`eval_rag.jsonl` 或 `eval_atommem.jsonl` 以及 `summary.json`。
summary 按 LoCoMo category 1-5 输出指标；category 5 (`adversarial_skip`) 只记录结果，不参与评分。
模型地址、生成模型、judge 模型和 embedding 模型分别配置在 `configs/baseline.yaml`、
`configs/rag.yaml`、`configs/atommem.yaml` 和 `configs/judge.yaml`。

embedding 服务需要以 embedding 模式启动：

```bash
sglang serve \
  --model-path /root/autodl-tmp/Train/model/Qwen3-Embedding-4B \
  --port 30001 \
  --is-embedding \
  --mem-fraction-static 0.30 \
  --context-length 4096 \
  --max-running-requests 32 \
  --chunked-prefill-size 4096
```

---

## 附录 A · 运行辅助

### A.1 一条命令跑通三步（`scripts/run_eval.py`）

search 是改动最频繁的一步，而完整流程是三步、各有自己的 CLI 与产物路径。这个脚本把它们串起来，
并把**该看的数字**摆成一张表：

```bash
python scripts/run_eval.py                        # 默认：1 个目录 × 前 5 条 QA，判分
python scripts/run_eval.py --tag v5-time-rule     # 打标记，便于和上一版对比
python scripts/run_eval.py --limit 0              # 0 = 全部 QA（慢）
python scripts/run_eval.py --no-answer            # 只跑 search，看 evidence 产出
python scripts/run_eval.py --no-judge             # 跑完三步但只算 CPU 指标（省 judge 调用）
python scripts/run_eval.py --reuse search_20260929_183148   # 复用 evidence，只重跑 answer+eval
python scripts/run_eval.py --quick                # 冒烟：--max-steps 8 --limit 3
```

输出长这样（含按 category 拆开那一栏 —— temporal 是时间规则改动的直接反馈）：

```
-- search --
  目录                           QA  evidence      步数      充分
  Caroline_Melanie       5/5             23      61    80%
-- eval（整体）--
  judge_accuracy   0.600     evidence_recall  0.550     unsupported_rate  0.200
-- eval（按 category）--
  category           n    judge   recall    unsup
  temporal           3    0.667    0.700    0.000
-- 失败归类 --
  missing_evidence      1   → 修 search：该找的没找到
```

**默认是"快反馈"配置**（小样本、步数偏紧），不是全量 —— 几十条 QA 就能看出 prompt 改动的
方向对不对。两处设计上的细节值得一说：

- **它接的是"这一次"的运行目录，不是"最近一次"。** 各步骤原本各自取最近一次，而调试时很容易
  接错 —— 比如你刚跑过一次无关的 `--dry-run`（它也会建目录）。这里改成"跑前记下已有目录、
  跑完取差集"。
- **eval 强制带 `--experiment eval`。** eval 默认把 `summary.json` 写进 `--output-dir`，
  而 search 已经在那里放了一份 —— 不加前缀就会**覆盖掉**，于是"这次 search 跑了多少步、
  多少条 evidence"当场丢失，而摘要正要读它。现在写的是 `eval_summary.json`，两份并存。

### A.2 自测

纯 CPU、零模型成本、零网络（假模型 + 真工具）：

```bash
python scripts/test_search.py       # 全链路断言（含两项静态检查）
python scripts/test_api.py          # 封装 API 自测（37 项）
```

`test_search.py` 开头是两项**静态检查**（都纯 AST，不用跑起来）：

- **无未定义引用**：确保没有"调用了但没定义/没导入"的名字 —— 重构最容易留下的就是它。
- **无过时关键字参数**：调用处传的 `kwarg=` 必须真的存在于签名里。
  这一条是**实测补上的**：删掉向量索引时去掉了 `run_qa` 的 `index_dir` 形参，但调用处还留着
  `index_dir=index_dir` —— 全量自测 304 项全绿（测试文件自己也传了这个参数，而测试里那个名字
  恰好还存在），一上真机就 `NameError`。**形参与实参是两个地方，删一个不等于删另一个。**

### A.3 常见问题

**路径里的反斜杠**：Git Bash 下 `H:\Project\...` 的反斜杠会被 shell 吞掉，请用正斜杠。

**输出被截断（`finish_reason=length`）**：某些模型会消耗大量 reasoning token，且这些 token
计入 `max_tokens`。日志出现 `TruncatedCompletion` 时调大 `max_tokens`。

**`chat_template_kwargs` 报 400**：该参数只对模型名含 `qwen3` 的端点发送，其它端点自动跳过。

**embedding 的两个实测坑**：
- **400 且提到 tokenizer 端口连不上** —— 并发压力下的瞬时故障，与请求内容无关，报错信息具
  误导性。已按 `chunk_size` 分批 + 每批退避重试；批越小越稳（批 64 实测 3/3，批 512 只有 1/3）。
- **`cudaMalloc failed: out of memory`** —— embedding 模型要占 GPU，腾显存后再跑。

**远端 API key**：`configs/*.yaml` 里的 key 是明文，`.gitignore` 已排除 `configs/`，注意不要提交。

---

## 附录 B · TODO

**已完成的架构重构**（2026-09-27）：围绕 add / search / answer / eval 四个步骤重组，
共享层提到 `codemem/` 顶层（`io` / `llm` / `log` / `dataset`）。
删掉了 v2 的动作空间版（`evomem.py` + `evoactions.py`，git HEAD 仍可找回）。

- [x] **S0** `evidence_to_ids` 解析 bug 已修（空格分隔多引用 `"D9:1 D4:4 D4:6"` 与 `"D:11:26"`
      都能解析）—— 现由 `dataset.evidence_to_ids` 唯一实现
- [x] **S1** 更严格的 CPU 不变量：`evidence.jsonl` 的校验（`content` 非空、
      `metadata.source` 非空且元素合法、重复 id 拒绝…）在 `search/evidence.py`
- [x] **S3** search 步骤（code-agent 式证据构建）：见 §2
- [x] **S4** answer / eval 两个步骤打通，`answers.jsonl` + `eval.jsonl` 端到端可跑
- [x] **S5** 静态回归网：`test_search.py` 用 AST 扫全包，杜绝"调用了但没导入"
- [x] **S6** add 只留 session summary（当时删掉了顶层 speakers summary）；search 简化为
      「检索就是 grep」的搜索 agent；补上时间精度规则（见 §2.3 / §2.4）；删掉向量检索与检索 CLI
- [x] **S7** `scripts/run_eval.py` 一条命令跑通三步并打印摘要；补一项静态检查
      （过时关键字参数 —— 它正是 S6 在真机上炸的那个错因）
- [x] **S8** 封装 API（Add / Search）：多用户隔离、request_id 幂等、复用 search agent 管线、
      多模态降级（见 §4）
- [x] **S9** 证据标准抽为独立 prompt 段 `EVIDENCE_STANDARD`；`edit` 加 append 模式、删掉 `write`
- [x] **S10** **从 bad case 反推的大改**（2026-09-30，见 §2.16）：删掉顶层 speakers summary
      （实测它把检索短路 —— 150/152 条 QA 一次 grep 都没做）、删掉 `next` 上下文块控制工具与
      整套 stage 机制，search 砍成**一个朴素 agent loop**；唯一引导是"关键词检索 vs 通篇读"
      的策略表（由 agent 自己选）。同目录同 judge：**31.6% → 59.1%**（temporal 10.8%→60%、
      single_hop 32.9%→63.5%、multi_hop 34.4%→57.7%）。顺带修两个 harness 级 bug：
      `grep` 缺 `-E` 自动补、`edit` 缺 `path` 默认 evidence.jsonl
- [x] **S11** **引入跨会话索引**（2026-10-01，见 §1.1 / §2.17）：删掉 `session_summaries`
      这一层，add 第二阶段改为逐会话抽 `(主体,关系,对象)` 三元组、合并成 `index.jsonl`
      （每条带它出现在哪些 session/msg/源话时间词）。索引是**导航不是摘要** —— 摘要（无论
      顶层印象还是逐会话）实测都让 agent 跳过检索。跨会话连接是它的核心价值（multi_hop
      平均跨 2.41 个 session）。search 侧继续做减法：不再有 `next`，只剩 read/edit/bash；
      引导改为"grep 索引 / grep 原文 / 读整段会话"三条；system prompt 明确核心目标
- [x] **S12**（已废弃，被 S13 取代）定位层升级为实体关系图：`nodes.jsonl` / `edges.jsonl`
- [x] **S13** **消解取代导航层**（2026-10-02，见 §1.1 / §2.18）：删掉 `nodes/edges` 图与
      `index.jsonl` 一路的"额外导航层"，add 第二阶段改为**逐消息消解**——
      **时间**锚定成绝对时间或带绝对锚点的相对时间（`time` + `time_kind` point/range/approx，
      粒度只减不增）、**指代**消解成真名（逐消息、喂前文消息作上下文）、**原话**保留在
      `source_content`/`source_time`（re-resolve 幂等的底）。产物改为
      `data/{dir}/sessions/session_N.jsonl`（**一次会话一个文件**）。search 侧：语料是
      `inputs/sessions/` 目录，两种检索策略（keyword 直达 / 通读相关 session）集成在一个
      agent loop 里、运行中自行切换；`evidence.jsonl` 同时是**笔记**（相关即可写，不必只写答案）
- [x] **S14** **三层记忆 + add agent 逐消息抽取**（2026-10-04，见 §1）：把"改写消息本身"
      升级为**分层**——`msgmem.jsonl`（`type=msg`，原话 + 会话时间锚点，纯 CPU 转换）/
      `summem.jsonl`（`type=summary`，逐 session 一次 LLM 总结）/ `atommem.jsonl`
      （`type=atom`，**add agent 逐 message 处理**：复用 search 的 `run_agent_round` /
      `build_tools` / `ToolCatalog`，默认 `max_step=8`，agent 只填 `memory` + `metadata.{time,
      role, tag}`，`id`/`source`/`type`/`changelog` 由 `finalize_file` 系统补、并支持合并演化）。记录格式对齐
      `memory_template.jsonl`。同目录逐 message **串行**（后面的 message 能 grep 到前面已生成的
      atommem，消解跨 session 指代）、不同目录并行；按 `source` 增量覆盖 = 幂等。search 语料换成
      硬链的三个记忆文件；api 的 `store.append` 在返回前重建该会话的 summem + atommem。
      删掉 `add/resolve.py` 与 `add/session.py`

待办：

- [ ] **N0**（最高优先）**open_domain（反事实题）从 77% 掉到 25%**（§2.16 末）：这类题没有
      单一事实、要拿全局信息推理，而 search 阶段硬找"那句不存在的话"只能返回 null。方向是把
      少量关键事实检索出来、让 **answer 阶段**从证据推理，而不是在 search 阶段逼它找反事实结论
- [ ] **N1** 效果评估：跑完整 10 个目录，对比 baseline 的 evidence recall 与 judge 准确率；
      盯 `sufficient_rate`、eval 的 `failures` 归类、`verifier_sufficient` vs `judge_correct`
      的背离。§2.15/§2.16 目前只有 Caroline_Melanie 一个目录的数字
- [ ] **N2** search 的多跳补强：`missing[]` 回灌已实现，可考虑让 verifier 能"点名要某条记忆"
      （当前只能描述缺口，由 agent 自己检索）—— 零证据的 23 条 QA 是这条的实证
- [ ] **N3** `ANSWER_PROMPT` 加 `used_ids` 字段（后续做 credit 归因要用）
- [ ] **N4** search 的强化学习：QA 驱动的证据构建已是一条完整 trajectory（状态 = evidence.jsonl，
      终局奖励 = judge 答案），可直接在其上做 oracle 轨迹采样 + GRPO
- [ ] **N5** 三个 baseline 脚本（`eval_baseline/rag/atommem`）与新链路的指标口径统一，
      使"msgmem / atommem / evidence / 全上下文"四者可横向对比
- [ ] **N6** 三层记忆的端到端效果评估（S14 尚未跑真实模型）：对比"三层（atommem 检索）"与
      "单层（消解后消息）"的 evidence recall 与 judge 准确率，验证 atommem 的拆分/消解质量

（注：`data/*/atommem.jsonl` 的 10 个目录为旧原子记忆体系的遗留数据，格式与 S14 不同。）

### S15: 优化 atommem 时间处理 + 原子粒度（2026-10-06）

**bad case 分析（search_20261005_235035）**：总体准确率 50.8%，temporal 只有 40%、multi_hop 31.8%。
逐条分析 104 条 INCORRECT，发现三类核心问题：

#### 问题 A: 时间处理过于机械（11/35 temporal 错）

**案例 A1**: 相对时间被硬算成绝对日期，丢失锚点

```
原始: "my school event last week" (9 June 2023)
atommem 产出: time="2023-06-02"
正确答案: "The week before 9 June 2023"
系统答案: "9 June 2023" ❌
```

**根因**: `parse_time_expression` 把 "last week" 用 `date -d` 算成了 2023-06-02，但丢掉了原始表达和锚点。

**案例 A2**: 持续时间信息完全丢失

```
原始: "5 years already!" (9 June 2023)
atommem 产出: time="2023-07-14"
正确答案: "5 years"
系统答案: "0 days" ❌
```

**根因**: pipeline 只处理了事件时间点，"5 years" 这个持续信息完全丢失。

#### 问题 B: 原子抽取粒度不当

**案例 B1**: 复合概念未拆分

```
原始: "Luna and Oliver! They are so sweet..."
atommem 产出: 只有 Oliver 和 Bailey 的独立 atom，Luna 没有
正确答案: "Oliver, Luna, Bailey"
系统答案: "Oliver and Bailey" (漏了 Luna)
```

**案例 B2**: 关键属性被掩盖

```
原始: "tough as a single parent"
atommem 产出: "Caroline is up for the challenge of being a single parent."
正确答案: "Single"
系统答案: "in a committed relationship" ❌
```

**根因**: 只关注动作/状态，"single" 这个关键属性被埋没了；没有独立抽一条 "Caroline is single"。

#### 优化方案（S15）

**方案 A: 改进时间处理（P0）**

1. 新增 `metadata.time_expr` 和 `metadata.time_anchor` 字段保留原始时间表达
2. 修改 `parse_time_expression` 策略：相对时间优先保留 "原表达 (锚点)" 形式，只有精确时点（yesterday）才算成绝对日期
3. 修改 EVIDENCE_STANDARD：优先使用 `time_expr + time_anchor` 而不是硬算的 `time`

```json
{
  "memory": "Caroline talked about her transgender journey at a school event.",
  "metadata": {
    "time": "last week (anchor: 9 June 2023)",
    "time_expr": "last week",
    "time_anchor": "7:55 pm on 9 June, 2023"
  }
}
```

**方案 B: 改进原子粒度（P0）**

1. 集合成员强制拆分：ATOM_EXTRACTION_SYSTEM 规则 6 新增 "Split compound entities"
2. 关键属性显式抽取：规则 7 新增 "Extract key attributes explicitly"
3. 示例补充：添加 "Luna and Oliver" 拆分、"single parent" 属性抽取的示例

**实测效果**：
- ✓ time_expr/time_anchor 正确产出
- ✓ "last week" 保留为 "last week (anchor: 9 June 2023)"
- ✓ "yesterday" 算成 "2023-06-08"
- ✓ prompt 新增集合拆分 + 关键属性抽取规则

**预期提升**: temporal +15-20%（从 40% → 55-60%），single_hop +10-15%（从 62% → 72-77%）。
