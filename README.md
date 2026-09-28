# CodeMem

面向代码/对话记忆的原子记忆系统。整个架构围绕**四个步骤**组织，依赖方向严格单向，
步骤之间**只通过数据文件耦合**（不互相 import）：

```
add ──► msgmem.jsonl ──► atommem.jsonl
                             │
                             ▼
search ──────────────► evidence.jsonl          （question + 记忆库 → 能回答该问题的记忆列表）
                             │
                             ▼
answer ──────────────► answers.jsonl           （question + evidence → 答案）
                             │
                             ▼
eval ────────────────► eval.jsonl / summary.json   （判断对错 + 各项指标）
```

| 步骤 | 做什么 | 入口 |
|---|---|---|
| **add** | 一段对话记忆 → `msgmem`（raw message memory）→ `atommem`（原子记忆） | `python -m codemem.add` |
| **search** | question + `atommem` → `evidence.jsonl`（一个 code agent 自主检索/推理/写作） | `python -m codemem.search` |
| **answer** | question + `evidence.jsonl` → 答案（只用证据，不检索） | `python -m codemem.answer` |
| **eval** | 答案正确性 + 指标（judge / token F1 / evidence recall / 失败归类） | `python -m codemem.eval` |

---

## 环境

```bash
pip install openai pyyaml tqdm
```

Python 3.12+。`jq` 需要可执行（search 的 agent 用它处理 JSON）。

---

## 目录结构

```
configs/
  add.yaml      # add 的模型与抽取参数（含 dpo 阶段）
  search.yaml   # search 的 agent/检索/预算参数
  answer.yaml   # answer 的模型参数
  eval.yaml     # eval 的 judge 参数
  tool.json     # search 的工具描述/schema/prompt/示例（agent 上下文的单一事实源）
data/
  correct_locomo10.json        # 原始对话数据（含 QA 与 evidence 标注）
  {speaker_a}_{speaker_b}/
    msgmem.jsonl               # add 产物：raw message memory（只读输入）
    atommem.jsonl              # add 产物：原子记忆（search 的输入）
  search_runs/{experiment}_{ts}/   # search/answer/eval 的运行目录
    {dir}/inputs/                   只读输入（硬链）+ 预建索引 + search/timecalc 包装
    {dir}/qa_{idx}/evidence.jsonl   search 的产物
    answers.jsonl                   answer 的产物（该次运行的全部回答）
    eval.jsonl / summary.json       eval 的产物
  dpo/atom_dpo.jsonl           # add --stage dpo 的产物（偏好训练数据）
  eval_runs/                   # 三个 baseline 评测脚本的输出
src/codemem/
  io.py llm.py log.py dataset.py    # 共享层：四个步骤都依赖，自身不依赖任何步骤
  add/    msgmem.py atommem.py dpo.py __main__.py
  search/ runner.py agent.py tools.py toolcfg.py verifier.py evidence.py
          searchmem.py index.py embedder.py searchctl.py timecalc.py __main__.py
  answer/ answer.py __main__.py
  eval/   runner.py judge.py metrics.py legacy.py __main__.py
scripts/
  test_search.py test_searchmem.py   # 纯 CPU 自测（假模型 + 真工具，零模型成本）
  eval_baseline.py eval_rag.py eval_atommem.py analyze_atommem_results.py  # baseline 对照（见 §5）
  regenerate_rejected.py transfer_data_to_msswift_standard.py locomo_data_loader.py
memory_template.json                 # 记忆条目字段定义（完整规范）
memory_template_init_extract.json   # 原子记忆抽取字段定义（注入 add 的 prompt）
memory_template_evomem.json         # 原子记忆演化字段定义（注入 search 的 prompt）
docs/search.md                      # search 步骤的设计文档
```

**共享层（`codemem/` 顶层四个模块）** —— 放在任何一个步骤里都会造成反向依赖，所以提出来：

| 模块 | 职责 |
|---|---|
| `io.py` | JSONL 读写（含"整份美化 JSON"兜底）、路径解析、memory item 访问器、`repair_json` |
| `llm.py` | chat completion：退避/重试/截断检测/qwen3 关思考 |
| `log.py` | 分级日志（终端 + 落盘，支持 `bind` 加前缀） |
| `dataset.py` | LoCoMo 数据模型：sample/QA/evidence 映射（`D1:3` → `session_1_3` 的唯一实现） |

---

## 1. add — 把对话记忆添加进来

```bash
python -m codemem.add                        # 两步都跑，全部目录
python -m codemem.add Caroline_Melanie       # 只处理指定目录
python -m codemem.add --stage msgmem         # 只跑第一步（纯 CPU，不调模型）
python -m codemem.add --stage atommem --limit 3   # 只跑第二步，每目录前 3 条（调试）
python -m codemem.add --stage atommem --resume    # 断点续跑，只重试失败项
```

**第一步 `msgmem`** —— 把 `correct_locomo10.json` 的每个 session 摊平成 raw memory：

| 字段 | 值 |
|---|---|
| `memory` | 原始消息文本 |
| `id` | `session_{会话号}_{消息号}` |
| `type` | `raw` |
| `time` | **会话首条**为该会话真实时间（ISO-8601）；其余为 `few minutes in {会话时间}` |
| `tag` | `["speaker:<name>"]` |
| `source` | `[]`（raw 无出处） |
| `changelog` | `[{"time": <真实创建时间>, "content": "created"}]` |

> `time` 的这条设计是故意的：让 search 阶段的 agent 从**任何一条**消息都能推出所在会话的
> 绝对时间 —— 相对时间推理（"about a month ago"）的锚点就是这么来的。

**第二步 `atommem`** —— 逐条读 raw、带上下文窗口调模型抽原子记忆（`context_window: 4` 表示前后各 4 条）。
`id`（`{raw_id}_{n}`）与 `changelog` 由系统补全；解析失败或截断的条目写入 `atom_errors.log`，
不影响其它条。`msgmem.jsonl` 始终保持不变。

`--stage dpo` 是可选的偏好训练数据生成（强模型出 chosen、弱模型出 rejected），
配置在 `configs/add.yaml` 的 `strong`/`weak` 两段。

---

## 2. search — 生成 evidence.jsonl

**目标**：对每条 question，产出一份**真正能回答它的记忆列表**，落盘为 `evidence.jsonl`。
设计文档：`docs/search.md`。

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
| `dirs` / `--sample NAME` | speaker 目录名或路径；缺省处理全部含 `atommem.jsonl` 的目录 |
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

设计文档：`docs/search.md`。

**为什么不是 v2 那套流程**：v2 实测（5 QA / 164 次调用）动作分布 `UPDATE 146 / NOOP 18 /
ADD 0 / DELETE 0`，库净变化 `+0 ~81 -0` —— 只会原地改写、从不新增、81 次无效改写。三条根因：
① 检索只做一次，候选集第一步就定死（多跳问题死在这）；② 动作空间只有"单条原地精修"，没有
能力**创造**把两条记忆连起来的事实；③ 成功判据（"更自足"）不可检验。

**code agent 一次解决三条**：检索是 shell 命令，想调几次调几次（→①）；`write`/`edit` 能写出
全新记忆（→②）；产物是文件，可被独立验证（→③）。外加一个附带的正确性收益：agent 只见到
`jq` 吐出的**原始 JSONL 行**，从来没有"展示串"这个概念 —— v2 把
`[id] (time) Speaker | type |` 抄进正文的格式 bug 从根上不可能发生。

### 3.1 核心抽象

```
每条 QA 一个私有目录（QA 之间零共享 → QA 级并发；base 库只读）

  <run>/{dir}/inputs/{atommem,msgmem}.jsonl   只读（硬链自 data/{dir}/）
  <run>/{dir}/inputs/index -> .index_cache/   检索索引（符号链接，跨运行共享，见 §2.3）
  <run>/{dir}/inputs/search                   search CLI 包装（PATH 里直接调）
  <run>/{dir}/qa_{idx}/                       agent 的工作目录（bash 的 cwd）
      evidence.jsonl                          **唯一产物**

  agent loop（read/write/edit/bash，≤ max_steps 步）
      ↓ 不再调用工具 = "我做完了"
  verifier（只看 question + evidence，看不到对话/编辑历史，**不看 reference**）
      ↓ 不足 → 把 missing[] 喂回去再跑一轮（≤ max_verify_rounds）
  校验 evidence.jsonl（§6.4）+ 校验 inputs/ 未被篡改
```

**Verifier 为什么必须独立**：一个既能编辑又能宣布成功的 agent，最省力的动作永远是宣布成功
—— v2 的 NOOP 泛滥就是这一偏差的温和版本。所以让一个**手里只有最终产物、看不到编辑过程**的
旁观者来回答；它无法被措辞说服，只能被内容说服。**不给它参考答案**是因为用 reference 当闸门
就是 oracle 泄漏（RL 会学会猜答案，推理时也拿不到答案键）—— 闸门检验**可回答性**，正确性只在
评测/奖励时用 judge 算。它输出的 `missing[]` 是下一轮最有效的观测，这就是"探索"的实际含义。

### 3.2 工具与 `configs/tool.json`

四个内置工具（`src/codemem/tools.py`）的执行语义**对齐 `reference/tools.py`**（含 2000 行/50KB
截断、`edit` 的唯一性校验、bash 的 `shell_command_prefix`、超时杀进程组），并加了两处约束：
`write`/`edit` 的目标路径必须落在工作目录内（软链逃逸会被 `resolve()` 拦下），因此
`qa_{idx}/inputs` 只能读不能写。

**`search` 是一条终端命令**（`python -m codemem.search.searchctl`），复用 `searchmem.py`
的三路 RRF，每行输出一条 JSON，直接喂 `jq`。它搜**三个语料**：

| `--source` | 搜什么 | 什么时候用 |
|---|---|---|
| `base` | `inputs/atommem.jsonl`（抽出的原子） | 找"已提炼过的结论" |
| **`raw`** | **`inputs/msgmem.jsonl`（原始消息）** | **答案只在原话里时（见下）** |
| `evidence` | 当前 `evidence.jsonl` | 查自己已收的证据有没有重复/缺口 |
| `all`（缺省） | 三者并集，evidence 优先 | 一般检索 |

还有 `--tag` / `--exclude-tag`（按标签**枚举**候选），用于"某人做过哪些 X"这类
**集合型**问题 —— 见 §2.2。

```bash
search "counseling mental health" -k 8                 # 原子 + 原话一起搜
search "counseling mental health" -k 8 --source raw     # 只在原话里搜关键词
search "support group" -k 5 --source evidence          # 只搜本 QA 已写出的证据
search "support group" -k 8 --lexical                  # 嵌入服务挂了时的词法回退
```

**为什么必须能搜原话**（这是实测修掉的一个真问题）：原子是**提炼过的**，常常把具体答案
泛化掉了。例如"Caroline 想读什么方向"，库里最强的原子只有一句泛泛的
`continue her education and check out career options`，而真正的答案
`I'm keen on counseling or working in mental health` **只存在于原始消息 `session_1_11` 里**。
只搜原子时，agent 找不到答案 → 只能猜消息 id → 猜到了别的 session → 重复查询被拒 →
空转到预算耗尽（实测 12 步、evidence 0 条）。让 `search` 覆盖原话，并按问题关键词检索，
这条 QA 现在 8 步内以正确答案收敛。

### 2.2 集合型问题要靠标签枚举，不能靠相似度

有一类问题的答案是**一个集合**："Melanie 参加哪些活动？"、"Caroline 有哪些爱好？"。
稠密检索对这类问题**结构性无能为力**：它按与 query 的相似度排序，而"她要去游泳了"
这条记录与"Melanie 参加什么活动"的相似度并不高（实测 dense 排名 **101**），于是永远
进不了 top-k —— 把 `-k` 调到 100 也救不回来，因为相关度天梯本身就不指向它。

但库里**已经有结构化标签**（`activity:pottery`、`topic:activities`），按标签枚举才是
这类问题的正确工具：

```bash
# 先看看实际有哪些 tag 值（词表不受控，得先探）
jq -r '.metadata.tag[]' inputs/atommem.jsonl | sort | uniq -c | sort -rn | head -30

# 按标签收全候选，再由 agent 判断哪些算"活动"
search "Melanie" -k 100 --source base --tag activity --tag event --exclude-tag action:ask
```

**实测效果**（qa15 "What activities does Melanie partake in?"，标注答案
`pottery, camping, painting, swimming`）：

| | 只靠相似度 | 加上标签枚举 |
|---|---|---|
| 4 个答案的 dense 排名 | 101 / 79 / 536 / 8 | 22 / 24 / 36 / 8（过滤后） |
| evidence_recall | 0 | **0.75** |
| 结局 | 12 步、evidence 0 条 | 14 步、39 条、sufficient |

**为什么仍收不全 —— 根因在 `add` 侧，不在检索侧**。同一个"活动"概念，四个标注答案
的原子分别用了**四种不同的 tag key**：

| 原子 | 标注 | 实际 tag |
|---|---|---|
| `session_5_4_1` | pottery | `action` |
| `session_9_1_2` | camping | `topic:activities` |
| `session_1_18_2` | swimming | `action:Swimming` |
| `session_1_12_1` | painting | `topic:art` |

`topic:` 有 1168 条、`action:` 225 条，外加 `activity:` / `event:` / `intent:` / `hobby:`
等 **30+ 个不同 key**，全是模型即兴发明的。没有哪一个 `--tag` 能一次收齐 ——
**这是标签词表不受控，不是检索算法的问题**。

已做的两件事：

1. **收紧抽取模板**（`memory_template_init_extract.json`）：tag 定义从"列几个 key 名"
   改成**固定 key 表 + 每个 key 的使用场景**（爱好一律 `activity:`；只是被谈到用 `topic:`；
   一次性言语行为用 `action:`），并明确"不要发明新 key"。
   **旧数据要重新跑 `add` 才能受益**（`python -m codemem.add --stage atommem`）。
2. **给 agent 一个枚举入口**：`search --tag-values --tag speaker:X --tag activity`
   直接列出标签值及计数，比让它读 100 条检索结果省事得多（实测 20KB → 1KB）。

> 这条也说明一个更一般的结论：**结构化的收益取决于结构本身的质量**。检索能按标签枚举，
> 前提是标签是可枚举的；标签由 `add` 阶段的模型生成，而那个阶段当时没有任何词表约束。

**两个方向都能走**（agent 的 `source`/`derived_from` 字段就是为此设计的）：

- **原子 → 原话（溯源）**：每条原子的 `metadata.source` 列出它来自哪些消息 id，读原话拿到精确措辞。
- **原话 → 原子（发散）**：`search "KEYWORDS" --source raw --derive` 给出这句话抽出了哪些原子
  （`derived_from` 字段）。

检索结果默认带 `kind`（`atom` / `raw`）与 `source`，所以溯源不必再多一轮 `jq`。

**`timecalc` 是第二命令行工具 —— 确定性日期算术。** 为什么要单独做一个命令：LoCoMo 里大量
问题问"什么时候"，而证据里是相对表述（"about a month ago"、"two weeks later"），8B 做日期
算术**不可靠**，算错就直接变成错误证据。所以把算术交给工具，模型只决定要算什么：

```bash
timecalc shift 2023-06-17 -1month     # -> 2023-05-17（月份越界取月末：01-31 +1month -> 02-28）
timecalc diff  2023-05-08 2023-06-17  # -> +40 days, ~1.3 months
timecalc range 2023-05-01 2023-05-31  # -> 2023-05-01/2023-05-31
timecalc info  2023-05-08T13:56:00    # -> 2023-05-08 (Monday), 2023-W19
```

**工具描述 / schema / prompt / 示例全部在 `configs/tool.json`（单一事实源）**，由
`src/codemem/toolcfg.py` 渲染注入。改 agent 行为 = 改 `tool.json`，不动代码。

#### 3.2.2 证据是**推导**出来的，不是抄的

本方案的核心，也是相对上一版最大的一处语义变化：

> **`evidence.jsonl` 不是库的子集，而是一份可以直接回答问题的记忆列表。** 下游只看到这个
> 文件 —— 看不到库、看不到原始消息。所以所需的推理必须**在记录里完成**。

因此 agent **可以也应该改写 `memory` 正文**：

- **消解指代**：`she`/`he`/`there` → 真实姓名/地点；
- **消解时间**：`yesterday` / `about a month ago` → 绝对日期（用 `timecalc`，锚点是
  `inputs/msgmem.jsonl` 里该 session 的时间戳）；
- **合并事实**：两条只有一起看才有意义的记录 → 一条陈述结论的记录；
- **推理**：时间题要算术、事件题要合并、**性格/属性题要把散落陈述汇总成一条**；
- **补一条库里没有任何单条记录陈述过的结论**，`source` 列出推理所依据的全部 id。

实测产出（性格/属性推理）：

```
memory: "Caroline is pursuing a career in counseling and mental health,
         driven by her interest in adoption and family planning."
source: ["session_1_9_1","session_1_9_2","session_2_8_1","session_2_8_2","session_1_17_2"]
```

反幻觉护栏不变：**`source` 必须非空**，且必须是**本会话真读过的** id（不是记得或猜的），
`source` 为空的记录一律 reject。

#### 3.2.3 标准 code-agent loop 的几件事

上一版实测暴露了四个问题，这一版逐一修掉（每条都有日志证据）：

| 实测问题 | 做法 |
|---|---|
| **93% 的观测是空的**（`jq ... >> file` 重定向走了 stdout），模型看不见自己刚写了什么，于是盲目重试 | 每次**变更型**调用后由 harness 附一段**证据摘要**：行数 / 唯一 id 数 / 末尾几行 |
| **60/60 步没有一次完成信号**，12~24 步全烧在重复追加 | **重复检测**：同一工具+参数、**且文件状态未变**时直接拒绝并给出脱困提示（状态感知 —— 文件变了就允许重跑） |
| 空转被算作"有进展"（往文件里追加重复行） | 进展判据改成**唯一记录集合的指纹**，不是行数、也不是字节数；重复追加不算进展 |
| prompt 随历史无限涨（3.3k→4.5k tok / 12 步） | **上下文压缩**：超预算即把中段坍缩成摘要，保留 system + 最近 N 条 + **工具用过的 id 集合**（保 id 是关键，否则模型会忘了已收过哪些而重复追加） |

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

### 2.3 存储：索引跨运行共享，产物分层存放

一次 search 的产物里**索引占了 99%**：一个目录 1688 条 × 2560 维 float32 ≈ **17MB**，
而它只取决于 `(语料, 嵌入模型)` —— 与哪次运行、哪条 QA 完全无关。早期实现把它建在**每次
运行的目录里**，于是跑 23 次就是 **334MB 完全相同的重复数据**（实测）。

现在分层，各自按真正的变化频率存放：

| 内容 | 位置 | 重复情况 |
|---|---|---|
| `atommem.jsonl` / `msgmem.jsonl` | 每次运行的 `inputs/` | **硬链**（inode 共享，0 额外空间） |
| **索引** `vectors.f32` | `data/search_runs/.index_cache/{corpus}-{model}/` | **一份，跨运行共享**；运行目录里只有**符号链接** |
| evidence / trajectory / 日志 | 每次运行目录 | 每次一份（本来就该有） |

于是同一目录反复跑（调试、`--qa`、`--repeat`）**只嵌一次**：

```
首次：建检索索引：1688 条（原子 1269 + 原话 419） -> 共享缓存 …/ae345cf000210e41-de1c6d0a
之后：复用共享索引缓存：1688 条 × 2560 维（1269 原子 + 419 原话）
```

**单次运行 19MB → 0.7MB。**

**缓存安全**：目录名是 `corpus_sha256` + 嵌入模型指纹。语料一变（重跑 `add` 抽取、正文被改、
条数变了）→ 指纹变 → 自动建新缓存。这一步不能省：若只检查"文件存在"就复用，重新抽取后的
旧向量会**静默**给出错误排序。`index.is_usable()` 三样都比对（`corpus_sha256` / `model` /
条数），测试对每种变化各有断言。

**第三处：每个 QA 的工作目录只放一个链接。**
早期实现给每个 `qa_{idx}/inputs/` **逐个**软链 5 个文件（`atommem.jsonl` / `msgmem.jsonl` /
`search` / `timecalc` / `index`）。但这些东西**在一个运行目录内对每个 QA 完全相同** ——
等于把同一组链接复制 152 遍。每个目录项占一个 4K 块，实测全量跑就浪费 **4.2MB**
（目录 458 个 + 软链 608 个），而全部证据加起来才 45KB。现在整目录一个软链，
每 QA 从 **6 个条目降到 2 个**。只读性不变（工具层 `resolve()` 后仍落在工作目录外）。

**第四处：轨迹里的观测要截断。**
`{dir}.qa_trajectories.jsonl` 实测 **4.8MB**，其中 `rounds[].steps[]` 占 95%，
而"原始工具观测"（`result_text`）一项就占 55%、`result_details` 再占 10%。
观测是**唯一**记录"模型当时看到什么"的地方（`model_calls.jsonl` 默认不写 messages），
所以是**截断而非删除** —— 默认每步保留 600 字符，足以看出是检索结果、报错、还是空输出。
要全文设 `trajectory_observation_chars: 0`。

**另外两处体积优化**：

- **`model_calls.jsonl` 默认不再写完整 `messages`**。那份 `messages` 是**整个不断增长的
  上下文**，相邻两次记录 95% 重复 —— 实测单条 21KB、一个 QA（16 次调用）**800KB**，
  全量 10 目录 × 200 QA 约 **1.6GB**。默认只记模型的**原始输出**（排错要的正是它）+ 上下文
  规模（`context_chars` / `context_messages`），降到 **8KB/QA**。要复现"模型当时看到什么"
  用 `--log-full-messages`（或看 debug 日志，它本来就打印完整 prompt 与观测）。
- **旧的内嵌索引可一键清理**（存储优化前留下的重复数据）：

  ```bash
  python -m codemem.search --clean-index-cache            # 只列出
  python -m codemem.search --clean-index-cache --yes      # 真正删除
  ```

  只删运行目录里**内嵌的** `inputs/index/`（跳过符号链接），共享缓存与 evidence/trajectory
  一概不动。实测回收 **322MB**（`351M → 28M`）。

- **历史运行目录可一键清理**：

  ```bash
  python -m codemem.search --prune-runs                      # 只列出
  python -m codemem.search --prune-runs --yes                # 真正删除
  python -m codemem.search --prune-runs --keep-last 3 --yes  # 保留最近 3 次
  ```

  默认保留**含 `full` 的那次运行**与 `.index_cache/`（共享索引**永不删** —— 它跨运行复用，
  删了下次要重嵌几十秒，且不属于任何单次运行）。实测清理 27 个调试跑次回收 **24.5MB**
  （`40M → 29M`）。

### 2.4 两条产物稳健性的实测坑

- **被 `max_tokens` 截断的数组必须抢救**。模型一次 `write` 写出 39 条记录、末尾的 `]`
  被截掉（7687 字节处断）。旧实现逐行读只看到"1 行、解析失败"，于是**把全部证据判成空** ——
  比"少一条"糟得多。现在按对象边界切开抢救，完整的收下、只丢最后一个残片，并在日志里
  明确写出"截断抢救（丢弃 N 个残片）"，**绝不静默**。
- **解析只有一处实现**。这个 bug 的根因是同一段解析逻辑存在三份（`io` / `evidence` 校验 /
  `evidence` 进展统计），对同一份文件三处判断不一致。现在全部走
  `io.parse_jsonl_text`，它用 `mode`（`linewise` / `document` / `salvaged`）明确说明
  文本是**怎么**解析出来的，而不是让调用方从"坏行数"去猜。

工具的执行语义**对齐 `reference/tools.py`**（2000 行/50KB 截断、`edit` 的唯一性校验、
`shell_command_prefix`、超时杀进程组），并加了两处约束：`write`/`edit` 的目标路径必须落在
工作目录内（软链逃逸会被 `resolve()` 拦下），因此 `qa_{idx}/inputs` 只能读不能写。

其实测过的一条时间推理轨迹（qa0，"When did Caroline go to the LGBTQ support group?"，
reference 是 **7 May 2023**）：

```bash
jq -c 'select(.metadata.type=="raw") | {id:.metadata.id, time:.metadata.time}' inputs/msgmem.jsonl  # 1 找锚点
search "Caroline LGBTQ support group" -k 8                          # 2 检索
jq -c 'select(.metadata.id=="session_1_3_1")' inputs/atommem.jsonl  # 3 读记录（库里的原文是
                                                                    #   "…went to a LGBTQ support group yesterday"）
jq -c 'select(.metadata.id=="session_1_3") | .metadata' inputs/msgmem.jsonl  # 4 取该 session 的时间戳
timecalc shift 2023-05-08 -1day                                     # 5 算绝对日期 -> 2023-05-07
```

然后**写一条改写过的记录**（不是抄第 3 步那行）：

```
memory: "Caroline went to an LGBTQ support group on 2023-05-07"
time:   "2023-05-07T13:56:00"
source: ["session_1_3_1", "session_1_3"]
```

一条记录，直接回答"什么时候"，下游不需要再做任何解析。**注意第 3→6 步的变化**：库里的
`yesterday` 在证据里变成了绝对日期 —— 这就是 §6.2.2 说的"推导"。

### 3.3 用法

```bash
# **先 dry-run**：只建 inputs/ + 渲染 system prompt 打印出来，不调模型
python -m codemem.evomem_v3 --sample Caroline_Melanie --dry-run

# 调试首选：只跑前 3 条 QA、每条最多 8 步，看完整数据流
python -m codemem.evomem_v3 --sample Caroline_Melanie --max-qa 3 --max-steps 8 --log-level debug

# 全部目录全部 QA（注意成本：每条 QA ≈ steps + verifier 次调用）
python -m codemem.evomem_v3 --experiment evomem_v3

# 纯 CPU 自测（假模型 + 真工具，221 项断言，零模型成本）
python scripts/test_evomem_v3.py
```

| 参数 | 说明 |
|---|---|
| `dirs` / `--sample NAME` | speaker 目录名或路径；缺省处理全部含 `atommem.jsonl` 的目录 |
| `--config PATH` | 演化配置（默认 `configs/evomem_v3.yaml`） |
| `--tool-config PATH` | 工具配置（默认 `configs/tool.json`） |
| `--max-qa N` | **调试闸门**：每个目录最多处理前 N 条 QA（0=全部） |
| `--max-steps N` | 覆盖 `max_steps`（默认 30） |
| `--max-verify-rounds N` | 覆盖 `max_verify_rounds`（默认 2） |
| `--concurrency N` | 覆盖 QA 级并发（默认 4） |
| `--dry-run` | 只渲染 prompt 后打印，不调模型 |
| `--log-level` / `--log-output` | 覆盖 `log.level` / `log.output` |

### 3.4 不变量（CPU 侧强制，违反即拒绝该动作、不中断 QA）

| # | 不变量 | 动机 |
|---|---|---|
| 1 | `evidence.jsonl` 每行必须是合法 `base_mem`：`memory` 非空、`metadata.id` 非空且**文件内唯一**、`type ∈ {inner,outer,raw}`、`tag` 含 `speaker:`、**`source` 非空** | `source` 非空是**反幻觉护栏**：从 base 拷来的行天然满足，只有合成的桥接记忆受约束 —— 恰好是最该约束的地方 |
| 2 | `write`/`edit` 路径必须落在 `qa_{idx}/` 内 | 只读输入不被误写（软链逃逸也会被拦） |
| 3 | 每 QA 收尾校验 `inputs/atommem.jsonl` 的 sha256；不一致 → `tampered=true`，该 QA 作废 | bash 可以绕开 #2，检测成本极低 |
| 4 | 只补全**系统自有字段**（缺 `changelog` → 补 `created`），其余不合规的行 **reject 并记账** | 补全语义问题会掩盖失败模式 |
| 5 | 工具 / 解析 / 模型调用失败 = 一次 failed step，记账后继续 | 所有失败路径都退化为无害一步 |
| 6 | 连续 `no_progress_patience` 步**唯一记录指纹**没有变化 → **提前收尾**（判据是唯一记录集合，不是行数/字节数；见 §6.2.3） | 把空转变成机制而不是 prompt 请求 |
| 7 | 产物解析容忍两种 `jq` 自然输出：一行一个数组、整份美化 JSON（否则规范化会把证据全删） | `jq` 是 agent 的主要工具，它的自然输出必须能被接住 |

### 3.5 输出与日志

`data/evomem_runs/{experiment}_{ts}/`：

- `{dir}/qa_{idx}/evidence.jsonl` — **唯一产物**（不合规时另存 `evidence.jsonl.raw`）
- `{dir}/inputs/` — 只读输入（硬链）+ 预建索引 + `search` 包装
- `{dir}/{dir}.qa_trajectories.jsonl` — 每 QA 一行：`evidence_ids` / `steps` / `counts_by_tool` /
  `verifications` / `rounds`（含每步的模型输出、工具调用、观测、evidence 行数变化）/ `tampered` /
  `stop_reason` / `final_answer`
- `{dir}/{dir}.model_calls.jsonl` — **每一次**模型调用的完整 messages 与原始输出（排错用）
- `summary.json` — 配置、每目录摘要（`qa_done` / `evidence_total` / `steps_total` /
  `sufficient_rate` / `tampered` / `lexical_fallback` / `model_calls`）

日志分级（`configs/search.yaml` 的 `log` 段，命令行可覆盖），**每行都带 `[目录:qaN]` 前缀**
便于并发时区分来源：

- **`info`（默认）**——每个 QA 开跑（含 question 原文与 category）、每个工具调用一行结果
  （含 evidence 行数）、verifier 判定、QA 收尾汇总。
- **`debug`**——上面全部 + 完整数据流：渲染后的 system prompt、每一步的观测、模型的原始输出、
  verifier 的输入与输出、evidence.jsonl 全文、异常 traceback。

`stop_reason` 取值：`sufficient`（verifier 判足够）/ `verify_rounds_exhausted`（判了几轮仍不足）
/ `no_progress(N)`（连续 N 步无进展提前收尾）/ `steps_exhausted` / `error`。

> ⚠️ **成本**：每条 QA ≈ `steps + verifier 次数` 次 `Qwen3-8B` 调用。system 段实测约
> **3.3k token（11449 字符）且固定** —— 可命中 prefix cache；随对话历史与工具观测增长，
> 每步再涨 ~1–2k。先用 `--max-qa 3 --max-steps 8` 试跑，`--dry-run` 可先看 prompt 全貌。
>
> ⚠️ **嵌入服务不可用时**会降级为**词法模式**（BM25 + tag，`search` 需带 `--lexical`），
> 并记 `lexical_fallback=true`；`configs/search.yaml` 的
> `embedding.allow_lexical_fallback: false` 可改成直接报错。索引每个目录只建一次、
> **并跨运行共享**（见 §2.3）。

**原子性**：只写 `data/search_runs/` 下的文件，`inputs/` 是硬链且收尾校验 sha256，
**绝不修改** `atommem.jsonl`。

---

### 2.5 首次全量跑（Caroline_Melanie，152 QA）的实测结论

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
与 search 阶段"相对时间交给 `timecalc`"是同一个取舍。效果：temporal 37.8% → **43.2%**，
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

## 3. answer — 根据 evidence 回答

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

## 4. eval — 判断对错与指标

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

## 5. Baseline 对照（三个独立评测脚本）

这三个脚本各自"检索 + 作答 + 评判"一条龙，**早于 answer/eval 两个步骤**，用来看
`msgmem` 与 `atommem` 的检索质量差异（README 里 28%→85% 那组数字就是它们产出的）。
新增能力请走四个步骤，不要往这里加。

### 4.1 Baseline 评测（完整上下文）

```bash
# 全上下文 baseline，使用所有 msgmem.jsonl 内容，实时显示 judge accuracy 和 ETA
python scripts/eval_baseline.py --experiment baseline_qwen3 --limit 10

# 中断后继续：指定同一个实验目录
python scripts/eval_baseline.py --experiment baseline_qwen3 --resume
```

### 4.2 RAG 评测（msgmem 检索）

```bash
# 通过 sglang /v1/embeddings API 做 embedding RAG，从 msgmem.jsonl 中检索最多 30 条记忆
python scripts/eval_rag.py --experiment rag_qwen3 --limit 10

# 中断后继续
python scripts/eval_rag.py --experiment rag_qwen3 --resume
```

### 4.3 AtomMem 评测（对比 msgmem 和 atommem）

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

### 4.4 输出文件

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

### 输出文件

每次运行创建 `data/eval_runs/{experiment}_{YYYYMMDD_HHMMSS}/`，含 `eval_baseline.jsonl`、
`eval_rag.jsonl` 或 `eval_atommem.jsonl` 以及 `summary.json`。summary 按 LoCoMo category 1-5
输出指标；category 5（`adversarial_skip`）只记录结果，不参与评分。

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

## 常见问题

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

## 自测

纯 CPU、零模型成本、零网络（假模型 + 真工具）：

```bash
python scripts/test_search.py       # 全链路 253 项断言（含静态"无未定义引用"检查）
python scripts/test_searchmem.py    # 混合检索（BM25 / tag / RRF 融合）
```

`test_search.py` 里的第 0 项是**静态检查**：用 AST 扫全包，确保没有"调用了但没定义/没导入"
的名字。重构最容易留下的就是这个（把函数搬到别处、忘了补 import，只在某个错误分支才炸）。

## TODO

**已完成的架构重构**（2026-09-27）：围绕 add / search / answer / eval 四个步骤重组，
共享层提到 `codemem/` 顶层（`io` / `llm` / `log` / `dataset`）。
删掉了 v2 的动作空间版（`evomem.py` + `evoactions.py`，git HEAD 仍可找回）。

- [x] **S0** `evidence_to_ids` 解析 bug 已修（空格分隔多引用 `"D9:1 D4:4 D4:6"` 与 `"D:11:26"`
      都能解析）—— 现由 `dataset.evidence_to_ids` 唯一实现
- [x] **S1** 更严格的 CPU 不变量：`evidence.jsonl` 的六项校验（`source` 非空、id 唯一、
      `tag` 含 `speaker:`、类型合法…）在 `search/evidence.py`
- [x] **S3** search 步骤（code-agent 式证据构建）：见 §2
- [x] **S4** answer / eval 两个步骤打通，`answers.jsonl` + `eval.jsonl` 端到端可跑
- [x] **S5** 静态回归网：`test_search.py` 用 AST 扫全包，杜绝"调用了但没导入"

待办：

- [ ] **N1** 效果评估：用本地 `Qwen3-8B` 跑 `--max-qa 3` 校准 search 的 prompt 与 `max_steps`，
      再跑完 10 个目录，对比 baseline 的 evidence recall 与 judge 准确率；盯 `sufficient_rate`、
      eval 的 `failures` 归类、以及 `verifier_sufficient` vs `judge_correct` 的背离
- [ ] **N2** search 的多跳补强：`missing[]` 回灌已实现，可考虑让 verifier 能"点名要某条记忆"
      （当前只能描述缺口，由 agent 自己检索）
- [ ] **N3** `ANSWER_PROMPT` 加 `used_ids` 字段（后续做 credit 归因要用）
- [ ] **N4** search 的强化学习：QA 驱动的证据构建已是一条完整 trajectory（状态 = evidence.jsonl，
      终局奖励 = judge 答案），可直接在其上做 oracle 轨迹采样 + GRPO
- [ ] **N5** 三个 baseline 脚本（`eval_baseline/rag/atommem`）与新链路的指标口径统一，
      使"msgmem / atommem / evidence / 全上下文"四者可横向对比

（注：`data/*/atommem.jsonl` 的 10 个目录均已生成完毕。）
