# CodeMem

面向代码/对话记忆的原子记忆系统。整个架构围绕**四个步骤**组织，依赖方向严格单向，
步骤之间**只通过数据文件耦合**（不互相 import）：

```
add ──► sessions.jsonl ──► session_summaries.jsonl
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
| **add** | 一段对话记忆 → 原始消息 + 每个 session 一份 summary | `python -m codemem.add` |
| **search** | question + 两层语料 → `evidence.jsonl`（搜索 agent：关键词多跳检索，语义兜底） | `python -m codemem.search` |
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
  add.yaml      # add 的模型与 summary 参数
  search.yaml   # search 的 agent/检索/预算参数
  answer.yaml   # answer 的模型参数
  eval.yaml     # eval 的 judge 参数
  tool.json     # search 的工具描述/schema/prompt/示例（agent 上下文的单一事实源）
data/
  correct_locomo10.json        # 原始对话数据（含 QA 与 evidence 标注）
  {speaker_a}_{speaker_b}/
    sessions.jsonl             # add 产物：原始消息（search 的输入，只读）
    session_summaries.jsonl    # 每个 session 一份 summary（粗筛层）
  search_runs/{experiment}_{ts}/   # search/answer/eval 的运行目录
    {dir}/inputs/                   只读输入（硬链）+ 语义索引 + search/timecalc 包装
    {dir}/qa_{idx}/evidence.jsonl   search 的产物
    answers.jsonl                   answer 的产物（该次运行的全部回答）
    eval.jsonl / summary.json       eval 的产物
  eval_runs/                   # 三个 baseline 评测脚本的输出
src/codemem/
  io.py llm.py log.py dataset.py    # 共享层：四个步骤都依赖，自身不依赖任何步骤
  add/    session.py summary.py __main__.py
  search/ runner.py agent.py tools.py toolcfg.py verifier.py evidence.py
          searchmem.py index.py embedder.py searchctl.py timecalc.py __main__.py
  answer/ answer.py __main__.py
  eval/   runner.py judge.py metrics.py legacy.py __main__.py
scripts/
  test_search.py test_searchmem.py   # 纯 CPU 自测（假模型 + 真工具，零模型成本）
  eval_baseline.py eval_rag.py eval_atommem.py analyze_atommem_results.py  # baseline 对照（见 §5）
  regenerate_rejected.py transfer_data_to_msswift_standard.py locomo_data_loader.py
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

## 1. add — 把对话记忆添加进来（session summary）

```bash
python -m codemem.add                      # session + summary，全部目录
python -m codemem.add Caroline_Melanie     # 只处理指定目录
python -m codemem.add --stage session      # 只跑第一步（纯 CPU，不调模型）
python -m codemem.add --stage summary --limit-sessions 3   # 只跑前 3 个 session（调试）
```

**两步，产出两层语料**（都在 `data/{speaker_a}_{speaker_b}/` 下）：

| 文件 | 内容 | 条数（Caroline_Melanie） |
|---|---|---|
| `sessions.jsonl` | 原始消息，一条一行 | 419 |
| `session_summaries.jsonl` | 每个 session 一份 summary | 19 |

记录格式统一为 `session_template.jsonl`：`{"msg_id", "role", "time", "content"}` ——
summary 记录只是把 `role` 标成 `"summary"`。**agent 只需理解一种格式**。

三个刻意的取舍：

- **`msg_id` 用 1-indexed**（`session_1_1`），与 LoCoMo 的 `dia_id`（`D1:1`）对齐 ——
  这样 evidence 映射是恒等变换，不需要每处 ±1（那是 bug 的温床）。
- **`role` 用真实人名**（`Caroline`/`Melanie`）而不是 `speaker1`/`speaker2` ——
  指代消解（"she"是谁）依赖它，丢掉就等于让 agent 从对话里猜。
- **`time` 保留原始形式**（`"1:56 pm on 8 May, 2023"`），不转 ISO ——
  原始形式是数据里唯一无歧义的时间来源；转换错误会被固化进下游。
  `timecalc` 能直接解析它：`timecalc shift "1:56 pm on 8 May, 2023" -1month`。

**为什么是 session summary 而不是原子记忆**：原子把会话结构打碎了 —— 一条条孤立事实无法回答
"他们什么时候认识的""这段关系怎么发展的"。session summary 保留"这次谈了什么、什么变了"，
并给检索一个**由粗到细**的入口：先读 session summary 定位到哪次会话，再进原始消息拿精确措辞。

**为什么取消了顶层 speakers summary**：它是把 19 份 summary 再压成 1 条的合成文本，
而检索需要的是"定位到哪次会话"——那是 session summary 就有的粒度。更要命的是它
**没有 `msg_id` 可回溯**，`source` 只能指向别的 summary 而非原始消息，直接破掉了
"证据必须能溯源到原话"这条不变量，每条 QA 还要多背一层没人用的语料。
`python -m codemem.add` 重跑时会自动删掉目录里历史遗留的 `speakers_summary.jsonl`。

> 旧的 `atommem`（原子抽取）与 `dpo`（为抽取模型造偏好数据）已删除 ——
> 它们训练的是不再使用的抽取模型。需要时从 commit `b1b5283` 取回。

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
| `dirs` / `--sample NAME` | speaker 目录名或路径；缺省处理全部含 `sessions.jsonl` 的目录 |
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

### 2.2 核心抽象

```
每条 QA 一个私有目录（QA 之间零共享 → QA 级并发；base 库只读）

  <run>/{dir}/inputs/{sessions,session_summaries}.jsonl  只读硬链
  <run>/{dir}/inputs/index -> .index_cache/   语义索引（符号链接，跨运行共享；--embed 才用）
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

### 2.3 一个搜索 agent：关键词多跳优先，语义兜底

**这一版把 agent 简化成"搜索 agent"**：它的核心动作只有一件事 —— **用关键词检索，从命中里
扣出新的关键词，再搜一轮**。多跳问题就是这么解的：第一跳找到的实体名，正是第二跳的检索词。

```bash
search "adoption counseling" --source summary -k 5   # ① 粗筛：定位哪次会话
jq -c 'select(.msg_id|startswith("session_8_"))' inputs/sessions.jsonl   # ② 读那次会话
search "pottery class" --source raw -k 8             # ③ 精确措辞只在原话里
search "Melanie husband" -k 5                        # ④ 多跳：拿刚读到的新名字再搜
search "feeling unsupported" --embed -k 8            # ⑤ 兜底：关键词确实搜不到时
```

**默认是关键词模式**（BM25 + tag 两路，不加载索引、不调嵌入服务），语义通道要显式
`--embed` 才打开。这是刻意的排序，理由有三条：

1. **快且确定** —— 没有网络、没有冷启动，agent 想调几次调几次；
2. **关键词才是多跳的主力** —— 从一条命中里抠出新人名再搜，这个动作本身不需要语义匹配；
3. **嵌入服务挂了不再等于检索挂了**。旧版这是"降级路径"（还要 `--lexical` 显式开关，
   忘了加就整个检索不可用）；现在关键词是默认，服务不可用只是少了一条 agent 主动选择才会用的通道。

`--source` 只有两层可选（外加 evidence 自身）：

| `--source` | 搜什么 | 什么时候用 |
|---|---|---|
| **`summary`** | `inputs/session_summaries.jsonl` | **第一步总是它** —— 从概览定位到哪次会话 |
| `raw` / `sessions` | `inputs/sessions.jsonl`（原始消息） | 精确的名字/数字/措辞，summary 改写掉了的那些 |
| `evidence` | 当前 `evidence.jsonl` | 查自己已收的证据有没有重复/缺口 |
| `all`（缺省） | 两层并集 + evidence | 还不知道答案在哪一层时 |

输出**每行一条 JSON**，字段只保留 agent 决策要用的：

```json
{"id":"session_3_14","kind":"session","session":"session_3","time":"7:55 pm on 9 June, 2023",
 "score":0.0328,"memory":"we've been married for five years now"}
```

`kind` / `session` / `time` 是刻意给出来的三样东西：**`kind`** 说明这是概览还是原话（决定要不要往下钻），
**`session`** 是立刻读同一次会话其它消息的入口（多跳最常用的一步），**`time`** 是相对时间推理的锚点。
通道明细 `rank_dense` / `rank_bm25` / `rank_tag` 默认**不打**——agent 不用它们做决策，
而一次 `-k 8` 就是 8 行；排错时加 `--all-fields` 拿回来。

**为什么必须能搜原话**（这是实测修掉的一个真问题）：summary 是**改写过的**，常常把具体答案
泛化掉了。例如"Caroline 想读什么方向"，真正的答案
`I'm keen on counseling or working in mental health` 只存在于原始消息 `session_1_11` 里。
只搜概览时 agent 找不到 → 只能猜消息 id → 重复查询被拒 → 空转到预算耗尽。让 `search` 覆盖原话，
这条 QA 现在 8 步内以正确答案收敛。

### 2.4 上下文预算：工具描述不能吃掉 agent 的步数

system prompt 是**每条 QA 都完整注入一次**的，所以它的每一句都在跟检索结果抢上下文。
旧版实测 **20503 字符**，其中大半是同一套规则在 system body、`workspace.rules`、
`writing_policy` 三处各写一遍。这一版压到 **11506 字符（-44%）**：

| 段落 | 旧 | 新 | 怎么省的 |
|---|---|---|---|
| system body | 9171 | 4409 | 规则去重：写一遍，不复述；只留两条 worked example（多跳 + 相对时间） |
| TOOLS | 3104 | 1795 | 精简 description；bash 的 guidelines 从 3 条压到 0 条（工具本身会报错，不必预告） |
| COMMANDS | 3838 | 2194 | jq 菜谱 9 → 4 条，只留真正常用的；删掉给已不存在语料的示例 |
| WORKSPACE | 920 | 520 | 布局只剩两层；规则从 4 条压到 2 条 |
| WRITING_POLICY | 2283 | 1548 | 合并同义条目；"证据是推导的不是抄的"仍在，只是说一遍 |
| SCHEMA | 774 | 611 | 字段说明写短 |
| PROTOCOL | 365 | 381 | 不变 |

`test_search.py` 里有一条断言把这个上限钉住（`< 13000 字符`）——**涨回去就等于白做这次简化**，
所以它该在 CI 里红，而不是等到某次跑完发现步数又不够用了。

**Verifier 同理**：它的 prompt 里没有 few-shot 示例。判定规则本身很短（能不能从记录里读出答案），
示例会让每条 QA 多背几百 token。唯一的例外是一条**日期等价性**说明 ——
"the weekend before 4 September 2023" 与 "2023-09-02" 指向同一天，那种误判最容易发生。

### 2.5 集合型问题要靠标签枚举，不能靠相似度

有一类问题的答案是**一个集合**："Melanie 参加哪些活动？"、"Caroline 有哪些爱好？"。
稠密检索对这类问题**结构性无能为力**：它按与 query 的相似度排序，而"她要去游泳了"
这条记录与"Melanie 参加什么活动"的相似度并不高（实测 dense 排名 **101**），于是永远
进不了 top-k —— 把 `-k` 调到 100 也救不回来，因为相关度天梯本身就不指向它。

`search` 因此保留了按标签**枚举**的入口（`--tag` / `--exclude-tag` / `--tag-values`），
用于收齐这类集合。**但这条路径的收益取决于标签词表的质量**，而词表由 `add` 阶段的模型生成、
且当前语料（`session_template` 格式）**没有 tag 字段** —— 所以它现在基本不命中任何东西，
保留是为了将来 summary 带上结构化标签时可以直接用。这仍是 `add` 侧待解决的问题。

> 这条也说明一个更一般的结论：**结构化的收益取决于结构本身的质量**。检索能按标签枚举，
> 前提是标签是可枚举的。

**两个方向都能走**：

- **概览 → 原话（下钻）**：命中带 `session` 字段，直接读该会话的全部消息。
- **原话 → 概览（拉远）**：原话的 `session` 同样给出去，读 `session_N_summary` 看它如何归位。

检索结果默认带 `kind` 与 `session`，所以这两个方向都不必再多一轮 `jq` 去找 id。

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

### 2.6 证据是**推导**出来的，不是抄的

这是 search 的核心语义，也是相对上一版最大的一处变化：

> **`evidence.jsonl` 不是库的子集，而是一份可以直接回答问题的记忆列表。** 下游只看到这个
> 文件 —— 看不到语料、看不到原始消息。所以所需的推理必须**在记录里完成**。

因此 agent **可以也应该改写正文**：

- **消解指代**：`she`/`he`/`there` → 真实姓名/地点；
- **消解时间**：`yesterday` / `about a month ago` → 绝对日期（用 `timecalc`，锚点是该会话
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

反幻觉护栏不变：**`source` 必须非空**，且必须是**本会话真读过的** id（不是记得或猜的），
`source` 为空的记录一律 reject。

### 2.7 code-agent loop 的几件事

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

### 2.8 存储：索引跨运行共享，产物分层存放

语义索引（`--embed` 才用得到）是运行产物里最大的一块：一个目录约 1650 条 × 2560 维 float32
≈ **17MB**，而它只取决于 `(语料, 嵌入模型)` —— 与哪次运行、哪条 QA 完全无关。早期实现把它
建在**每次运行的目录里**，于是跑 23 次就是 **334MB 完全相同的重复数据**（实测）。

现在分层，各自按真正的变化频率存放：

| 内容 | 位置 | 重复情况 |
|---|---|---|
| `sessions.jsonl` / `session_summaries.jsonl` | 每次运行的 `inputs/` | **硬链**（inode 共享，0 额外空间） |
| **索引** `vectors.f32` | `data/search_runs/.index_cache/{corpus}-{model}/` | **一份，跨运行共享**；运行目录里只有**符号链接** |
| evidence / trajectory / 日志 | 每次运行目录 | 每次一份（本来就该有） |

于是同一目录反复跑（调试、`--qa`、`--repeat`）**只嵌一次**：

```
首次：建检索索引：1650 条（session summary 19 + 原始消息 419）
      -> 共享缓存 …/ae345cf000210e41-de1c6d0a
之后：复用共享索引缓存：1650 条 × 2560 维（session summary 19 + 消息 419）
```

**缓存安全**：目录名是 `corpus_sha256` + 嵌入模型指纹。语料一变（重跑 `add`、正文被改、
条数变了）→ 指纹变 → 自动建新缓存。这一步不能省：若只检查"文件存在"就复用，重新生成后的
旧向量会**静默**给出错误排序。`index.is_usable()` 三样都比对（`corpus_sha256` / `model` /
条数），测试对每种变化各有断言。

**每个 QA 的工作目录只放一个链接。**
早期实现给每个 `qa_{idx}/inputs/` **逐个**软链 5 个文件。但这些东西**在一个运行目录内对每个
QA 完全相同** —— 等于把同一组链接复制 152 遍。每个目录项占一个 4K 块，实测全量跑就浪费
**4.2MB**（目录 458 个 + 软链 608 个），而全部证据加起来才 45KB。现在整目录一个链接，
每 QA 从 **6 个条目降到 2 个**。

链接优先用**符号链接**：工具层的路径约束会 `resolve()`，解析后落在工作目录之外，所以 agent 写
`inputs/...` 会被直接拒绝。Windows 上普通账户建符号链接需要开发者模式/管理员权限（`WinError 1314`），
`link_dir_or_copy` 会退回**逐文件硬链**。硬链下 `bash >>` 会写穿到源文件 —— 实测确认过，
而收尾的 sha256 校验能抓到并把该 QA 作废；`write`/`edit` 的原子替换则只会悄悄换掉自己那份链接，
源文件不受影响。

**轨迹里的观测要截断。**
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

### 2.9 两条产物稳健性的实测坑

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
search "Caroline LGBTQ support group" --source raw -k 8        # 1 检索
# -> {"id":"session_1_3_1","session":"session_1","time":"1:56 pm on 8 May, 2023",
#     "memory":"Caroline: I went to a LGBTQ support group yesterday"}
jq -c 'select(.msg_id=="session_1_3_1")' inputs/sessions.jsonl # 2 读原话（确认措辞）
timecalc shift "1:56 pm on 8 May, 2023" -1day                  # 3 算绝对日期 -> 2023-05-07
```

然后**写一条改写过的记录**（不是抄第 2 步那行）：

```
content: "Caroline went to an LGBTQ support group on 2023-05-07."
source:  ["session_1_3_1"]
```

一条记录，直接回答"什么时候"，下游不需要再做任何解析。**注意第 2→3 步的变化**：语料里的
`yesterday` 在证据里变成了绝对日期 —— 这就是 §2.6 说的"推导"。

### 2.10 用法

```bash
# **先 dry-run**：只建 inputs/ + 渲染 system prompt 打印出来，不调模型
python -m codemem.search --sample Caroline_Melanie --dry-run

# 调试首选：只跑前 3 条 QA、每条最多 8 步，看完整数据流
python -m codemem.search --sample Caroline_Melanie --max-qa 3 --max-steps 8 --log-level debug

# 全部目录全部 QA（注意成本：每条 QA ≈ steps + verifier 次调用）
python -m codemem.search --experiment search

# 纯 CPU 自测（假模型 + 真工具，398 项断言，零模型成本）
python scripts/test_search.py
```

| 参数 | 说明 |
|---|---|
| `dirs` / `--sample NAME` | speaker 目录名或路径；缺省处理全部含 `sessions.jsonl` 的目录 |
| `--config PATH` | 检索配置（默认 `configs/search.yaml`） |
| `--tool-config PATH` | 工具配置（默认 `configs/tool.json`） |
| `--max-qa N` | **调试闸门**：每个目录最多处理前 N 条 QA（0=全部） |
| `--max-steps N` | 覆盖 `max_steps`（默认 30） |
| `--max-verify-rounds N` | 覆盖 `max_verify_rounds`（默认 2） |
| `--concurrency N` | 覆盖 QA 级并发（默认 4） |
| `--dry-run` | 只渲染 prompt 后打印，不调模型 |
| `--log-level` / `--log-output` | 覆盖 `log.level` / `log.output` |

### 2.11 不变量（CPU 侧强制，违反即拒绝该动作、不中断 QA）

| # | 不变量 | 动机 |
|---|---|---|
| 1 | `evidence.jsonl` 每行必须是 `{"content": 非空, "metadata.source": 非空数组}`；`score` / `changelog` 系统可补，**其余字段一律不替 agent 造** | `source` 非空是**反幻觉护栏**：它必须列出**真读过**的 msg_id |
| 2 | `write`/`edit` 路径必须落在 `qa_{idx}/` 内 | 只读输入不被误写（软链逃逸也会被拦） |
| 3 | 每 QA 收尾校验**两层语料**的 sha256；不一致 → `tampered=true`，该 QA 作废 | bash 可以绕开 #2，检测成本极低 |
| 4 | 只补全**系统自有字段**（缺 `changelog` → 补 `created`；缺 `score` → 补 0.5），其余不合规的行 **reject 并记账** | 补全语义问题会掩盖失败模式 |
| 5 | 工具 / 解析 / 模型调用失败 = 一次 failed step，记账后继续 | 所有失败路径都退化为无害一步 |
| 6 | 连续 `no_progress_patience` 步**唯一记录指纹**没有变化 → **提前收尾**（判据是唯一记录集合，不是行数/字节数；见 §2.7） | 把空转变成机制而不是 prompt 请求 |
| 7 | 产物解析容忍两种 `jq` 自然输出：一行一个数组、整份美化 JSON（否则规范化会把证据全删） | `jq` 是 agent 的主要工具，它的自然输出必须能被接住 |

### 2.12 输出与日志

`data/search_runs/{experiment}_{ts}/`：

- `{dir}/qa_{idx}/evidence.jsonl` — **唯一产物**（不合规时另存 `evidence.jsonl.raw`）
- `{dir}/inputs/` — 只读输入（硬链）+ 语义索引 + `search` / `timecalc` 包装
- `{dir}/{dir}.qa_trajectories.jsonl` — 每 QA 一行：`evidence_ids` / `steps` / `counts_by_tool` /
  `verifications` / `rounds`（含每步的模型输出、工具调用、观测、evidence 行数变化）/ `tampered` /
  `stop_reason` / `final_answer`
- `{dir}/{dir}.model_calls.jsonl` — 每次模型调用的原始输出（排错用；`--log-full-messages`
  才带完整 messages）
- `summary.json` — 配置、每目录摘要（`qa_done` / `evidence_total` / `steps_total` /
  `sufficient_rate` / `tampered` / `embed_available` / `model_calls`）

日志分级（`configs/search.yaml` 的 `log` 段，命令行可覆盖），**每行都带 `[目录:qaN]` 前缀**
便于并发时区分来源：

- **`info`（默认）**——每个 QA 开跑（含 question 原文与 category）、每个工具调用一行结果
  （含 evidence 行数）、verifier 判定、QA 收尾汇总。
- **`debug`**——上面全部 + 完整数据流：渲染后的 system prompt、每一步的观测、模型的原始输出、
  verifier 的输入与输出、evidence.jsonl 全文、异常 traceback。

`stop_reason` 取值：`sufficient`（verifier 判足够）/ `verify_rounds_exhausted`（判了几轮仍不足）
/ `no_progress(N)`（连续 N 步无进展提前收尾）/ `steps_exhausted` / `error`。

> ⚠️ **成本**：每条 QA ≈ `steps + verifier 次数` 次 `Qwen3-8B` 调用。system 段实测
> **11506 字符（约 3k token）且固定** —— 可命中 prefix cache；随对话历史与工具观测增长，
> 每步再涨 ~1–2k（见 §2.7 的压缩机制）。先用 `--max-qa 3 --max-steps 8` 试跑，
> `--dry-run` 可先看 prompt 全貌。
>
> ⚠️ **嵌入服务不可用时**只是少一条通道：`search` 仍以关键词模式（BM25 + tag）正常工作，
> 只有 `--embed` 会报错。索引每个目录只建一次、**并跨运行共享**（见 §2.8）。
> `configs/search.yaml` 的 `build_index: false` 可完全跳过建索引。

**原子性**：只写 `data/search_runs/` 下的文件，`inputs/` 是硬链且收尾校验 sha256，
**绝不修改** `data/{dir}/` 下的任何语料。

### 2.13 首次全量跑（Caroline_Melanie，152 QA）的实测结论

> ⚠️ **这组数字来自删掉 speakers summary 之前的语料（三层），保留它是为了记录三个诊断结论**
> —— 那三条结论与语料层数无关，仍然是当前设计（evidence 软上限、temporal 交给代码算日期、
> recall 不能当失败信号）的依据。用两层语料重跑后应当更新这张表。

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

这三个脚本各自"检索 + 作答 + 评判"一条龙，**早于 answer/eval 两个步骤**。它们读的是
`msgmem.jsonl` / `atommem.jsonl` —— **旧原子记忆体系的产物，当前 add 流程已不再生成**
（代码仍在：它与 `eval/legacy.py` 一起构成历史对照，`atommem.jsonl` 的数据也还在盘上）。
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
python scripts/test_search.py       # 全链路 398 项断言（含静态"无未定义引用"检查）
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
- [x] **S1** 更严格的 CPU 不变量：`evidence.jsonl` 的校验（`content` 非空、
      `metadata.source` 非空且元素合法、重复 id 拒绝…）在 `search/evidence.py`
- [x] **S3** search 步骤（code-agent 式证据构建）：见 §2
- [x] **S4** answer / eval 两个步骤打通，`answers.jsonl` + `eval.jsonl` 端到端可跑
- [x] **S5** 静态回归网：`test_search.py` 用 AST 扫全包，杜绝"调用了但没导入"
- [x] **S6** add 只留 session summary（删掉顶层 speakers summary）；search 简化为
      「关键词多跳优先、`--embed` 语义兜底」的搜索 agent；system prompt 20503 → 11506 字符
      （-44%，见 §2.4）

待办：

- [ ] **N1** 效果评估：用本地 `Qwen3-8B` 跑 `--max-qa 3` 校准 search 的 prompt 与 `max_steps`，
      再跑完 10 个目录，对比 baseline 的 evidence recall 与 judge 准确率；盯 `sufficient_rate`、
      eval 的 `failures` 归类、以及 `verifier_sufficient` vs `judge_correct` 的背离。
      §2.13 那张表还是三层语料时代的数字，重跑后要更新
- [ ] **N2** search 的多跳补强：`missing[]` 回灌已实现，可考虑让 verifier 能"点名要某条记忆"
      （当前只能描述缺口，由 agent 自己检索）
- [ ] **N3** `ANSWER_PROMPT` 加 `used_ids` 字段（后续做 credit 归因要用）
- [ ] **N4** search 的强化学习：QA 驱动的证据构建已是一条完整 trajectory（状态 = evidence.jsonl，
      终局奖励 = judge 答案），可直接在其上做 oracle 轨迹采样 + GRPO
- [ ] **N5** 三个 baseline 脚本（`eval_baseline/rag/atommem`）与新链路的指标口径统一，
      使"msgmem / atommem / evidence / 全上下文"四者可横向对比

（注：`data/*/atommem.jsonl` 的 10 个目录均已生成完毕，属旧原子记忆体系的遗留数据。）
