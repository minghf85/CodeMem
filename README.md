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
    {dir}/inputs/                   只读输入（硬链的两层语料）
    {dir}/qa_{idx}/evidence.jsonl   search 的产物
    answers.jsonl                   answer 的产物（该次运行的全部回答）
    eval.jsonl / summary.json       eval 的产物
  eval_runs/                   # 三个 baseline 评测脚本的输出
src/codemem/
  io.py llm.py log.py dataset.py    # 共享层：四个步骤都依赖，自身不依赖任何步骤
  add/    session.py summary.py __main__.py
  search/ runner.py agent.py tools.py toolcfg.py verifier.py evidence.py
          prune.py __main__.py
  answer/ answer.py __main__.py
  eval/   runner.py judge.py metrics.py legacy.py __main__.py
scripts/
  run_eval.py                        # 一条命令跑通 search → answer → eval 并打印摘要
  test_search.py                     # 纯 CPU 自测（假模型 + 真工具，零模型成本）
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
  agent 用 `date -d "8 May 2023 -1 month" +"%d %b %Y"` 现场折算它。

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

### 2.3 一个搜索 agent：检索就是 `grep`，按问题类型选策略

**这一版把 agent 砍到底**：没有向量索引、没有 embedding 服务、没有检索 CLI、没有日期工具。
语料是两个 JSONL 文件（一行一条记录），检索就是 **`grep`**，时间算术就是 **`date -d`**。
`README` 里其它地方提到的 `search` 命令、`timecalc`、索引缓存都已随这次改动删除。

**为什么要删**：向量检索换来的能力是"换个说法也能搜到"，代价却是一个必须常驻的外部服务、
一次几十秒的索引构建（一个目录 17MB）、以及一个**无法解释的排序** —— 它为什么把这条排
101 名，你查不出来。而多跳问题靠的是**词面匹配**（第一跳找到的实体名，正是第二跳的检索词），
那正是 grep 的强项。删掉之后 agent 需要学的思维模型只剩一条：
**一切都是一条对 JSONL 文件的命令**。

```bash
grep -in "husband" inputs/session_summaries.jsonl    # ① 哪次会话相关（先看概览）
grep -in "husband" inputs/sessions.jsonl | head      # ② 原话在这里
read inputs/sessions.jsonl offset=120 limit=40       # ③ 精读那一次会话
date -d "9 Jun 2023 -5 year" +%Y                     # ④ 算时间 -> 2018
```

**两种策略，写进 system prompt 让模型先选再动手**（实测 LoCoMo 的标注证据里，
1985 条 QA 中 1187 条只用 1 个 session，但有 330 条跨 2 个以上、最多跨 15 个，
另有 57 条是集合型问题 —— 两类问题的最优动作相反）：

| | 什么时候用 | 怎么做 |
|---|---|---|
| **A. 关键词多跳** | 事实型：一个名字、一个数字、一件事 | grep 定位会话 → 读它 → **拿新词再 grep** |
| **B. 遍历** | 集合型/散布型："她参加过哪些活动"、"他读过哪些书" | grep 出所有相关会话，**逐个走完**，边读边收成员 |

只写一句"检索要彻底"会让模型在 B 上过早收手（收不全成员），在 A 上过早发散。

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
这正是用户指出的那个点：带锚点的相对表述本身就是完整答案，而模型把它抹平成了锚点。

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



这是 `when` 类问题的核心规则，也是这一版改得最实的一处。看几个真实例子 ——
**源语料的表述往往比参考答案还粗**：

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

**verifier 也要同步**，否则它会反向逼 agent 编造：它在"够不够"这一关很容易把锚点表述判成
不完整（"这不是一个日期"），于是 agent 只好去猜一个更精确的。所以它的 prompt 里明确写了
*Match the precision of the evidence -- do NOT demand more*。

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

system prompt 是**每条 QA 都完整注入一次**的，所以它的每一句都在跟检索结果抢上下文。
实测现在 **12977 字符**：

| 段落 | 字符 | 说明 |
|---|---|---|
| system body | 5039 | 两条策略的判据 + 三条 worked example（多跳 / 相对时间 / 遍历） |
| TOOLS | 2006 | 四个工具的 description + schema |
| COMMANDS | 1485 | grep 与 date 的用法与示例 |
| WRITING_POLICY | 1410 | 证据是推导、不是抄的 |
| TIME_POLICY | 1307 | 时间精度规则（§2.4） |
| SCHEMA | 611 | evidence 记录字段 |
| WORKSPACE | 557 | 目录布局与只读规则 |
| PROTOCOL | 381 | 响应格式 |

**这个数字比删掉向量检索之前反而大了一点** —— 因为多了一整套时间精度规则和第二条策略。
换来的是不再需要维护索引与嵌入服务。

`test_search.py` 里有一条断言把这个上限钉住（`< 14000 字符`）——**涨回去就等于白做这次简化**，
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

这一版换了个更直接的办法：**遍历**（策略 B）。grep 出所有相关会话，逐个走完 ——
没有"相关度天梯"这回事，命中的就是字面包含的那些行，覆盖是可数的、可验证的。

### 2.8 证据是**推导**出来的，不是抄的

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

反幻觉护栏不变：**`source` 必须非空**，且必须是**本会话真读过的** id（不是记得或猜的），
`source` 为空的记录一律 reject。

### 2.9 code-agent loop 的几件事

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

### 2.10 存储与运行目录

旧版这一节是整条链路最重的一块：向量索引（一个目录 17MB）、共享缓存、按 `corpus_sha256`
判失效、跨运行复用、三个清理命令。**现在这些全都不存在了** —— 语料就是两个 JSONL 文件，
硬链进运行目录（inode 共享，0 额外空间），agent 直接 grep。

| 内容 | 位置 | 说明 |
|---|---|---|
| `sessions.jsonl` / `session_summaries.jsonl` | 每次运行的 `inputs/` | **硬链**（inode 共享） |
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
grep -in "support group" inputs/sessions.jsonl
# -> session_1_3: "I went to a LGBTQ support group yesterday"
#    该记录的 time 是 "1:56 pm on 8 May, 2023" —— 这就是锚点
date -d "8 May 2023 -1 day" +"%d %b %Y"                        # -> 07 May 2023
```

然后**写一条改写过的记录**（不是抄那一行）：

```
content: "Caroline went to an LGBTQ support group on 7 May 2023."
source:  ["session_1_3"]
```

一条记录，直接回答"什么时候"，下游不需要再做任何解析。**注意这中间发生的事**：语料里的
`yesterday` 在证据里变成了绝对日期 —— 这就是 §2.8 说的"推导"。
（`yesterday` 相对于该消息自己的时间，算术精确、锚点已知 —— 属于 §2.4 说的"该算就算"的
那一类；如果源说的是 "last year"，答案就该是年份而不是日期。）

### 2.12 用法

```bash
# **先 dry-run**：只建 inputs/ + 渲染 system prompt 打印出来，不调模型
python -m codemem.search --sample Caroline_Melanie --dry-run

# 调试首选：只跑前 3 条 QA、每条最多 8 步，看完整数据流
python -m codemem.search --sample Caroline_Melanie --max-qa 3 --max-steps 8 --log-level debug

# 全部目录全部 QA（注意成本：每条 QA ≈ steps + verifier 次调用）
python -m codemem.search --experiment search

# 纯 CPU 自测（假模型 + 真工具，322 项断言，零模型成本）
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

### 2.13 不变量（CPU 侧强制，违反即拒绝该动作、不中断 QA）

| # | 不变量 | 动机 |
|---|---|---|
| 1 | `evidence.jsonl` 每行必须是 `{"content": 非空, "metadata.source": 非空数组}`；`score` / `changelog` 系统可补，**其余字段一律不替 agent 造** | `source` 非空是**反幻觉护栏**：它必须列出**真读过**的 msg_id |
| 2 | `write`/`edit` 路径必须落在 `qa_{idx}/` 内 | 只读输入不被误写（软链逃逸也会被拦） |
| 3 | 每 QA 收尾校验**两层语料**的 sha256；不一致 → `tampered=true`，该 QA 作废 | bash 可以绕开 #2，检测成本极低 |
| 4 | 只补全**系统自有字段**（缺 `changelog` → 补 `created`；缺 `score` → 补 0.5），其余不合规的行 **reject 并记账** | 补全语义问题会掩盖失败模式 |
| 5 | 工具 / 解析 / 模型调用失败 = 一次 failed step，记账后继续 | 所有失败路径都退化为无害一步 |
| 6 | 连续 `no_progress_patience` 步**唯一记录指纹**没有变化 → **提前收尾**（判据是唯一记录集合，不是行数/字节数；见 §2.7） | 把空转变成机制而不是 prompt 请求 |
| 7 | 产物解析容忍两种 `jq` 自然输出：一行一个数组、整份美化 JSON（否则规范化会把证据全删） | `jq` 是 agent 的主要工具，它的自然输出必须能被接住 |

### 2.14 输出与日志

`data/search_runs/{experiment}_{ts}/`：

- `{dir}/qa_{idx}/evidence.jsonl` — **唯一产物**（不合规时另存 `evidence.jsonl.raw`）
- `{dir}/inputs/` — 只读输入（硬链的两层语料）
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
> **12977 字符（约 3.2k token）且固定** —— 可命中 prefix cache；随对话历史与工具观测增长，
> 每步再涨 ~1–2k（见 §2.8 的压缩机制）。先用 `--max-qa 3 --max-steps 8` 试跑，
> `--dry-run` 可先看 prompt 全貌。
>
> ⚠️ **没有外部依赖**：没有嵌入服务、没有索引、没有检索 CLI。agent 用系统自带的 `grep`
> 检索、用 `date -d` 做算术。整条链路只在调模型时联网。

**原子性**：只写 `data/search_runs/` 下的文件，`inputs/` 是硬链且收尾校验 sha256，
**绝不修改** `data/{dir}/` 下的任何语料。

### 2.15 首次全量跑（Caroline_Melanie，152 QA）的实测结论

> **这组数字是最早那次全量跑（三层语料时代）的结果**，保留它是为了记录下面三个诊断结论 ——
> 它们与语料层数无关，至今仍是当前设计的依据。
>
> 最新一次（两层语料 + grep agent，150 条 QA、`search_20260929_185935`）的结果是：
> `judge_accuracy 58.0%`，其中 **temporal 只有 27%**（详见 §2.4 的逐条归因），
> single_hop 69.6% / multi_hop 61.3% / open_domain 76.9%。

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

## 一条命令跑通三步（`scripts/run_eval.py`）

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

## 自测

纯 CPU、零模型成本、零网络（假模型 + 真工具）：

```bash
python scripts/test_search.py       # 全链路 322 项断言（含两项静态检查）
```

`test_search.py` 开头是两项**静态检查**（都纯 AST，不用跑起来）：

- **无未定义引用**：确保没有"调用了但没定义/没导入"的名字 —— 重构最容易留下的就是它。
- **无过时关键字参数**：调用处传的 `kwarg=` 必须真的存在于签名里。
  这一条是**实测补上的**：删掉向量索引时去掉了 `run_qa` 的 `index_dir` 形参，但调用处还留着
  `index_dir=index_dir` —— 全量自测 304 项全绿（测试文件自己也传了这个参数，而测试里那个名字
  恰好还存在），一上真机就 `NameError`。**形参与实参是两个地方，删一个不等于删另一个。**

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
      「检索就是 grep、按问题类型选策略」的搜索 agent；补上时间精度规则
      （见 §2.3 / §2.4）；删掉向量检索与检索 CLI（约 1670 行）
- [x] **S7** `scripts/run_eval.py` 一条命令跑通三步并打印摘要；补一项静态检查
      （过时关键字参数 —— 它正是 S6 在真机上炸的那个错因）

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
