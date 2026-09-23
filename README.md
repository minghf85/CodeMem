# CodeMem

面向代码/对话记忆的原子记忆系统。当前已实现四个阶段：

1. **`msgmem`** — 把原始对话数据格式化为 raw message memory
2. **`atommem`** — 调用大模型从 raw memory 抽取原子记忆
3. **`gen_dpo_data`** — 为原子记忆抽取模型生成 DPO 偏好训练数据
4. **`evomem`** — 用 prompt 逐步演化原子记忆库（当前为纯 prompt 版，不做强化学习）

---

## 环境

```bash
pip install openai pyyaml tqdm
```

Python 3.12+。

---

## 目录结构

```
configs/
  model.yaml          # atommem 的模型接口配置
  dpo_data.yaml       # gen_dpo_data 的模型接口配置
  grpo_data.yaml      # 强模型 gpt-5.6-luna（gen_dpo 的 chosen 侧）
  atommem.yaml        # atommem 评测配置（embedding / generator / judge）
  evomem.yaml         # evomem 演化配置（embedding / generator / QA 闸门 / 混合检索权重）
  baseline.yaml rag.yaml judge.yaml
data/
  correct_locomo10.json        # 原始对话数据（含 QA 和 evidence）
  {speaker_a}_{speaker_b}/
    msgmem.jsonl               # msgmem 产物：raw message memory（只读输入）
    atommem.jsonl              # atommem 产物：抽取出的原子记忆（评测与 evomem 的输入）
    memory.jsonl               # 旧命名遗留（= 早期 msgmem），不再使用
    raw_memory.jsonl           # 旧命名遗留（= 早期 atommem），不再使用
    atom_errors.log            # 失败样本日志（有失败时生成）
  dpo/
    atom_dpo.jsonl             # DPO/ORPO 训练数据（输出）
    atom_dpo_errors.log        # DPO 生成失败日志
  embedding_cache/
    embedding_cache.jsonl      # (model, text) -> 向量 的磁盘缓存（仅供 eval_atommem；evomem 不用缓存）
  eval_runs/                   # 每次评测的 {experiment}_{ts}/ 输出
  evomem_runs/                 # 每次 evomem 演化的 {experiment}_{ts}/ 输出
    {dir}.atommem.evolved.jsonl  # 演化后仍存活的记忆库
    {dir}.evolution.jsonl        # 逐轮 trace（含原始模型输出、增删改 id、errors）
    summary.json                 # 全部目录的汇总
src/codemem/
  msgmem.py atommem.py gen_dpo_data.py
  prompts.py                   # 全部 prompt（含 EVOMEM_PROMPT 的 {{QUESTION}} 槽）
  searchmem.py                 # 混合检索：dense + BM25 + tag 三路，RRF 融合（唯一检索实现）
  evoactions.py                # evomem 动作的解析与执行（纯 CPU）
  evomem.py                    # evomem 流程编排：QA 驱动三层循环（QA / 候选原子 / agent loop）
  evolog.py                    # 分级日志：级别过滤 + 终端 + 落盘
  eval_utils.py judge.py metrics.py
scripts/
  eval_baseline.py eval_rag.py eval_atommem.py analyze_atommem_results.py
  test_searchmem.py            # 混合检索的自测（纯 CPU，含假 embedder）
  test_evoactions.py           # 动作解析与执行的纯 CPU 自测
  smoke_evomem_logging.py      # 日志、QA 驱动循环与逐轮上报的纯 CPU 自测
  smoke_evomem_main.py         # async_main 端到端（假模型）的纯 CPU 自测
memory_template.json                 # 记忆条目字段定义（完整规范）
memory_template_init_extract.json   # 原子记忆抽取字段定义（注入 atommem 的 prompt）
memory_template_evomem.json         # 原子记忆演化字段定义（注入 evomem 的 prompt）
docs/evomem_plan.md                  # EvoMem 设计（当前权威版本）
```

> 注：`data/*/atommem.jsonl` 的 10 个目录均已生成完毕。

---

## 1. msgmem — 初始化 raw memory

将 `data/correct_locomo10.json` 里的 session 转换为 memory item，按
`data/{speaker_a}_{speaker_b}/msgmem.jsonl` 存储。

```bash
python -m codemem.msgmem
```

生成的每条 raw memory：

| 字段 | 值 |
|---|---|
| `memory` | 原始消息文本 |
| `id` | `session_{session_index}_{message_index}` |
| `type` | `raw` |
| `time` | 会话首条为 session 时间（ISO-8601，无时区）；其余为 `few minutes in {session_time}` |
| `tag` | `["speaker:<name>"]` |
| `source` | `[]`（raw 类型无 source） |
| `changelog` | `[{"time": <真实创建时间>, "content": "created"}]` |

**记忆条目格式**（参考 `memory_template.json`）：
```json
{
  "memory": "记忆内容文本",
  "metadata": {
    "id": "唯一标识符",
    "type": "raw|inner|outer",
    "time": "ISO-8601 时间戳或时间范围",
    "tag": ["speaker:name", "topic:...", "entity:...", ...],
    "source": ["源记忆ID列表"],
    "changelog": [
      {"time": "修改时间", "content": "修改前内容"}
    ]
  }
}
```

---

## 2. atommem — 抽取原子记忆

逐条读取 `msgmem.jsonl` 中的 raw memory，带上下文窗口调用大模型，抽取原子记忆并写入
`atommem.jsonl`。原始 `msgmem.jsonl` 始终保持不变。

```bash
# 处理全部 speaker 目录
python -m codemem.atommem

# 只处理指定目录
python -m codemem.atommem Caroline_Melanie
python -m codemem.atommem H:/Project/CodeMem/data/Calvin_Dave   # 也支持完整路径

# 调试：每个目录只处理前 N 条 raw
python -m codemem.atommem --limit 4 Caroline_Melanie

# 覆盖并发数与上下文窗口
python -m codemem.atommem --concurrency 4 --context-window 3

# 断点续跑：跳过已完成的 raw，只重试失败或未完成项
python -m codemem.atommem --resume Caroline_Melanie
```

**参数**

| 参数 | 说明 |
|---|---|
| `dirs` | speaker 目录名或路径；缺省处理全部 |
| `--limit N` | 每个目录只处理前 N 条 raw（调试用） |
| `--concurrency N` | 覆盖配置里的并发数 |
| `--context-window N` | 覆盖配置里的上下文窗口（前后各 N 条） |
| `--resume` | 跳过已完成的 raw，只重试失败或未完成项 |

**生成的原子记忆**：模型填写 `memory` / `type` / `time` / `tag`；
`id`（`{raw_id}_{n}`）、`changelog`（真实创建时间）由系统补全。
模型输出解析失败或截断时，该条写入 `atom_errors.log`，不影响其它条。合法的空 `atoms`
结果也会记录为已完成；使用 `--resume` 时只重试失败项。

**原子记忆字段说明**：
- `type`: `inner`（核心身份、长期事实）或 `outer`（临时状态、偏好、计划）
- `time`: ISO-8601 格式，优先使用绝对时间；无法解析时保留原描述（如 "two days before Alan was born"）
- `tag`: 必须包含 `speaker:<name>`，其他可选标签：`topic:`, `entity:`, `relation:`, `action:` 等

其余字段为系统生成

---

## 3. gen_dpo_data — 生成 DPO 训练数据

为每个 raw message memory 构造一对偏好数据：

- **chosen** — 强模型 + 完整 prompt 生成
- **rejected** — 小模型 + 简化 prompt 生成；若重试后仍无法解析出合法 JSON，
  保留其**原始错误输出**作为负样本
- **messages** — 存强模型使用的完整 prompt，作为训练时的输入 prompt

```bash
# 处理全部目录
python -m codemem.gen_dpo_data

# 只处理指定目录
python -m codemem.gen_dpo_data Caroline_Melanie

# 调试：每个目录只取前 50 条
python -m codemem.gen_dpo_data --limit 50

# 断点续跑：跳过已生成过的样本，追加到已有输出
python -m codemem.gen_dpo_data --resume

# 覆盖强/弱模型并发数
python -m codemem.gen_dpo_data --strong-concurrency 8 --weak-concurrency 1
```

**参数**

| 参数 | 说明 |
|---|---|
| `dirs` | speaker 目录名或路径；缺省处理全部 |
| `--limit N` | 每个目录只取前 N 条 raw（调试用） |
| `--strong-concurrency N` | 覆盖强模型并发数 |
| `--weak-concurrency N` | 覆盖小模型并发数 |
| `--context-window N` | 覆盖上下文窗口大小 |
| `--output PATH` | 覆盖输出文件路径 |
| `--resume` | 跳过已生成过的样本，追加到已有输出 |
| `--include-identical` | 保留 `chosen == rejected` 的样本 |

**关于 `--resume`**

- 增量落盘：每处理完一个目录立即追加写入，中断不丢已生成数据
- 以 `(目录, message_id)` 为去重键（不同 speaker 目录会有同名 id，如都从 `session_1_1` 开始）
- 不加 `--resume` 时会**清空旧输出**重新生成

**输出格式（TRL / LLaMA-Factory 兼容）**

```json
{
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "chosen":   "<强模型输出的 JSON 字符串>",
  "rejected": "<小模型输出的 JSON 字符串；解析失败时为原始错误输出>",
  "meta": {
    "id": "session_1_2",
    "dir": "Caroline_Melanie",
    "chosen_model": "gpt-4.1-mini",
    "rejected_model": "qwen3:14b",
    "rejected_parse_failed": false
  }
}
```

`rejected_parse_failed: true` 表示该 rejected 是小模型解析失败的原始输出（DPO 中正是要抑制的行为）。
weak prompt 会有意保留上下文泄漏、事实合并、遗漏隐含事实、代词未消解、重复输出等
典型错误，以便构造更有区分度的 rejected；chosen 和训练输入 `messages` 使用完整的
`ATOM_EXTRACTION_SYSTEM_PROMPT`。

> 这一阶段训练的是**抽取**能力（把消息拆成原子记忆）。记忆的**演化**是另一个任务，
> 见 §5 与 `docs/evomem_plan.md`。当前 §5 是纯 prompt 版（不训练）；计划中更远的
> 强化学习版本里，`rejected` 的含义会变成"没能把该 QA 变成可回答"。

## 4. 评测

评测直接读取 `data/correct_locomo10.json`。每条 QA 的 `evidence`（例如 `D1:3`）会映射为
`session_1_3`（即原始对话的第1个session第3条消息），并统计答案等价性、exact match、token F1 和 evidence recall。

**Evidence 映射规则**：`D{session_index}:{message_index}` → `session_{session_index}_{message_index}`
- 示例：`D1:3` → `session_1_3`
- 示例：`D2:5` → `session_2_5`

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

## 配置

### `configs/model.yaml`（atommem 使用）

```yaml
base_url: https://api.openlux.ai/v1   # OpenAI 兼容 /v1/chat/completions
api_key: sk-...
model: gpt-4.1-mini
temperature: 0.2
max_tokens: 8192       # 注意：reasoning token 也计入，过小会导致输出被截断
concurrency: 4
timeout: 300
max_retries: 6
backoff_base: 2.0
rate_limit_backoff: 10.0
max_backoff: 120.0
backoff_jitter: 0.5
context_window: 4
enable_thinking: false # 仅 ollama/qwen3 生效，其它服务忽略
```

### `configs/dpo_data.yaml`（gen_dpo_data 使用）

```yaml
strong:                # 生成 chosen
  base_url: https://api.openlux.ai/v1
  api_key: sk-...
  model: gpt-4.1-mini
  temperature: 0.2
  concurrency: 4       # 强模型独立并发数
weak:                  # 生成 rejected
  base_url: http://localhost:11434/v1
  api_key: ollama
  model: qwen3:14b
  temperature: 0.7
  concurrency: 1       # 小模型独立并发数（本地模型通常调小）
max_tokens: 8192
timeout: 300
max_retries: 6
backoff_base: 2.0
rate_limit_backoff: 10.0
max_backoff: 120.0
backoff_jitter: 0.5
context_window: 4
enable_thinking: false # 仅 ollama/qwen3 生效，其它服务忽略
output: data/dpo/atom_dpo.jsonl
error_log: data/dpo/atom_dpo_errors.log
```

### `configs/evomem.yaml`（evomem 使用）

字段含义见文件内注释。要改的关键项：`embedding`（本地 ollama）、`generator`（演化用强模型）、
**QA 驱动闸门** `max_qa`（调试上限，0 = 全部 QA）/ `qa_candidates`（每条 QA 取几个候选，K）/
`skip_category5`、**混合检索** `search.rrf_k` 与 `search.weights`（`{dense, bm25, tag}`，
设成 `{dense: 1.0, bm25: 0.0, tag: 0.0}` 即退化为纯 embedding 检索）、
`top_k` / `context_window` / `max_evomem_turn`（**每条候选原子**的内层轮数上限，默认 2）/
`concurrency`（目录级）/ `library_max_tokens`、以及 `log`（见 §5.4）。

> ⚠️ **成本**：QA 驱动下调用次数 ≈ `QA 数 × qa_candidates × max_evomem_turn`。全 1986 条 QA、
> K=10、轮数 2 时最坏约 4 万次强模型调用。不过每次 prompt 只含**一条目标 + top_k 条证据**
> （不是整个库），所以单次比早期版本小得多。真跑前先用 `--max-qa 20` 或 `--max-turns 1` 试。

---

## 常见问题

**路径里的反斜杠**：在 Git Bash 下 `H:\Project\...` 的反斜杠会被 shell 吞掉，请用正斜杠 `H:/Project/...`。

**输出被截断（`finish_reason=length`）**：某些模型（如 gemini 系列）会消耗大量 reasoning token，
且这些 token 计入 `max_tokens`。若日志出现 `TruncatedCompletion`，调大 `max_tokens`。

**`chat_template_kwargs` 报 400**：该参数仅对 ollama/qwen3 端点发送（用于关闭思考过程），
其它端点会自动跳过，无需手动配置。

**远端 API key**：`configs/*.yaml` 里的 key 是明文，`.gitignore` 已排除 `configs/`，注意不要提交。

## 5. EvoMem — 用 prompt 逐步演化原子记忆

atommem 是"从消息里抽出事实"，evomem 是"把抽出来的记忆库改得更好"：更准确、更自足、
更完整。当前阶段**不做强化学习**，直接用一个强模型按 prompt 反复演化。

**演化单位是"一条 QA 问题"**：不再逐条精修全库原子，而是**按问题驱动** —— 用 QA 的
question 在 atommem 上做**混合检索**得到一个候选原子集合，再对这个集合内的每条原子做
演化，最后得到"更可能回答这个问题"的原子集合。这样既有成本上的好处（调用只花在与问题
相关的原子上），也对后续 RL 友好（一条 QA 一条 trajectory）。

### 5.1 流程

对每个 speaker 目录（**目录之间并行，目录之内串行** —— 第 N 轮动作依赖前 N-1 轮改完的
记忆库）。**QA 驱动的三层循环**：

```
base = atommem.jsonl（按 atom_key 排好）;  anchor = 最后一条 raw 消息的 id

for 每条 QA（跳过 category 5）:                # 外层：问题驱动
    候选集 = searchmem.search(question, base, top_k=K)   # 混合检索 dense+BM25+tag（RRF）
    for current in 候选集内每条原子:            # 中层：候选集内逐条
        召回（**每条候选只做一次**）: current 文本 + 最近动作摘要 → query
              → 从「位置早于 current」的存活原子里按 embedding 相似度取 top_k
              → 每条附上它 source 对应的 msgmem + 前后 context_window 条原始消息
        for inner in 1..max_evomem_turn:       # 内层 agent loop，默认每条最多 2 轮
            构建 prompt（question + current + 召回证据 + HISTORY）→ 调模型 → 恰好一个动作
            → 执行（纯 CPU）→ 把执行结果写进 HISTORY，作为下一轮的反馈
            输出本轮结果（status / 动作 / 增删改 / 错误）
            动作是 NOOP → 结束**这一条原子**，候选集里下一条继续
            用满 max_evomem_turn → 同样结束这一条，候选集里下一条继续
    落一条 QA trajectory（问题 / 候选 / 演化后集合 / 逐轮轨迹）

输出: {dir}.qa_trajectories.jsonl + {dir}.atommem.evolved.jsonl
      + {dir}.evolution.jsonl（逐轮明细）+ summary.json
```

**每条 QA 从同一份 base 库出发**（QA 之间互不影响）—— 这是"一条 QA 一条 trajectory"的
前提，也让 base `atommem.jsonl` 始终不被修改。代价是不同 QA 可能演化出重复的新原子
（都挂在同一个 anchor 下），第一版接受这一点，重复度只在目录级并集快照里体现。

`max_evomem_turn` 是**每条原子**的内层轮数上限，**不是全目录总轮数**。`NOOP` 只结束当前
这条原子的内层循环（prompt 里写明这是大多数条目的正常结果），候选集里下一条继续。坏输出
（解析不出动作）同样只结束这一条，不终止整个 QA、更不终止目录。

**每条原子只召回一次**：内层各轮共享同一份证据，上一轮改了什么靠 `HISTORY` 传给模型，而
不是重新检索 —— 避免把模型自己刚写出的措辞当成"独立证据"又召回来，绕成自我指涉。

每个目录跑完返回 `status`（`OK` / `ERROR` / `SKIPPED`）与逐轮结果，便于跑完再看下一步。

**候选池不足 `top_k` 不做特殊处理**：库里前 `top_k` 条的候选天然少于 `top_k`，第一条更是空池
（召回 0 条，prompt 的 `{{RECALLED}}` 渲染成 `(no recalled memories)`）。这些照常走完整流程 ——
embedding 在这里的作用本来就是**排序**，候选不够就拿到多少算多少。

**硬约束**：`library_max_tokens`（默认 40000）超限时从最旧的开始软删除；动作只作用于
召回的 top_k 子区；embedding 失败**直接报错终止该目录**，不静默退化（退化只能按文件顺序
取前 top_k，那不是语义召回，模型会对着无关证据回 NOOP —— 看起来像"库已很好"，实际是嵌入挂了）。

### 5.1.1 混合检索（`src/codemem/searchmem.py`）

`searchmem` 是本项目唯一的检索实现，**两种模式共用同一套打分**：

| 模式 | 用在哪 | query 文本 |
|---|---|---|
| atom → atom | evomem 给当前原子找证据 | 该原子的 `memory` 文本 + 最近动作摘要 |
| question → atommem | evomem 选候选集 / QA 回答 | 问题的文本 |

**三路通道，RRF 融合**（`k = 60`）：

```
score(d) = Σ_r  w_r / (k + rank_r(d))
    r = dense   余弦相似度排名                权重 1.0
    r = bm25    BM25 词法排名（纯 stdlib 实现） 权重 1.0
    r = tag     tag 命中计数排名              权重 0.5
```

**为什么用 RRF 而不是加权分数**：dense 余弦、BM25 分数、tag 计数三者量纲完全不同，直接
加权要先做分数归一化，而归一化对分布很敏感（换库、换 query 长度就得重调）。RRF 只看
**名次**，天然可比、免调参；以后加 time 通道就是多一路，不影响其它路。

**为什么 tag 单列一路而不是拼进被嵌入的文本**：`speaker:Caroline` 这种值对 embedding
几乎不携带信息（会被整句话的语义平均掉），但对标签匹配很关键；单列一路才能给它独立权重。

**为什么零信号候选要兜底填充**：RRF 只给"有信号的候选"排名，所以纯词法模式下一条在
BM25/tag 上都不命中的原子会完全消失。但调用方要的是**恰好 K 条**候选去演化 —— 没有词法
信号不代表语义不相关（那正是 dense 通道的意义）。所以零信号候选按库中顺序接在有效候选
之后**补足 `top_k`**（`score=0`、无通道明细，可辨认），不会被静默截短。

纯 CPU 自测：`python scripts/test_searchmem.py`（无需 embedding 端点，用假 embedder 测
dense 通道）。

### 5.2 动作空间

模型每次只输出**一个**动作块（`src/codemem/prompts.py` 的 `EVOMEM_PROMPT`）：

```
<ADD>[atommem, ...]</ADD>       新增（从候选集内 + 召回证据派生）
<UPDATE>[atommem, ...]</UPDATE> 就地改写已有条目
<DELETE>[atommem, ...]</DELETE> 只删冗余重复
<NOOP></NOOP>                   不需要改动（多数轮次的正常结果）
```

`EVOMEM_PROMPT` 的占位符：`{{QUESTION}}`（为什么这条原子此刻被演化）/ `{{CURRENT}}` /
`{{RECALLED}}` / `{{HISTORY}}` / `{{SCHEMA}}`。

三种载荷动作共用**同一个形状**（完整 atommem 对象数组）：

| 动作 | `id` | `source` | `changelog` | 不变量 |
|---|---|---|---|---|
| `ADD` | 留空 `""`，系统分配 `{anchor}_{n}` | **必须非空** | 留空 `[]`，系统记 `created` | 不与库内重复；每轮最多 3 条 |
| `UPDATE` | **必须原样回填**库中 id | 与旧值取并集 | 留空 `[]`，系统归档旧 `memory`/`time` | `token-F1(新,旧) ≥ 0.6`；每轮最多 3 条 |
| `DELETE` | **原样回填** id，且 `memory` 逐字复制库中文本 | —— | —— | **唯一理由**是冗余（另一条已表达同一事实）；过时值用 UPDATE 而非 DELETE；每轮最多 3 条 |

两处"设计 vs 实现"的取舍：DELETE 计划说不靠 id（怕误删），但没有 id 无法定位，所以仍然
要求回填 id **且逐字复制被删文本**，`(id, memory)` 成对出现便于核对，`verify_payload_matches`
在不一致时给 warning；不设 `INSERT`（全新事实归 atommem 抽取阶段），"联合两条已有记忆"走 `ADD`。

### 5.3 用法

```bash
# **调试首选**：只跑前 20 条 QA（每条 QA 取 10 个候选、每条最多 2 轮），零风险试跑
python -m codemem.evomem --sample Caroline_Melanie --max-qa 20

# 最省的一次试跑：20 条 QA × 每条候选仅 1 轮
python -m codemem.evomem --sample Caroline_Melanie --max-qa 20 --max-turns 1

# 看完整数据流（混合检索候选 / 召回 / prompt / 模型原始输出 / 动作解析）
python -m codemem.evomem --sample Caroline_Melanie --max-qa 5 --max-turns 1 --log-level debug

# 全部目录全部 QA（注意成本：≈ QA 数 × 候选数 × 轮数 次强模型调用）
python -m codemem.evomem --experiment evomem_dev

# 纯 CPU 自测（零模型成本）
python scripts/test_searchmem.py           # 混合检索（BM25 / tag / RRF 融合）
python scripts/test_evoactions.py          # 动作解析与执行
python scripts/smoke_evomem_logging.py     # 日志、QA 驱动循环与逐轮上报
python scripts/smoke_evomem_main.py        # async_main 端到端（假模型）
```

| 参数 | 说明 |
|---|---|
| `dirs` | speaker 目录名或路径；缺省处理全部含 `atommem.jsonl` 的目录 |
| `--config PATH` | 配置文件路径（默认 `configs/evomem.yaml`） |
| `--sample NAME` | 只处理指定目录 |
| `--max-turns N` | 覆盖 `max_evomem_turn`（**每条原子**的内层轮数上限） |
| `--max-qa N` | **调试用**：每个目录最多处理前 N 条 QA（不传或 0 = 全部） |
| `--candidates K` | 覆盖 `qa_candidates`：每条 QA 用混合检索取几个候选原子来演化 |
| `--top-k / --concurrency N` | 覆盖给候选找证据时的召回条数 / 目录级并发 |
| `--output-dir PATH` | 输出目录（默认 `data/evomem_runs/`） |
| `--experiment NAME` | 实验名，作为输出子目录前缀 |
| `--log-level LEVEL` | 覆盖 `log.level`：`debug`/`info`/`warn`/`error`/`silent` |
| `--log-output PATH` | 覆盖 `log.output`（日志保存目录） |

> ⚠️ **成本**：调用数 ≈ `QA 数 × qa_candidates × max_evomem_turn`。全 1986 条 QA、
> 每条 10 个候选、每条 2 轮时最坏约 4 万次强模型调用。先用 `--max-qa 20` 或
> `--max-turns 1` 小范围试跑，确认候选检索与 prompt 符合预期再放开。`summary.json` 的
> `stop_reason` 会记录这次为何停下。

### 5.4 日志

日志分两级，`configs/evomem.yaml` 的 `log` 段控制（命令行可覆盖）：

- **`info`（默认）**——每条 QA 一行、每个候选每轮一行结果、每条 QA 收尾一行、目录汇总。形如：

  ```
  INFO  [Caroline_Melanie] QA 12/199：候选 10/1273（category 2）question='When did Caroline go to the LGBTQ support group?'
  INFO  [Caroline_Melanie] QA 12 候选 3/10 current=session_5_2_1 第 1/2 轮：候选 39 召回 30 prompt≈4200tok 调用模型…
  INFO  [Caroline_Melanie] turn 78 qa=12 OK current=session_5_2_1 action=UPDATE ~3 warns=1 | library=1273 active=1270
  INFO  [Caroline_Melanie] QA 12 完成：候选 10 演化 10（+4 ~6 -1）turns=17
  ```

  第一行说明**这是第几条问题、检索到几个候选、问题原文与类别**；第二行说明**这条 QA 内
  正在处理第几个候选、内层第几轮**；第三行是本轮结果。`status` 含义：`OK` 改了库 /
  `NOOP` 模型明确不改（正常收敛，且会**结束这一条原子**，候选集里下一条继续）/
  `EMPTY` 解析出的动作一条都没执行成功 / `ERROR` 模型调用或召回失败。
  收尾再汇总每个目录的 status、`qa_done` 与 `stop_reason`。

- **`debug`**——上面全部，外加**完整真实数据流**，逐条 QA 打印：混合检索候选（id + RRF 分 +
  各路名次）、召回 query、召回了哪些记忆（id + 文本）、完整 prompt（system/user）、模型
  原始输出、动作解析与执行明细（kind/payload/repairs/errors/增删改 id/软删除 id）。
  排查"为什么这条问题只检索到这些原子""为什么回 NOOP""为什么删错"看这个级别。

`log.output` 设成目录时日志同时落盘为 `evomem_{level}_{时间戳}.log`；留空则只打终端。
并行跑多目录时每行都带 `[目录名]` 前缀，日志写入加锁，不会交错乱行。

**embedding 的两个实测坑**（都已处理，出问题时先看这两条）：

- **400 且提到 tokenizer 端口连不上** —— ollama 在并发压力下的瞬时故障，与请求内容无关，
  报错信息具误导性。已按 `embedding_chunk_size` 分批 + 每批退避重试；批越小越稳
  （批 64 实测 3/3 成功，批 512 只有 1/3）。
- **`cudaMalloc failed: out of memory`** —— embedding 模型要占 GPU。12G 卡上开
  Chrome / Steam / VS Code 后常只剩 3–4G，llama-server 起不来。腾显存后再跑。

两种情况下 evomem 都会**明确报错终止该目录并计入 `status=ERROR`**，不会静默退化。

**混合检索的说明**：`searchmem` 按 id 去重候选池（演化中 ADD 可能落下重复条目），RRF 免
归一化所以不用调 score scale；权重在 `configs/evomem.yaml` 的 `search.weights` 可调，
设成 `{dense: 1.0, bm25: 0.0, tag: 0.0}` 即退化为旧的纯 embedding 检索，便于对照。

### 5.5 输出

`data/evomem_runs/{experiment}_{YYYYMMDD_HHMMSS}/`：

- `{dir}.qa_trajectories.jsonl` — **每 QA 一行**：`qa_index` / `question` / `category` /
  `reference` / `evidence` / `candidate_ids`（混合检索到的候选）/ `evolved_ids`（演化后仍
  存活的）/ `library_size` / `turns` / `actions` / `turn_status` / 增删改计数。这是后续 RL
  取 trajectory 的入口
- `{dir}.atommem.evolved.jsonl` — 目录级**并集快照**：base 库 + 各 QA 演化出的新原子
  （按 id 合并去重）
- `{dir}.evolution.jsonl` — 逐轮 trace（`turn` / `qa_index` / `current_id` / `inner` / status、
  动作、增删改 id、错误、warnings、召回 id、token 估算、模型原始输出）。`inner` 是这条原子
  的内层第几轮，`turn` 是全局第几次模型调用，`qa_index` 说明属于哪条问题
- `summary.json` — 配置、日志路径、embedding 统计、每目录摘要（`qa_total` / `qa_done` /
  `qa_skipped` / `candidates_total` / `atoms_done` 演化过的候选数 / `turns` 模型调用总数 /
  `stop_reason` 本次为何停止；不含 traces 以免体积过大）

> **原子性**：evomem 只写 `data/evomem_runs/` 下的文件，**绝不修改** `atommem.jsonl`，
> 所以不需要提前备份、事后恢复。


---

## TODO
[x] `src/codemem/judge.py`: 使用 LLM 判断候选答案与参考答案是否语义等价（已优化为更严格的评判标准）。
[x] `src/codemem/metrics.py`: 提供 LLM judge accuracy、exact match、token F1 和 evidence recall。
[x] `scripts/eval_baseline.py`: 使用完整 session 上下文进行 baseline 评测（已集成 msgmem.jsonl 加载和时间信息）。
[x] `scripts/eval_rag.py`: 使用 Qwen3-Embedding-4B 检索最多 30 条 msg memory 后评测（已集成时间戳和 speaker 信息）。
[x] `scripts/eval_atommem.py`: 对比 msgmem 和 atommem 的检索召回和答案准确率，配置在 configs/atommem.yaml 中（测试显示 atommem evidence recall 85% vs msgmem 28%，judge accuracy 48% vs 16%）。
[x] `src/codemem/evoactions.py` + `prompts.py` + `evomem.py`: EvoMem 基本版（动作解析与执行、`EVOMEM_PROMPT`、召回与多轮演化），含分级日志与逐轮结果上报
[x] **CRITICAL FIX** EvoMem 收敛问题修复（2026-09-23）：增强HISTORY信号强度 + 强化prompt停止条件 + context_window增加到2。收敛率从35%提升到90%，卡死率从30%降为0%，摆动率从20%降为10%。详见 `EVOMEM_COMPLETE_REPORT.md`
[x] **`src/codemem/searchmem.py`**: 混合检索（dense + BM25 + tag 三路，RRF 融合），支持 atom→atom 与 question→mem 两种查询；纯 CPU 自测 `scripts/test_searchmem.py`
[x] **QA 驱动的 EvoMem**：外层从"走遍全库逐条演化"改成"按 QA 问题混合检索候选集 → 候选集内逐条演化"，`EVOMEM_PROMPT` 加 `{{QUESTION}}` 槽，输出 `{dir}.qa_trajectories.jsonl`（每 QA 一条 trajectory，便于 RL）；日志与自测同步更新
[ ] **S0** `src/codemem/eval_utils.py`: 修 `evidence_to_ids` 解析 bug —— 空格分隔多引用（`"D9:1 D4:4 D4:6"`）与 `"D:11:26"` 目前被静默丢弃，影响 4 条 QA（Evan_Sam 3 / Tim_John 1）
[ ] **S1** `src/codemem/evocheck.py`: 更严格的 CPU 不变量（4-gram 重叠、`token-F1 ≥ 0.6`、时间可解析性、删除必须能指出同文本的另一条）
[ ] **S2** EvoMem 强化学习：QA 驱动的候选检索子区**已是主流程**（见 §5.1），在其上加 oracle 轨迹采样 + GRPO 训练（奖励 `w_ans·Judge + w_attr·credit + w_proc·mean proc − w_len − w_size`），详见 `docs/evomem_plan.md`
[ ] **S3** 纯 prompt 版效果评估：跑完 10 个目录，对比演化前后的记忆库与 QA 准确率（混合检索 dense-only vs hybrid 的 evidence recall 也要对照，ceiling 参考 ~78%）

[ ] **S4** `ANSWER_PROMPT` 加 `used_ids` 字段并校准准确率（后续 `credit` 归因依赖它）

（注：`data/*/atommem.jsonl` 的 10 个目录均已生成完毕。）
