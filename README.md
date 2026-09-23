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
  evomem.yaml         # evomem 演化配置（embedding / generator / 轮数 / token 预算）
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
  prompts.py                   # 全部 prompt（含 EVOMEM_PROMPT）
  evoactions.py                # evomem 动作的解析与执行（纯 CPU）
  evomem.py                    # evomem 流程编排：两层循环（外层走库 / 内层 agent loop）
  evolog.py                    # 分级日志：级别过滤 + 终端 + 落盘
  eval_utils.py judge.py metrics.py
scripts/
  eval_baseline.py eval_rag.py eval_atommem.py analyze_atommem_results.py
  test_evoactions.py           # 动作解析与执行的纯 CPU 自测
  smoke_evomem_logging.py      # 日志、嵌套循环与逐轮上报的纯 CPU 自测
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
`top_k` / `context_window` / `max_evomem_turn`（**每条原子**的内层轮数上限）/
`max_atommem`（调试上限，0 = 全库）/ `concurrency`（目录级）/ `library_max_tokens`、
以及 `log`（见 §5.4）。

> ⚠️ **成本**：嵌套循环下调用次数 = 每条原子的内层轮数之和，最坏 `原子数 × max_evomem_turn`。
> 不过每次 prompt 只含**一条目标 + top_k 条证据**（不是整个库），所以单次比早期版本小得多。
> 真跑前先用 `--max-atommem 30` 或 `--max-turns 1` 试。

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

### 5.1 流程

对每个 speaker 目录（**目录之间并行，目录之内串行** —— 第 N 轮动作依赖前 N-1 轮改完的
记忆库）。**两层循环**：

```
记忆库 = atommem.jsonl（按 id 顺序排好）;  active = 全部 id;  anchor = 最后一条 raw 消息

for current in 按 id 顺序的每条原子:            # 外层游标，默认走遍全库
    召回（**每条原子只做一次**）: current 文本 + 最近动作摘要 → query
          → 从「位置早于 current」的存活原子里按 embedding 相似度取 top_k
          → 每条附上它 source 对应的 msgmem + 前后 context_window 条原始消息
    for inner in 1..max_evomem_turn:          # 内层 agent loop，默认每条最多 4 轮
        构建 prompt（current + 召回证据 + HISTORY）→ 调用模型 → 解析出恰好一个动作
        → 执行（纯 CPU）→ 把执行结果写进 HISTORY，作为下一轮的反馈
        输出本轮结果（status / 动作 / 增删改 / 错误）
        动作是 NOOP → 结束**这一条原子**，游标前进到下一条
        用满 max_evomem_turn → 同样结束这一条，游标前进

输出: {dir}.atommem.evolved.jsonl + {dir}.evolution.jsonl（逐轮明细）+ summary.json
```

`max_evomem_turn` 是**每条原子**的内层轮数上限，**不是全目录总轮数**。`NOOP` 只结束当前
这条原子的内层循环（prompt 里写明这是大多数条目的正常结果），游标继续走 —— 库里的原子都会
被看一遍。坏输出（解析不出动作）同样只结束这一条，不终止整个目录。

**每条原子只召回一次**：内层各轮共享同一份证据，上一轮改了什么靠 `HISTORY` 传给模型，而
不是重新检索 —— 避免把模型自己刚写出的措辞当成"独立证据"又召回来，绕成自我指涉。

每个目录跑完返回 `status`（`OK` / `ERROR` / `SKIPPED`）与逐轮结果，便于跑完再看下一步。

**候选池不足 `top_k` 不做特殊处理**：库里前 `top_k` 条的候选天然少于 `top_k`，第一条更是空池
（召回 0 条，prompt 的 `{{RECALLED}}` 渲染成 `(no recalled memories)`）。这些照常走完整流程 ——
embedding 在这里的作用本来就是**排序**，候选不够就拿到多少算多少。

**硬约束**：`library_max_tokens`（默认 40000）超限时从最旧的开始软删除；动作只作用于
召回的 top_k 子区；embedding 失败**直接报错终止该目录**，不静默退化（退化只能按文件顺序
取前 top_k，那不是语义召回，模型会对着无关证据回 NOOP —— 看起来像"库已很好"，实际是嵌入挂了）。

### 5.2 动作空间

模型每次只输出**一个**动作块（`src/codemem/prompts.py` 的 `EVOMEM_PROMPT`）：

```
<ADD>[atommem, ...]</ADD>       新增（从库内 + 召回证据派生）
<UPDATE>[atommem, ...]</UPDATE> 就地改写已有条目
<DELETE>[atommem, ...]</DELETE> 只删冗余重复
<NOOP></NOOP>                   不需要改动（多数轮次的正常结果）
```

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
# 先跑通一个目录（默认走遍全库，注意成本）
python -m codemem.evomem Caroline_Melanie

# **调试首选**：只跑前 30 条原子（每条最多 4 轮），零风险试跑
python -m codemem.evomem --sample Caroline_Melanie --max-atommem 30

# 看完整数据流（召回 / prompt / 模型原始输出 / 动作解析）
python -m codemem.evomem --sample Caroline_Melanie --max-atommem 5 --log-level debug

# 全部目录（注意成本：每条原子最多 max_evomem_turn 次强模型调用）
python -m codemem.evomem --experiment evomem_dev

# 纯 CPU 自测（零模型成本）
python scripts/test_evoactions.py          # 动作解析与执行
python scripts/smoke_evomem_logging.py     # 日志、嵌套循环与逐轮上报
python scripts/smoke_evomem_main.py        # async_main 端到端（假模型）
```

| 参数 | 说明 |
|---|---|
| `dirs` | speaker 目录名或路径；缺省处理全部含 `atommem.jsonl` 的目录 |
| `--config PATH` | 配置文件路径（默认 `configs/evomem.yaml`） |
| `--sample NAME` | 只处理指定目录 |
| `--max-turns N` | 覆盖 `max_evomem_turn`（**每条原子**的内层轮数上限） |
| `--max-atommem N` | **调试用**：最多处理前 N 条 atommem（按 id 顺序取；不传或 0 = 走遍全库） |
| `--top-k / --concurrency N` | 覆盖召回条数 / 目录级并发 |
| `--output-dir PATH` | 输出目录（默认 `data/evomem_runs/`） |
| `--experiment NAME` | 实验名，作为输出子目录前缀 |
| `--log-level LEVEL` | 覆盖 `log.level`：`debug`/`info`/`warn`/`error`/`silent` |
| `--log-output PATH` | 覆盖 `log.output`（日志保存目录） |

> ⚠️ **成本**：嵌套循环下模型调用 = 每条原子的内层轮数之和。全库 1273 条、`max_evomem_turn`
> 为 4 时最坏约 5000 次强模型调用。先用 `--max-atommem 30` 或 `--max-turns 1` 小范围试跑，
> 确认召回与 prompt 符合预期再放开。`summary.json` 的 `stop_reason` 会记录这次为何停下。

### 5.4 日志

日志分两级，`configs/evomem.yaml` 的 `log` 段控制（命令行可覆盖）：

- **`info`（默认）**——每次模型调用一行结果 + 收尾汇总。每次调用一行形如：

  ```
  INFO  [Caroline_Melanie] current session_5_2_1（本次第 40/1273 条，全库 1273 条）第 1/4 轮：候选 39 召回 30 prompt≈4200tok 调用模型…
  INFO  [Caroline_Melanie] turn 78 OK current=session_5_2_1 action=UPDATE ~3 warns=1 | library=1273 active=1270
  INFO  [Caroline_Melanie] turn 79 NOOP current=session_5_2_1 action=NOOP | library=1273 active=1270
  ```

  第一行说明**正在处理哪条原子、这是本次第几条、内层第几轮**；第二行是本轮结果。
  `status` 含义：`OK` 改了库 / `NOOP` 模型明确不改（正常收敛，且会**结束这一条原子**，
  游标前进到下一条）/ `EMPTY` 解析出的动作一条都没执行成功 / `ERROR` 模型调用或召回失败。
  收尾再汇总每个目录的 status、`atoms_done` 与 `stop_reason`。

- **`debug`**——上面全部，外加**完整真实数据流**，逐轮打印：召回 query、召回了哪些
  记忆（id + 文本）、完整 prompt（system/user）、模型原始输出、动作解析与执行明细
  （kind/payload/repairs/errors/增删改 id/软删除 id）。排查"为什么回 NOOP""为什么删错"
  看这个级别。

`log.output` 设成目录时日志同时落盘为 `evomem_{level}_{时间戳}.log`；留空则只打终端。
并行跑多目录时每行都带 `[目录名]` 前缀，日志写入加锁，不会交错乱行。

**embedding 的两个实测坑**（都已处理，出问题时先看这两条）：

- **400 且提到 tokenizer 端口连不上** —— ollama 在并发压力下的瞬时故障，与请求内容无关，
  报错信息具误导性。已按 `embedding_chunk_size` 分批 + 每批退避重试；批越小越稳
  （批 64 实测 3/3 成功，批 512 只有 1/3）。
- **`cudaMalloc failed: out of memory`** —— embedding 模型要占 GPU。12G 卡上开
  Chrome / Steam / VS Code 后常只剩 3–4G，llama-server 起不来。腾显存后再跑。

两种情况下 evomem 都会**明确报错终止该目录并计入 `status=ERROR`**，不会静默退化。

### 5.5 输出

`data/evomem_runs/{experiment}_{YYYYMMDD_HHMMSS}/`：

- `{dir}.atommem.evolved.jsonl` — 演化后仍存活的记忆库
- `{dir}.evolution.jsonl` — 逐轮 trace（`turn`/`current_id`/`inner`/status、动作、增删改
  id、错误、warnings、召回 id、token 估算、模型原始输出）。`inner` 是这条原子的内层第几轮，
  `turn` 是全局第几次模型调用 —— 同一条原子的多轮靠 `current_id` 相同、`inner` 递增辨认
- `summary.json` — 配置、日志路径、embedding 统计、每目录摘要（`atoms_done` 走完内层循环的
  原子数、`turns` 模型调用总数、`stop_reason` 本次为何停止、逐轮 status；不含 traces 以免体积过大）

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
[ ] **S0** `src/codemem/eval_utils.py`: 修 `evidence_to_ids` 解析 bug —— 空格分隔多引用（`"D9:1 D4:4 D4:6"`）与 `"D:11:26"` 目前被静默丢弃，影响 4 条 QA（Evan_Sam 3 / Tim_John 1）
[ ] **S1** `src/codemem/evocheck.py`: 更严格的 CPU 不变量（4-gram 重叠、`token-F1 ≥ 0.6`、时间可解析性、删除必须能指出同文本的另一条）
[ ] **S2** EvoMem 强化学习：在基本版之上加 QA+Evidence 召回子区的 oracle 轨迹采样 + GRPO 训练（奖励 `w_ans·Judge + w_attr·credit + w_proc·mean proc − w_len − w_size`），详见 `docs/evomem_plan.md`
[ ] **S3** 纯 prompt 版效果评估：跑完 10 个目录，对比演化前后的记忆库（自足性、冗余、过时条目）
[ ] **S4** `ANSWER_PROMPT` 加 `used_ids` 字段并校准准确率（后续 `credit` 归因依赖它）

（注：`data/*/atommem.jsonl` 的 10 个目录均已生成完毕。）
