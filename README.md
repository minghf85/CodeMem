# CodeMem

面向代码/对话记忆的原子记忆系统。当前已实现两个阶段：

1. **`init_mem`** — 把原始对话数据格式化为 raw message memory
2. **`atommem`** — 调用大模型从 raw memory 抽取原子记忆
3. **`gen_dpo_data`** — 为原子记忆抽取模型生成 DPO 偏好训练数据

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
data/
  correct_locomo10.json        # 原始对话数据（含 QA 和 evidence）
  {speaker_a}_{speaker_b}/
    msgmem.jsonl               # 原始 message memory（只读输入）
    atommem.jsonl              # 模型生成的原子记忆
    evomem.jsonl               # 演化后的记忆（ADD/UPDATE 操作结果）
    atommem.done.json          # atommem --resume 的完成状态
    atom_errors.log            # 失败样本日志（有失败时生成）
  dpo/
    atom_dpo.jsonl             # DPO 训练数据（输出）
    atom_dpo_errors.log        # DPO 生成失败日志
  grpo/
    evomem_training.jsonl      # GRPO 训练数据（记忆操作 + 奖励）
memory_template.json                 # 记忆条目字段定义（完整规范）
memory_template_init_extract.json   # 原子记忆抽取字段定义（注入 prompt）
```

---

## 1. init_mem — 初始化 raw memory

将 `data/correct_locomo10.json` 里的 session 转换为 memory item，按
`data/{speaker_a}_{speaker_b}/msgmem.jsonl` 存储。

```bash
python -m codemem.init_mem
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
| `target` | `[]`（由系统自动生成） |
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
    "target": ["目标记忆ID列表（系统生成）"],
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
`id`（`{raw_id}_{n}`）、`target`（空，后续由系统维护）、`changelog`（真实创建时间）由系统补全。
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

## 4. 评测

评测直接读取 `data/correct_locomo10.json`。每条 QA 的 `evidence`（例如 `D1:3`）会映射为
`session_1_3`（即原始对话的第1个session第3条消息），并统计答案等价性、exact match、token F1 和 evidence recall。

**Evidence 映射规则**：`D{session_index}:{message_index}` → `session_{session_index}_{message_index}`
- 示例：`D1:3` → `session_1_3`
- 示例：`D2:5` → `session_2_5`

```bash
# 全上下文 baseline，使用所有 msgmem.jsonl 内容，实时显示 judge accuracy 和 ETA
python scripts/eval_baseline.py --experiment baseline_qwen3 --limit 10

# 中断后继续：指定同一个实验目录
python scripts/eval_baseline.py --experiment baseline_qwen3 --resume

# 通过 sglang /v1/embeddings API 做 embedding RAG，从 msgmem.jsonl 中检索最多 30 条记忆
python scripts/eval_rag.py --experiment rag_qwen3 --limit 10

# 中断后继续
python scripts/eval_rag.py --experiment rag_qwen3 --resume
```

每次运行创建 `data/eval_runs/{experiment}_{YYYYMMDD_HHMMSS}/`，目录包含对应的
`eval_baseline.jsonl` 或 `eval_rag.jsonl` 以及 `summary.json`。summary 按 LoCoMo
category 1-5 输出指标；category 5 (`adversarial_skip`) 只记录结果，不参与评分。
模型地址、生成模型、judge 模型和 embedding 模型分别配置在 `configs/baseline.yaml`、
`configs/rag.yaml` 和 `configs/judge.yaml`。

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

---

## 常见问题

**路径里的反斜杠**：在 Git Bash 下 `H:\Project\...` 的反斜杠会被 shell 吞掉，请用正斜杠 `H:/Project/...`。

**输出被截断（`finish_reason=length`）**：某些模型（如 gemini 系列）会消耗大量 reasoning token，
且这些 token 计入 `max_tokens`。若日志出现 `TruncatedCompletion`，调大 `max_tokens`。

**`chat_template_kwargs` 报 400**：该参数仅对 ollama/qwen3 端点发送（用于关闭思考过程），
其它端点会自动跳过，无需手动配置。

**远端 API key**：`configs/*.yaml` 里的 key 是明文，`.gitignore` 已排除 `configs/`，注意不要提交。

## TODO
[x] `src/codemem/judge.py`: 使用 LLM 判断候选答案与参考答案是否语义等价（已优化为更严格的评判标准）。
[x] `src/codemem/metrics.py`: 提供 LLM judge accuracy、exact match、token F1 和 evidence recall。
[x] `scripts/eval_baseline.py`: 使用完整 session 上下文进行 baseline 评测（已集成 msgmem.jsonl 加载和时间信息）。
[x] `scripts/eval_rag.py`: 使用 Qwen3-Embedding-4B 检索最多 30 条 msg memory 后评测（已集成时间戳和 speaker 信息）。
[ ] `src/codemem/searchmem.py`: 实现记忆检索（双模式：为 evomem 服务 + 为 eval 服务）
[ ] `src/codemem/evomem.py`: 实现记忆演化（GRPO 训练，双层奖励：全局 QA 准确率 + 局部强模型评分）

---

## 5. 记忆检索 (`src/codemem/searchmem.py`)

实现两种类型的记忆检索，用于记忆演化和评估。

### 5.1 AtomMem 检索 Mem（为 EvoMem 服务）

**目的**：为记忆演化提供相关上下文

**输入**：单个 atommem 条目

**输出**：与该 atommem 相关的 msgmem 和 atommem 内容

**检索流程**：

```
1. 元数据初筛
   ├─ 时间约束：只检索目标 atommem 时间之前的记忆
   ├─ Speaker 过滤：根据 tag 中的 speaker 信息
   └─ 图结构筛选：利用 source/target 关系链
   
2. Embedding 相似度筛选
   ├─ 对候选池计算与目标 atommem 的语义相似度
   └─ 按 cosine similarity 降序排列
   
3. Top-K 截取
   └─ 返回最多 30 条最相关的记忆（msgmem + atommem）
```

**示例**：
```python
# 为 atommem "John went to Paris in 2022" 检索相关记忆
target_atom = {
    "memory": "John went to Paris in 2022",
    "metadata": {
        "id": "session_3_5_1",
        "time": "2022-05-10T14:30:00",
        "tag": ["speaker:John"]
    }
}

related_memories = search_for_evomem(
    target_atom,
    all_memories,
    embedding_client,
    top_k=30
)
# 返回：时间在 2022-05-10 之前、与 John/Paris/travel 相关的记忆
```

### 5.2 Query 检索 Mem（为 Eval 服务）

**目的**：为问答评估提供检索增强（RAG）

**输入**：QA 对中的 query

**输出**：相关的 atommem 和 msgmem，用于回答问题

**检索流程**：

```
1. Embedding 检索
   ├─ 使用 query 与所有记忆计算语义相似度
   └─ 按相似度排序
   
2. Top-K 截取
   └─ 返回最多 30 条记忆
   
3. 格式化输出
   └─ 带时间戳和 speaker 信息的上下文
```

**示例**：
```python
# 检索问题 "When did John go to Paris?" 的相关记忆
query = "When did John go to Paris?"

retrieved = search_for_eval(
    query,
    all_memories,
    embedding_client,
    top_k=30
)

# 用于评估检索召回率和答案准确率
evidence_recall = calculate_recall(retrieved, ground_truth_evidence)
```

### 5.3 检索策略细节

**候选池构建**：
```python
def build_candidate_pool(target_memory, all_memories):
    candidates = []
    for mem in all_memories:
        # 时间约束
        if mem["metadata"]["time"] >= target_memory["metadata"]["time"]:
            continue
        
        # Speaker 约束（可选）
        target_speakers = extract_speakers(target_memory)
        mem_speakers = extract_speakers(mem)
        if not speakers_overlap(target_speakers, mem_speakers):
            continue
        
        # 图约束：source/target 关系链
        if is_related_by_graph(target_memory, mem):
            candidates.append(mem)
        else:
            candidates.append(mem)
    
    return candidates
```

**相似度计算**：
```python
def compute_similarity(query_embedding, memory_embeddings):
    similarities = []
    for mem_emb in memory_embeddings:
        sim = cosine_similarity(query_embedding, mem_emb)
        similarities.append(sim)
    return similarities
```

**结果限制**：
- Maximum: 30 条记忆
- 包含 msgmem（原始对话）和 atommem（提取的原子记忆）
- 格式：`[timestamp] speaker: memory_text`

---

## 6. 记忆演化 (`src/codemem/evomem.py`)

针对每个 atommem 进行细粒度操作演化优化记忆。

### 6.1 核心流程

```
对于每个 atommem:
├─ 1. 检索相关记忆池（使用 searchmem）
│   ├─ 元数据筛选（时间、speaker、图结构）
│   ├─ Embedding 相似度排序
│   └─ Top-30 相关记忆
│
├─ 2. 构建演化提示
│   ├─ 目标 atommem
│   ├─ 相关上下文记忆
│   └─ 操作指令
│
├─ 3. 大模型生成记忆操作
│   ├─ ADD: 创建新记忆
│   ├─ DELETE: 删除冗余/错误记忆
│   └─ UPDATE: 更新现有记忆
│
└─ 4. 执行操作并更新记忆库
    ├─ 验证操作合法性
    ├─ 更新 source/target 图结构
    └─ 写入 changelog
```

### 6.2 记忆操作类型

#### ADD（添加）
**场景**：
- 发现新的隐含信息
- 关系推理（例如从 "A 是 B 的朋友" 和 "B 是 C 的同事" 推断 A 和 C 可能认识）
- 时间推断（例如从 "去年去过印度" 和时间戳推断具体年份）

**示例**：
```json
{
  "operation": "ADD",
  "memory": "John went to India in 2021",
  "metadata": {
    "type": "outer",
    "time": "2021-07-15T00:00:00",
    "tag": ["speaker:John", "inferred:true"],
    "source": ["session_2_3"],
    "target": []
  },
  "reason": "从 'John说去年去过印度'（2022-07-15）推断出具体时间为2021年"
}
```

#### DELETE（删除）
**场景**：
- 重复信息（已有更详细或更准确的记忆）
- 过时信息（被后续对话更新）
- 错误提取（原子记忆抽取错误）

**示例**：
```json
{
  "operation": "DELETE",
  "target_id": "session_1_2_1",
  "reason": "与 session_3_5_1 重复，且后者包含更详细的时间信息"
}
```

#### UPDATE（更新）
**场景**：
- 信息补充（添加缺失的细节）
- 精度提升（将模糊信息具体化）
- 消歧（解决代词或指代问题）

**示例**：
```json
{
  "operation": "UPDATE",
  "target_id": "session_2_4_1",
  "memory": "John went to Paris in May 2022",
  "metadata": {
    "type": "outer",
    "time": "2022-05-10T14:30:00",
    "tag": ["speaker:John", "updated:true"],
    "source": ["session_2_4", "session_3_5"],
    "target": []
  },
  "reason": "将 '他去了那里' 更新为明确的 'John went to Paris'，并添加时间信息"
}
```

### 6.3 操作格式规范

```json
{
  "operation": "ADD|DELETE|UPDATE",
  "target_id": "session_X_Y_Z",  // UPDATE/DELETE 时必需
  "memory": "记忆内容文本",        // ADD/UPDATE 时必需
  "metadata": {
    "type": "inner|outer",
    "time": "YYYY-MM-DDTHH:MM:SS",
    "tag": ["speaker:X", "relation:Y", "inferred:true"],
    "source": ["source_memory_id_1", "source_memory_id_2"],
    "target": ["target_memory_id_1"]
  },
  "reason": "操作原因的简短说明"
}
```

### 6.4 强化学习训练（GRPO）

使用 **Group Relative Policy Optimization** 训练 Qwen3-8B 模型学习最优记忆操作策略。

#### 挑战：超长程任务的信用分配

记忆演化是一个**超长程任务**：
- 只有完全形成记忆库后，才能通过 QA 评估获得最终奖励
- 每个 atom 的上下文环境都不一样（依赖时间、speaker、图结构）
- 单个操作的好坏难以立即评估

#### 双层奖励设计：全局 + 局部

为解决长程信用分配问题，设计**全局奖励**（延迟、稀疏）和**局部奖励**（即时、密集）的组合：

##### 1. 全局奖励（Global Reward）
在所有 atom 处理完成后，通过 QA 评估计算：

```python
def compute_global_reward(updated_memory_db, qa_pairs):
    """处理完所有 atom 后，评估整体效果"""
    total_score = 0.0
    
    # 1. QA 准确率（主要指标）
    correct_count = 0
    for qa in qa_pairs:
        # 使用更新后的记忆库回答问题
        answer = answer_with_memory(qa["question"], updated_memory_db)
        if judge_answer(answer, qa["answer"]) == "CORRECT":
            correct_count += 1
    qa_accuracy = correct_count / len(qa_pairs)
    
    # 2. Evidence 召回率
    evidence_recalls = []
    for qa in qa_pairs:
        # evidence 如 ["D1:3", "D2:5"] 映射为 ["session_1_3", "session_2_5"]
        expected_ids = [evidence_to_id(e) for e in qa["evidence"]]
        retrieved = retrieve_for_qa(qa["question"], updated_memory_db, top_k=30)
        retrieved_ids = [mem["metadata"]["id"] for mem in retrieved]
        recall = len(set(expected_ids) & set(retrieved_ids)) / len(expected_ids)
        evidence_recalls.append(recall)
    avg_evidence_recall = sum(evidence_recalls) / len(evidence_recalls)
    
    # 3. 记忆库质量指标
    consistency = check_memory_consistency(updated_memory_db)
    redundancy = calculate_redundancy_ratio(updated_memory_db)
    
    # 综合全局奖励
    global_reward = (
        0.5 * qa_accuracy +           # QA 准确率（最重要）
        0.2 * avg_evidence_recall +   # Evidence 召回
        0.2 * consistency +            # 一致性
        0.1 * (1.0 - redundancy)      # 简洁性
    )
    return global_reward

def evidence_to_id(evidence_str):
    """将 evidence 如 'D1:3' 转换为 'session_1_3'"""
    match = re.match(r"D(\d+):(\d+)", evidence_str)
    if match:
        session_idx, msg_idx = match.groups()
        return f"session_{session_idx}_{msg_idx}"
    return evidence_str
```

##### 2. 局部奖励（Local Reward）
针对单个 atom 的操作，利用**强大模型**和**QA evidence**即时评估：

```python
def compute_local_reward(operation, target_atom, context_memories, qa_evidence):
    """单个 atom 操作的即时奖励"""
    
    # 1. 操作合法性（基础分）
    validity_score = validate_operation_quality(operation, target_atom, context_memories)
    
    # 2. Evidence 相关性（关键分）
    # 检查此操作是否涉及 QA evidence 中的关键记忆
    evidence_relevance = 0.0
    if operation["operation"] in ["ADD", "UPDATE"]:
        new_memory_id = operation.get("memory_id") or operation.get("target_id")
        # 检查是否与某个 QA 的 evidence 相关
        for qa in qa_evidence:
            expected_ids = [evidence_to_id(e) for e in qa.get("evidence", [])]
            # 检查 source 是否命中 evidence
            if any(src in expected_ids for src in operation["metadata"].get("source", [])):
                evidence_relevance = 1.0
                break
    
    # 3. 强模型评分（操作合理性）
    # 让强模型评估此操作是否合理、是否改进了记忆质量
    llm_score = strong_model_evaluate(
        operation=operation,
        target_atom=target_atom,
        context=context_memories,
        criteria="Is this operation reasonable and does it improve memory quality?"
    )
    
    # 4. 时间一致性
    time_consistency = check_time_consistency(operation, context_memories)
    
    # 5. 图结构合理性
    graph_consistency = check_graph_consistency(operation, context_memories)
    
    # 综合局部奖励
    local_reward = (
        0.2 * validity_score +        # 基础合法性
        0.3 * evidence_relevance +    # Evidence 相关（重要）
        0.3 * llm_score +              # 强模型评分（重要）
        0.1 * time_consistency +       # 时间一致性
        0.1 * graph_consistency        # 图结构合理性
    )
    return local_reward

def strong_model_evaluate(operation, target_atom, context, criteria):
    """使用强大模型（如 GPT-4）评估操作质量"""
    prompt = f"""
Given the target atom and its context, evaluate the following memory operation:

Target Atom:
{json.dumps(target_atom, indent=2)}

Context Memories:
{format_context(context)}

Operation:
{json.dumps(operation, indent=2)}

Criteria: {criteria}

Rate this operation on a scale of 0.0 to 1.0.
Output only a JSON: {{"score": 0.0-1.0, "reason": "brief explanation"}}
"""
    response = strong_model.generate(prompt)
    return parse_score(response)
```

#### GRPO 训练流程

```python
def train_grpo(model, all_atommems, qa_pairs, epochs=10, group_size=4):
    """GRPO 训练：结合全局和局部奖励"""
    
    for epoch in range(epochs):
        epoch_operations = []  # 收集所有 atom 的操作
        epoch_local_rewards = []
        
        # === 第一阶段：对每个 atom 生成操作并计算局部奖励 ===
        for atommem in all_atommems:
            # 1. 检索相关上下文（时间之前 + embedding 相似）
            context = search_for_evomem(atommem, top_k=30)
            
            # 2. 生成多个候选操作（sampling）
            candidates = []
            for _ in range(group_size):
                operation = model.generate(
                    atommem, context, 
                    temperature=0.8, 
                    do_sample=True
                )
                candidates.append(operation)
            
            # 3. 计算局部奖励
            local_rewards = []
            for op in candidates:
                local_r = compute_local_reward(
                    op, atommem, context, qa_pairs
                )
                local_rewards.append(local_r)
            
            # 4. 使用局部奖励计算 group advantage（即时反馈）
            local_baseline = np.mean(local_rewards)
            local_advantages = [r - local_baseline for r in local_rewards]
            
            # 5. 中间更新：基于局部奖励
            loss_local = compute_grpo_loss(candidates, local_advantages)
            optimizer.step(loss_local)
            
            # 6. 选择局部奖励最高的操作应用到记忆库
            best_idx = np.argmax(local_rewards)
            best_operation = candidates[best_idx]
            apply_operation(memory_db, best_operation)
            
            epoch_operations.append(best_operation)
            epoch_local_rewards.append(local_rewards[best_idx])
        
        # === 第二阶段：全局评估并二次更新 ===
        # 所有 atom 处理完毕，计算全局奖励
        global_reward = compute_global_reward(memory_db, qa_pairs)
        
        print(f"Epoch {epoch}: Local Reward Avg = {np.mean(epoch_local_rewards):.3f}, "
              f"Global Reward = {global_reward:.3f}")
        
        # 7. 全局奖励反向传播：重新计算所有操作的优势
        # 将全局奖励作为额外信号，与局部奖励结合
        for i, operation in enumerate(epoch_operations):
            # 混合奖励：局部即时 + 全局延迟
            mixed_reward = 0.6 * epoch_local_rewards[i] + 0.4 * global_reward
            
            # 重新生成该 atom 的候选，用混合奖励更新
            atommem = all_atommems[i]
            context = search_for_evomem(atommem, top_k=30)
            candidates = [model.generate(atommem, context, temperature=0.8) 
                          for _ in range(group_size)]
            
            # 为候选分配奖励（最佳操作获得混合奖励）
            mixed_advantages = [-0.1] * group_size  # 其他候选给负奖励
            mixed_advantages[0] = mixed_reward  # 假设第一个是重新生成的最佳
            
            loss_global = compute_grpo_loss(candidates, mixed_advantages)
            optimizer.step(loss_global)
        
        # 8. 保存检查点
        if (epoch + 1) % 5 == 0:
            model.save(f"evomem_qwen3_epoch{epoch+1}.pt")

def compute_grpo_loss(candidates, advantages):
    """Group Relative Policy Optimization 损失"""
    loss = 0.0
    for candidate, advantage in zip(candidates, advantages):
        log_prob = model.log_prob(candidate)
        loss -= log_prob * advantage  # Policy gradient
    return loss / len(candidates)
```

#### 训练优势

1. **双层奖励解决长程问题**：
   - 局部奖励提供即时反馈，加速学习
   - 全局奖励确保最终目标对齐

2. **利用强模型作为即时评判**：
   - 无需等待完整流程即可评估操作质量
   - 降低训练方差

3. **Evidence 作为关键信号**：
   - QA evidence 指明哪些记忆对回答至关重要
   - 优先优化 evidence 相关的 atom

4. **Group-based baseline**：
   - 同一 atom 的多个候选相互比较
   - 减少方差，提高训练稳定性

### 6.5 实现要点

**时间一致性**：
```python
def validate_time_consistency(operation, existing_memories):
    if operation["operation"] == "ADD":
        new_time = parse_time(operation["metadata"]["time"])
        # 确保新记忆的时间不晚于引用它的记忆
        for source_id in operation["metadata"]["source"]:
            source_time = get_memory_time(source_id)
            assert new_time <= source_time, "Time inconsistency"
```

**图结构维护**：
```python
def update_graph_structure(operation, memory_db):
    if operation["operation"] == "ADD":
        # 添加新节点和边
        memory_db.add_node(operation["memory_id"])
        for source_id in operation["metadata"]["source"]:
            memory_db.add_edge(source_id, operation["memory_id"])
    
    elif operation["operation"] == "DELETE":
        # 删除节点，重新连接相关边
        target_id = operation["target_id"]
        sources = memory_db.get_sources(target_id)
        targets = memory_db.get_targets(target_id)
        memory_db.remove_node(target_id)
        # 可选：连接 sources 到 targets
```

**操作验证**：
```python
def validate_operation(operation, memory_db):
    # 1. 检查必需字段
    assert "operation" in operation
    assert operation["operation"] in ["ADD", "DELETE", "UPDATE"]
    
    # 2. 检查 target_id 存在性
    if operation["operation"] in ["DELETE", "UPDATE"]:
        assert operation["target_id"] in memory_db
    
    # 3. 检查时间合理性
    validate_time_consistency(operation, memory_db)
    
    # 4. 检查 speaker 一致性
    validate_speaker_consistency(operation, memory_db)
    
    return True
```

**增量更新**：
```python
def incremental_evolution(new_atommem, memory_db):
    # 只对新添加的 atommem 进行演化
    context = search_for_evomem(new_atommem, memory_db, top_k=30)
    operation = model.generate(new_atommem, context)
    
    if validate_operation(operation, memory_db):
        apply_operation(memory_db, operation)
        update_changelog(new_atommem, operation)
```

### 6.6 使用示例

```bash
# 对所有 atommem 进行演化（使用预训练模型）
python -m codemem.evomem

# 只处理指定目录
python -m codemem.evomem Caroline_Melanie

# 训练 GRPO 模型（从零开始或继续训练）
python -m codemem.evomem --train \
  --model qwen3-8b \
  --epochs 10 \
  --group-size 4 \
  --local-weight 0.6 \
  --global-weight 0.4

# 使用训练好的模型进行演化
python -m codemem.evomem --model-path ./output/evomem-qwen3-8b

# 评估演化后的记忆库（对比演化前后的 QA 准确率）
python -m codemem.evomem --eval-only --before atommem.jsonl --after evomem.jsonl
```

**训练参数说明**：
- `--group-size`: 每个 atom 生成的候选操作数（默认 4）
- `--local-weight`: 局部奖励权重（默认 0.6）
- `--global-weight`: 全局奖励权重（默认 0.4）
- `--strong-model`: 用于局部评分的强模型（默认 gpt-4o-mini）

---