# CodeMem

面向代码/对话记忆的原子记忆系统。当前已实现两个阶段：

1. **`init_mem`** — 把原始对话数据格式化为 raw message memory
2. **`atommem`** — 调用大模型从 raw memory 抽取原子记忆
3. **`gen_dpo_data`** — 为原子记忆抽取模型生成 DPO 偏好训练数据

---

## 环境

```bash
pip install openai pyyaml
```

Python 3.12+。

---

## 目录结构

```
configs/
  model.yaml          # atommem 的模型接口配置
  dpo_data.yaml       # gen_dpo_data 的模型接口配置
data/
  correct_locomo10.json        # 原始对话数据
  {speaker_a}_{speaker_b}/
    memory.jsonl               # raw memory + 生成的原子记忆
    raw_memory.jsonl           # raw 备份（首次运行时生成）
    atom_errors.log            # 失败样本日志（有失败时生成）
  dpo/
    atom_dpo.jsonl             # DPO 训练数据（输出）
    atom_dpo_errors.log        # DPO 生成失败日志
memory_template_init_extract.json   # 原子记忆字段定义（注入 prompt）
```

---

## 1. init_mem — 初始化 raw memory

将 `data/correct_locomo10.json` 里的 session 转换为 memory item，按
`data/{speaker_a}_{speaker_b}/memory.jsonl` 存储。

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
| `source` / `target` | `[]` |
| `changelog` | `[{"time": <真实创建时间>, "content": "created"}]` |

---

## 2. atommem — 抽取原子记忆

逐条处理 raw memory，带上下文窗口调用大模型，抽取原子记忆并写回 `memory.jsonl`。
首次运行会把原始文件备份为 `raw_memory.jsonl`。

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
```

**参数**

| 参数 | 说明 |
|---|---|
| `dirs` | speaker 目录名或路径；缺省处理全部 |
| `--limit N` | 每个目录只处理前 N 条 raw（调试用） |
| `--concurrency N` | 覆盖配置里的并发数 |
| `--context-window N` | 覆盖配置里的上下文窗口（前后各 N 条） |

**生成的原子记忆**：模型只填 `memory` / `type` / `time` / `tag`；
`id`（`{raw_id}_{n}`）、`source`（固定为源 message id）、`target`（空）、`changelog`（真实创建时间）由系统补全。
模型输出解析失败或截断时，该条写入 `atom_errors.log`，不影响其它条。

---

## 3. gen_dpo_data — 生成 DPO 训练数据

为每个 raw message memory 构造一对偏好数据：

- **chosen** — 强模型 + 完整 prompt 生成
- **rejected** — 小模型 + 简化 prompt 生成；若重试后仍无法解析出合法 JSON，
  保留其**原始错误输出**作为负样本
- **messages** — 存小模型实际使用的简化 prompt，使训练分布与部署时一致

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
