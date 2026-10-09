# add —— 把一段对话记忆建成"三层记忆"

add 是整条链路的入口。它把一段两人对话（`msgmem` 原话）加工成**三层可检索的记忆文件**，
供 search 用同一套 `grep`/`read` 检索。三层都是同一种 JSONL 格式（一行一条），只是 `type`
不同：`msg` / `summary` / `atom`。

```
一段对话 ──► msgmem.jsonl ──► summem.jsonl ──► atommem.jsonl
              (纯 CPU)         (逐会话 LLM)      (逐 message pipeline)
```

产物落在 `data/{speaker_a}_{speaker_b}/`。

| 文件 | `type` | 内容 | 生成方式 |
|---|---|---|---|
| `msgmem.jsonl` | `msg` | 每条原始消息（**原话**），`time` 是会话时间戳（锚点），`role` 是说话人 | 纯 CPU，由 dataset 直接转换 |
| `summem.jsonl` | `summary` | 每个会话一份总结（`id = session_N_summary`） | LLM，逐 session 一次 |
| `atommem.jsonl` | `atom` | 逐 message 抽出的**原子事实**（自足、指代消解、时间锚定） | **pipeline**（单次 LLM + 代码级后处理） |

**为什么是三层，而不是"改写消息本身"**：`atommem` 是检索的**主语料** —— 一条 atom 是自足、
指代消解、时间锚定的事实，`grep` 命中的就是能回答问题的原料；`msgmem` 保留**原话**（补具体答案
与溯源，也是时间/指代的锚点）；`summem` 提供会话粒度的路标（"哪次会话谈过这件事"）。
三条链格式统一，search 用同一套 `grep`/`read` 检索它们。

---

## 三层怎么建

### 1. `msgmem` —— 纯 CPU

`python -m codemem.add --stage msgmem` 把数据集里的一段对话逐条转成 `msg` 记录：`id` 是
`session_{s}_{i}`，`role` 用**说话人真名**，`time` 是会话时间戳（后续所有相对时间的锚点）。

### 2. `summem` —— 逐会话总结

`--stage summem` 每个 session 调一次 LLM，产出 `session_N_summary`。它只用来**定位**哪次会话
谈过某件事，不承载具体答案。

### 3. `atommem` —— 默认 pipeline（核心）

`--stage atommem`（默认模式 `pipeline`）逐 message 抽原子事实，三阶段：

1. **召回**（纯 CPU）：取同会话的前 N 条消息（`context_window`，默认 12）作为消解指代的上下文。
2. **抽取**（单次 LLM）：一次调用产出整个 JSON 数组
   `[{memory, time_expr, role, tag}, ...]`。`memory` 要求**不含任何时间词**（时间走 `time_expr`），
   且要**抓具体值**（书名/频率/用途/枚举成员都别抽象成类别）。
3. **后处理**（纯 CPU）：`time_expr` + 会话锚点 → 最终 `metadata.time`；`tag` 收敛到固定 key 集合；
   补 `id`/`type`/`source`/`changelog`；最后按 memory 归一化**全局去重**（同事实只留一条，`source` 取并集）。

**整目录并行**：每条 message 的抽取互不依赖（召回只读 `msgmem`、id 按各自 source 分配、去重是
收尾一次全局 pass），所以全部 message 并发抽取后一次性去重落盘 —— 419 条 message 从串行
~45 分钟降到 ~1 分钟。

#### 时间策略（消解的要点）

`memory` 里**不写时间词**，时间全部进 `metadata.time`，由代码（`add/atommem_time.py`）解析：
**相对表达保留原样 + 附锚点**，不折算成算出的日期（折算既丢精度、也未必能被复核算术）：

| 源话 | `time_expr` | 落盘 `metadata.time` |
|---|---|---|
| "I got married yesterday" | `yesterday` | `7 May 2023`（**只有** yesterday/today/tomorrow 折成绝对日期） |
| "I did it last week" | `last week` | `the week before 2023-05-08`（保留相对 + 锚点） |
| "last Saturday" | `last Saturday` | `the Saturday before 25 May 2023` |

会话时间戳只在没有时间表达时留空 —— **绝不**把会话时间当成事件时间。

---

## 命令与配置

```bash
python -m codemem.add                       # 全部目录，三层全跑
python -m codemem.add Caroline_Melanie      # 指定目录
python -m codemem.add --stage atommem        # 只跑某一层
python -m codemem.add --stage atommem --mode pipeline   # 显式指定 pipeline（默认）
```

配置在 `configs/add.yaml`（**注意：`*.yaml` 被 gitignore，不入库**）：

| 键 | 说明 |
|---|---|
| `model` / `base_url` / `api_key` | summem 与 atommem 共用的模型；可用 `summem:` / `atommem:` 子段分别覆盖 |
| `concurrency` | pipeline 模式下同目录并行抽取的 message 上限 |
| `context_window` | 逐消息抽取时附上的同会话前文条数 |
| `max_tokens` | 输出上限（reasoning token 也计入，留足余量） |

---

## 与 agent 变体的关系

历史上 `atommem` 还有**逐 message 的 code agent**（`read`/`edit`/`bash` 自己 grep 前文消解指代，
`--mode agent` / `agent-v2`）。它更灵活，但弱模型在 loop 里不稳定，故**不作默认**。

**主分支只保留 pipeline。** agent 变体的实现单独放在 **`add-agent` 分支**（`git switch add-agent`），
需要时再取。

## 封装 API 里的 add

`POST /add` 复用**同一条 pipeline**（`api/store.py` 的 `_refresh_atoms` 调
`atommem_pipeline.run_message_pipeline`），只针对本次写入的会话增量重建 `summem` 与 `atommem`，
兑现"返回即立即可检索"。见 [`docs/core.md` §4](core.md)。
