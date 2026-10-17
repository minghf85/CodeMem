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
2. **抽取**（LLM）：一次调用产出整个 JSON 数组
   `[{memory, time_expr, role, tag}, ...]`。`memory` 要求**不含任何时间词**（时间走 `time_expr`），
   且要**抓具体值**（书名/频率/用途/枚举成员都别抽象成类别）。**放不下就切成多次**（见下）。
3. **后处理**（纯 CPU）：`time_expr` + 会话锚点 → 最终 `metadata.time`；`tag` 收敛到固定 key 集合；
   补 `id`/`type`/`source`/`changelog`；最后把新抽出的 atom **并进已有集合**（`merge_atoms`：
   同事实只留一条、`source` 取并集；候选用 token 倒排索引筛，**别拿整个库重跑两两比较** ——
   那是 O(n²)，而 API 每次 Add 都要做一遍，库大了就会把服务拖住）。

**整目录并行**：每条 message 的抽取互不依赖（召回只读 `msgmem`、id 按各自 source 分配、去重是
收尾一次合并），所以全部 message 并发抽取后一次性合并落盘 —— 419 条 message 从串行
~45 分钟降到 ~1 分钟。

#### 输入预算：一条消息装不进模型上下文时怎么办

平台喂进来的"一条 message"可能是**一整篇文档**（`document` / `narration` 形态见过几万字），
而 `context_window` 条前文也可能是长消息。整段塞进 prompt 的结果不是"抽得糙"，而是**一条 atom
都没有**：端点回 400 `maximum context length`，`llm.py` 把 4xx 判成不可重试，整条 message 静默
归零。所以 `atommem_pipeline` 对 prompt 的输入有硬预算（`max_input_tokens`，system + user 一起算）：

```
预算 max_input_tokens
  ├─ 扣掉 system prompt 与模板骨架（模型侧的固定开销）
  └─ 余下按比例分给「前文上下文」（35%）与「目标消息」
       ├─ 上下文：按**行**从最新往回留（消解指代靠最近的几轮），丢掉的部分留一行标记
       └─ 目标消息：超预算就**切块**（按行切，单行太长则按字符硬切；块间重叠 10%，
          免得一个事实正好被切在边界上、两边都抽不出来），逐块抽取后合并
```

token 数是**估**的（`llm.estimate_tokens`：CJK 一字一个，其余 4 字符一个，宁高勿低），
估不准时靠两条**自适应重切**兜底：

| 端点怎么报的 | 处理 |
|---|---|
| 输入超上下文（400 / `context length`） | 识别出来（`llm.is_context_length_error`），把预算**砍半**、重新切块再试（最多 3 次） |
| 输出被 `max_tokens` 截断（`TruncatedCompletion`） | 先按对象边界**抢救**已吐出的记录（`io.salvage_json_records`），再把这一块切细重试（块小 → 一次要吐的 atom 少 → 放得下） |

两种情况都只影响自己那一块，别的块照跑。真有一块怎么切都不行时返回 `ok=False` —— **API 侧据此
不记账"这条已经抽过"**，下次刷新会重试，而不是把它永久钉成 0 条（见 `core.md` §4.3）。

**`summem` 用同一把尺子、同一个键。** 会话总结原来是**整段塞**：一个会话 15 万 token 就直接
400，那个会话一条总结都没有（而它正是检索要用的路标）。现在超出预算就**分段总结再拼起来**
（连续切、段间重叠一条消息，段首标 `(part k/n)`；单条消息本身就超预算时按"保头 + 保尾"截断），
各段拼成一条 summary。端点仍判超长（估算偏乐观）就把预算砍半重切。

预算关掉（`max_input_tokens: 0`）即退回旧行为：整段上下文 + 整条消息一次发出去。

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
| `max_input_tokens` | **prompt 的 token 预算**（system + user）；超出就截上下文 / 切消息块。按模型的上下文长度设（8k→6000、32k→24000、128k→60000）；0 = 不控制 |
| `max_tokens` | 输出上限（reasoning token 也计入，留足余量） |

---

## 与 agent 变体的关系

历史上 `atommem` 还有**逐 message 的 code agent**（`read`/`edit`/`bash` 自己 grep 前文消解指代，
`--mode agent` / `agent-v2`）。它更灵活，但弱模型在 loop 里不稳定，故**不作默认**。

**主分支只保留 pipeline。** agent 变体的实现单独放在 **`add-agent` 分支**（`git switch add-agent`），
需要时再取。

## 封装 API 里的 add

`POST /add` 复用**同一条 pipeline**（`api/store.py` 的 `_refresh_atoms` 调
`atommem_pipeline.run_message_pipeline`），只针对本次写入的会话重建 `summem` 与 `atommem`，
兑现"返回即立即可检索"。两处与离线不同的地方：

- **增量抽取**：只抽这个会话里**还没抽过**的 message（进度记在 `atom_state.json`）。Add 是追加
  式的，而一条 message 的抽取只依赖它**之前**的原文（`recall_context` 只往前看），所以早抽过的
  结论不会因为后面又追加了消息而变 —— 不必每次重抽整个会话。`rebuild_api_store.py` 走
  `force=True` 绕过它（重建的意义就是整会话重算）。
- **模型并发有闸门**：一次 Add 会按 message 并发抽取，所以"同时接 32 个 Add"不等于"32 个模型
  请求"。所有 add 侧调用（summem + atommem）共享一个进程级信号量，额度 = `add_concurrency`。
- **失败要记账、要重试**：一条 message 抽取失败时，Add 请求**仍然是成功的**（消息可检索，只是
  少几条 atom），所以它不会留在 `meta.pending` 里、重启的续跑也捞不到它。三层兜住：Add 内
  **就地重试**（`atom_retry_rounds`，默认 2 轮）；把失败记进 `atom_state.json` 的 `failed`，
  由**补漏**（启动后台任务 / `python -m codemem.api --repair`）重刷；而**端点不接受这个 prompt**
  的那些（内容策略拦截、非法请求）记进 `blocked` —— 重发同一份 prompt 还是同样结果，所以**不**
  自动重试，换端点后 `--repair --retry-blocked` 重来。补漏只认显式记账的失败，常态下一个模型
  调用都不发。

服务侧的并发与锁的作用域见 [`docs/core.md` §4.1](core.md)。
