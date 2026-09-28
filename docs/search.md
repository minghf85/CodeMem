# search 步骤 — Code-Agent 式证据构建（核心设计）

**目标**：对每条 question，演化出一份**真正能回答它的 `base_mem` 格式记忆列表**，落盘为
`evidence.jsonl`。

**载体**：一个 code agent。它工作在**一个目录**里，只会四个通用工具
`read` / `write` / `edit` / `bash`，其中 `bash` 的两个核心用法是 **`jq`**（JSON 工具，
见 §3.3）和 **`search`**（检索 CLI，见 §3.2）。工具描述与 schema 由**统一的 `tool.json`**
管理，渲染进上下文（§4）。

---

## 0. 定位：为什么是 code agent

上一版的实测失败（5 条 QA / 164 次调用）：`UPDATE 146 / NOOP 18 / ADD 0 / DELETE 0`，
库净变化 `+0 ~81 -0`，65 errors 多为 `unchanged, skipped`。三条根因：

| # | 根因 | 表现 |
|---|---|---|
| 1 | **检索只做一次**，候选集在第一步定死 | bridging 原子没进候选 → 后面几轮注定空转（多跳问题死在这） |
| 2 | **动作空间只有"单条原地精修"** | 只会 UPDATE，没有能力**创造**把两条记忆连起来的事实 —— 而缺的正是它 |
| 3 | **成功判据不可检验**（"更自足"） | 模型无法知道自己做好了没有 → NOOP 泛滥 + 81 次无效改写 |

还有一个此前没说清的**格式 bug**：模型把展示串
`[session_16_9_4] (2023-09-13T00:09:00) Caroline | outer | ` 当正文抄了回去。
根因是**渲染文本与可写字段是同一个通道**。

**code agent 一次解决这四条**，因为一个真实的工作目录天然具备这四个性质：

- 检索是 shell 命令，**想调几次调几次**，还能 `search | jq | grep` 组合 → 治根因 1；
- `write` / `edit` 能写出**全新的记忆**（合成桥接事实），不限于原地改写 → 治根因 2；
- 产物是**文件**，跑完可以被独立校验、独立回答 → 治根因 3（见 §6）；
- agent 只见到 `jq` 吐出的**原始 JSONL 行**，从来没有"展示串"这个概念 → 格式 bug 不可能发生。

代价是放弃了结构化动作空间带来的"每步可解析、可奖励"的规整性。这个代价可以接受：终局产物
是一个文件，仍然有明确的状态和终局奖励，只是中间步变自由了（§8）。

---

## 1. 核心抽象

```
                       ┌──────────────────── mine 的运行目录（一次实验）────────────────────┐
                       │                                                                    │
  data/{dir}/atommem.jsonl ──硬链──► inputs/atommem.jsonl   ← 只读（base 记忆库，base_mem）  │
  data/{dir}/msgmem.jsonl  ──硬链──► inputs/msgmem.jsonl    ← 只读（原始消息，供 source 溯源）│
                                     inputs/index/          ← 只读（预建向量索引，§9）        │
                       │                                                                    │
                       │   qa_{idx}/                    ← workspace == bash 的 cwd           │
                       │       evidence.jsonl           ← 唯一产物：能回答该 question 的记忆列表│
                       └────────────────────────────────────────────────────────────────────┘
                                          ▲                    │
                                          │ 写文件             │ 读文件 / 跑 shell
                                          │                    ▼
                                     Agent（read/write/edit/bash）──「我不再调用工具了」──► Verifier
                                          ▲                                                  │
                                          └────────── 不足以回答：注入 missing[] ◄───────────┘
```

- **每个 QA 一个 workspace**。QA 之间不共享任何可变状态 → **QA 级并发**（§9）。base 库只读。
- **Agent 没有"提交"工具**：写完 `evidence.jsonl` 就是完成。这是刻意的 —— 产物应该是数据，
  不是一次 RPC。
- **Verifier 不属于 agent 的工具箱**，它是 harness 的节点，agent 只能被它检查、不能调它（§6）。

---

## 2. 目录布局与产物

```
data/search_runs/{experiment}_{ts}/{dir}/
    inputs/
        atommem.jsonl      # 只读，硬链到 data/{dir}/atommem.jsonl
        msgmem.jsonl       # 只读，硬链到 data/{dir}/msgmem.jsonl
        index/             # 预建检索索引（ids.json + vectors.f32 + meta.json）
        search             # 可执行的 search CLI 包装（§3.2）
    qa_{idx}/
        evidence.jsonl     # 产物（agent 用 write/edit/bash 生成）
    {dir}.qa_trajectories.jsonl   # 每 QA 一行：轨迹、步数、工具调用序列、reject、tampered…
    {dir}.steps.jsonl             # 每次模型调用的原始输出与解析结果（debug 级排错用）
    summary.json
```

`inputs/` 用**硬链**而不是拷贝/软链：省空间，且写穿软链就会改到源文件，硬链不会。再加一道
保险 —— **每个 QA 跑完校验 `inputs/atommem.jsonl` 的 sha256**，不一致则记
`tampered=true` 并把该 QA 的结果作废（不计入统计）。这是"检测而非防御"：bash 理论上能写
任何地方，但"检测 + 作废"足够便宜，不值得为此做沙箱。

---

## 3. 工具契约

四个工具的执行语义**对齐 `reference/tools.py`**（该文件依赖的 `tau_agent` 未安装，故移植其
实现到 `src/codemem/tools.py`，保留其关键行为）：

| 工具 | 参数 | 关键行为（沿用 reference 语义） |
|---|---|---|
| `read` | `path, offset?, limit?` | 文本按 UTF-8 读；输出截断到 **2000 行 / 50KB**，附续读提示 |
| `write` | `path, content` | 覆盖写，自动建父目录；**落在 workspace 之外则拒绝**（§7 新增约束） |
| `edit` | `path, edits[{oldText,newText}]` | 每个 `oldText` 必须**唯一且不重叠**，全部校验通过才写；返回 diff |
| `bash` | `command, description, timeout?` | stdout+stderr 合并，**tail** 截断 2000 行 / 50KB，超时杀进程组 |

`bash` 的沙箱 = `cwd` 指向 `qa_{idx}/`，并通过 `shell_command_prefix` 注入环境
（这条正是 reference 实现已有的能力）：

```bash
export PATH=<run>/{dir}/inputs:$PATH     # 让 `search` 成为一条命令
export PYTHONPATH=<repo>/src
export CODEMEM_INDEX=<run>/{dir}/inputs/index
export CODEMEM_EMBED_URL=http://127.0.0.1:30001/v1
```

### 3.2 `search` —— 检索做成终端工具

`python -m codemem.searchctl`（`inputs/search` 是它的 shebang 包装），**唯一检索入口**，
复用现有 `searchmem.py`（dense + BM25 + tag 三路 → RRF 融合）：

```bash
search "when did Caroline go to the LGBTQ support group" -k 8
search "painting" -k 5 --source evidence        # 只搜本 QA 已写出的证据
```

输出**每行一条 JSON**，直接喂给 `jq`：

```json
{"id":"session_5_2_1","score":0.0328,"rank_dense":1,"rank_bm25":4,"rank_tag":null,"memory":"Caroline joined an LGBTQ support group in June 2023"}
```

**为什么做成 CLI 而不是一个"检索工具"**：让它能和 `jq` / `grep` / `head` 自由组合，而这正是
这个 agent 唯一需要学的思维模型 —— **一切都是一条对 JSONL 文件的命令**。代价有两条，都可接受：
① 每次调用要 load 一次预建索引（float32 二进制，1273×4096 ≈ 20MB，约 50ms，见 §9）；
② 检索以子进程失败的形式暴露（stderr + 非零退出码）—— 反而**比旧版更好**：旧版 embedding 挂了
就直接终止整个目录，现在 agent 看得见 `search: embedding service unreachable`，可以重试，或者
改用 `jq`/`grep` 走词法路径绕过去。

检索范围默认 = **base 库 ∪ 当前 `evidence.jsonl`**（按 id 去重，evidence 侧优先）。所以 agent
写出的新记忆立刻可被自己检索到，不需要重建索引 —— evidence 通常只有几条，实时嵌入即可。

### 3.3 `jq` —— 这就是"编辑记忆库"的方式

`jq` 不是我们实现的工具，是 `bash` 的核心用法，写进 prompt 的 few-shot。典型动作：

```bash
# 1) 看看这条 question 相关的高分记忆
search "Caroline LGBTQ support group" -k 8 | jq -c 'select(.rank_dense) | {id,score,memory}'

# 2) 把命中的原子原样取进证据（槽位组合，最常用的一步）
jq -c 'select(.metadata.id=="session_5_2_1" or .metadata.id=="session_5_2_3")' \
   inputs/atommem.jsonl >> evidence.jsonl

# 3) 顺着 source 回到原始消息核对（防止改写失真）
jq -c 'select(.metadata.id=="session_5_2")' inputs/msgmem.jsonl

# 4) 自检产物：逐行校验必填字段，列出不合规的行号
jq -c 'select((.memory|type)!="string" or (.metadata.source|length)==0)' evidence.jsonl

# 5) 去重
jq -sc 'unique_by(.metadata.id)[]' evidence.jsonl > t && mv t evidence.jsonl
```

**关键设计点：`evidence.jsonl` 的每一行就是 `inputs/atommem.jsonl` 的一行**（同一个
`base_mem` schema）。所以"把某条记忆纳入证据"退化成 `jq` 的槽位组合 —— 这是整个方案里最省
token、最不容易出错的一步，也把**改写失真**（旧版把展示串写进正文那类）从根上排除了。
只有需要**合成新的桥接记忆**时，才动用 `write`（追加）或 `edit`。

---

## 4. `tool.json`：统一工具配置与上下文注入

单一事实源。既用于渲染注入 prompt，也保留 JSON Schema 以便将来切换为模型原生 `tools=` 参数。

```json
{
  "protocol": {
    "response_format": "one JSON object per turn, nothing else",
    "call_schema":   {"tool": "<name>", "args": { "...": "..." }},
    "no_call":       "reply with plain text (no JSON) to signal you are done",
    "example":       {"tool": "bash", "args": {"command": "search \"painting\" -k 5 | jq -r .id", "description": "Searching library"}}
  },
  "tools": {
    "read": {
      "description": "Read the contents of a file. Output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files.",
      "guidelines": ["Use read to examine files instead of cat or sed."],
      "input_schema": {
        "type": "object",
        "properties": {
          "path":   {"type": "string",  "description": "Path to the file to read"},
          "offset": {"type": "integer", "description": "Line number to start reading from"},
          "limit":  {"type": "integer", "description": "Maximum number of lines to read"}
        },
        "required": ["path"]
      }
    },
    "write": { "...": "同 reference/tools.py 的 write 定义" },
    "edit":  { "...": "同上（edits[{oldText,newText}]）" },
    "bash":  { "...": "同上（command/description/timeout）" }
  },
  "commands": {
    "jq": {
      "description": "JSON processor. The ONLY way you should select, filter, or inspect memory records.",
      "recipes": [
        "search \"Q\" -k 8 | jq -c '{id,score,memory}'",
        "jq -c 'select(.metadata.id==\"session_5_2_1\")' inputs/atommem.jsonl >> evidence.jsonl",
        "jq -c 'select(.metadata.source|length==0)' evidence.jsonl"
      ]
    },
    "search": {
      "description": "Hybrid retrieval (dense + BM25 + tag, RRF) over the base library plus your current evidence.jsonl. Prints one JSON object per hit.",
      "usage": "search \"QUERY\" [-k N] [--source base|evidence|all]",
      "output_fields": ["id", "score", "rank_dense", "rank_bm25", "rank_tag", "memory"]
    }
  },
  "system_prompt_template": "...（见下）..."
}
```

注入方式（`src/codemem/prompt.py` 渲染，宿主模型无原生 tool API，走文本协议）：

```
<tools>
### read
Read the contents of a file. Output is truncated to 2000 lines or 50KB …
  - Use read to examine files instead of cat or sed.
<input_schema>
### write
…
</tools>

<command_line_tools>
### jq — JSON processor. The ONLY way you should select, filter, or inspect memory records.
    search "Q" -k 8 | jq -c '{id,score,memory}'
    jq -c 'select(.metadata.id=="session_5_2_1")' inputs/atommem.jsonl >> evidence.jsonl
### search — Hybrid retrieval over the base library plus your current evidence.jsonl. …
</command_line_tools>

<response_protocol>
Each turn, output ONE JSON object: {"tool": "<name>", "args": {…}}
When evidence.jsonl is complete, reply in plain text with no JSON. That means "I am done".
</response_protocol>
```

**为什么走文本协议而不是模型原生 `tool_calls`**：生成模型是本地 `Qwen3-8B`，原生 tool-calling
的格式遵循度不稳定，而 v2 已经积累了一整套稳健的文本解析经验（围栏剥离、括号修复、裸 JSON、
`{"atoms": [...]}` 包装；见 `evoactions.py`）——**所有解析失败路径都退化为"无害的一步"**，
这条路更可靠。`tool.json` 同时保留 JSON Schema，所以将来换成原生 `tools=` 只改一个适配层。

---

## 5. Agent loop

```python
for qa in questions:                            # 每条 QA 一个 workspace，QA 间并发
    ws = prepare(qa)                            # 硬链 inputs/、建空 evidence.jsonl、注入 env
    msgs = [system(tool_json, workspace, question), user("Begin.")]
    for step in 1..max_steps:                   # 默认 30
        reply = model(msgs)                     # Qwen3-8B，temperature 0.2
        call  = parse_tool_call(reply)          # 稳健解析；解析不出则视为「无工具调用」

        if call is None:                        # agent 说它做完了
            verdict = verify(question, ws.evidence)          # §6
            if verdict.sufficient or rounds_used >= max_verify_rounds:
                break
            rounds_used += 1
            msgs += [assistant(reply), user(gaps(verdict.missing))]
            continue

        result = execute(call, ws)              # 见下：路径约束 / 无进展检测 / 失败处理
        msgs += [assistant(reply), user(render(result))]

    record_trajectory(qa, ws, msgs)              # + tampered 校验（§2）
```

**执行期的三条规则**：

1. **无进展检测**：连续 `no_progress_patience`（默认 6）步既没有 `write`/`edit`、也没有
   `evidence.jsonl` 的行数变化 → 提前结束这一条 QA，直接进 verifier。旧版的空转正是这样被
   省掉的，但这次是**机制**而不是 prompt 请求。
2. **失败不中断**：工具报错（`jq` 语法错、`edit` 的 `oldText` 不唯一、`search` 连不上）都是
   一条 `user` 消息返回给模型，模型自己修。这和 `write`/`edit` 的"全部校验通过才落盘"是同一个
   原则 —— **宁可拒绝，不可静默做错**。
3. **路径约束**：`write`/`edit` 的目标路径解析后必须落在 `qa_{idx}/` 内，否则拒绝并返回错误。
   只读输入不会被一个手滑的 `write` 覆盖（bash 层面靠 §2 的 sha256 校验兜底）。

---

## 6. Verifier：agent 不能同时是编辑者和裁判

一个既能编辑又能宣布成功的 agent，最省力的动作永远是**宣布成功**——旧版 NOOP 泛滥就是这一
偏差温和的版本。所以判定必须由一个**独立角色**来做：

- **输入只有 `(question, evidence.jsonl 的解析结果)`**，看不到 agent 的对话、检索过程、
  编辑历史。它无法被措辞说服，只能被内容说服。
- 被要求：**仅用这些记忆回答问题**；若不足以回答，列出**缺什么**（`missing[]`）。
  `missing[]` 是下一轮最有效的观测 —— 这条反馈回路就是"探索"的实际含义。
- **不给它参考答案**。用 reference 当闸门就是 oracle 泄漏：RL 会直接学会猜答案，推理时也拿
  不到答案键。所以闸门检验的是**可回答性**，不是正确性；正确性只在评测/奖励时用 judge 算。

`verifies_at_most = 2` 轮：第一次不足 → 注入 `missing[]` 让 agent 补 → 第二次不管结果都收尾。
所以每条 QA 的模型调用 = `steps + verifies_at_most`。

**最终交付**：`verdict.answer`（只由 evidence 生成的答案）+ `evidence.jsonl` 本身。前者用于
评测正确率，后者（作为记忆列表）用于评测 evidence recall 与 RL 终局状态。

---

## 7. 不变量（CPU 侧强制，违反即拒绝该动作、不中断 QA）

| # | 不变量 | 动机 |
|---|---|---|
| 1 | `evidence.jsonl` 每行必须是合法 `base_mem`：`memory` 非空字符串、`metadata.id` 非空且文件内唯一、`type ∈ {inner,outer,raw}`、`tag` 非空且含 `speaker:`、**`source` 非空** | `source` 非空是**反幻觉护栏**：从 base 拷来的行天然满足，只有合成的桥接记忆受约束 —— 恰好是我们最想约束的地方 |
| 2 | `write`/`edit` 路径必须落在 `qa_{idx}/` 内 | 只读输入不被误写 |
| 3 | 每 QA 收尾校验 `inputs/atommem.jsonl` sha256；不一致 → `tampered=true`，该 QA 作废 | bash 可以绕开 #2，检测成本极低 |
| 4 | 收尾只补全**系统自有字段**（缺 `changelog` 则补 `[{time, content:"created"}]`；`id` 形如 `evidence_{anchor}_{n}` 则保留），其余不合规的行 **reject 并在轨迹里记账** | 不让补全掩盖格式错误 |
| 5 | 工具/解析/模型调用失败 = 一次 failed step，记账后继续 | 沿用 `evoactions.py` 的原则：所有失败路径都退化为无害一步 |

---

## 8. 与旧版的关系：删掉什么

**删掉**（它们的价值已被"文件 + 通用工具"吸收）：

| 旧版机制 | 为什么不再需要 |
|---|---|
| `ADD/UPDATE/DELETE/NOOP` 动作空间 + `evoactions.py` 的全部校验 | 被 `write`/`edit`/`jq` 取代；`jq` 槽位组合本身就保证了"纳入的证据与库里的一致" |
| v2 的 `max_evomem_turn`、`qa_candidates`(K)、候选集"每条都必须访问一遍" | 循环不再预设候选，agent 自己决定检索几次、看什么 |
| `EVOMEM_PROMPT` 的 `{{CURRENT}}/{{RECALLED}}/{{HISTORY}}` 三段式 | 被工具协议 + 对话历史取代（HISTORY 天然就是 messages） |
| `NOOP` 动作 | 终止由"不再调用工具 + verifier 判定"决定 |
| 目录内串行 | base 只读 + 产物 per-QA → QA 级并发（§9） |
| 展示串渲染（`format_current` / `build_recalled_block`） | agent 只看原始 JSONL，展示串不存在了 —— 顺带消灭了旧版的正文污染 bug |

**保留复用**：`searchmem.py`（唯一检索实现）、`Embedder`（分批 + 单批重试）、
`memory_template*.json`（字段定义，`evidence.jsonl` 的 schema）、`evolog.py`（日志）、
`judge.py`/`metrics.py`（评测）、`eval_utils.load_samples`、`atommem.chat_completion`（带
`finish_reason=length` 重试）、reference/tools.py 的工具语义与 `shell_command_prefix`。

**新增**：`tool.json`（§4）、`prompt.py`（注入渲染）、`tools.py`（工具移植 + 路径约束）、
`agent.py`（§5 的 loop）、`evidence.py`（§7 的校验/规范化）、`verifier.py`（§6）、
`searchctl.py`（§3.2 CLI）、`prompts.py` 加 `EVOLVER_PROMPT` / `VERIFIER_PROMPT`；
`search/runner.py` 负责编排（建 workspace → 逐 QA 并发跑 agent）。

---

## 9. 并发与成本

**QA 级并发是这次最大的结构收益**。旧版必须目录内串行，只因为所有 QA 共享一份可变的记忆库；
现在 base 只读、产物 per-QA，QA 之间零共享 → 可以按 `concurrency` 直接并发（唯一共享是
只读索引与 embedding 服务，前者只读、后者已有分批与退避）。

**成本**：每条 QA ≈ `max_steps(≤30) + verifies(≤2)` 次 `Qwen3-8B` 调用，prompt 约 3–8k
token（system + tool.json 注入是固定的，可命中 prefix cache）。单目录 199 条 QA 最坏约 6000 次
本地 8B 调用 —— 比旧版同量级但**没有一次是无效重复**。先用 `--max-qa 20` 试跑。

**索引只建一次**（关键前提）：旧版 `searchmem.search` 每轮都嵌入 `[query, *全库]`，实测一个
目录跑出 21342 条文本 / 698 批。现在改为**每个运行目录预建一次**：

```
inputs/index/ids.json       # ["session_1_1_1", ...]
inputs/index/vectors.f32    # float32[N, D]，行主序（stdlib array，无 numpy 依赖）
inputs/index/meta.json      # {"model":…, "dim":4096, "n":1273, "corpus_sha256":…}
```

每次 `search` = load 20MB 二进制（≈50ms）+ 嵌入 1 条 query + 实时嵌入 `evidence.jsonl`
的少量条目 + 排序。**这是多轮检索在成本上可行的前提。**

---

## 10. 评测

同一套检索与 judge 下对照三条件：`base atommem`（不演化）/ `旧版 QA 驱动` / 本方案。

| 指标 | 含义 |
|---|---|
| `Judge(verdict.answer) vs reference` | 最终正确率（verifier 只依据 evidence 生成的答案） |
| `evidence_recall(evidence.jsonl)` | 回答集覆盖 evidence 指向原子的比例 —— 检索侧硬指标 |
| `|evidence.jsonl|` / `steps` | 简洁性与成本，防止"把半个库塞进去"刷 recall |
| `verifier_sufficient` vs `judge_correct` | 背离 = 闸门失准（"自认为够了但答错"），下一个要修的信号 |
| `reject_count` / `tampered` | 格式与只读约束的健康度 |

---

## 11. 两个决策点

1. **`search` 做成 CLI（本方案）而非一个"检索工具"**：换来与 `jq`/`grep` 的自由组合，
   代价是每次调用 load 索引 + 以子进程失败形式暴露错误。后者在本设计里反而更好（见 §3.2）。
2. **文本工具协议（本方案）而非模型原生 `tool_calls`**：对本地 8B 更稳，且 `tool.json`
   里已存 JSON Schema，日后切换只改适配层。
