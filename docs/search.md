# search 步骤 — 搜索 agent 的证据构建（核心设计）

**目标**：对每条 question，产出一份**真正能回答它的记忆列表**，落盘为 `evidence.jsonl`。

**载体**：一个**搜索 agent**。它工作在**一个目录**里，只会四个通用工具
`read` / `write` / `edit` / `bash`，其中 `bash` 的两个核心用法是 **`jq`**（JSON 工具，见 §4.3）
和 **`search`**（检索 CLI，见 §4.2）。工具描述与 schema 由**统一的 `tool.json`** 管理，
渲染进上下文（§5）。

**这个 agent 的核心动作只有一件事：用关键词检索，从命中里抠出新的关键词，再搜一轮。**
多跳问题就是这么解的 —— 第一跳找到的实体名，正是第二跳的检索词。语义检索（`--embed`）是
**兜底**而不是默认：关键词快、确定、不依赖网络，而且多跳本身靠的就是词面匹配。

---

## 0. 定位：为什么是 code agent

上一版的实测失败（5 条 QA / 164 次调用）：`UPDATE 146 / NOOP 18 / ADD 0 / DELETE 0`，
库净变化 `+0 ~81 -0`，65 errors 多为 `unchanged, skipped`。三条根因：

| # | 根因 | 表现 |
|---|---|---|
| 1 | **检索只做一次**，候选集在第一步定死 | bridging 事实没进候选 → 后面几轮注定空转（多跳问题死在这） |
| 2 | **动作空间只有"单条原地精修"** | 只会 UPDATE，没有能力**创造**把两条记忆连起来的事实 —— 而缺的正是它 |
| 3 | **成功判据不可检验**（"更自足"） | 模型无法知道自己做好了没有 → NOOP 泛滥 + 81 次无效改写 |

还有一个此前没说清的**格式 bug**：模型把展示串
`[session_16_9_4] (2023-09-13T00:09:00) Caroline | outer | ` 当正文抄了回去。
根因是**渲染文本与可写字段是同一个通道**。

**code agent 一次解决这四条**，因为一个真实的工作目录天然具备这四个性质：

- 检索是 shell 命令，**想调几次调几次**，还能 `search | jq | grep` 组合 → 治根因 1；
- `write` / `edit` 能写出**全新的记忆**（合成桥接事实），不限于原地改写 → 治根因 2；
- 产物是**文件**，跑完可以被独立校验、独立回答 → 治根因 3（见 §7）；
- agent 只见到 `jq` 吐出的**原始 JSONL 行**，从来没有"展示串"这个概念 → 格式 bug 不可能发生。

代价是放弃了结构化动作空间带来的"每步可解析、可奖励"的规整性。这个代价可以接受：终局产物
是一个文件，仍然有明确的状态和终局奖励，只是中间步变自由了（§10）。

---

## 1. 核心抽象

```
                       ┌──────────────────── 一次运行的目录 ────────────────────┐
                       │                                                        │
  data/{dir}/sessions.jsonl            ──硬链──► inputs/sessions.jsonl            │
  data/{dir}/session_summaries.jsonl   ──硬链──► inputs/session_summaries.jsonl   │
                                            inputs/index/  语义索引（--embed 才用） │
                       │                                                        │
                       │   qa_{idx}/                    ← workspace == bash 的 cwd │
                       │       evidence.jsonl           ← 唯一产物               │
                       └────────────────────────────────────────────────────────┘
                                          ▲                    │
                                          │ 写文件             │ 读文件 / 跑 shell
                                          │                    ▼
                                     Agent（read/write/edit/bash）──「我不再调用工具了」──► Verifier
                                          ▲                                                  │
                                          └────────── 不足以回答：注入 missing[] ◄───────────┘
```

- **每个 QA 一个 workspace**。QA 之间不共享任何可变状态 → **QA 级并发**（§10）。语料只读。
- **Agent 没有"提交"工具**：写完 `evidence.jsonl` 就是完成。这是刻意的 —— 产物应该是数据，
  不是一次 RPC。
- **Verifier 不属于 agent 的工具箱**，它是 harness 的节点，agent 只能被它检查、不能调它（§7）。

---

## 2. 语料：为什么只有两层

| 文件 | 粒度 | 条数（Caroline_Melanie） | 用途 |
|---|---|---|---|
| `session_summaries.jsonl` | 概览（每 session 一份） | 19 | **粗筛**：定位"哪次会话谈过这件事" |
| `sessions.jsonl` | 原始消息 | 419 | 精确措辞、时间锚点 |

**删掉了顶层的 speakers summary。** 它是把 19 份 summary 再压成 1 条的合成文本，而检索真正
需要的是"定位到哪次会话"—— 那是 session summary 就有的粒度。更要命的是它**没有 `msg_id`
可回溯**，`source` 只能指向别的 summary 而非原始消息，直接破掉了 §8 的溯源不变量。
每多一层，还多一次模型调用、多一次全量重跑，以及每条 QA 都要背的一份没人用的语料。

`session_template.jsonl` 是唯一的记录格式：`{"msg_id", "role", "time", "content"}`，
summary 只是把 `role` 标成 `"summary"`。**agent 只需理解一种格式。**

---

## 3. 目录布局与产物

```
data/search_runs/{experiment}_{ts}/{dir}/
    inputs/
        session_summaries.jsonl   # 只读，硬链
        sessions.jsonl            # 只读，硬链
        index/                    # 语义索引（符号链接到共享缓存；--embed 才用得到）
        search                    # 可执行的 search CLI 包装（§4.2）
        timecalc                  # 可执行的日期算术 CLI 包装
    qa_{idx}/
        evidence.jsonl            # 产物（agent 用 write/edit/bash 生成）
    {dir}.qa_trajectories.jsonl   # 每 QA 一行：轨迹、步数、工具调用序列、reject、tampered…
    {dir}.model_calls.jsonl       # 每次模型调用的原始输出（debug 级排错用）
    summary.json
```

`inputs/` 用**硬链**而不是拷贝：省空间。再加一道保险 —— **每个 QA 跑完校验两层语料的
sha256**，不一致则记 `tampered=true` 并把该 QA 的结果作废（不计入统计）。这是"检测而非防御"：
bash 理论上能写任何地方，但"检测 + 作废"足够便宜，不值得为此做沙箱。

`qa_{idx}/inputs` 是**一个**指向公共 `inputs/` 的目录链接（不是每 QA 建一整套软链）：
那些文件在一个运行目录内对每个 QA 完全相同，逐个建链等于把同一组链接复制 152 遍，
实测浪费 4.2MB（目录 458 个 + 软链 608 个），而全部证据加起来才 45KB。

链接优先用**符号链接** —— 工具层的路径约束会 `resolve()`，解析后落在工作目录之外，所以
写 `inputs/...` 会被直接拒绝。Windows 上普通账户建符号链接需要开发者模式/管理员权限
（`WinError 1314`），`runner.link_dir_or_copy` 会退回**逐文件硬链**：此时 `bash >>` 会写穿到
源文件（实测确认），收尾的 sha256 校验负责抓到它并把该 QA 作废；`write`/`edit` 的原子替换
则只会悄悄换掉自己那份链接，源文件不受影响。

---

## 4. 工具契约

四个工具的执行语义**对齐 `reference/tools.py`**（该文件依赖的 `tau_agent` 未安装，故移植其
实现到 `src/codemem/search/tools.py`，保留其关键行为）：

| 工具 | 参数 | 关键行为（沿用 reference 语义） |
|---|---|---|
| `read` | `path, offset?, limit?` | 文本按 UTF-8 读；输出截断到 **2000 行 / 50KB**，附续读提示 |
| `write` | `path, content` | 覆盖写，自动建父目录；**落在 workspace 之外则拒绝**（§8 新增约束） |
| `edit` | `path, edits[{oldText,newText}]` | 每个 `oldText` 必须**唯一且不重叠**，全部校验通过才写；返回 diff |
| `bash` | `command, description, timeout?` | stdout+stderr 合并，**tail** 截断 2000 行 / 50KB，超时杀进程组 |

`bash` 的沙箱 = `cwd` 指向 `qa_{idx}/`，并通过 `shell_command_prefix` 注入环境
（这条正是 reference 实现已有的能力）：

```bash
export PATH=<run>/{dir}/inputs:$PATH     # 让 `search` / `timecalc` 成为命令
export PYTHONPATH=<repo>/src
export CODEMEM_INDEX=<run>/{dir}/inputs/index
export CODEMEM_EMBED_URL=http://127.0.0.1:30001/v1
```

命令统一由 `bash -c` 执行（`shutil.which("bash")` 解析绝对路径），不走
`create_subprocess_shell(executable=...)`：前缀是 POSIX 语法，且 Windows 会把 executable
拼进 `cmd.exe /c` 的命令行，把 `C:\Program Files\...\bash.EXE` 里的空格拆坏。

### 4.2 `search` —— 检索做成终端工具

`python -m codemem.search.searchctl`（`inputs/search` 是它的 shebang 包装），**唯一检索入口**，
复用 `searchmem.py`（dense + BM25 + tag 三路 → RRF 融合）：

```bash
search "adoption counseling" --source summary -k 5   # ① 粗筛：定位哪次会话
search "pottery class" --source raw -k 8             # ③ 精确措辞只在原话里
search "Melanie husband" -k 5                        # ④ 多跳：拿刚读到的新名字再搜
search "feeling unsupported" --embed -k 8            # ⑤ 兜底：关键词确实搜不到时
```

**默认是关键词模式**（BM25 + tag，不加载索引、不请求嵌入服务），`--embed` 才打开 dense。
这个排序是刻意的，见本文开头的说明。检索池 = **两层语料 ∪ 当前 `evidence.jsonl`**
（按 id 去重，evidence 侧优先）。所以 agent 写出的新记忆立刻可被自己检索到，不需要重建索引 ——
evidence 通常只有几条，实时嵌入即可。

| `--source` | 搜什么 | 什么时候用 |
|---|---|---|
| `summary` | `inputs/session_summaries.jsonl` | **第一步总是它** —— 从概览定位到哪次会话 |
| `raw` / `sessions` | `inputs/sessions.jsonl`（原始消息） | 精确的名字/数字/措辞，summary 改写掉了的那些 |
| `evidence` | 当前 `evidence.jsonl` | 查自己已收的证据有没有重复/缺口 |
| `all`（缺省） | 两层并集 + evidence | 还不知道答案在哪一层时 |

输出**每行一条 JSON**，字段只保留 agent 决策要用的：

```json
{"id":"session_3_14","kind":"session","session":"session_3","time":"7:55 pm on 9 June, 2023",
 "score":0.0328,"memory":"we've been married for five years now"}
```

`kind` / `session` / `time` 是刻意给出来的：**`kind`** 说明这是概览还是原话（决定要不要往下钻），
**`session`** 是立刻读同一次会话其它消息的入口（多跳最常用的一步），**`time`** 是相对时间推理
的锚点。通道明细 `rank_dense` / `rank_bm25` / `rank_tag` 默认不打 —— agent 不用它们做决策，
而一次 `-k 8` 就是 8 行；排错时用 `--all-fields` 拿回来。

**为什么做成 CLI 而不是一个"检索工具"**：让它能和 `jq` / `grep` / `head` 自由组合，而这正是
这个 agent 唯一需要学的思维模型 —— **一切都是一条对 JSONL 文件的命令**。代价有两条，都可接受：
① `--embed` 时每次调用要 load 一次索引（float32 二进制，约 20MB，约 50ms，见 §10）；
② 检索以子进程失败的形式暴露（stderr + 非零退出码）—— 反而**比旧版更好**：旧版 embedding 挂了
就直接终止整个目录，现在 agent 看得见 `search: embedding service unreachable`，改用关键词
（默认行为）绕过去。

### 4.3 `jq` —— 这就是"读记忆库"的方式

`jq` 不是我们实现的工具，是 `bash` 的核心用法，写进 prompt 的 few-shot。典型动作：

```bash
# 1) 读一条已知记录（输入语料的 id 是 msg_id，不是 metadata.id）
jq -c 'select(.msg_id=="session_1_11")' inputs/sessions.jsonl

# 2) 读一整个会话（summary -> 原文的下钻）
jq -c 'select(.msg_id|startswith("session_8_"))' inputs/sessions.jsonl

# 3) 看看自己已经写进去什么（别重复追加）
jq -r '.content' evidence.jsonl

# 4) 自检产物：逐行校验必填字段
jq -c 'select((.content|type)!="string" or (.metadata.source|length)==0)' evidence.jsonl

# 5) 去重
jq -sc 'unique_by(.content)[]' evidence.jsonl > t && mv t evidence.jsonl
```

**关键设计点：`evidence.jsonl` 的行形状与输入语料**不同**。** 输入是
`{msg_id, role, time, content}`（没有 `metadata`）；evidence 是
`{content, metadata: {source, score}}`。**两者不能直接拷贝** —— 从输入直接搬一行过来会因为
没有 `source` 而被拒。这条是刻意的：搬过来的是"原始片段"（带悬空代词、相对时间），
而下游要的是"可直接使用的陈述"。这个约束把"避免改写失真"（旧版把展示串写进正文那类）
变成了**机制**，而不是 prompt 里的一句请求。

---

## 5. `tool.json`：统一工具配置与上下文注入

单一事实源。既用于渲染注入 prompt，也保留 JSON Schema 以便将来切换为模型原生 `tools=` 参数。
加载与渲染在 `src/codemem/search/toolcfg.py`。

**上下文预算**：这段文本**每条 QA 都完整注入一次**，所以它的每一句都在跟检索结果抢上下文。
旧版实测 **20503 字符**，其中大半是同一套规则在 system body、`workspace.rules`、
`writing_policy` 三处各写一遍。现在压到 **11506 字符（-44%）**：

| 段落 | 旧 | 新 | 怎么省的 |
|---|---|---|---|
| system body | 9171 | 4409 | 规则去重；只留两条 worked example（多跳 + 相对时间） |
| TOOLS | 3104 | 1795 | 精简 description；bash 的 guidelines 从 3 条压到 0（工具自己会报错） |
| COMMANDS | 3838 | 2194 | jq 菜谱 9 → 4；删掉指向已不存在语料的示例 |
| WORKSPACE | 920 | 520 | 布局只剩两层；规则 4 → 2 条 |
| WRITING_POLICY | 2283 | 1548 | 合并同义条目，每条只说一遍 |
| SCHEMA | 774 | 611 | 字段说明写短 |
| PROTOCOL | 365 | 381 | 不变 |

`scripts/test_search.py` 里有一条断言把这个上限钉住（`< 13000 字符`）——
**涨回去就等于白做这次简化**，所以它该在 CI 里红，而不是等到某次跑完发现步数又不够用了。

**为什么走文本协议而不是模型原生 `tool_calls`**：生成模型是本地 `Qwen3-8B`，原生 tool-calling
的格式遵循度不稳定，而项目已经积累了一整套稳健的文本解析经验（围栏剥离、括号修复、裸 JSON，
见 `toolcfg.parse_tool_call`）——**所有解析失败路径都退化为"无害的一步"**，这条路更可靠。
`tool.json` 同时保留 JSON Schema，所以将来换成原生 `tools=` 只改一个适配层。

---

## 6. Agent loop

```python
for qa in questions:                            # 每条 QA 一个 workspace，QA 间并发
    ws = prepare(qa)                            # 硬链 inputs/、建空 evidence.jsonl、注入 env
    msgs = [system(tool_json, workspace, question), user("Begin.")]
    for step in 1..max_steps:                   # 默认 30
        reply = model(msgs)                     # Qwen3-8B，temperature 0.2
        call  = parse_tool_call(reply)          # 稳健解析；解析不出则视为「无工具调用」

        if call is None:                        # agent 说它做完了
            verdict = verify(question, ws.evidence)          # §7
            if verdict.sufficient or rounds_used >= max_verify_rounds:
                break
            rounds_used += 1
            msgs += [assistant(reply), user(gaps(verdict.missing))]
            continue

        result = execute(call, ws)              # 见下：路径约束 / 无进展检测 / 失败处理
        msgs += [assistant(reply), user(render(result))]

    record_trajectory(qa, ws, msgs)              # + tampered 校验（§3）
```

**执行期的三条规则**：

1. **无进展检测**：连续 `no_progress_patience`（默认 6）步**唯一记录指纹**没有变化 →
   提前结束这一条 QA，直接进 verifier。判据是**唯一记录集合**而不是行数/字节数 ——
   实测模型会用 `>>` 反复追加同样的行，行数在涨但一个事实都没多。这仍然是一条**机制**，
   而不是 prompt 里的一句请求。
2. **失败不中断**：工具报错（`jq` 语法错、`edit` 的 `oldText` 不唯一、`search` 连不上）都是
   一条 `user` 消息返回给模型，模型自己修。这和 `write`/`edit` 的"全部校验通过才落盘"是同一个
   原则 —— **宁可拒绝，不可静默做错**。
3. **路径约束**：`write`/`edit` 的目标路径解析后必须落在 `qa_{idx}/` 内，否则拒绝并返回错误。
   只读输入不会被一个手滑的 `write` 覆盖（bash 层面靠 §3 的 sha256 校验兜底）。

### 6.2 上下文压缩

prompt 会随历史无限涨（实测 12 步从 3.3k 涨到 4.5k token）。超预算时把**中段**坍缩成一段摘要，
保留 system + 最近 N 条。摘要**不是模型生成的**（那要额外花钱、还可能编造），而是从被丢掉的
观测里机械提取：调用过的工具、碰过的 id 集合。**保留 id 是关键** —— 否则模型会忘了自己已经
收过哪些记忆，转而重复追加。

### 6.3 终止语义

**解析失败 ≠ 完成。** 只有模型**明确回复纯文本**才算"我做完了"。以 `{` 开头却解析失败
（通常是被 `max_tokens` 截断）会**注入纠正消息重试**（`max_parse_retries`），空回复同样重试。
混在一起会让 verifier 对着一份半成品去判。

### 6.4 观测里的机制性插话

三类情况由 harness 主动在观测里加一句，而不是指望 prompt 的软约束（实测 8B 会无视）：

- **变更型调用之后附证据摘要**（行数 / 唯一 id 数 / 末尾几行）。`jq ... >> file` 的 stdout
  是空的，模型看不见自己刚写了什么，于是盲目重试 —— 实测 93% 的观测是空的。
- **烧了 N 步还没动笔** → 明确说"把你已经查到的写下来"。实测 qa15 12 步里读了 12 次、
  evidence 始终为空。
- **命令成功但无输出** → 把"`jq` 选不到东西"翻译出来。这是最隐蔽的坑：`jq 'select(...)'`
  匹配不到时打印空、退出码 0，模型以为环境坏了，再跑一次同样的命令又被重复检测拒掉。

---

## 7. Verifier：agent 不能同时是编辑者和裁判

一个既能编辑又能宣布成功的 agent，最省力的动作永远是**宣布成功**——旧版 NOOP 泛滥就是这一
偏差温和的版本。所以判定必须由一个**独立角色**来做：

- **输入只有 `(question, evidence.jsonl 的解析结果)`**，看不到 agent 的对话、检索过程、
  编辑历史。它无法被措辞说服，只能被内容说服。
- 被要求：**仅用这些记忆回答问题**；若不足以回答，列出**缺什么**（`missing[]`）。
  `missing[]` 是下一轮最有效的观测 —— 它同时就是下一轮的**检索关键词**，这就是"探索"的实际含义。
- **不给它参考答案**。用 reference 当闸门就是 oracle 泄漏：RL 会直接学会猜答案，推理时也拿
  不到答案键。所以闸门检验的是**可回答性**，不是正确性；正确性只在评测/奖励时用 judge 算。
- **prompt 里没有 few-shot 示例**（判定规则本身很短，示例会让每条 QA 多背几百 token）。
  唯一的例外是一条**日期等价性**说明 —— "the weekend before 4 September 2023" 与
  "2023-09-02" 指向同一天，那种误判最容易发生。

`max_verify_rounds = 2`：第一次不足 → 注入 `missing[]` 让 agent 补 → 第二次不管结果都收尾。
所以每条 QA 的模型调用 = `steps + verifies_at_most`。

**空 evidence 直接判不足，且不问模型** —— 最有用的反馈是"你跑了 12 步却什么都没写"这个事实
本身（实测这是最常见的失败：一直在读、从不动笔）。

**最终交付**：`verdict.answer`（只由 evidence 生成的答案）+ `evidence.jsonl` 本身。前者用于
评测正确率，后者（作为记忆列表）用于评测 evidence recall 与 RL 终局状态。

---

## 8. 不变量（CPU 侧强制，违反即拒绝该动作、不中断 QA）

| # | 不变量 | 动机 |
|---|---|---|
| 1 | `evidence.jsonl` 每行必须是 `{content: 非空, metadata.source: 非空数组}`；`score` / `changelog` 系统可补，其余不合规的行 **reject 并在轨迹里记账** | `source` 非空是**反幻觉护栏**：它必须列出真读过的 msg_id，而不是记得或猜的 |
| 2 | `write`/`edit` 路径必须落在 `qa_{idx}/` 内 | 只读输入不被误写 |
| 3 | 每 QA 收尾校验**两层语料**的 sha256；不一致 → `tampered=true`，该 QA 作废 | bash 可以绕开 #2，检测成本极低 |
| 4 | 只补全**系统自有字段**（缺 `changelog` → 补 `created`；缺 `score` → 补 0.5） | 补全语义问题会掩盖失败模式 |
| 5 | 工具/解析/模型调用失败 = 一次 failed step，记账后继续 | 所有失败路径都退化为无害一步 |
| 6 | 连续 `no_progress_patience` 步唯一记录指纹没变 → 提前收尾 | 把空转变成机制而不是 prompt 请求 |
| 7 | 产物解析容忍两种 `jq` 自然输出：一行一个数组、整份美化 JSON | `jq` 是主要工具，它的自然输出必须能被接住（否则规范化会把证据**全删**） |

---

## 9. 与旧版的关系：删掉什么

**删掉**（它们的价值已被"文件 + 通用工具"吸收）：

| 旧版机制 | 为什么不再需要 |
|---|---|
| `ADD/UPDATE/DELETE/NOOP` 动作空间 + 逐动作校验 | 被 `write`/`edit`/`jq` 取代 |
| 候选集 K 与"每条都必须访问一遍" | 循环不再预设候选，agent 自己决定检索几次、看什么 |
| `{{CURRENT}}/{{RECALLED}}/{{HISTORY}}` 三段式 prompt | 被工具协议 + 对话历史取代（HISTORY 天然就是 messages） |
| `NOOP` 动作 | 终止由"不再调用工具 + verifier 判定"决定 |
| 目录内串行 | 语料只读 + 产物 per-QA → QA 级并发 |
| 展示串渲染（`format_current` / `build_recalled_block`） | agent 只看原始 JSONL，展示串不存在了 —— 顺带消灭了正文污染 bug |
| 顶层 speakers summary | 见 §2：合成文本，无 msg_id 可回溯，破掉溯源不变量 |
| 原子记忆（`atommem`）/ 原子抽取模型 / 为其造的 dpo 数据 | 原子把会话结构打碎，且抽取阶段无词表约束 |

**保留复用**：`searchmem.py`（唯一检索实现）、`Embedder`（分批 + 单批重试）、
`log.py`（日志）、`judge.py` / `metrics.py`（评测）、`reference/tools.py` 的工具语义与
`shell_command_prefix`。

---

## 10. 并发与成本

**QA 级并发是这次最大的结构收益**。旧版必须目录内串行，只因为所有 QA 共享一份可变的记忆库；
现在语料只读、产物 per-QA，QA 之间零共享 → 可以按 `concurrency` 直接并发（唯一的共享是
只读索引与 embedding 服务，前者只读、后者已有分批与退避）。

**成本**：每条 QA ≈ `max_steps(≤30) + verifies(≤2)` 次 `Qwen3-8B` 调用，system 段约
**11506 字符（约 3k token）且固定**，可命中 prefix cache。单目录 199 条 QA 最坏约 6000 次
本地 8B 调用。先用 `--max-qa 3 --max-steps 8` 试跑。

**索引只建一次（且跨运行共享）**：旧版 `searchmem.search` 每轮都嵌入 `[query, *全库]`，
实测一个目录跑出 21342 条文本 / 698 批。现在改为每次运行预建一次，落在
`data/search_runs/.index_cache/{corpus}-{model}/`，运行目录里只有符号链接：

```
inputs/index/ids.json       # ["session_1_1", ...]
inputs/index/vectors.f32    # float32[N, D]，行主序
inputs/index/meta.json      # {"model":…, "dim":2560, "n":1650, "corpus_sha256":…}
```

目录名绑定 `corpus_sha256` + 模型指纹：语料一变（重跑 `add`、正文被改、条数变了）→ 指纹变
→ 自动建新缓存。这一步不能省 —— 若只检查"文件存在"就复用，重新生成后的旧向量会**静默**
给出错误排序。索引只服务 `--embed`，`configs/search.yaml` 的 `build_index: false` 可完全跳过。

---

## 11. 评测

同一套检索与 judge 下对照三条件：`只给原始消息`（不演化）/ `旧版原子记忆` / 本方案。

| 指标 | 含义 |
|---|---|
| `Judge(verdict.answer) vs reference` | 最终正确率（verifier 只依据 evidence 生成的答案） |
| `evidence_recall(evidence.jsonl)` | 覆盖标注证据的比例 —— 检索侧硬指标，但**不是可靠的失败信号**（标注只列 1-4 条，同一事实常有其他消息佐证） |
| `|evidence.jsonl|` / `steps` | 简洁性与成本，防止"把半个库塞进去"刷 recall |
| `verifier_sufficient` vs `judge_correct` | 背离 = 闸门失准（"自认为够了但答错"），下一个要修的信号 |
| `reject_count` / `tampered` | 格式与只读约束的健康度 |

---

## 12. 三个决策点

1. **`search` 做成 CLI（本方案）而非一个"检索工具"**：换来与 `jq`/`grep` 的自由组合，
   代价是每次调用 load 索引 + 以子进程失败形式暴露错误。后者在本设计里反而更好（见 §4.2）。
2. **默认关键词、`--embed` 兜底（本方案）而非默认语义**：关键词更快、更确定、不依赖网络，
   而且多跳本身靠词面匹配；语义只在关键词确实搜不到时才有用。副作用是嵌入服务从"关键路径"
   变成了"可选依赖"。
3. **文本工具协议（本方案）而非模型原生 `tool_calls`**：对本地 8B 更稳，且 `tool.json`
   里已存 JSON Schema，日后切换只改适配层。
