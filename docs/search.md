# search 步骤 — 搜索 agent 的证据构建（核心设计）

**目标**：对每条 question，产出一份**真正能回答它的记忆列表**，落盘为 `evidence.jsonl`。

**载体**：一个**搜索 agent**。它工作在**一个目录**里，只会四个通用工具
`read` / `write` / `edit` / `bash`。检索就是 **`grep`** 一条命令 —— 没有任何检索 CLI、
没有向量索引、没有 embedding 服务。时间算术用 bash 的 **`date -d`**。工具描述与 schema
由**统一的 `tool.json`** 管理，渲染进上下文（§5）。

**检索为什么用 grep 而不是向量检索**：语料是 JSONL，一行一条记录，所以 `grep` 直接可用、
行号稳定（`grep -n` 定位 → `read offset/limit` 精读）。而向量检索的那套（索引、嵌入服务、
相似度排序）换来的是"换个说法也能搜到"，代价却是：一个必须常驻的外部服务、一次几十秒的
索引构建、以及一个**无法解释的排序**（它为什么排 101 名，你查不出来）。多跳问题靠的是
**词面匹配**（第一跳找到的实体名，正是第二跳的检索词），那正是 grep 的强项。

**两种策略，按问题类型选**（写进 system prompt，见 §6.1）：

| | 什么时候用 | 怎么做 |
|---|---|---|
| **A. 关键词多跳** | 事实型：一个名字、一个数字、一件事 | grep 定位会话 → 读它 → 拿新词再 grep |
| **B. 遍历** | 集合型/散布型："她参加过哪些活动"、"他读过哪些书" | grep 出所有相关会话，**逐个走完** |

实测 LoCoMo 的标注证据里，**1187/1985 条 QA 的证据只用到一个 session**，但有 **330 条
跨 2 个以上、最多跨 15 个**；集合型问题（"what activities / books / places..."）有 57 条。
两类问题的正确打法不同，一个 prompt 说不清 —— 所以显式给判据，让模型先选再动手。

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

- 检索是 shell 命令，**想调几次调几次**，还能 `grep | head | read` 组合 → 治根因 1；
- `write` / `edit` 能写出**全新的记忆**（合成桥接事实），不限于原地改写 → 治根因 2；
- 产物是**文件**，跑完可以被独立校验、独立回答 → 治根因 3（见 §7）；
- agent 只见到文件里的**原始 JSONL 行**（grep/read 直接读原文），从来没有"展示串"这个概念 → 格式 bug 不可能发生。

代价是放弃了结构化动作空间带来的"每步可解析、可奖励"的规整性。这个代价可以接受：终局产物
是一个文件，仍然有明确的状态和终局奖励，只是中间步变自由了（§10）。

---

## 1. 核心抽象

```
                       ┌──────────────────── 一次运行的目录 ────────────────────┐
                       │                                                        │
  data/{dir}/sessions.jsonl            ──硬链──► inputs/sessions.jsonl            │
  data/{dir}/session_summaries.jsonl   ──硬链──► inputs/session_summaries.jsonl   │
                       │   （没有索引、没有 CLI —— agent 用 grep 检索）             │
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
export LC_ALL=C                          # date 的输出必须是英文月份/星期名
export PYTHONPATH=<repo>/src
export CODEMEM_INPUTS=<run>/{dir}/inputs # 语料目录的绝对路径
```

命令统一由 `bash -c` 执行（`shutil.which("bash")` 解析绝对路径），不走
`create_subprocess_shell(executable=...)`：前缀是 POSIX 语法，且 Windows 会把 executable
拼进 `cmd.exe /c` 的命令行，把 `C:\Program Files\...\bash.EXE` 里的空格拆坏。

### 4.2 `grep` —— 检索就是它

语料是两个 JSONL 文件，一行一条记录，所以 `grep` 直接可用、行号稳定：

语料是两个文件，各按 line 切：

| 文件 | 内容 | 怎么用 |
|---|---|---|
| `inputs/session_summaries.jsonl` | 每个 session 一行概览 | **第一步总是 grep 它** —— 定位到哪次会话 |
| `inputs/sessions.jsonl` | 每条消息一行，带 `msg_id` 与 `time` | 精确措辞、时间锚点；`grep -n` 拿到行号后 `read` |

```bash
grep -in "husband" inputs/session_summaries.jsonl     # ① 哪次会话相关
grep -in "husband" inputs/sessions.jsonl | head       # ② 原话在这里
grep -Ein "pottery|camping|painting" inputs/sessions.jsonl | head   # ③ 几个词一起
read inputs/sessions.jsonl offset=120 limit=40        # ④ 精读一次会话
```

每条记录自带 `time`（会话时间）与 `msg_id`，所以 agent 从任何一行都能拿到时间锚点，
不必非去找会话首条。

**grep 的退出码 1 是个坑，观测层替它翻译。** `grep` 没匹配时打印空、退出码 1 —— 工具层会附
一句 `Command exited with code 1`，于是观测**非空**，只看"输出为空"判不出来。模型看到
"exited with code 1"的第一反应往往是环境坏了、命令写错了，然后重试同一个命令（再被重复检测
拒掉，白烧两步）。所以 `agent.grep_no_match_notice()` 专门讲清"退出码 1 = 没匹配，不是崩溃"，
并给出下一步（换文件 / 换词 / `grep -c .` 看看里面到底有什么）。

**时间算术用 `date -d`，并且 locale 被钉死。** harness 注入 `LC_ALL=C`：`date` 的月份名/
星期名跟随系统 locale，中文环境下 `date -d "8 May 2023 -1 day" +"%d %b %Y"` 会吐
`07 5月 2023` —— 那是一条下游读不了的记录。实测确认固定后输出 `07 May 2023`。

```bash
date -d "8 May 2023 -1 day" +"%d %b %Y"          # 07 May 2023
date -d "15 Jul 2023 -2 day" +"%A %d %b %Y"      # Thursday 13 Jul 2023（源说 "the Friday before"）
echo $(( ( $(date -d "2023-06-17" +%s) - $(date -d "2023-05-08" +%s) ) / 86400 )) days
```

**为什么不做成一个"检索工具"**：`grep` 本来就能和 `sed` / `head` / `sort` / `uniq` 自由组合，
而这正是这个 agent 唯一需要学的思维模型 —— **一切都是一条对 JSONL 文件的命令**。
我们不需要实现任何检索代码，也就不需要维护它、测试它、给它建索引。

### 4.3 evidence 的行形状与语料**不同**

这是必须说清的一点，也是"拷贝 vs 推导"那条规则的技术基础：

| | 形状 |
|---|---|
| 输入语料 | `{"msg_id", "role", "time", "content"}`（没有 `metadata`） |
| `evidence.jsonl` | `{"content", "metadata": {"source": [...], "score": ...}}` |

**两者不能直接拷贝** —— 从输入搬一行过来会因为缺少 `source` 而被拒。这条是刻意的：
搬过来的是"原始片段"（带悬空代词、相对时间、没有说话人以外的上下文），而下游要的是
"可直接使用的陈述"。校验器把这个约束变成了**机制**，而不是 prompt 里的一句请求。

（agent 当然可以用 `jq` 做辅助 —— 它就在 bash 里。但工具描述里主推的是 `grep` + `read`：
读记录不需要构造 selector，`grep -n` 拿行号再 `read` 更短、更不容易写错。）

---

## 5. `tool.json`：统一工具配置与上下文注入

单一事实源。既用于渲染注入 prompt，也保留 JSON Schema 以便将来切换为模型原生 `tools=` 参数。
加载与渲染在 `src/codemem/search/toolcfg.py`。

**上下文预算**：这段文本**每条 QA 都完整注入一次**，所以它的每一句都在跟检索结果抢上下文。
实测 **12977 字符**，分配如下：

| 段落 | 字符 | 说明 |
|---|---|---|
| system body | 5039 | 两条策略的判据 + 三条 worked example（多跳 / 相对时间 / 遍历） |
| TOOLS | 2006 | 四个工具的 description + schema |
| COMMANDS | 1485 | grep 与 date 的用法与示例 |
| WRITING_POLICY | 1410 | 证据是推导、不是抄的 |
| TIME_POLICY | 1307 | **时间精度**规则（见 §6.2） |
| SCHEMA | 611 | evidence 记录字段 |
| WORKSPACE | 557 | 目录布局与只读规则 |
| PROTOCOL | 381 | 响应格式 |

这个数字比删掉向量检索之前**反而大了一点** —— 因为多了一整套时间精度规则和第二条策略。
换来的是：不再需要维护索引与嵌入服务，也不再需要为"排序为什么是这样"做解释。

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
2. **失败不中断**：工具报错（`grep` 语法错、`edit` 的 `oldText` 不唯一）都是
   一条 `user` 消息返回给模型，模型自己修。这和 `write`/`edit` 的"全部校验通过才落盘"是同一个
   原则 —— **宁可拒绝，不可静默做错**。
3. **路径约束**：`write`/`edit` 的目标路径解析后必须落在 `qa_{idx}/` 内，否则拒绝并返回错误。
   只读输入不会被一个手滑的 `write` 覆盖（bash 层面靠 §3 的 sha256 校验兜底）。

### 6.1 两种策略（写进 system prompt）

`tool.json` 的 system prompt 里有一段 `# PICK YOUR STRATEGY FIRST`，让模型**先选策略再动手**：

| | 判据（问题长什么样） | 动作 |
|---|---|---|
| **A. KEYWORD HOP** | 一个名字/数字/标题/具体事件；"when did X"、"where did X" | grep 定位会话 → 读它 → **拿新词再 grep** |
| **B. SWEEP** | 答案是**一个集合**或散落在多处；"what activities does X do"、"what books has X read"、"how much has X done with Y" | grep 出所有相关会话，**逐个走完**，边读边收成员 |

**为什么必须显式给这两条**：实测 LoCoMo 的标注证据里 **1187/1985 条 QA 只用到 1 个 session**
（策略 A 的主场），但有 **330 条跨 2 个以上、最多跨 15 个**，另有 57 条是集合型问题。
两类问题的最优动作相反 —— A 要快、要跳；B 要慢、要覆盖。只写一句"检索要彻底"会让模型在
B 上过早收手（收不全成员），在 A 上过早发散。

**不做的选择：加一个显式的 plan 阶段**（先跑一次模型调用输出 `{strategy, keywords}`）。
那会让每条 QA 多一次调用，而 8B 有可能把策略也选错 —— 先把判据写进 prompt，
用实测判断值不值得显式化（记在 §12）。

prompt 里对 B 有一个额外的机制性保护：`evidence_soft_limit`（默认 12）会在写太多条时提醒
精简，而 writing policy 明确说**集合型问题应该写进一条记录**（"Melanie's activities include
pottery, camping, painting and swimming"），而不是每个成员一条。

### 6.2 时间精度：不得编造源语料没有的精度

这是这一版针对 `when` 类问题的核心规则。实测 LoCoMo 全部 **262 条 when 问题**的参考答案
形态：

| 答案形态 | 条数 | 例 |
|---|---|---|
| 相对锚点（`... before X`） | 94 | "The week before 9 June 2023"、"two weekends before 17 July 2023" |
| 绝对点/区间 | 143 | "7 May 2023"、"June 2023"、"2022"、"13 August" |
| 周/周末/月/年区间 | ~65 | "September 2023"、"the week of 23 August 2023" |
| `N ago` / `since Y` | 数十 | "A few years ago"、"Since 2016" |

而**源语料的表述往往比参考答案还粗**。看几个真实例子：

| 源消息（带会话时间） | 参考答案 |
|---|---|
| "I painted that lake sunrise **last year**!"（8 May 2023） | **2022** |
| "a friend made it for my 18th birthday **ten years ago**"（27 Jun 2023） | **10 years ago** |
| "I ran a charity race **last Saturday**"（25 May 2023） | "The Saturday before 25 May 2023" |
| "my daughter's birthday **last night**"（14 Aug 2023） | **13 August** |
| "we went camping **two weekends ago**"（17 Jul 2023） | "two weekends before 17 July 2023" |

所以要写进 prompt 的规则是**四件不同的事**：

1. **先看源用了什么单位，就按那个单位答**，不要细化。源说 "last year" → 答案是年份，
   不是编出来的日期。说 "last Saturday" → 是星期几，不是时刻。
2. **相对表述本身就是正确答案，别抹平它**。"the week before 9 June 2023"、"the Friday
   before 15 July 2023" 是精确且无歧义的 —— 把它压成一个猜出来的日期是**信息损失**。
3. **只在算术精确且锚点已知时才算出绝对日期**（"yesterday" + 该消息自己的时间），
   并且**用 `date -d` 算，不准心算**（见 §4.2 的 locale 说明）。
4. **区间是合法答案**（"June 2023"、"the week of 23 August 2023"）。

还有一条**与检索强相关**的推论：evidence 记录必须**自带锚点**。下游读者看不到会话，
所以一条写着 "Caroline went to the support group the week before" 的记录是没用的 ——
锚点日期必须写进正文。prompt 还要求在记录里把源的原话照录一份
（`... what she called "last week", i.e. the week before 9 June 2023`），这样判定和复核都能
两边对照。

**verifier 也要同步。** 它在"够不够"这一关很容易把"锚点表述"判成不完整（"这不是一个日期"），
于是逼着 agent 去编一个更精确的 —— 正好是我们要避免的。所以它的 prompt 里写了：
*Match the precision of the evidence -- do NOT demand more*，并明确"the weekend before
4 September 2023" 与 "2023-09-02"指向同一段时间，不算不匹配。

### 6.3 上下文压缩

prompt 会随历史无限涨（实测 12 步从 3.3k 涨到 4.5k token）。超预算时把**中段**坍缩成一段摘要，
保留 system + 最近 N 条。摘要**不是模型生成的**（那要额外花钱、还可能编造），而是从被丢掉的
观测里机械提取：调用过的工具、碰过的 id 集合。**保留 id 是关键** —— 否则模型会忘了自己已经
收过哪些记忆，转而重复追加。

### 6.4 终止语义

**解析失败 ≠ 完成。** 只有模型**明确回复纯文本**才算"我做完了"。以 `{` 开头却解析失败
（通常是被 `max_tokens` 截断）会**注入纠正消息重试**（`max_parse_retries`），空回复同样重试。
混在一起会让 verifier 对着一份半成品去判。

### 6.5 观测里的机制性插话

三类情况由 harness 主动在观测里加一句，而不是指望 prompt 的软约束（实测 8B 会无视）：

- **变更型调用之后附证据摘要**（行数 / 唯一 id 数 / 末尾几行）。`cmd >> file` 的 stdout
  是空的，模型看不见自己刚写了什么，于是盲目重试 —— 实测 93% 的观测是空的。
- **烧了 N 步还没动笔** → 明确说"把你已经查到的写下来"。实测 qa15 12 步里读了 12 次、
  evidence 始终为空。
- **`grep` 退出码 1（= 无匹配）** → 专门翻译成"没搜到，不是崩溃"，并给出下一步。
  见 §4.2 的说明 —— 这是这一版新增的，因为 grep 是主检索手段。

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
  唯一的例外是**时间精度的两条说明**，因为那里的误判会**反向伤害** agent：
  ① 它很容易把"锚点表述"判成不完整（"这不是一个日期"），于是逼 agent 去编一个更精确的 ——
  正是 §6.2 要避免的；② 同一个时间点写成不同样子（"the weekend before 4 Sep 2023" vs
  "2023-09-02"）不该算不匹配。所以规则是 *Match the precision of the evidence -- do NOT
  demand more*，外加一条等价性说明。

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
| 7 | 产物解析容忍两种 `jq` 自然输出：一行一个数组、整份美化 JSON | agent 可能用 `jq` 组装证据，它的自然输出必须能被接住（否则规范化会把证据**全删**） |

---

## 9. 与旧版的关系：删掉什么

**删掉**（它们的价值已被"文件 + 通用工具"吸收）：

| 旧版机制 | 为什么不再需要 |
|---|---|
| `ADD/UPDATE/DELETE/NOOP` 动作空间 + 逐动作校验 | 被 `write`/`edit`/`bash` 取代 |
| 候选集 K 与"每条都必须访问一遍" | 循环不再预设候选，agent 自己决定检索几次、看什么 |
| `{{CURRENT}}/{{RECALLED}}/{{HISTORY}}` 三段式 prompt | 被工具协议 + 对话历史取代（HISTORY 天然就是 messages） |
| `NOOP` 动作 | 终止由"不再调用工具 + verifier 判定"决定 |
| 目录内串行 | 语料只读 + 产物 per-QA → QA 级并发 |
| 展示串渲染（`format_current` / `build_recalled_block`） | agent 只看原始 JSONL，展示串不存在了 —— 顺带消灭了正文污染 bug |
| 顶层 speakers summary | 见 §2：合成文本，无 msg_id 可回溯，破掉溯源不变量 |
| 原子记忆（`atommem`）/ 原子抽取模型 / 为其造的 dpo 数据 | 原子把会话结构打碎，且抽取阶段无词表约束 |

**保留复用**：`log.py`（日志）、`judge.py` / `metrics.py`（评测）、`reference/tools.py`
的工具语义与 `shell_command_prefix`。

---

## 10. 并发与成本

**QA 级并发是这次最大的结构收益**。旧版必须目录内串行，只因为所有 QA 共享一份可变的记忆库；
现在语料只读、产物 per-QA，QA 之间零共享 → 可以按 `concurrency` 直接并发（唯一的共享是
只读索引与 embedding 服务，前者只读、后者已有分批与退避）。

**成本**：每条 QA ≈ `max_steps(≤30) + verifies(≤2)` 次 `Qwen3-8B` 调用，system 段约
**12977 字符（约 3.2k token）且固定**，可命中 prefix cache。单目录 199 条 QA 最坏约 6000 次
本地 8B 调用。先用 `--max-qa 3 --max-steps 8` 试跑。

**没有索引，也就没有索引的成本**。旧版这一步是整条链路最重的一块：向量索引（一个目录
17MB）、共享缓存、按 `corpus_sha256` 判失效、跨运行复用。现在这一切都不存在 ——
语料就是两个 JSONL 文件，硬链进运行目录（inode 共享，0 额外空间），agent 直接 grep。
删掉的代码：`index.py`（344 行）、`embedder.py`（115 行）、`searchmem.py`（434 行）、
`searchctl.py`（531 行）、`timecalc.py`（247 行），合计约 1670 行，以及它们对应的测试。

**代价是诚实的**：语义检索能命中"换了种说法"的表述，grep 不能。这个取舍在 LoCoMo 上是划算的
（多跳靠词面匹配、时间靠算术），但如果将来语料变成用户自由输入的长文本，值得重新评估。

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

## 12. 决策点

1. **检索用 `grep` 而不是向量检索（本方案）**：换来零外部依赖、可解释的排序（命中的就是
   字面包含的那几行）、以及不用维护的 1600 行代码。代价是失去语义召回 —— 见 §10 末尾的说明。
2. **时间推理用 `date -d` 而不是自建工具（本方案）**：`date` 是系统自带的确定性算术，
   我们只需要把 locale 钉死（`LC_ALL=C`）。旧版的 `timecalc` 是为了"让模型不自己算"而
   存在的 —— 这个目的靠 prompt 里明确写"用 `date -d`，不要心算"同样能达到。
3. **按问题类型选策略写进 prompt，而不是加一个 plan 阶段**：多一次模型调用换更稳的策略，
   对 8B 不一定划算（它可能把策略也选错）。先把判据写清楚，看实测再决定要不要显式化。
4. **文本工具协议（本方案）而非模型原生 `tool_calls`**：对本地 8B 更稳，且 `tool.json`
   里已存 JSON Schema，日后切换只改适配层。
