# 形态适配层：把平台的报文归一化成三层记忆能吃的消息

这一份写 **`src/codemem/adapters.py`** —— 报文进 `codemem.add` 之前的预处理。它回答三个问题：
平台到底喂了些什么形状、哪些形状会**静默毁掉检索**、以及每种形状怎么处理。

系统设计与契约在 [`core.md`](core.md)；这一份是它的一个子模块的展开（与 [`add.md`](add.md) /
[`search.md`](search.md) 同级）。

---

## 0. 一句话

**平台把"谁说的""什么时候"编码在报文的各个角落，而不是 `role` 字段里。**
适配层把这些信息**无损地搬到该在的字段**（`role` / `time`），并摘掉平台自己的记账标签；
正文一字不改。

---

## 1. 不处理会怎样（两个实测到的故障）

### 故障一：说话人丢失，`grep` 直接落空

`/add` 的报文长这样：

```json
{"role": "user", "timestamp": 1704153600000,
 "content": "[beam_100k][session_1][D1:21][text] Christina Baker: I'm trying to implement ..."}
```

真实说话人是 **Christina Baker**，协议 `role` 只表示"这条是用户侧发上来的"。旧代码把
`role` 原样写进 `msgmem`，于是原子记忆全变成：

```
"user is trying to implement city autocomplete using OpenWeather's Geocoding API v1"
```

而平台问的是 `"What fields would Caroline be likely to pursue in her education?"` ——
**`Caroline` 这个名字一条原子记忆里都没有**，agent grep 它只会命中 `msgmem` 里那句被标签
裹着的原话，`atommem`（首选语料）整个落空。

统计：全量 **7268 条** message 里 **6396 条（88%）** 的协议 `role` 是 `user`，
**624 个 `/add` 里有 320 个**的正文前缀带着真名 —— 也就是说，近一半的语料在旧管线里
"说话人"是丢的。

### 故障二：平台记账标签污染正文

`[beam_100k][session_1][D1:21][text]` 这四个方括号是平台自己的记账（数据集名 / 会话号 /
数据集内 id / 消息类型），它们进了 `memory` 正文之后：

- `grep "session_1"` 命中**全库**（每条消息都带这个前缀）；
- atommem 的 `source` 已经写了 `session_1_1`，正文里的 `D1:21` 是**第二份**彼此对不上的 id；
- 文档类语料的真实开头（`Document: xxx.pdf`）被推到第 40 个字符之后。

---

## 2. 平台实际喂了什么（2026-10-14 全量普查）

数字来自 `python scripts/analyze_api_formats.py`。**这是快照** —— 平台在持续灌数据，
目录数与请求数每次跑都会涨。

```
user 目录总数 121 · 有 requests.jsonl 的 100 · add 请求 624 · search 请求 216
```

### 2.1 第一个 `/add` 的前两条 message

| 现象 | 数字 |
|---|---|
| 协议 `role` 分布（前两条） | `user` 153 / `assistant` 24 |
| 首条 add 的消息条数 | 20 条 35 次、1 条 23 次、18 条 10 次、2 条 9 次 …… |

**协议 `role` 几乎恒为 `user`** —— 它不携带说话人信息，这正是故障一的来源。

### 2.2 记账标签的两种排列

| 排列 | 请求数 | 例子 |
|---|---|---|
| 无标签 | 215 | `Gina: Hey Jon! Good to see you.` |
| `[Time: ...]` | 170 | `[Time: Sep 04, 2025, 09:05:32] Hello! I'd like to share …` |
| `[数据集][session_N][Dx:y][text]` | 125 | `[memtrapbench][session_1][D1:1][text] user: I want to configure …` |
| `[数据集][session_N][Dx:y][dialogue]` | 50 | `[angry][session_1][D1:2][dialogue] JUDGE: Pardon me, gentlemen.` |
| `[数据集][session_N][Dx:y][narration]` | 45 | `[angry][session_1][D1:1][narration] NARRATION: (SCENE.—…)` |

第四格是**消息类型**：`text` / `dialogue` / `narration`。第一格是数据集 slug（无空格）。

### 2.3 正文前缀（剥掉记账标签之后的头 24 名）

| 次数 | 前缀 | 是什么 |
|---|---|---|
| 884 | `NARRATION:` | 剧本旁白 / 舞台指示（12 Angry Men） |
| 325 / 280 / 209 / 190 / 120 / 90 / 70 / 60 | `FOREMAN:` `#7:` `#3:` `#8:` `#12:` `#4:` `#2:` `#9:` | 剧本角色（含编号角色） |
| 300 / 260 | `user:` `assistant:` | 人机对话的通用角色标记 |
| 292 / 200 / 184 / 160 / 104 / 100 / 70 | `Dr. Stockmann:` `Mrs. Stockmann:` `Peter Stockmann:` `Hovstad:` `Billing:` `Petra:` `Gina:` / `Jon:` / `Audrey:` / `Andrew:` | **真名** |
| 145 / 145 | `Caroline:` `Melanie:` | **真名**（LoCoMo） |
| 59 / 55 / 45 | `Corpus:` `User:` `Christina Baker:` | 知识图谱三元组 / 通用角色 / **真名** |
| 25 / 20 / 20 | `document:` `CLERK:` `STAGE:` | 文档分页 / 剧本角色 / 舞台指示 |
| 35 等 | `CLOSE UP - #8` `MEDIUM SHOT - THE JURY` | 镜头说明（**没有冒号**） |

### 2.4 适配层判出的形态（全部 624 个 add）

| 形态 | add 数 | 是什么 |
|---|---|---|
| `dialogue` | 340 | 逐条对话，说话人在正文前缀里（真名或 `user:`） |
| `plain` | 170 | 没有说话人前缀的人机对话（含中文医患对话） |
| `narration` | 50 | 剧本旁白 / 舞台指示为主 |
| `triples` | 26 | 知识图谱三元组 `Corpus: {...}` |
| `qa-feedback` | 20 | 流式问答反馈（JSON） |
| `trajectory` | 8 | agent 轨迹（JSON，内含 `trajectory[]`） |
| `document` | 5 | 文档分页（`document: Document: xxx.pdf`） |
| `facts` | 5 | 整段陈述句（`source: …`） |

### 2.5 各形态的完整样例

**① tagged-dialogue（真名）**

```
原始  role=user  "[beam_100k][session_1][D1:21][text] Christina Baker: I'm trying to …"
  →   role="Christina Baker"  "I'm trying to …"
```

**② 真名对话，无标签**（`Gina` / `Jon` / `Audrey` / `Andrew`，协议 role 全是 `user`）

```
原始  role=user  "Audrey: Hey Andrew! Good to see ya! What's been up?"
  →   role="Audrey"  "Hey Andrew! Good to see ya! What's been up?"
```

**③ 剧本**

```
原始  role=user  "[Time: Unknown] NARRATION: FOREMAN: 35 years old. Assistant high school coach."
  →   role="FOREMAN"  "35 years old. Assistant high school coach."
原始  role=user  "[Time: Unknown] NARRATION: (SCENE.—DR. STOCKMANN'S sitting-room. It is evening.)"
  →   role="narration"  "(SCENE.—DR. STOCKMANN'S sitting-room. It is evening.)"
```

**④ 文档**

```
原始  role=user  "[docpp][session_1][D1:1][text] document: Document: e79deb…clip_0010.pdf\nNative clip page: 1 of 5\n\n```text\n…"
  →   role="document"  "Document: e79deb…clip_0010.pdf\nNative clip page: 1 of 5\n\n```text\n…"
```

**⑤ 知识图谱三元组**

```
原始  role=user  'Corpus: {"subject": "Latvia", "relation": "official language", "object": "Latvian", "subject_id": "Q211", …}'
  →   role="corpus"  "Latvia | official language | Latvian"
原始  role=user  'Corpus: UPDATE: replace the prior value …\n{"subject": "India", "relation": "continent", "object": "Africa"}'
  →   role="corpus"  "UPDATE: India | continent | Africa (this replaces the prior value for this subject and relation)"
```

**⑥ `[Time: ...]` 标签里的时间搬到 `timestamp`**

```
原始  role=user  ts=None   "[Time: Sep 04, 2025, 09:05:32] Hello! I'd like to share my info."
  →   role="user"  ts=1756976732000  "Hello! I'd like to share my info."
```

旧管线里这一类 `timestamp` 是 `None`，于是 `msgmem.time` 是空串 —— **整份语料没有任何时间
锚点**，`date -d` 折算全部无从谈起。

**⑦ 流式 QA 反馈**（转义 JSON → 可读多行）

```
原始  role=user  'Streaming online QA feedback:\n{"qa_id": "q1", "question": "When …\\n\\nA. 2022\\nB. 2023", "predicted_answer": "(B)", "score": 0.0}'
  →   role="feedback"
      Streaming online QA feedback
      qa_id: q1
      question: When …

      A. 2022
      B. 2023
      answer given at the time (not authoritative): (B)
      score: 0.0
```

**⑧ agent 轨迹**（展开成逐条消息）

```
原始  role=user  'Streaming intra-transfer train trajectory:\n{"dataset": "ALF", "task_index": 86, "trajectory": [{"role": "system", "content": "…"}, …]}'
  →   role="system"  "[ALF task index 86] Interact with a household to solve a task. …"
      role="user"    "[ALF task index 86] Here is your task. You are in the middle of a room. …"
      …（`trajectory[]` 有几条就落几条 msg 记录）
```

平台对这类语料的查询是 **`"ALF task index 86"`** —— 它要的就是那条轨迹。所以展开出来的每条
都带上 `ALF task index 86` 这个**字面串**，否则 `grep "task index 86"` 永远搜不到
（`"task_index": 86` 埋在 JSON 里，中间隔着下划线和引号）。

---

## 3. 三条判据（为什么这么做）

### 3.1 剥标签：**只剥认得出是平台记账的**

**不能无脑剥所有前导 `[...]`**：正文里本来就有方括号 ——

```
[clbench_16_32k][session_1][D1:2][text] user: [part 1/2] Background
                                                ^^^^^^^^^^^ 这是正文
```

所以只剥这几种：

| 形状 | 例 |
|---|---|
| `session_\d+` | `[session_1]` |
| `[Dd]\d+:\d+` | `[D1:21]` |
| `Time: …` | `[Time: Sep 04, 2025, 09:05:32]` |
| 类型词 | `text` / `dialogue` / `narration` / `table` / `image` / … |
| **无空格的 slug，且只在第一位** | `[beam_100k]` `[locomo_refined]` |

`[part 1/2]` 含空格 → 不合 slug 规则 → 不作为标签。剥到第一个不合格的就停手。

### 3.2 认说话人：**判错的代价不对称**

把说话人误判成内容（漏认），代价是 `role` 退化成协议 `role` —— 可接受。
把内容误判成说话人，代码会**从正文里删掉这段前缀** —— 不可接受。

所以 `split_speaker` 偏保守，三道闸：

1. **黑名单**（`NON_SPEAKER_PREFIXES`）：内容类型（`Corpus` / `document` / `source` / `question`…）、
   舞台指示（`CLOSE UP` / `MEDIUM SHOT` / `NOTE` / `STAGE`…）、对话语气词（`Wait` / `Well` / `Yes`…）。
   **只做整词匹配** —— 一旦改成前缀匹配，`Andrew:` 会被 `and` 吃掉、`Henry:` 会被 `he` 吃掉。
   （这个 bug 真在第一次跑的时候出现过，`Audrey`/`Andrew` 那两份语料整批丢说话人。）
2. **形状**：≤ 40 字符、≤ 4 个词、不以纯数字开头、首字母大写或 `#`。
3. **语料内证据**：一个前缀在**本批次里重复出现**，就允许放宽（`Gina:` 出现 20 次必然是说话人，
   `Wait:` 出现一次就只是正文里的冒号）。

正例：`Christina Baker` `Dr. Stockmann` `Mrs. Stockmann` `Juror #2` `#8` `Caroline` `JUDGE` `CLERK`。
反例：`Corpus` `document` `source` `NARRATION` `NOTE` `STAGE` `CLOSE UP` `Available Actions`。

`user:` / `assistant:` / `User:` / `Assistant:` 收敛成通用角色 `user` / `assistant`（不是人名，
但也不是没信息 —— 它说明这是人机对话的哪一侧）。

### 3.3 判形态：**看正文，不看协议 role、也不看数据集名**

- 协议 `role` 恒为 `user`，判不出东西；
- 数据集名（`beam_100k` / `memtrapbench`）是平台内部 key，换个名字就失效；
- 只有**正文前缀**是语料自己长出来的。

结构性形态（三元组 / 流式记录 / 文档 / 整段陈述）由首条消息判定；剩下的一律先当"对话/剧本"，
再用**整批**的说话人证据分 `dialogue`（拆得出说话人）与 `plain`（拆不出）。剧本的第一条常是
演员表 `NOTE:`，没有说话人 —— 所以判据必须看整批，不能只看首条。

---

## 4. 各形态的处理一览

| 形态 | `role` 怎么来 | 正文怎么来 | 备注 |
|---|---|---|---|
| `dialogue` | 正文前缀（真名/通用角色） | 剥标签 + 剥说话人前缀 | 剧本角色（`JUDGE`）也走这条 |
| `narration` | 前缀是 `NARRATION:` → `narration`；里面连着的 `FOREMAN:` 优先 | 剥标签 + 剥 `NARRATION:` | 舞台指示记成 `narration` 而不是 `user` |
| `plain` | 协议 `role` | 剥标签 | 中文医患对话、无人名的人机对话 |
| `document` | `document` | 剥标签 + 剥 `document:` 标记，**保留** `Document: <名>` 与页码 | 文档名是溯源信息，不能丢 |
| `triples` | `corpus` | JSON → `Subject \| relation \| object`，**丢掉 Q-id** | `UPDATE:` 前缀保留更新语义 |
| `facts` | `corpus` | 剥标签 + 剥 `source:` | 整段陈述句 |
| `qa-feedback` | `feedback` | JSON → 多行 `key: value` | 旧答案标成 "not authoritative" |
| `trajectory` | 轨迹内部每条自己的 `role` | 展开成逐条，各带 `[<dataset> task index N]` | 让 `task index N` 可 grep |

**三元组为什么渲染成 `A | relation | B` 而不是一句英文**：`relation` 是数据集自己的词
（`country of citizenship` / `official language`），任何"改写成自然语言"的模板都会把它动词化错
（"is the country of citizenship of" 还是 "is a citizen of"？）。逐字保留三段既不会改写错，
又能被 grep 到（查询里就是 `"official language"` 这三个词）。

---

## 5. 适配之后，信息流到哪

```
/add 报文
   │  server.add     normalize_content（多模态 → 文本）
   ▼
store.append
   │  adapters.adapt_messages        ← 本文这一层
   ▼
msgmem.jsonl          role = 真名 / 剧本角色 / user / corpus / document / narration / feedback
   │
   ├─► adapters.speakers_of(records)  → 本会话的真实说话人名单
   │        │
   │        ├─► summem    prompt 的 {speakers} 槽（"参与者是 Caroline 和 Melanie"）
   │        └─► atommem   prompt 的 {speakers} 槽 + "role 只能从这份名单里选"
   │
   └─► search  inputs/
            └─► runner.render_participants  → user_begin 的 {{PARTICIPANTS}} 槽
```

三处下游都改了：

**① atommem 的 prompt 拿到名单。** 旧版把 `speaker_a` / `speaker_b` 硬替换成 `user` / `assistant`，
模型于是产出 `"user is trying to implement …"`。现在填的是 `Christina Baker`（单人的语料就是
一个名字），剧本则是 14 个角色的名单。prompt 里还多了一条规则：

> **不是每条记录都是人在说话**。记录可能是文档、数据库三元组（`Subject | relation | object`）、
> 旁白、评测记录。对这些：照记录本身陈述事实，`role` 用它自己的值，**不要发明一个说话人**。

**② summem 的 prompt 不再写死两个人。** 旧版 `between {speaker_a} and {speaker_b}` 在
14 个陪审员的剧本上是错的。现在是一个 `{speakers}` 槽 + "不是每次会话都是对话"的说明。

**③ search 的 `user_begin` 多了一段"这份语料有谁"。** 由
`runner.render_participants(inputs_dir)` 从**这份语料真实的 `metadata.role`** 渲染：

```
Who this material is about -- every record carries `metadata.role` naming the person or
character it belongs to: Caroline, Melanie.
When the question names a person, the fact must be about THAT person: grep the name,
and keep the name in the record you write …

Records whose `role` is one of 'corpus', 'document' has no speaker. A "session" here may be a
document, a set of `Subject | relation | object` facts (read it as: the subject's relation is
the object), or an agent trajectory. Use the material as it is and never invent a speaker.
```

**为什么是"渲染真实数据"而不是"再写一段 prompt 说教"**：与 `render_tag_index` 同一个理由 ——
实测小模型不会因为被**描述**就采用一个它没见过的字段，但把真实的值摆出来它就会用。
**语料里全是 `user`/`assistant` 时整段不渲染**（那段话对它是纯噪声，白占 prompt）。

---

## 6. 存量语料怎么重跑

`msgmem.jsonl` 是**已经落盘**的产物 —— 修好适配层之后，用旧代码写出来的语料不会自己变好。
拿 `requests.jsonl`（每次 `/add` 的原始报文留痕）当输入重放：

```bash
# ① 先看要动什么（不动文件）
python scripts/rebuild_api_store.py --all --dry-run --only-with-adds

# ② 只重建 msgmem：纯 CPU、零模型成本、秒级。先把"谁能被 grep 到"修对
python scripts/rebuild_api_store.py --all --stage msgmem --only-with-adds

# ③ 再重建 summem / atommem：要调模型（逐会话 + 逐消息）
python scripts/rebuild_api_store.py --all --stage memories --only-with-adds
```

- **id 与时间戳重放后逐条一致**（`session_1_1` 还是 `session_1_1`，`time` 一字不差）——
  下游的 evidence `source`、`session_N_summary` 全都不用改。已验证。
- 覆盖前先把原文件备份成 `msgmem.jsonl.bak-<时间戳>`。
- **重放条数少于现有条数时一律拒写**（要 `--allow-loss` 才覆盖）。实测真有这种 user：
  `requests.jsonl` 的留痕是**尽力而为**的（写失败不阻断请求），所以重放会比语料少一整批。

---

## 7. 重新跑一次普查

```bash
python scripts/analyze_api_formats.py                  # 统计 + 每形态抽样前后对照
python scripts/analyze_api_formats.py --samples 3      # 每形态多看几个 user
python scripts/analyze_api_formats.py --json fmt.json  # 落一份机器可读统计
```

脚本用的是**适配层自己的**判据（`split_lead_tags` / `split_speaker` / `adapt_messages`），
所以它报出来的分类与预处理时看到的**是同一件事** —— 不会出现"统计说 50 份对话，
实际处理只认出 40 份"这种两套判据各说各话的情况。

单测在 `scripts/test_api.py` 的 `[8]`–`[10]` 段：每个形态一条用例，外加"贯通到三层记忆"
（真的走一遍 `store.append`，断言 atommem 的 prompt 里收到了真名）。
