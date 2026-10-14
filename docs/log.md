# CodeMem 实验日志与运行手册

> 这份文件记录**过程**：每一轮的实测数字与失败归因、被取代过的设计、里程碑与待办、运行与调试手册。
>
> **当前系统的设计与契约在 [`core.md`](core.md)** —— 想理解系统怎么工作、为什么这样设计，读那一份；
> 想查"当时为什么这么改""上一轮的数字是多少""这个坑是怎么踩的"，读这一份。
>
> 标题里带 `原 §2.x` 的几节，原先嵌在 `core.md` 的第 2 章里。它们记录的是**当时的**迭代过程
> （其中一部分设计后来被取代），保留原编号是为了让旧笔记与引用仍然对得上。

## 目录

| 部分 | 内容 |
|---|---|
| 一 · 运行手册 | 一条命令跑通三步、自测、常见问题、search 的 CLI |
| 二 · 里程碑与待办 | 早期里程碑 S0–S14 清单、待办 N0–N7 |
| 三 · 实验记录 | 时间推理的三轮迭代、三个阶段实录、S15–S21 |

---

## 第一部分 · 运行手册

### A.1 一条命令跑通三步（`scripts/run_eval.py`）

search 是改动最频繁的一步，而完整流程是三步、各有自己的 CLI 与产物路径。这个脚本把它们串起来，
并把**该看的数字**摆成一张表：

```bash
python scripts/run_eval.py                        # 默认：1 个目录 × 前 5 条 QA，判分
python scripts/run_eval.py --tag v5-time-rule     # 打标记，便于和上一版对比
python scripts/run_eval.py --limit 0              # 0 = 全部 QA（慢）
python scripts/run_eval.py --no-answer            # 只跑 search，看 evidence 产出
python scripts/run_eval.py --no-judge             # 跑完三步但只算 CPU 指标（省 judge 调用）
python scripts/run_eval.py --reuse search_20260929_183148   # 复用 evidence，只重跑 answer+eval
python scripts/run_eval.py --quick                # 冒烟：--max-steps 8 --limit 3
```

输出长这样（含按 category 拆开那一栏 —— temporal 是时间规则改动的直接反馈）：

```
-- search --
  目录                           QA  evidence      步数      充分
  Caroline_Melanie       5/5             23      61    80%
-- eval（整体）--
  judge_accuracy   0.600     evidence_recall  0.550     unsupported_rate  0.200
-- eval（按 category）--
  category           n    judge   recall    unsup
  temporal           3    0.667    0.700    0.000
-- 失败归类 --
  missing_evidence      1   → 修 search：该找的没找到
```

**默认只跑前 5 条 QA**，不是全量 —— 几十条就能看出 prompt 改动的方向对不对；全量给
`--limit 0`。**步数上限默认 16**（与 `configs/search.yaml` 的 30 同一量级；这个值实测过得
+3.7 个点，见 §S20），要"快反馈"用 `--quick`（= 8 步 / 3 条）。两处设计上的细节值得一说：

- **它接的是"这一次"的运行目录，不是"最近一次"。** 各步骤原本各自取最近一次，而调试时很容易
  接错 —— 比如你刚跑过一次无关的 `--dry-run`（它也会建目录）。这里改成"跑前记下已有目录、
  跑完取差集"。
- **eval 强制带 `--experiment eval`。** eval 默认把 `summary.json` 写进 `--output-dir`，
  而 search 已经在那里放了一份 —— 不加前缀就会**覆盖掉**，于是"这次 search 跑了多少步、
  多少条 evidence"当场丢失，而摘要正要读它。现在写的是 `eval_summary.json`，两份并存。

### A.2 自测

纯 CPU、零模型成本、零网络（假模型 + 真工具）：

```bash
python scripts/test_search.py       # 全链路断言（含两项静态检查）
python scripts/test_api.py          # 封装 API 自测（37 项）
```

`test_search.py` 开头是两项**静态检查**（都纯 AST，不用跑起来）：

- **无未定义引用**：确保没有"调用了但没定义/没导入"的名字 —— 重构最容易留下的就是它。
- **无过时关键字参数**：调用处传的 `kwarg=` 必须真的存在于签名里。
  这一条是**实测补上的**：删掉向量索引时去掉了 `run_qa` 的 `index_dir` 形参，但调用处还留着
  `index_dir=index_dir` —— 全量自测 304 项全绿（测试文件自己也传了这个参数，而测试里那个名字
  恰好还存在），一上真机就 `NameError`。**形参与实参是两个地方，删一个不等于删另一个。**

### A.3 常见问题

**路径里的反斜杠**：Git Bash 下 `H:\Project\...` 的反斜杠会被 shell 吞掉，请用正斜杠。

**输出被截断（`finish_reason=length`）**：某些模型会消耗大量 reasoning token，且这些 token
计入 `max_tokens`。日志出现 `TruncatedCompletion` 时调大 `max_tokens`。

**`chat_template_kwargs` 报 400**：该参数只对模型名含 `qwen3` 的端点发送，其它端点自动跳过。

**embedding 的两个实测坑**：
- **400 且提到 tokenizer 端口连不上** —— 并发压力下的瞬时故障，与请求内容无关，报错信息具
  误导性。已按 `chunk_size` 分批 + 每批退避重试；批越小越稳（批 64 实测 3/3，批 512 只有 1/3）。
- **`cudaMalloc failed: out of memory`** —— embedding 模型要占 GPU，腾显存后再跑。

**远端 API key**：`configs/*.yaml` 里的 key 是明文，`.gitignore` 已排除 `configs/`，注意不要提交。

---

### A.4 search 的 CLI 用法（速查）

```bash
# **先 dry-run**：只建 inputs/ + 渲染 system prompt 打印出来，不调模型
python -m codemem.search --sample Caroline_Melanie --dry-run

# 调试首选：只跑前 3 条 QA、每条最多 8 步，看完整数据流
python -m codemem.search --sample Caroline_Melanie --max-qa 3 --max-steps 8 --log-level debug

# 全部目录全部 QA（注意成本：每条 QA ≈ steps + verifier 次调用）
python -m codemem.search --experiment search

# 纯 CPU 自测（假模型 + 真工具，零模型成本）
python scripts/test_search.py
```

| 参数 | 说明 |
|---|---|
| `dirs` / `--sample NAME` | speaker 目录名或路径；缺省处理全部含 `msgmem.jsonl` 的目录 |
| `--config PATH` / `--tool-config PATH` | 配置（默认 `configs/search.yaml` / `configs/tool.json`） |
| `--qa 3,7,12` | **只跑指定的 QA 原始下标**（逗号/空格分隔）。优先于 `--max-qa` 与步数预算：点名了就一定跑完。配 `--list-qa` 查下标 |
| `--repeat N` | 每条 QA 重复跑 N 次（默认 1）。用来反复测同一条 QA 看结果稳不稳、定位偶发问题 |
| `--list-qa` | 只列出该目录的 QA 下标就退出，不跑演化 |
| `--tag NAME` | 给这次运行打标记，写进输出目录名与 summary（如 `--tag promptA`） |
| `--max-qa N` | **调试闸门**：每个目录最多处理前 N 条 QA（0=全部） |
| `--max-steps N` / `--max-verify-rounds N` | 覆盖步数上限 / verifier 轮数 |
| `--concurrency N` | 覆盖 QA 级并发（生效值见 `configs/search.yaml`） |
| `--native-tools` / `--no-native-tools` | 强制开关原生函数调用（覆盖配置） |
| `--experiment NAME` / `--output-dir PATH` | 输出目录前缀 / 指定输出目录 |
| `--prune-runs` / `--keep-last K` | 清理历史运行目录 |
| `--dry-run` | 只渲染 prompt 后打印，不调模型 |
| `--log-level` / `--log-output` | 覆盖 `log.level` / `log.output` |

---

## 第二部分 · 里程碑与待办

**已完成的架构重构**（2026-09-27）：围绕 add / search / answer / eval 四个步骤重组，
共享层提到 `codemem/` 顶层（`io` / `llm` / `log` / `dataset`）。
删掉了 v2 的动作空间版（`evomem.py` + `evoactions.py`，git HEAD 仍可找回）。

- [x] **S0** `evidence_to_ids` 解析 bug 已修（空格分隔多引用 `"D9:1 D4:4 D4:6"` 与 `"D:11:26"`
      都能解析）—— 现由 `dataset.evidence_to_ids` 唯一实现
- [x] **S1** 更严格的 CPU 不变量：`evidence.jsonl` 的校验（`content` 非空、
      `metadata.source` 非空且元素合法、重复 id 拒绝…）在 `search/evidence.py`
- [x] **S3** search 步骤（code-agent 式证据构建）：见 §2
- [x] **S4** answer / eval 两个步骤打通，`answers.jsonl` + `eval.jsonl` 端到端可跑
- [x] **S5** 静态回归网：`test_search.py` 用 AST 扫全包，杜绝"调用了但没导入"
- [x] **S6** add 只留 session summary（当时删掉了顶层 speakers summary）；search 简化为
      「检索就是 grep」的搜索 agent；补上时间精度规则（见 §2.2 / §2.3）；删掉向量检索与检索 CLI
- [x] **S7** `scripts/run_eval.py` 一条命令跑通三步并打印摘要；补一项静态检查
      （过时关键字参数 —— 它正是 S6 在真机上炸的那个错因）
- [x] **S8** 封装 API（Add / Search）：多用户隔离、request_id 幂等、复用 search agent 管线、
      多模态降级（见 §4）
- [x] **S9** 证据标准抽为独立 prompt 段 `EVIDENCE_STANDARD`；`edit` 加 append 模式、删掉 `write`
- [x] **S10** **从 bad case 反推的大改**（2026-09-30，见 原 §2.16）：删掉顶层 speakers summary
      （实测它把检索短路 —— 150/152 条 QA 一次 grep 都没做）、删掉 `next` 上下文块控制工具与
      整套 stage 机制，search 砍成**一个朴素 agent loop**；唯一引导是"关键词检索 vs 通篇读"
      的策略表（由 agent 自己选）。同目录同 judge：**31.6% → 59.1%**（temporal 10.8%→60%、
      single_hop 32.9%→63.5%、multi_hop 34.4%→57.7%）。顺带修两个 harness 级 bug：
      `grep` 缺 `-E` 自动补、`edit` 缺 `path` 默认 evidence.jsonl
- [x] **S11** **引入跨会话索引**（2026-10-01，见 core.md §1.1 / 原 §2.17）：删掉 `session_summaries`
      这一层，add 第二阶段改为逐会话抽 `(主体,关系,对象)` 三元组、合并成 `index.jsonl`
      （每条带它出现在哪些 session/msg/源话时间词）。索引是**导航不是摘要** —— 摘要（无论
      顶层印象还是逐会话）实测都让 agent 跳过检索。跨会话连接是它的核心价值（multi_hop
      平均跨 2.41 个 session）。search 侧继续做减法：不再有 `next`，只剩 read/edit/bash；
      引导改为"grep 索引 / grep 原文 / 读整段会话"三条；system prompt 明确核心目标
- [x] **S12**（已废弃，被 S13 取代）定位层升级为实体关系图：`nodes.jsonl` / `edges.jsonl`
- [x] **S13** **消解取代导航层**（2026-10-02，见 core.md §1.1 / 原 §2.18）：删掉 `nodes/edges` 图与
      `index.jsonl` 一路的"额外导航层"，add 第二阶段改为**逐消息消解**——
      **时间**锚定成绝对时间或带绝对锚点的相对时间（`time` + `time_kind` point/range/approx，
      粒度只减不增）、**指代**消解成真名（逐消息、喂前文消息作上下文）、**原话**保留在
      `source_content`/`source_time`（re-resolve 幂等的底）。产物改为
      `data/{dir}/sessions/session_N.jsonl`（**一次会话一个文件**）。search 侧：语料是
      `inputs/sessions/` 目录，两种检索策略（keyword 直达 / 通读相关 session）集成在一个
      agent loop 里、运行中自行切换；`evidence.jsonl` 同时是**笔记**（相关即可写，不必只写答案）
- [x] **S14** **三层记忆 + add agent 逐消息抽取**（2026-10-04，见 §1）：把"改写消息本身"
      升级为**分层**——`msgmem.jsonl`（`type=msg`，原话 + 会话时间锚点，纯 CPU 转换）/
      `summem.jsonl`（`type=summary`，逐 session 一次 LLM 总结）/ `atommem.jsonl`
      （`type=atom`，**add agent 逐 message 处理**：复用 search 的 `run_agent_round` /
      `build_tools` / `ToolCatalog`，默认 `max_step=8`，agent 只填 `memory` + `metadata.{time,
      role, tag}`，`id`/`source`/`type`/`changelog` 由 `finalize_file` 系统补、并支持合并演化）。记录格式对齐
      `memory_template.jsonl`。同目录逐 message **串行**（后面的 message 能 grep 到前面已生成的
      atommem，消解跨 session 指代）、不同目录并行；按 `source` 增量覆盖 = 幂等。search 语料换成
      硬链的三个记忆文件；api 的 `store.append` 在返回前重建该会话的 summem + atommem。
      删掉 `add/resolve.py` 与 `add/session.py`

待办：

- [ ] **N0**（最高优先）**open_domain（反事实题）从 77% 掉到 25%**（原 §2.16 末）：这类题没有
      单一事实、要拿全局信息推理，而 search 阶段硬找"那句不存在的话"只能返回 null。方向是把
      少量关键事实检索出来、让 **answer 阶段**从证据推理，而不是在 search 阶段逼它找反事实结论
- [ ] **N1** 效果评估：跑完整 10 个目录，对比 baseline 的 evidence recall 与 judge 准确率；
      盯 `sufficient_rate`、eval 的 `failures` 归类、`verifier_sufficient` vs `judge_correct`
      的背离。原 §2.15/§2.16 目前只有 Caroline_Melanie 一个目录的数字
- [ ] **N2** search 的多跳补强：`missing[]` 回灌已实现，可考虑让 verifier 能"点名要某条记忆"
      （当前只能描述缺口，由 agent 自己检索）—— 零证据的 23 条 QA 是这条的实证
- [ ] **N3** `ANSWER_PROMPT` 加 `used_ids` 字段（后续做 credit 归因要用）
- [ ] **N4** search 的强化学习：QA 驱动的证据构建已是一条完整 trajectory（状态 = evidence.jsonl，
      终局奖励 = judge 答案），可直接在其上做 oracle 轨迹采样 + GRPO
- [ ] **N5** 三个 baseline 脚本（`eval_baseline/rag/atommem`）与新链路的指标口径统一，
      使"msgmem / atommem / evidence / 全上下文"四者可横向对比
- [ ] **N6** 三层记忆的端到端效果评估（S14 尚未跑真实模型）：对比"三层（atommem 检索）"与
      "单层（消解后消息）"的 evidence recall 与 judge 准确率，验证 atommem 的拆分/消解质量
- [ ] **N7**（S20 实测得出的新方向）**瓶颈已从"检索覆盖"转到"检索到了但答不对"**：
      Deborah_Jolene 上把步数上限放到 24 后，evidence_recall 42%、judge 58%，而"参考答案实词
      确实出现在 evidence 里、答案仍然错"的 QA 仍有约 **20 条**（改动前 15 条）—— 多检索到的
      事实没有转化成正确答案。下一轮该动 **answer / 判定**（含"同一事实在 evidence 里有多条
      不同措辞、答案挑了次优那条"这类），而不是继续加检索

（注：`data/*/atommem.jsonl` 的 10 个目录为旧原子记忆体系的遗留数据，格式与 S14 不同。）

---

## 第三部分 · 实验记录

_下面按时间顺序。带 `原 §2.x` 的是从 `core.md` 第 2 章移出的实录。_

### 原 §2.4.1 一次"效果变差"的完整归因：58% → 47% 里有多少是真的

改完时间规则后重跑，分数**从 58% 掉到 47%**。逐条归因之后，结论是：
**这 11 个点里绝大部分是判分口径变了，不是答案变差了。**

做法：把两次运行（`search_20260929_185935` / `search_20260929_214956`）的逐条结果按
**同一套规则**重新判一遍，再比较。三件事同时发生了：

| 现象 | 影响 | 性质 |
|---|---|---|
| **旧 judge 把"空答案"判成 CORRECT** —— 实测 **11 条**（"The prediction correctly identifies that the answer is unsupported…" → 竟然判对） | 旧分数**虚高** | 旧 judge 的 bug，新 judge 修对了 |
| **新 judge 把"集合型问题多给成员"判成错** —— 集合型 67% → 26%，而参考答案只是**不完整的列表** | 新分数**虚低** | 新 judge 引入的过严，已修 |
| **temporal 真的变好了** 27.8% → 50.0% | 真实提升 | 时间规则的功劳 |

**统一判据后**（把两边都用同一套规则重判）：

```
                 旧 run      新 run
multi_hop      18/30 60% →  9/30 30%
temporal       10/36 28% → 18/36 50%     ← 真实提升
open_domain     5/13 38% →  5/13 38%
single_hop     43/69 62% → 39/69 57%
总体           76/148 51% → 71/148 48%
```

**剩下的 20 条"变差"里，14 条的答案覆盖率不低于旧答案**（措辞变了、实质等价，被模型判成
不同），只有 6 条是真的变差。也就是说：**净变化基本在噪声范围内，唯一确凿的移动是
temporal +22 个点。**

**为什么判分口径会漂移这么多**，根因是上一版把 judge 的 `reason` 挪到了第一位
（为了 CoT），副作用是模型从"直接比较"变成"先复述判据标题再比较" —— 点名判据的比例
**42% → 82%**，而且反射性地去点 `CORE FACTUAL ACCURACY`，在那条判据下"多给内容 = 加错误
主张 = INCORRECT"。这一条现在明确写了：**集合型问题上超集是 CORRECT**，且 **reason 里不要
复述判据标题**。

**教训**：judge 是被测量的一部分。它的 prompt 一改，**历史分数就不能直接比了** ——
改 judge 之前，要么用同一套判据把旧结果重判一遍，要么明确标注"跨版本不可比"。
（早期为此写过 `scripts/judge_impact.py` 做影响面扫描，那是一次性工具，已完成使命并删除。）

### 原 §2.4.2 时间处理的最优方案：保留相对表达，不折算（2026-10-06）

§2.3 的规则里留着一条**自相矛盾**的指导：一边说"带锚点的短语就是答案"（规则 4），一边又写
"**Compute when you can: lead with the computed date**"（旧规则 3）。8B 在矛盾下倾向于执行
"尽量折算"，于是把 "last week" 算成了**会话日期本身** —— 恰好踩中规则 1 明令禁止的错误。
全量跑后 temporal 只有 36%，但 evidence_recall 高达 83% —— **检索到了，是作答/判定把时间搞错了**。

#### 三个叠加的 bug（逐条实测确认）

**① search 把相对表达错误折算成会话日期。**

```
源消息 (session_3_1, 时间 9 June 2023): "my school event last week"
atommem (正确): time = "last week (anchor: 9 June 2023)"  ← pipeline 已保留
search 写的 evidence (错): "Caroline gave a speech at a school on 9 June 2023"  ← "last week" 被吞成锚点本身
参考答案: "The week before 9 June 2023"
```

**② judge 折算 candidate 时丢掉了修饰语，把逐字相同的答案判成不同日期。**

`judge._parse_date("the week before 6 July 2023")` 走的是**绝对日期**分支 —— 它抠出
"6 July 2023" 算成 `2023-07-06`，**完全无视 "the week before"**。而 reference 走
`resolve_relative_date` 算出 `2023-06-29`。于是：

```
reference: "The week before 6 July 2023"  → 2023-06-29
candidate: "the week before 6 July 2023"  → 2023-07-06   ← 逐字相同却算成差 7 天
judge: "These differ by 7 day(s)" → INCORRECT
```

逐字相同的正确答案被判错。**这是 judge 最隐蔽、也最该先修的 bug** —— 它污染测量，
让"答案其实对了"的题显示为错。

**③ answer 把锚点当成事件时间作答。**

#### 最优方案（用户定的口径）

> **相对时间表达本身就是一个完整、精确、充沛的答案。** "The week before 6 July 2023" 自带锚点
> 6 July，任何人（包括 judge）都能验证两个答案是否相同 —— **不需要、也不应该折算成绝对日期**。
> 折算是有损的：它丢掉"一周内哪天"的信息（"the week before X" 是一个**周区间**，不是一个点），
> 还会引入算错的风险。而"上周发生的"（模糊描述）= 一个周内发生，**不能具体到某一天**。
> **不要随意理解日期。**

落成三条规则：

| 表达 | 处理 | 理由 |
|---|---|---|
| `yesterday` / `today` / `tomorrow` | **折算成绝对日期**（`date -d`，单步精确） | 唯一"精确且平凡"的折算 |
| `last week` / `last Friday` / `the week before X` / `N years ago` | **保留原表达 + 锚点**，不折算 | 自带锚点即完整答案；折算是信息损失 |
| `a few weeks ago` / `recently` | 保留为**粗粒度窗口**，绝不细化成日期 | 源就是模糊的，细化 = 编造 |

**具体修改**：

1. **`tool.json` 的 `time_policy`**（search 侧）：删掉"compute when you can"，改成
   "**preserve, don't convert**" + "只有 yesterday/today/tomorrow 才折算" + "按源的单位作答，绝不细化"。
2. **`tool.json` 的 worked examples**：把"lead with the computed date"的例子改成
   "keep 'last week' + anchor" 的正确形态。
3. **`prompts.ANSWER_PROMPT`**（answer 侧）：加"keep the source's own time wording, anchored --
   do NOT flatten it into a date" + "read the unit the source used and answer at that unit"。
4. **`prompts.EVIDENCE_STANDARD`**：第 3 条改为 "TIME ANCHORED, KEPT AS WRITTEN"。
5. **`judge._parse_date`**：**先试相对表达**（复用 `resolve_relative_date` 的同一套逻辑），
   识别不到再回退到绝对日期解析 —— 保证 reference 与 candidate **口径完全一致**。
   把折算逻辑抽成 `_resolve_relative_expression` 供两个函数共用。

#### 实测（Caroline_Melanie 的 3 条 temporal QA）

| QA | 修改前 | 修改后 |
|---|---|---|
| qa8 | `9 June 2023`（吞成锚点）❌ | `the week before 9 June 2023` ✅ CORRECT |
| qa9 | `9 June 2023` ❌ | `the week before 9 June 2023` ✅ CORRECT |
| qa28 | `15 July 2023` ❌ | `the week before 15 July 2023` ⚠️（差在 "Friday" vs "week"，非时间方案问题）|

token_f1 从 ~0.5 升到 **0.944**。新生成的 evidence 正是期望的形态：

```
Caroline gave a speech at a school 'last week' (message dated 7:55 pm on 9 June, 2023)
  -- i.e. the week before 9 June 2023.
```

### 原 §2.4.3 全量跑后的错因分析：最大瓶颈是"集合题检索不到"（2026-10-06）

时间方案落地后全量跑（`search_20261006_112146`）：**judge 55.4%**，temporal 从 36% 升到 **61.1%**
（时间方案生效），但 single_hop 62.1%、multi_hop 40.7%、open_domain 33.3%。

#### 按 recall 拆开 58 个错误，瓶颈一目了然

| 错误形态 | 条数 | 含义 |
|---|---|---|
| **recall=0（检索不到）** | **29** | evidence 完全没覆盖标注证据 —— 最大瓶颈 |
| unsupported（没答） | 13 | search 判"证据不足"直接放弃 |
| recall=1（检索到但答错） | 12 | 证据对，作答/判定错 |
| recall 部分 | 4 | 集合收不全 |

**29 条 recall=0 里绝大多数是"集合型问题"**（"what activities/pets/books/items does X have"、
"where has X camped"、"what types of pottery…"）—— 标注证据是一**组散点**，search 收漏一条就大量丢分。

#### 三个具体错因（逐条看轨迹确认）

**① 集合题，agent 只做关键词 grep，不读会话。**

qa18「Where has Melanie camped?」（真答案 `beach, mountains, forest`）的 12 步轨迹全是
`grep "camp"` / `grep "mountains"` / `grep "campfire"` —— **它从没用过 Strategy 2 的"读会话"**。
成员散落在多个会话，关键词法永远抓不全。

**② 具体内容被抽象化。**

qa127「What did Caroline make for a local church?」原子层**是对的**（`Caroline made a stained
glass window`），但 search 写成 `"Caroline created a painting"` —— **"stained glass window" 被抽象成了
"painting"**。qa112（`a sunset with a palm tree` → `a nature-themed artwork`）同理。
这正是 `ATOM_AGENT_STANDARD` 规则 13 禁止的，但 search 的 `EVIDENCE_STANDARD` 没有对应条款。

**③ answer 复述 prompt 模板占位符。**

qa18 的 answer 输出 `"the week before <session date>"`、qa71 输出 `"the year before session_7_8"`
—— 因为 `ANSWER_PROMPT` 的工作示例里写的是 `<session date>` / `<year-1>` 这种**模板标记**，
模型逐字照抄了，还把它当成了答案。

#### 修改

1. **`strategy`（策略选择）**：加一条判据 —— 集合/计数题**必须走 Strategy 2**，并警告
   "grep 人名会匹配几乎每一行，`head` 只看到噪声，不是成员"。
2. **`EVIDENCE_STANDARD` 第 8 条**：新增 `SPECIFICS PRESERVED` —— 保留具体内容（stained glass
   window / sunset with a palm tree / 标语原文），禁止抽象成类别。
3. **`verifier`**：加两条 —— 集合题要**具体成员**（不是类别）、答案要**保留具体细节**。
4. **`ANSWER_PROMPT` 示例**：把 `<session date>` / `<year-1>` 占位符换成**真实示意日期**
   （9 June 2023 / 2022），并明确"示例里的日期是示意值，必须替换成真实日期；绝不复制占位符、
   session id 或示例值"。
5. `system prompt` 上限 19500 → **19700**（上述规则是有意的、针对最大瓶颈的增长）。

#### 诚实记录：一次失败的方向

最初把规则写成"you MUST use Strategy 2"，结果 **充分率从 42% 掉到 0%** —— 8B 把它理解成
"要 grep 人名"，于是 `grep "Melanie" | head -20` 反复重试、卡住不写 evidence。
**加规则 ≠ 改行为**：抽象描述（"list the sessions, then read"）8B 会误读，必须给**具体命令形态**
和**明确的反例**（"grep 人名只会看到噪声"）。这是 add / search agent 反复验证过的教训。

#### 实测（20 条混合题，中等规模验证）

| 指标 | 全量（改前） | 20 条（改后） |
|---|---|---|
| judge_accuracy | 55.4% | **60.0%** |
| unsupported_rate | 10.0% | **0.0%** |
| token_f1 | 0.404 | **0.442** |
| sufficient_rate | 42% | **75%** |

（20 条样本小、判分条数少，只作方向参考；全量验证见下。）


**一句话总结**：相对时间**保留原样 + 附锚点**是答案，绝对日期只是"锚点 + 偏移"的另一写法；
judge 两边都要用同一套折算逻辑，否则会把逐字相同的答案判成不同。

---

### 原 §2.15 首次全量跑（Caroline_Melanie，152 QA）的实测结论

> **这组数字来自更早一次全量跑**（当时是"会话时间当事件时间"那一版 prompt），保留它是为了
> 记录下面三个诊断结论 —— 它们与语料层数、与后来的分阶段改动都无关，至今仍是当前设计的依据。
> **当前朴素 loop（core.md §2.1）的实测见 原 §2.16**，这组数字是更早版本的，不能当作它的成绩。
>
> **两次完整运行的对比在 原 §2.4.1** —— 那里说明了为什么"58% → 47%"这个表面跌幅
> 绝大部分是判分口径变化，而不是答案变差（用统一判据重判后是 51% → 48%，
> 唯一确凿的移动是 temporal 28% → 50%）。
>
> 所以：**跨版本的分数不能直接比**，除非用同一套 judge 判据重判一遍。

```
search   152/152 QA，0 错误，verifier 判"足够" 64%，473 条 evidence
answer   147 条（5 条无 evidence 可答），证据不足 22 条
eval     judge_accuracy 55.1%   token_f1 0.310   evidence_recall 43.6%
```

| category | n | 准确率 | recall |
|---|---|---|---|
| single_hop | 68 | 66.2% | 41.2% |
| open_domain | 12 | 75.0% | 22.5% |
| multi_hop | 30 | 43.3% | 23.1% |
| temporal | 37 | 37.8% | **70.3%** |

**三个诊断结论**（都是查了数据才敢下的）：

**① `evidence_recall` 不是可靠的失败信号。** 74 条 recall=0 里有 **36 条答案是对的** ——
标注 evidence 只列 1-4 条消息，而同一事实往往有其他消息佐证。所以"recall 低"不等于
"检索差"；判断链路好坏必须看 judge 准确率，recall 只能当参考。

**② temporal 的 recall 70% 但准确率 38% —— 瓶颈已从检索转到作答与判定。** 其中一部分是
**judge 的测量缺陷**：参考用相对表述（"the week before 6 July 2023"）而预测给绝对日期
（`2023-06-29`，**两者在时间点上完全相等**），judge 却判错。8B 推算"13 Sep 2023 是周三、
之前的周末是 9-10 号"这类算术**不可靠**（实测同一条判 3 次都错）。

修法：**日期算术交给代码**（`judge.resolve_relative_date`），把算好的结果作为
`[PRECOMPUTED] ... SAME date` 提示塞给 judge，让它只做比较、不做推算 ——
与 search 阶段"相对时间交给 `date -d`"是同一个取舍。效果：temporal 37.8% → **43.2%**，
且 5 个已知误判里 4 个稳定转正。

**③ 每一条 QA 只写 3.2 条 evidence（均值），且 4-8 条是准确率甜点**：

| evidence 条数 | n | 准确率 |
|---|---|---|
| 0（空） | 8 | 0% |
| 1-3 | 99 | 54% |
| **4-8** | **27** | **81%** |
| 9+ | 13 | 46% |

塞太多会引入噪声（9+ 掉到 46%），塞太少则信息不足。`evidence_soft_limit: 40` 太宽松，
**调到 12 左右更贴合实测的最优区间**。

---

### 原 §2.16 从 bad case 反推的一次大改：31.6% → 59.1%

上一节那张表是"印象判答 + 逐 session"版本的。这一节记录**怎么从 bad case 找到根因、
把 search 砍成一个朴素 loop** 的全过程 —— 这是本项目最实的一次方法改进，也是"范畴判断
优先于堆流程"的最好例证。

#### 先看数据：104 条 INCORRECT 是什么样的

拿旧版全量跑（152 QA）的 `eval_eval.jsonl` 逐条分类：

| 现象 | 条数 | 性质 |
|---|---|---|
| **一次 `grep` 都没做的 QA** | **150 / 152** | 检索被整个短路 |
| INCORRECT 里"证据在但答错"（= 编造） | **97 / 104** | 不是检索不到，是压根没去检索 |
| verifier 判 sufficient 的 / 其中错的 | 131 / **87** | verifier 被"看起来完整"的编造骗过 |
| 集合型问题 `recall=0` | 17 / 18 | 只给一两个成员就收工 |

#### 典型 bad case：0 步检索 + 编造 + 假来源

qa92「What country is Caroline's grandma from?」（参考 **Sweden**）的完整轨迹只有两步：

```
step1 edit: {"content": "Caroline's grandma is from Canada, as Melanie mentioned
             during their conversation...", "metadata":{"source":["session_13"]}}
step2 纯文本 = 完成
```

它**从没读过 session_13**（正确答案在 `session_4_3`），编了一个国家名、还配了个看似
可溯源的假 source，verifier 判 sufficient。同形态的还有：
- qa106「新鞋用来干什么」→ 编"pottery"（答案 running），证据里写着 "**likely required** the use of her new shoes"（臆测）；
- qa93「奶奶的礼物」→ 编"一幅马的画"（答案 necklace）；
- qa104「推荐的书」→ 编"The Power of Now"（答案 "Becoming Nicole"）；
- qa88「期待收养过程的什么」→ 把问题复述一遍当答案。

#### 根因：**顶层概览太像"成品"，把检索短路了**

这些 QA 的共同点是**开头读了一眼 speakers summary 就直接作答**。speakers summary 是
"整段关系的浓缩印象"，里面提过几乎所有主线（家庭、职业、兴趣）。agent 看到问题沾边，
就凭印象凑一条答案 —— 它**有把握**（因为印象里确实提过"家庭背景"），于是不去核对。
浓缩成品 = 一条"看起来够用"的捷径，而捷径一旦存在，模型就会走。

**这不是加规则能修的。** "检索要彻底"这类叮嘱在 8B 上反复失效（§2.8 已多次记录）。
能修的是**类别判断**：既然"给成品"会诱导跳步，那就**不给成品**。

#### 改法：删掉成品，砍成一个朴素 loop

1. **删掉 speakers summary 这一层**（add 回到两步）。开场只给一份**会话清单**
   （`session_N (时间)`，不含任何事实）—— 它是路标，不是答案。
2. **删掉 `next(discard, add)` 上下文块控制工具与整套 stage 机制**，回到**一个 agent loop**：
   `read` / `edit` / `bash` 三个工具 + 一份简短引导（这题该关键词检索还是通篇读，**由 agent
   自己选**）。
3. **唯一新增的引导**是一段 8 行的策略表（§2.1）：关键词检索 vs 通篇读，各给**判据 + 一句做法**，
   **不给流程**。加一句 `write from what you READ, never from what you assume` 针对编造。

#### 效果（同一目录、同一 judge、Caroline_Melanie）

| | 旧（印象判答 + 逐 session） | 新（朴素 loop） |
|---|---|---|
| **judge_accuracy** | 31.6% (48/152) | **59.1% (78/132)** |
| evidence_recall | — | 0.633 |
| unsupported_rate | — | 0.083 |
| token_f1 | — | 0.426 |
| single_hop | 32.9% | **63.5%** |
| temporal | 10.8% | **60.0%**（recall 0.80） |
| multi_hop | 34.4% | **57.7%** |

**检索密度**：旧版全量 **2 次 bash 调用**；新版单目录 **500+ 次**（每个 QA 平均 3-20 步
真检索）。**这就是全部差别**：同样是 8B 模型、同样的语料，只是把它从"看一眼印象就答"
逼回"必须真去 grep"。

#### 修掉的两个 harness 级 bug（与模型能力无关）

跑完新流程后仍有 23 条**零证据** QA 直接失败。逐条看轨迹，其中两个根因是**我们的工具**：

- **`grep "a|b"` 没加 `-E`**：基本正则里 `|` 是**字面竖线**，这条命令永远匹配不到东西、
  返回空，模型据此认定"库里没有"。实测一次全量 **95 次带 `|` 的 grep 有 63 次没加 `-E`**
  （qa2 连查 12 次全是这种空 grep）。**harness 现在自动补 `-E`** 并在观测里说明修正
  （`tools._fix_grep_alternation`）—— 这是"工具该替模型兜住的错"，不是 prompt 能解决的。
- **`edit` 缺 `path`**：8B 写证据时常常只给 `append`、忘了 `path`，旧行为是报
  `path must be a string, got NoneType`，然后它把**整条一模一样的调用重复 10 次**（每次
  被重复检测拒掉），12 步烧光、0 条证据。现在缺 `path` 默认 `evidence.jsonl`。

#### 仍未解决：open_domain（反事实题）从 77% 掉到 25%

诚实记录：这一类（"Caroline 会不会信教""会不会写书"）旧版反而更高（77%）。原因是旧版
凭印象作答时，对这类**没有单一事实、要拿全局印象做推理**的题恰好蒙对得多；新版逼它去
检索，而库里根本没有一条消息直接说"她信不信教"，它检索一圈后要么返回 `null`（证据不足）、
要么给个无锚点回答（qa77 答 "soon"）。**这类题的正确解法是"检索少量关键事实 + 让 answer
阶段从证据推理"，而不是在 search 阶段硬找"那句不存在的话"** —— 属于 §2.1 说的
"哪些判断该留给下游"的问题，待办。但代价是值得的：用它的 4 个点换 temporal +49 点、
single_hop +31 点、multi_hop +23 点。


### 原 §2.17 又一次大改：引入跨会话索引，31.6%→59.1%→等待新数字

原 §2.16 把 search 从"印象判答"救回"必须真检索"（31.6%→59.1%）。但 59.1% 的跑次
（`search_20260930_173758`）逐条看 INCORRECT，**检索已经发生了**（141 条 QA 里 0 条零
bash），瓶颈换成了两处：

1. **时间推理仍不稳**：temporal 类 16 条答错、9 条找到了却**算错**；只有 7/37 条用了
   `date -d` —— 模型习惯把会话日期直接当事件日期写（正是 §2.3 反复强调、却仍会犯的错）。
2. **跨会话联系收不全**：cat1 multi_hop 平均要跨 **2.41 个 session**（最多 7 个），而 agent
   逐 session 读时"读到后面忘了前面"；只有 33/152 条 QA 读过定位层。散落在几次会话里的
   同一件事（"Caroline 为帮孩子做过的事"分散在 4 次会话）总是只收到一部分。

#### 改法：把定位层从"摘要"换成"索引"

**删掉 `session_summaries`，改成 `index.jsonl`**（设计见 §1.1）：逐会话抽 `(主体, 关系, 对象)`
三元组，合并同一关系在不同会话的出现，每条标注它出现在哪些 session/msg_id/源话时间词。

| | 摘要（已删） | 索引（当前） |
|---|---|---|
| 形态 | 一段浓缩成品 | 一条条 (s,r,o) + 来源标注 |
| agent 拿它干什么 | "哦大致讲了这些" → 常不读原文 | "这事在 session 2、8、13 都出现过" → 去读原文核对 |
| 跨会话连接 | 无（每会话独立） | **合并后有**（一条记录带多个 session 号） |

同时按 原 §2.16 的教训继续**做减法**：search agent 不再有 `next`，只剩 `read`/`edit`/`bash`；
引导由"关键词检索 vs 通篇读"两条改为"grep 索引 / grep 原文 / 读整段会话"三条（§2.1）；
system prompt 明确**核心目标**是产出符合证据标准的 `evidence.jsonl`。

（本节待新跑次数字回来补齐效果对比表。）

### 原 §2.18 又一次大改：不做导航层，直接把消息消解成上下文无关

原 §2.17 的 `index.jsonl`（扁平三元组）与 §2.18（原）的 `nodes/edges` 图，都是为"跨会话定位"
**额外造一层导航**：agent 先 grep 指针，再回头读原文。这条路有效，但始终是**两层**——指针
会漏、会压缩、措辞与原文不一致，agent 得在"指针"和"原文"之间来回对。这一版**把导航层整个
去掉**，改成**把每条消息本身改写成自足的**（`add/resolve.py`）：

- **时间消解**：把相对时间锚定成**绝对时间**或**带绝对锚点的相对时间**，粒度严格照源话
  （`last year` → `2022`，不编月日），并标注 `time_kind` = point / range / approx。
- **指代消解**：`she`/`it`/`there` → 真名 / 真实地点，逐消息、喂前文消息作上下文。
- **原话保留**：消解结果进 `content`/`time`，原话进 `source_content`/`source_time`
  （溯源 + re-resolve 幂等的底）。

于是 grep 命中的**就是能回答问题的原料**，不需要指针，也不需要回头对原文；同一实体跨会话
都写成同一个真名，`grep -rin NAME inputs/sessions/` 一次命中它的全部会话——这正是图"边带
session 号"想做的事，但不需要单独一层。

**为什么值得再把图也砍掉**（三条，都是"少一层结构"的收益）：

1. **单一事实源。** 图是原文的**压缩**，压缩就有损；消解是原文的**改写**，无损（原话还在
   `source_*`）。agent 只需读一种东西。
2. **两种检索策略更自然。** 关键词直达（策略①）直接命中自足句；通读（策略②）时每个会话
   文件也自足——不会"读到后面忘了前面"。而图路线下，两种策略都要先经"指针"这一跳。
3. **增量更简单。** 图要"逐会话重抽 + 全局重新合并"；消解只需"重新消解那一个会话文件"
   （读原始视图，天然幂等），没有跨会话的合并态要维护。

代价说清楚：消解是 **O(消息数)** 次 LLM 调用（图是 O(会话数)×逐消息，两者同量级），且
**消解质量直接决定检索质量**——消解错了（把事件时间写成会话时间、把代词解成错的人）就会
固化进语料。所以 prompt 里把三条时间硬约束（自足 / 精度只减不增 / 区分点段大概）写成规则，
并对每条消息都给出**会话时间锚点**（`source_time`）。

（本节待新跑次数字回来补齐效果对比表。）

---

### S15: 优化 atommem 时间处理 + 原子粒度（2026-10-06）

**bad case 分析（search_20261005_235035）**：总体准确率 50.8%，temporal 只有 40%、multi_hop 31.8%。
逐条分析 104 条 INCORRECT，发现三类核心问题：

##### 问题 A: 时间处理过于机械（11/35 temporal 错）

**案例 A1**: 相对时间被硬算成绝对日期，丢失锚点

```
原始: "my school event last week" (9 June 2023)
atommem 产出: time="2023-06-02"
正确答案: "The week before 9 June 2023"
系统答案: "9 June 2023" ❌
```

**根因**: `parse_time_expression` 把 "last week" 用 `date -d` 算成了 2023-06-02，但丢掉了原始表达和锚点。

**案例 A2**: 持续时间信息完全丢失

```
原始: "5 years already!" (9 June 2023)
atommem 产出: time="2023-07-14"
正确答案: "5 years"
系统答案: "0 days" ❌
```

**根因**: pipeline 只处理了事件时间点，"5 years" 这个持续信息完全丢失。

##### 问题 B: 原子抽取粒度不当

**案例 B1**: 复合概念未拆分

```
原始: "Luna and Oliver! They are so sweet..."
atommem 产出: 只有 Oliver 和 Bailey 的独立 atom，Luna 没有
正确答案: "Oliver, Luna, Bailey"
系统答案: "Oliver and Bailey" (漏了 Luna)
```

**案例 B2**: 关键属性被掩盖

```
原始: "tough as a single parent"
atommem 产出: "Caroline is up for the challenge of being a single parent."
正确答案: "Single"
系统答案: "in a committed relationship" ❌
```

**根因**: 只关注动作/状态，"single" 这个关键属性被埋没了；没有独立抽一条 "Caroline is single"。

##### 优化方案（S15）

**方案 A: 改进时间处理（P0）**

1. 新增 `metadata.time_expr` 和 `metadata.time_anchor` 字段保留原始时间表达
2. 修改 `parse_time_expression` 策略：相对时间优先保留 "原表达 (锚点)" 形式，只有精确时点（yesterday）才算成绝对日期
3. 修改 EVIDENCE_STANDARD：优先使用 `time_expr + time_anchor` 而不是硬算的 `time`

```json
{
  "memory": "Caroline talked about her transgender journey at a school event.",
  "metadata": {
    "time": "last week (anchor: 9 June 2023)",
    "time_expr": "last week",
    "time_anchor": "7:55 pm on 9 June, 2023"
  }
}
```

**方案 B: 改进原子粒度（P0）**

1. 集合成员强制拆分：ATOM_EXTRACTION_SYSTEM 规则 6 新增 "Split compound entities"
2. 关键属性显式抽取：规则 7 新增 "Extract key attributes explicitly"
3. 示例补充：添加 "Luna and Oliver" 拆分、"single parent" 属性抽取的示例

**实测效果**：
- ✓ time_expr/time_anchor 正确产出
- ✓ "last week" 保留为 "last week (anchor: 9 June 2023)"
- ✓ "yesterday" 算成 "2023-06-08"
- ✓ prompt 新增集合拆分 + 关键属性抽取规则

**预期提升**: temporal +15-20%（从 40% → 55-60%），single_hop +10-15%（从 62% → 72-77%）。

---

### S16: 从 bad case 逆向的四阶段优化：judge → answer → search → add（2026-10-08）

**起点**：`search_20261008_070515`（atommem pipeline + search agent 全量 152 QA）——
judge_accuracy **57.6%**、token_f1 0.393、evidence_recall 37.1%、unsupported 11.0%。
按用户要求**逆向分阶段**（judge → answer → search → add）逐层修，每层用现有数据回归。

#### 逐条 bad case 的归类（50 条 INCORRECT / 118）

| 形态 | 条数 | 根因 |
|---|---|---|
| **时间坍缩**：evidence/answer 把"X 之前的那周/那天"写成了**会话当天的日期** | 11 | **add 侧**（见下）——不是 search 的错 |
| **集合成员不全/被抽象**：`stained glass window`→`a piece`、`Oliver, Luna, Bailey`→`Oliver and pup` | 12 | add 的**去重把成员名合并没了** + search 早停 |
| **检索不到（unsupported）** | 13 | search 对集合/计数题收不全 |
| **session id 泄漏进正文**：answer `the week before session_3_11` | 2 | evidence 用 session id 当日期锚点 |
| **judge 把集合子集判成 CORRECT** | 4 | judge prompt 自相矛盾（"多给不算错"压过了 COMPLETENESS） |

#### 各阶段改动与实测（`--sample Caroline_Melanie`）

| 阶段 | 改动 | 实测 |
|---|---|---|
| **judge** | 删掉一整段冗余 worked examples（行为中性）；新增 `_resolve_anchor_form`：把 add 侧落盘的规范形态 `"last week (anchor: 27 June 2023)"` 按**相对表达同一套逻辑**折算（旧版只抠出锚点 `27 June`，差一周）。 | 基线重判不变（57.6%，基线答案没有 anchor-form）；修完后 answer 侧 +3pt |
| **answer** | `ANSWER_PROMPT` 大幅精简（~78→~40 行）；时间规则重写：锚定短语**原样保留**、只有精确一天偏移才折算、**禁止把锚点当答案**、**禁止回答里出现 session id/占位符** | 固定 evidence 上 58.5% → **60.2%** |
| **search** | `time_policy` 加两条：**atom.time 等于它自己的会话日期＝红旗**（add 侧可能坍缩过）、**禁止拿 session id 当日期**；`EVIDENCE_STANDARD` 加第 11 条"正文不写 id/占位符" | 单靠 search 修不动时间题（原子本身是错的），必须先修 add |
| **add** | ① `atommem_pipeline.parse_time_expression` 改为**委托**给单一事实源 `atommem_time.parse_time_expression`；② 去重加**专有名词守卫**（见下） | 见下 |

#### add 的两处根因（本轮的**关键**）

**① 时间解析器是坏的（`date -d` 的陷阱）。** pipeline 自己那份 `parse_time_expression` 直接
`date -d "<anchor> <expr>"`，两个实测缺陷都直接造成 temporal 答错：

- `date -d "25 May 2023 last saturday"` → **`2023-05-25`**（GNU date 返回的正是会话日期本身），
  于是 "last Saturday" 被错记成会话当天 —— qa5/qa28/qa29/qa79… 一批题全错在这。
- 粗粒度表述（"last week"）被压成**单日日期**，丢掉锚点，下游只能照抄这个日期。

修正版 `atommem_time.parse_time_expression` 早就写好了（保留锚点、星期几用 Python 精确折算、
精度只减不增），但**只有 agent 模式在用**；默认的 pipeline 走的是自己那份坏的。改为委托后：
`last Saturday`(25 May) → `2023-05-20`、`last week`(9 June) → `"last week (anchor: 9 June 2023)"`。
全量 atommem：`flat_iso` 106→**37**、`anchored` 79→**136**。

**② 去重把枚举成员合并没了。** `_is_same_fact` 的 token-Jaccard（≥0.6）会把
`"Melanie has a pet named pup"` 与 `"...named Luna"` 判成同一事实（Jaccard 0.67）——于是
**Luna / Oliver / Bailey 三条被并进"pup"那条，成员名全丢**（qa52/qa124 的 pet 题死在这）。
修法：**专有名词不同 ⇒ 不同事实**（新 `_proper_nouns` 守卫）。名字正是区分枚举成员的关键，
措辞再像也不能并。修完后 atommem 里 `named Luna / Oliver / Bailey` 三条齐全。

#### 实测（同一目录、同一 judge）

| 指标 | 优化前 | 优化后 |
|---|---|---|
| **judge_accuracy** | 57.6% | **70.9%** |
| evidence_recall | 37.1% | **51.0%** |
| unsupported_rate | 11.0% | **5.5%** |
| token_f1 | 0.393 | 0.395 |

| category | 前 | 后 |
|---|---|---|
| multi_hop | 35.0% | **76.2%** |
| temporal | 57.1% | **72.2%** |
| single_hop | 65.4% | **71.7%** |
| open_domain | 62.5% | 50.0%（n=8→10，样本小） |

**同集合口径**（两跑共有的 106 条 QA）：59.4% → **70.8%**（20 条变好 / 8 条变差）。

#### 诚实记录：本轮的三个教训

1. **judge 的措辞极度敏感，不能当"顺手简化"的对象。** 试着把 JUDGE_PROMPT 重写得更短，
   分数在 **46.6% ↔ 61.9%** 之间来回跳（同一批答案）——8B 对"默认 CORRECT/extra 不算错/
   空答案必错"这些措辞极其敏感。最终**只删了冗余示例块**（行为中性），逻辑一字未动。
2. **测量口径要先校准。** judge 不能读 `"(anchor: X)"` 形态是**真 bug**，但它同时压低了
   基线和新跑——所以要先修 judge，否则 new/old 不可比。
3. **改下游 prompt 修不动上游的数据。** search 的 time_policy 再怎么强调"别用会话日期当事件
   日期"，只要 atom 的 `time` 本身就是会话日期（add 的错），agent 就会照抄。**根因在哪一层，
   就必须在哪一层修。**

---

### S17: 清理死代码 / 过时配置 + atommem 并行化（2026-10-08）

三件事：**atommem pipeline 并行化**、**删确定死代码**、**修过时/重复的 yaml 与一处配置 bug**。

#### atommem pipeline 并行化（效率）

原来是**逐 message 串行**（`run_dir` 的 `for target in targets: await run_message_pipeline(...)`），
每处理一条就把全库重写一遍。但**每条 message 的抽取其实互不依赖**：

- 召回（`recall_context`）只读 `msgmem`，**不需要**"前 N-1 条已产出的 atom"；
- id 按各自 `source_id` 独立分配（`_max_suffix(set(), source_id)` 对全新 source 恒为 0）；
- 去重本就可以是收尾的**一次全局 pass**（保留首次出现，与逐条增量等价）。

改成：全部 message 并发抽取（`asyncio.gather` + `Semaphore(concurrency)`）→ `_dedup_atoms` 一次
→ 落盘一次。**实测 419 条 message：串行 ~45 分钟 → 并行 ~1 分钟**。agent / agent-v2 模式保持原样
（串行/并行各自成立）。同时把管道模式不再使用的 `summem` 读取移到 agent 分支里。

#### 删除的确定死代码（AST 全仓扫描，含 `scripts/` 与 `__all__` 复核）

| 文件 | 删除 |
|---|---|
| `io.py` | `append_jsonl` / `ensure_jsonl` / `resolve_dir` / `default_config_path` / `msg_role` / `msg_kind` / `source_content` / `source_time` / `make_session_record` / `memory_tags` / `memory_speaker` / `dedupe_by_id` / `format_time`（全部 0 引用） |
| `search/tools.py` | `create_write_tool`（`write` 工具早已删除）+ 级联的 `shrink_notice` |
| `search/agent.py` | `steps_for_record` |
| `search/runner.py` | `corpus_files` |
| `search/evidence.py` | `_try_whole_file` |
| `add/memory.py` | `load_memories` / `recent_window` |
| `add/msgmem.py` | `parse_session_datetime` |
| `eval/judge.py` | 同步版 `judge_answer` |
| `llm.py` | `complete_sync` |
| `api/store.py` | `debug_dump` |
| 各文件 | 十余处未使用 import（`json` / `sys` / `Path` / `field` / 未用的 `_TAG_ALIASES` 等） |

`eval/legacy.py` 与 `scripts/eval_baseline|rag|atommem` 是 core.md §6 记录的**历史对照链路**，保留。

#### 修掉的过时 / 重复配置（含一处真 bug）

- **`eval/judge.py` 有两处 `CONFIG_FILE` 赋值**：第一处指向 `configs/eval.yaml`，第二处（模块末尾）
  却指向**并不存在的** `configs/judge.yaml` —— 于是 `load_config` 一直在读一个不存在的文件，
  **`eval.yaml` 的 `judge:` 段被静默忽略**。删掉那行后，`eval.yaml` 的 judge 配置才真正生效。
- **`search.yaml`** 顶部注释写的是"消解后的会话语料（sessions/）"——`sessions/` 早被三层记忆文件
  （`atommem`/`msgmem`/`summem`）取代。重写抬头；冗余长注释压到 `runner.py` 的 `DEFAULT_CONFIG`（单一事实源）。
- **`add.yaml`**：`enable_thinking` **重复出现两次**（YAML 后者覆盖前者）；`tool_config` 键**无人读取**
  （CLI 走 `--tool-config`）——两者都删。注释里"逐 message agent"与默认（pipeline）不符，已更正，
  并把只被 agent 模式读取的键归到一段。
- **`answer.yaml` / `eval.yaml`**：删掉过时的远程模型（openlux / gpt-4o-mini，含明文 key）注释块。
- **`configs/atommem.yaml`**：只被历史 baseline 脚本使用，加抬头注明"历史对照、不参与当前管线"。

#### 验证

`python scripts/test_search.py`（470 项）与 `python scripts/test_api.py`（44 项）全绿；
五个 CLI（add/search/answer/eval/api）`--help` 均可跑；`python -m compileall src/codemem` 通过。

---

### S18: 时间形态对齐参考语料 + answer 风格 + 全量 run_eval（2026-10-08）

用户反馈三点，逐一处理。

#### ① 时间答案要**照参考的形态**，不要折算成算出的日期

实测（`search_20261008_090757`）：qa28/qa29 参考是 ``"The Friday before 15 July 2023"``，而系统答
``"2023-07-14"`` —— **算对了，但形态不对**。问题在于：

- judge 未必能复核日期算术（§2.3 早已反复踩）；
- token F1 直接受损（两个答案 token 几乎不重叠）；
- 参考语料里 "when" 类答案**绝大多数本身就是** ``"the week/day/weekend before <日期>"`` 这种形态
  （全库 72 条 anchored-relative，见下）。

**改法：** ``atommem_time.parse_time_expression`` 不再把相对表达折算成绝对日期，而是产出**与参考
同形的锚定表述**：

| 源（会话时间锚点） | 旧产出 | 新产出 |
|---|---|---|
| "last Saturday"（25 May 2023） | `2023-05-20` | `the Saturday before 25 May 2023` |
| "last week"（9 June 2023） | `2023-06-02` | `the week before 9 June 2023` |
| "Last Fri"（15 July 2023） | `2023-07-14` | `the Friday before 15 July 2023` |
| "last weekend"（17 July 2023） | `2023-07-15` | `the weekend before 17 July 2023` |
| "yesterday"（8 May 2023） | `2023-05-07` | `7 May 2023`（非 ISO） |
| "last night"（14 August 2023） | `2023-08-13` | `13 August 2023` |
| "two days ago"（12 July 2023） | `2023-07-10` | `10 July 2023` |
| "last year" | `2022` | `2022`（不变） |

绝对日期统一写成 ``"7 May 2023"`` 形态（``date -d ... +"%-d %B %Y"``），不再写 ISO。
另补两处漏洞：数词（"two days ago"）与星期缩写（"last Tues"）之前识别不了。

**answer 侧同步**（``prompts.ANSWER_PROMPT``）：显式禁止把 "the Friday before X" 替换成算出的日期
（"2023-07-14" 是更差的答案）；日期写 ``"7 May 2023"`` 而非 ISO；并要求**答案要短**
（短语而非整句，不复述问题）——参考语料很多答案就是一个短语（"mental health" / "Sweden" / "5 years"），
贴风格既提升 token F1 也更容易判对。

**实测（10 条 temporal QA）**：改前多数答成算出的日期或会话日期；改后 **10/10 与参考逐字同形**：

```
qa0  7 May 2023                          ✓
qa5  the Saturday before 25 May 2023     ✓
qa8  the week before 9 June 2023         ✓
qa28/29 the Friday before 15 July 2023   ✓
qa41 the Tuesday before 20 July 2023     ✓
qa44 13 August 2023                      ✓
qa54 the week before 23 August 2023      ✓
qa74 the weekend before 20 October 2023  ✓
qa79 the Friday before 22 October 2023   ✓
```

#### ② open_domain（LoCoMo cat3，反事实题）偏弱

实测 37.5%（n=8）。逐条看：一半是**弃答**（"Would Caroline be religious?" → null），一半是**方向答反**
（"Would Melanie enjoy Vivaldi?" 参考 "Yes; classical music"，答 "Unlikely"）。改法是把
``ANSWER_PROMPT`` 的 counterfactual 段写死：**"would X" 题除非语料对 X 一无所知，否则禁止弃答**；
要从 X 已显露的偏好/价值**判断方向**（喜欢/做过 → Yes/Likely yes；明确拒绝/冲突 → No/Likely no），
不要默认 "Unlikely"；输出短句 "<Yes|No|Likely …>; <一句理由>"（与参考同形）。

#### ③ `run_eval.py --limit 0` 只跑了 Caroline_Melanie

根因：``scripts/run_eval.py`` 的 ``--dirs`` **默认值写死了** ``["Caroline_Melanie"]``。改为默认
``[]``（= 全部含 ``msgmem.jsonl`` 的目录），并在文档里说明 ``--limit 0`` 才是全量。

#### ④ add 并行的**全局**并发闸门（顺带修的）

上一版把 atommem 改成整目录并行后，``async_main`` 的"目录级 semaphore" + ``run_dir`` 的"message 级
semaphore"是**两层独立**的 → 跑 10 个目录会有 ``concurrency × 10`` 个并发请求打在模型端点上。
改成：一个**全局** semaphore，所有目录、所有 message 任务共用（单一上限 = ``concurrency``）。

#### 验证

``scripts/test_search.py``（478 项）与 ``test_api.py``（44 项）全绿；``parse_time_expression``
的 8 个新模式各有断言；all-dirs 重新生成 atommem 走通。跨全部 10 个目录的全量评测留给
``python scripts/run_eval.py --limit 0``（现在会真的跑全部目录）。

#### S18 实测（Caroline_Melanie 全量，`search_anchorfix_20261008_100103`）

| 指标 | 改前（090757） | 改后 |
|---|---|---|
| **token_f1** | 0.408 | **0.473** |
| **open_domain** | 37.5% | **75.0%** |
| evidence_recall | 48.4% | 52.3% |
| unsupported_rate | 8.9% | **2.3%** |
| judge_accuracy | 69.1% | 66.4% |
| temporal | 76.5% | 70.3%（**定向 10 条 temporal 全对**，见上表）|
| multi_hop | 71.4% | 52.4% |
| single_hop | 68.3% | 67.7% |

**同集合口径**（两跑共有的 113 条）：judge 69.0% → 68.1%（**基本不变**）；分项
open_domain 0.50→0.67、single_hop 0.67→0.69、temporal 0.76→0.76、multi_hop 0.67→0.50。

**诚实说明 multi_hop 的落差**：它**不是**这几处 prompt 改动造成的，而是 search 的逐跑随机性
（temp 0.2）—— 两次运行 **113 条里有 89 条答案不同**。抽查 qa37（"what did Melanie paint
recently?"，参考 "sunset"）与 qa24（"ways to destress"，参考 "Running, pottery"）：新跑的
**evidence 里就没有** sunset / pottery（search 没检索到），answer 无米下炊，与 answer prompt 无关。
answer 侧本身也有 ±2~3pt 的方差（同一 evidence、只改 STYLE 措辞，multi_hop 在 47.6%~57.1% 之间跳）。

**结论**：锚定时间形态（+token F1）与 counterfactual 强化（+open_domain）是**确定的正向**；
judge 整体在噪声内持平；STYLE 段保留"先给完整答案"的写法以规避集合题漏成员。

---

### S19: S18 的一次回归及其根因（2026-10-08）

S18 落地后用户全量复测：judge **69.1% → 61.7%**，且 `reference="2022"` 答成 `8 May 2023`、
`"June 2023"` 答成 `next month`、`"July 2023"` 答成 `this month` —— 明显回归。逐条查证后是**两处**
改动叠加，都定位到了具体机制。

#### 根因 1：`EVIDENCE_STANDARD` 第 3 条被改写，把 search agent 带偏

S18 把第 3 条从
> "Keep a relative phrase AS WRITTEN **and attach its anchor when a reader ... would need it**: '<person>
> did X 'last week' -- i.e. the week before <session date>'"

改成
> "Keep a relative phrase AS WRITTEN **with its anchor**: 'last week' -> 'the week before <session date>'"

措辞看着等价，但 8B 的实际行为变了：它开始**照抄 msgmem 的原始相对词**（`last year` / `next month`），
而不是照抄 atommem 里**已经解算好**的 `time`（`2022` / `September 2023`）。**对照实验**（把旧 atommem 换回来、
prompt 不变）证明这不是语料问题：qa1 在新旧 atommem 下都答错 → 是 **prompt** 的锅。

**修法：把第 3 条回退到 S18 之前的措辞**（保留 "attach its anchor when a reader would need it" 的原句）。

#### 根因 2：`atommem_time` 把月份写成 `the month after 28 August 2023`

S18 让相对表达产出"与参考同形"的锚定表述，但**月份的参考形态是月份名**（`June 2023` / `September 2023`），
不是 `the month after X`。更糟的是 judge 的 `resolve_relative_date` 只认 `the <unit> **before** X`
（不认 `after`），于是 `the month after 28 August 2023` 既非参考形态、judge 也解析不出。

**修法：`last/this/next month` 直接产出月份名 + 年份**（`_shift_month`）——`next month` @ 28 Aug → `September 2023`。
week/weekend/weekday 仍保留 `the <unit> before <anchor>`（那正是参考形态）。

#### 实测（Caroline_Melanie 全量）

| 运行 | 状态 | judge |
|---|---|---|
| `search_090757` | S17（改动前） | 69.1% |
| `search_123213` | S18（回归） | 61.7% |
| `search_fix2` | S18 + 这两处修复 | **72.3%** |

**同集合口径**（fix2 vs 090757 共有的 115 条）：68.7% → **73.9%**；分项 temporal 0.76→**0.85**、
single_hop 0.67→**0.74**、multi_hop 0.67→0.61（噪声）、open_domain 0.50→0.50。

#### 教训

- **search agent（8B）对 prompt 措辞极其敏感** —— 一句"看着等价"的改写就能让它整体换一种行为。
  任何 search prompt 改动**必须全量跑一次**才能宣布有效，不能用"读起来更清楚"来判断。
- **"与参考同形"要按每条参考的实际形态定**：日期是 `7 May 2023`、周是 `the week before X`、
  **月是 `September 2023`** —— 不能一刀切。
- judge 的解析能力是"答案形态"的硬约束：产出 judge 认不出的形态（`the month after X`）等于答错。

> **另一处试过但**没生效**的修法（记录以免重试）**：qa6/qa17 这类"evidence 里就是一个裸相对词
> （`next month` / `this month`）"的情况，试过在 answer prompt 里加"用该条记忆里的任何日期把它解析出来"
> 的规则 —— 实测**不生效**（8B answer 仍原样输出 `next month`），且 0.723 → 0.708 轻微拖累，已回退。
> 要根治只能在 **search 侧**让 agent 产出带锚点的 evidence（风险高，需整轮复测）。

---

### S20: 从 Deborah_Jolene 全量 bad case 反推的一轮审计（2026-10-13）

**起点**：用户报的一组数（`search_20261013_022233`，Deborah_Jolene 全量 165 QA，gpt-4o-mini）——
judge_accuracy **55.6%**、evidence_recall 32.5%、unsupported 11.5%、token_f1 0.408；
失败归类 wrong_answer 52 / missing_evidence 19 / judge_error 5。

#### 先量噪声：同一份代码跑两遍能差多少

`search_baserep_20261013_030406` 是**同配置重跑**：judge 53.7%（-1.9 点）。逐条比对，
**165 条里有 27 条的判对/判错翻转**。所以本节所有结论都以**多次重跑的均值**为准 ——
单次差 2 个点不构成证据。下面每个配置都跑了 3~4 次。

#### 逐条 bad case 归因（71 条非 judge_error 的错例）

先按"错在哪一层"分类：以参考答案的实词能否在 **atommem** 里找到为准。

| 形态 | 条数 | 含义 |
|---|---|---|
| **答案就在 atommem 里** | **42** | 纯检索行为问题 —— 能找，没去找 |
| 全语料都没有（含只在图片里的多模态题） | 28 | 语料里确实没有 |
| 只在 msgmem 里 | 1 | 原子层漏抽了这条 |

72% 错在检索行为上，继续拆"为什么没找到"：

| 现象 | 量 | 证据 |
|---|---|---|
| **第一步是 `grep -rin "<说话人>" inputs/`** | **90/165 条 QA** | 说话人几乎出现在每条记录里 → 命中几百行、观测被截断；`broad_grep` 提示全量触发 **81 次** |
| **从没 grep 过参考答案里那个区分性实词** | 35 条 | qa118 问"Deborah 跟猫做什么"，全程 grep `activity:` 而**从没 grep 过 `cats`** |
| **疑问句 atom 被当成答案** | 原子层 **6.4%** 是 `"X asks about Y"` | qa88 命中 `session_5_4_3 "Deborah asks about the accomplishments…"` 就收工，而回答那句（session_5_5）**原子层压根没抽出来** |
| **同域名实体只找到"一个"就停** | ≥8 条 | 游戏题（qa117/125/147/154）、瑜伽姿势题（qa22）：语料里有 5 个游戏名，命中哪个答哪个 |

#### 一个不属于"模型不听话"的机制：命中顺序

```
qa62  "When did Jolene and her partner return home from Rio de Janeiro?"
  grep -rin "Rio de Janeiro" inputs/   -> 4206 字符；命中顺序 msgmem, summem, atommem
  atommem（正确）: session_23_1_1  time = "29 August 2023"
  summem （误导）: "On 30 August 2023, Jolene shared ... had returned from a trip ..."
  模型写的:        "returned home from Rio de Janeiro on 30 August 2023"   ❌（抄了会话日期）
```

`grep -r inputs/` 的命中顺序取决于 **readdir**（实测同一个目录里换一个 pattern 顺序都能变），
而 `summem` 是**整段会话总结**：实测单条 4KB，七条就能把 8000 字符的观测预算吃光，
`atommem` 里那些**时间已经消解好**的记录一条都轮不到被看到。qa19（atom 明明写着 `2023-01-28`，
模型却把 `next Saturday` 自己算成 4 February）、qa27（summem 说 "mini retreat on 1 February"，
atom 说 "the Wednesday before 9 February"）、qa42 全是这个形态：**先看到哪条，就用哪条。**

#### 改动（五条）

| # | 位置 | 改动 |
|---|---|---|
| 1 | `search/tools.py` | **语料 grep 的命中按记忆层重排**：`atommem -> msgmem -> summem`。只重排、不删改；同层稳定排序；只在"每行都属于这三层、至少命中两层、未被截断"时动，否则原样返回。实测同一 pattern：开头 8000 字符从"6 条 summem 长段"变成"15 条 atom + 7 条 msgmem" |
| 2 | `audit.py` | **引号里的专名不再漏检**。证据标准本来就鼓励引用源话（`recommending 'Stardew Valley'`），而 `fullmatch("[A-Z][a-zA-Z]{2,}")` 要求整词恰好是大写开头的字母串 —— 带引号就匹配不上，**编造专名里最该抓的那类整类漏掉**。剥掉词首/词尾非字母字符后，同一批 1020 条 evidence 的命中数 24 -> 29 |
| 3 | `audit.py` | **新增"抄了消息时间戳当事件日期"检查**：被引消息原文里有**精确天偏移**（yesterday / "N days ago" / last night…），而记录写的日期**正好是那条消息自己的会话日期** —— 事件必然发生了偏移，记录里那个日期只能是"说这话的时间"。判据全确定，而且**把正确日期算出来**直接告诉 agent（"the day before yesterday" @ 26 June -> 24 June）。单次 run 上命中 13 条 / 9 个 QA，抽查全是真阳性（qa10/33/43/47/62…，正是前面测出的那一批"差一天"错） |
| 4 | `tool.json` strategy / time_policy | ① 关键词取问题里**最稀有的实词**，明确**不许拿说话人名字当关键词**；② **疑问句 atom 是路标不是答案**；③ 停下前确认命中的是**问题问的那个**实体；④ **atom 的 `time` 就是结果，别自己重算**（例外仍是"time 等于自己会话日期"这个红旗）；⑤ `next <weekday>` = 消息当天之后**第一个**该星期几 |
| 5 | `add/atommem_time.py` | 两个**静默失效**的精确折算：`"yesterday morning"` 被 `_normalize_expr` 变成 `"-1 day morning"`，`date -d` 直接解析失败 → 退回"保留 + 锚点"（qa43 的 yoga 因此错成 6 June）；`"the day before yesterday"` **压根不在偏移表里**（qa47 差两天）。两条都补上，并让长表达优先替换（否则 `the day before yesterday` 只剩 `-1 day`）。全库 12806 条 atom 里命中 11 条；**要下次 `add` 才会落进语料** |

配套：`test_search.py` 从 449 项加到 **476 项** —— 新增 §16 audit（这一段此前**一条测试都没有**，
而它治的是"编造事实配假 source"这类最贵的错）与 §17 命中重排（钉住"只换顺序、不增删内容、
非语料输出一字不动"）。system prompt 预算 21000 -> 22800（有意增长，注释写明来源）。

#### 实测（同目录、同 165 QA、gpt-4o-mini；每个配置 3~4 次重跑）

| 配置 | judge_accuracy（逐次） | **均值** | evidence_recall | sufficient |
|---|---|---|---|---|
| A 改动前 @12 步 | 55.6 / 53.7 / 51.5 / 53.4 | **53.6** | 32.7 | 0.32~0.40 |
| B 仅 prompt（#4）@12 | 53.7 / 52.8 / 54.9 | **53.8** | 34.0 | 0.22~0.28 |
| C B + 命中重排（#1）@12 | 50.9 / 54.9 / 54.3 / 51.5 | **52.9** | **37.8** | 0.26~0.35 |
| D C + 步数 12→24 | 59.3 / 58.6 / 54.0 | **57.3** | 40.6 | **0.52** |
| **E D + 抄锚点检查（#2/#3）** | 55.5 / 59.0 / 59.8 | **58.1** | **42.4** | 0.45~0.55 |

A → E：judge **53.6% → 58.1%（+4.5）**，evidence_recall **32.7% → 42.4%（+9.7）**。
（合并标准误约 1.57，差 4.5 ≈ t 2.9；n 只有 3~4，属于"方向明确、幅度待更大语料复核"。）

**三条诚实的结论：**

1. **这一轮最大的单点收益是"把评测的步数上限调回配置该有的值"，不是任何 prompt 技巧。**
   `configs/search.yaml` 写的是 `max_steps: 30`，但 `scripts/run_eval.py` **默认把它压到 12**
   （那是"快反馈"时代的遗留），而**用户报的那组数字全部来自 12 步的 run**。放到 24：
   judge 52.9 → 57.3、sufficient 32% → 52%、missing_evidence 23 → 18。步数均值只从 10.0 涨到
   16.5（agent 一般自己会停），代价是模型调用 +65%。**已把 `run_eval.py` 的默认值改成 24**
   （注释写明来历），`--quick` 仍是 8。**这条不改代码，只改一个默认值。**

2. **prompt 那五条规则机制上生效、分数上没动。** 行为确实变了：第一步就是说话人名 grep 的
   QA 从 **90/165 掉到 24/165**，说话人名 grep 占比 22% → 14%，`read` 从 93 次升到 106 次。
   但 judge 均值 53.8 vs 53.6 是**噪声内的持平** —— 省下来的那一步没有被用在更深的检索上。
   这几条保留下来是因为它们各自指向一条确凿的失败形态（且不占多少预算），不是因为分数。

3. **命中重排换来的是 recall，不是准确率。** recall 32.7 → 37.8（四次重跑**每一次**都高于
   基线对应值），judge 却在噪声内略降。也就是说**瓶颈已经不在"检索覆盖"**：C 配置里
   "参考答案实词出现在 evidence 里、答案仍错"的 QA 从 15.3 涨到 20.3 条 —— 多检索到的事实
   没有转化成正确答案，压力转移到了 answer 与判定一侧。**这是下一轮该动的地方**
   （也是 原 §2.16 末 N0 那条的同一个方向）。

> **样本量提醒**：Deborah_Jolene 只有 165 QA，单次 judge 的标准差约 **2 个点**；上面的
> "均值"是 3~4 次重跑出来的。要判定 1 个点级别的差异，得继续加重复或换更大语料
> （`--limit 0` 跑全部 10 个目录，约 1300 QA）。**跨配置比较时不要拿单次数字对齐。**

---

### S21: 清理 + prompt 瘦身一轮（2026-10-13）

**起点**：用户报的 `search_20261013_083410`（Caroline_Melanie 全量 138 QA，gpt-4o-mini）——
judge **63.8%**、evidence_recall 41.1%、unsupported 9.4%、token_f1 0.419；
其中 **multi_hop 只有 29.2%**，是最差的一类。

#### 先做的不是改 prompt，是逐条看错在哪一层

138 条里挑出 multi_hop 的 17 条错例与 single_hop 的 19 条错例逐条读，失败形态收敛成两类：

| 形态 | 典型例子 | 该改哪儿 |
|---|---|---|
| **检索到了，但被写成了类别** | 参考 "a cup with a dog face on it" → 证据写"pottery"；"Trans Lives Matter" → 证据写"posters featured artistic expressions" | search 的证据标准 |
| **一条事实换十几种措辞重复写** | 某条 QA 写出 16 条证据，多数是同一件事的改写，"真正回答问题的记录被埋在中间" | search 的记录纪律 |
| **答了相关的事实，不是问的那件** | 问"在会上看到了什么" → 答了会议的**日期** | answer 的 prompt |

注意第一类的代价**不在检索，而在写法**：`evidence_recall` 有时是 1.0（该找的找到了），
但写出来的记录抽象掉了具体值，下游就答不出来。所以这一轮的改动集中在**"什么算一条合格的
证据"**与**"答问题问的那件事"**，而不是检索策略。

#### 改了什么

**search 侧**（`prompts.py` 的 `EVIDENCE_STANDARD` + `configs/tool.json`）：

- EVIDENCE_STANDARD 新增**第 12 条「NEW INFORMATION ONLY」**：每条记录必须带来新事实，
  禁止换措辞重复（对应上表第二类）。
- 第 8 条 SPECIFICS PRESERVED 加重：明确指出这是"一条记录不再能回答问题的**最常见**方式"。
- `strategy` 开头新增**「Write the SPECIFIC, never the category」**，并给出实测原话作反例。
- `writing_policy` 的记录条数从 "5-15 条" 收到 **"3-8 条"**。

**answer 侧**（`ANSWER_PROMPT`）：

- 新增 **`# SPECIFICITY`**：具体细节不得抽象成类别，配三个实测反例；
  并写明"what/which/who/where/why 的问题不得用日期或时间短语回答"（对应上表第三类）。
- 新增 **「Answer the fact the question ASKS FOR」** 与 **「Abstain LAST, not first」**
  （实测有 5 条 `evidence_recall=1.0` 的 QA 被 answer 判成 unsupported）。

**同时瘦身**：system prompt **22632 → 21053 字符（−7%）**。删掉的是重复：
COMMANDS 的两条空转配方、WORKSPACE 与 `user_begin` 重复的三层语料描述、
TIME_POLICY 里同一规则说四遍、EVIDENCE_STANDARD 每条的长解释。
`scripts/test_search.py` 的上限断言从 22800 收到 **21500** —— 新增规则前先想想能不能删一条旧的。
（同时把那段 16 行的"阈值历史沿革"注释换成 2 行。）

#### 实测（Caroline_Melanie 全量 138 QA，同一模型 gpt-4o-mini，跑两遍）

| category | 基线 | 新#1 | 新#2 | 新均值 | Δ |
|---|---|---|---|---|---|
| **multi_hop** | 0.292 | 0.542 | 0.625 | **0.583** | **+29.2 点** |
| temporal | 0.667 | 0.667 | 0.694 | 0.681 | +1.4 |
| open_domain | 0.778 | 0.667 | 0.444 | 0.556 | −22.2（n=9，噪声） |
| single_hop | 0.725 | 0.652 | 0.681 | 0.667 | −5.8 |
| **整体** | 0.638 | 0.638 | 0.659 | 0.649 | **+1.1** |
| **token_f1** | 0.419 | 0.447 | 0.449 | **0.448** | **+0.029** |

**三条结论：**

1. **multi_hop 的提升是可复现的，其余都在噪声内。** +29 点、两遍同向（+25 / +33）、
   逐条看有 3 条 QA 在**两遍里都**由错转对（q11 / q42 / q45）、没有一条反向 —— 方向明确。
   single_hop 的 −5.8 点按 n=69 算标准误约 7.6 点（0.76σ），**不构成证据**：
   两遍里"都错"的只有 6 条、"都对"的 5 条，净 −1 条，剩下的都是两遍之间自己翻转的噪声。
   open_domain 只有 9 条，两遍从 0.667 摆到 0.444，不用读。
2. **answer 侧的 prompt 改动基本不动分数 —— 这是单独测出来的。** 用**同一份基线 evidence**
   只重跑 answer（新 prompt）再评一次：gained 1 / lost 2，且两条"lost"是把原答案换了个说法
   （q89 的 `hope for her own path and felt connected` vs `felt connected and gained hope`）。
   所以 answer 侧那三条新规则**保留是因为它指明确凿的失败形态，不是因为分数** ——
   和 §S20 结论 2 是同一个现象：这一层已经不在瓶颈上。
3. **单跳题上"证据越多越容易答错"，这次的数据又一次指向它。** 三次 run 合起来按证据条数分桶：

   | 证据条数 | 1-4 | 5-8 | 9-12 | ≥13 |
   |---|---|---|---|---|
   | single_hop 正确率 | 67% | **75%** | 68% | 65% |

   5-8 条是甜点，≥13 条反而掉回 65%。这与 S20 结论 3（"瓶颈已不在检索覆盖"）一致，
   而 `evidence_soft_limit` 现在还是 12、prompt 却写"3-8 条"—— **两者不一致，下一轮该对齐**。
   （这一轮没动它，因为它是配置改动、会同时影响 multi_hop 的召回，值得单独跑一次 A/B。）

#### 这一轮顺带做的清理

删掉 7 个孤儿文件（`scripts/monitor_eval.sh` 里写死 `/root/autodl-tmp/...` 的路径、
`locomo_data_loader.py`、DPO/训练链路的 `regenerate_rejected.py` /
`transfer_data_to_msswift_standard.py` / `train_orpo.sh` / `judge_impact.py`、
被 `memory_template.jsonl` 取代的 `memory_template.json`），
删掉 5 个从未被调用的死方法 + 4 个 legacy 死函数 + 一批重构遗留的别名与死配置键
（`add_tool_config`、add.yaml 的 agent 模式段），
并把 `docs/search.md` 里还写着 `inputs/sessions/session_N.jsonl` 那一版架构的段落改到当前形态。

**文档本身也拆了一次。** `core.md` 原本把「现在的设计」和「每一次迭代的实测与归因」混在
一份文件里，读设计的人要先翻过几百行历史。现在按「这份内容的**半年后还成立吗**」划线：
`core.md` 只留**仍然成立**的设计与契约（2275 → 1136 行），过程记录 —— 逐轮实测数字、
失败归因、被取代过的设计、里程碑与待办、运行手册 —— 全部移到本文件（1187 行）。
搬家之后第 2 章留下了编号空洞（原 §2.1 早就缺号，原 §2.12 是这次搬走的），顺手把整章
重编成连续的 §2.1–§2.12，散在两个文件里的 `§2.x` 引用全部跟着改号并逐条核对过。

---

## S22: 先修测量，再修行为 —— 两个静默丢数据的 bug（2026-10-13）

**起点**：用户要求"每个 category 都达到高正确率且稳定、抗噪声扰动"。在动任何 prompt 之前，
先把**噪声本身**量出来，结果先撞上两个与模型能力无关的 bug。

这一轮把目标模型换成了**本地 Qwen3-8B**（用户定的）。**注意：S15–S21 的全部数字都是
gpt-4o-mini 的，两者不可直接比较**。

### 先把噪声拆开（同一份产物重跑，不改任何东西）

| 噪声源 | 逐条翻转率 | 结论 |
|---|---|---|
| judge（同一份 answers 评两遍） | **0.7%** | 可忽略 —— judge 不是不稳定来源 |
| answer（同一份 evidence 答两遍） | **2.2%** | 可忽略 |
| search（同配置跑两遍，逐条 evidence） | **15.2%** | **不稳定几乎全在这一层** |

这个分解否掉了一个直觉：以前怀疑 judge 抖（S20 记过"单次标准差约 2 个点"），实际上
**judge 稳得很**，分数摆动来自 search 的随机性 + 小样本。

### Bug 1：评测会**静默改变分母**（最严重）

Qwen3-8B 在 Caroline_Melanie 全量上，138 条里有 **18 条**（13%）烧完两轮 verifier、
一个字没写就收工 —— `qa_{idx}/evidence.jsonl` 根本不存在。而 `answer.collect_jobs`
遇到"找不到 evidence.jsonl"时是 `continue`：**这 18 条直接从评测里消失了**。

后果不是"少了 18 条"，而是**分母从 138 变成 120，准确率被虚高**：同一份结果，
按 120 条算是 **67.5%**，把 18 条补回分母算是 **58.0%** —— 差了 **9.5 个点**。
更糟的是它**按模型变化**：gpt-4o-mini 一条都没缺（138/138），Qwen 缺 18 条 ——
于是"两个模型横向比"这件事在这个 bug 下根本不成立。

**修法**：`answer.py` 不再跳过，把缺失当成"证据为空"照常入分母、照常计分。
**评测里静默改分母比答错危险得多**，因为它不报错、还能让结果看起来更好。

### Bug 2：模型写对了，harness 把调用丢了

18 条空产物的轨迹逐条看，最后一步都是这样：

```
step tool=None  reply='{"tool": "edit", "args": {"append": "{\"content\": \"Caroline is ...'
```

`parsed: False`。但这条 reply **本身是完整合法的 JSON 对象**，只是末尾多吐了一个 `}`：

```
...0.5}}\n"}}}
        ↑ 需要 2 个，模型给了 3 个
```

`json.loads` 报 `Extra data`，`_first_json_object` 的兜底正则只支持两层嵌套、也匹配不上，
于是**整条工具调用被丢掉** —— 那条记录永远没写进 evidence。

统计 fixB 那一轮：**31 步**栽在这里，其中 **28 步是 `edit` append**，
`json.JSONDecoder().raw_decode` 能救回 **25 步（81%）**。

**修法**：`_first_json_object` 在严格解析之后，加一层 `raw_decode` ——
"从某个 `{` 起解一个完整 JSON 文档，后面的尾巴不管"。它**不会**把非法 JSON 修成合法
（未转义引号、截断的 JSON 仍然被拒，两条都补了断言），所以原先那条"宁可判成没有工具调用，
也不执行一条猜出来的命令"的安全边界没有被吃掉。

### Bug 3（机制）：空产物也能收工

上面 18 条的共同形态是"**读了一堆、一个字没写、回复纯文本收工**"。prompt 里明明写着
必须写，`write_nudge_after` 也插过提醒 —— 但那是**劝**。实测劝不动。

所以在机制层加了 `max_empty_finish_refusals`（默认 2）：**产物为空时，拒绝这次完成信号**，
注入一段"现在就把已经读到的写一条"的提示（带可直接照抄的 `edit` 调用），拦满次数才放行。
拦满放行是必须的 —— 否则模型在这里死循环，而空产物至少还能被记一条失败。

配合 Bug 2 的修法之后，**空产物从 18/138 降到 0/138**。

### 另外两条 prompt 规则（同一轮，来自逐条 bad case）

- **前提缺失 ≠ 答案缺失**（`ANSWER_PROMPT`）。实测 qa67：evidence 原文就是答案
  （"Melanie realized that self-care is important."），模型却因为**问题里提到的
  "charity race" 在 evidence 里没有**而弃答。补一条：证据不必复述问题的框架，
  只有"被问的那个事实"必须在。
- **不许把不相关的事实并进一条记录**（`EVIDENCE_STANDARD` 第 7 条）。实测 qa69 的
  evidence 一条里同时写了"researching adoption agencies"和"planning to go camping"，
  下游挑了后者，参考答的是前者。一条记录两个事实 = 下游可以挑错那半。

### 实测（Caroline_Melanie 全量 138 QA，Qwen3-8B，同一 judge）

| run | n | 整体 | multi_hop | temporal | open_domain | single_hop |
|---|---|---|---|---|---|---|
| base1 | 138 | 0.580 | 0.458 | 0.694 | 0.333 | 0.594 |
| base2 | 138 | 0.572 | 0.375 | 0.667 | 0.556 | 0.594 |
| **base 均值** | | **0.576** | **0.417** | **0.681** | **0.444** | **0.594** |
| fixB | 138 | 0.623 | 0.458 | 0.750 | 0.556 | 0.623 |
| fixAll | 138 | 0.630 | 0.333 | 0.722 | 0.444 | 0.710 |
| fixAll2 | 138 | 0.652 | 0.583 | 0.639 | 0.556 | 0.696 |
| **fix 均值** | | **0.635** | **0.458** | **0.704** | **0.519** | **0.676** |
| **Δ** | | **+0.059** | **+0.042** | **+0.023** | **+0.074** | **+0.082** |
| **fix 内部波动** | | 0.029 | **0.250** | 0.111 | 0.111 | 0.087 |

（base1/base2 是**同一份代码**重跑两遍；fix 是 fixB / fixAll / fixAll2 三次。
全部按 Bug 1 修好之后的口径统计，138 条都在分母里。）

**四条诚实的结论：**

1. **整体 +5.9 点，且每个 category 的均值都涨了** —— 但**涨幅最稳的是 single_hop（+8.2）**，
   因为它是最大的一个类（n=69），三次重跑依次 0.623 / 0.710 / 0.696，方向一致。
2. **multi_hop 与 open_domain 还**说不了**。** fix 组内部 multi_hop 从 0.333 摆到 0.583
   （**波动 25 个点**），open_domain 从 0.444 摆到 0.556 —— **这两类的 run-to-run 波动
   比要测的效果还大**。n 只有 24 和 9，单条 QA 就值 4.2 / 11.1 个点。要判定它们，
   必须上全量语料（10 个目录 1382 条，multi_hop n=213、open_domain n=68）。
3. **"高正确率"目前只对 temporal（0.704）与 single_hop（0.676）成立**，
   multi_hop（0.458）与 open_domain（0.519）还是最弱的两类。
4. **"抗噪声"这一轮真正拿到的是"少丢东西"**：空产物 18 → 0、弃答率 20.0% → 5.1–13.0%、
   每轮救回 25 条被丢掉的工具调用、分母不再说谎。这些是**确定性的**改善，
   和上面那些还在噪声里的分数不是一回事。

---

## S23: 平台报文形态 —— 88% 的消息"说话人"是丢的（2026-10-14）

### 起因

平台的数据是**多个数据集混着灌进来的**。抽查 `/add` 报文发现同一个 `role` 字段承载不了它要
承载的信息：

```json
{"role": "user", "timestamp": 1704153600000,
 "content": "[beam_100k][session_1][D1:21][text] Christina Baker: I'm trying to implement …"}
```

### 普查（`scripts/analyze_api_formats.py`，全量快照）

| 事实 | 数字 |
|---|---|
| 有 `requests.jsonl` 的 user / add 请求 / search 请求 | 100 / 624 / 216 |
| message 总数 | 7268 |
| 其中协议 `role == "user"` | 6396（**88%**） |
| 正文前缀带**真名**的 add | 320 / 624（**51%**） |
| 记账标签排列 | 无标签 215、`[Time: …]` 170、`[数据集][session_N][Dx:y][text]` 125、`[…][dialogue]` 50、`[…][narration]` 45 |

八种形态：`dialogue` 340 / `plain` 170 / `narration` 50 / `triples` 26 / `qa-feedback` 20 /
`trajectory` 8 / `document` 5 / `facts` 5。

### 两个会静默毁掉检索的故障

1. **说话人丢失**。旧代码把协议 `role` 原样写进 `msgmem`，于是 atommem 全是
   `"user is trying to implement city autocomplete …"`。而平台问的是
   `"What fields would Caroline be likely to pursue?"` —— **这个名字一条原子记忆里都没有**，
   agent grep `atommem`（首选语料）整个落空，只能退到 `msgmem` 里那句被标签裹着的原话。
   **一半的语料是这种情况。**
2. **记账标签进正文**。`[beam_100k][session_1][D1:21][text]` 让 `grep "session_1"` 命中全库，
   还在正文里塞了第二份与 `source` 对不上的 id。
3. （附带）**`[Time: Sep 04, 2025, 09:05:32]` 这一类语料 `timestamp` 是 `None`** ——
   整份语料 `msgmem.time` 是空串，**没有任何时间锚点**，`date -d` 折算无从谈起。

### 做了什么

新增 `src/codemem/adapters.py`：**只做无损换位**（平台编码 → `role` / `time`），摘掉平台记账
标签，**正文一字不改**。接在 `api.store.append` 建 msg 记录之前。

下游跟着改了三处（这是它必须做在 add 之前的原因）：

- `msgmem.role` 变成真名 / 剧本角色 / `corpus` / `document` / `narration` / `feedback`；
- summem + atommem 的 prompt 用 `{speakers}` 槽吃**本会话真实的说话人名单**
  （旧版硬替换成 `speaker_a`→`user`、`speaker_b`→`assistant`；剧本有 14 个角色，写成两个人是错的），
  并新增一条"不是每条记录都是人在说话 —— 不要发明说话人"；
- search 的 `user_begin` 加了一段由 `runner.render_participants` 从**这份语料真实的 `role`**
  渲染的说明（人名清单 + 三元组记法 + 别编造说话人）。

### 踩到的坑（两个都真出现过）

- **前缀匹配吃掉真名**：黑名单 `NON_SPEAKER_PREFIXES` 里短词（`and` / `he` / `no`）用
  `startswith` 判，`Andrew:` 被 `and` 吃掉、`Henry:` 被 `he` 吃掉 —— `Audrey`/`Andrew` 那两份
  语料整批丢说话人。改成**只做整词匹配**。判错的代价不对称：漏认只是 `role` 退化成 `user`，
  误认会**从正文里删掉前缀**。
- **无脑剥前导方括号**：`user: [part 1/2] Background` 里的 `[part 1/2]` 是正文。所以只剥
  `session_N` / `D1:1` / `Time: …` / 类型词 / **无空格的 slug**，剥到第一个不合格的就停手。

### 存量语料

`msgmem.jsonl` 是已落盘的产物，修适配层不会让它自己变好。`scripts/rebuild_api_store.py`
拿 `requests.jsonl`（原始报文留痕）重放：id 与时间戳**逐条一致**（已验证 3 个 user），
覆盖前备份；**重放条数少于现有条数时一律拒写** —— 实测真有这种 user（留痕是尽力而为的，
写失败不阻断请求），重放会比语料少整整一批。

### 数字

尚无端到端 judge 数字（这一轮改的是**输入层**，跑分要等重跑完存量语料 + 端点)。
可确定的是两件**决定性**的事：`atommem` 里终于有真名可 grep（此前 51% 的 add 没有），
以及 `[Time: …]` 那 170 个请求终于有了时间锚点。

普查口径与适配层**共用同一套判据**（`analyze_api_formats.py` 直接 import
`adapters.split_speaker`），不会出现"统计说 50 份对话、实际只认出 40 份"这种两套各说各话。
