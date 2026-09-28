#!/usr/bin/env python
"""search 步骤的纯 CPU 自测：假模型 + 真工具，零模型成本、零网络。

覆盖：

1. ``toolcfg``：catalog 加载与校验、prompt 渲染（槽位全部替换）、响应解析的稳健性。
2. ``tools``：read/write/edit/bash 的正常路径 + 全部失败路径 + 路径约束。
3. ``index`` + ``searchmem``：纯词法检索（含 RRF 与兜底填满 top_k）。
4. ``evidence``：逐行校验、补全 changelog、拒绝缺 source / 重复 id / 坏类型。
5. ``verifier``：解析稳健性 + 解析失败**保守判不足**。
6. ``agent``：用假模型脚本驱动一轮 agent loop，验证无进展检测与失败不中断。
7. ``searchctl``：CLI 端到端（词法模式，真子进程）。
8. ``runner``：dry-run 渲染，以及用假模型跑通一条 QA 的端到端（含篡改检测）。

跑：``python scripts/test_evomem_v3.py``
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codemem.search import agent as agent_mod   # noqa: E402
from codemem.search import evidence as evidence_mod  # noqa: E402
from codemem.search import index as index_mod   # noqa: E402
from codemem.search import searchmem            # noqa: E402
from codemem.search import toolcfg              # noqa: E402
from codemem.search import tools as tools_mod   # noqa: E402
from codemem.search import verifier as verifier_mod  # noqa: E402
from codemem.log import Logger                  # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f"  -- {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# 1. toolcfg
# ---------------------------------------------------------------------------

def test_toolcfg(tmp: Path) -> None:
    section("1. toolcfg：加载 / 校验 / 渲染 / 解析")
    catalog = toolcfg.load_catalog(PROJECT_ROOT / "configs/tool.json")
    check("catalog 加载成功", bool(catalog.tools), str(catalog.tools))
    check("四个工具都有描述", set(catalog.tools) == {"read", "write", "edit", "bash"})
    check("命令含 jq / search / timecalc",
          set(catalog.commands) == {"jq", "search", "timecalc"}, str(set(catalog.commands)))

    catalog.validate(["read", "write", "edit", "bash"])
    try:
        catalog.validate(["read", "write", "edit", "bash", "nope"])
        check("校验能发现缺描述的工具", False, "没有报错")
    except toolcfg.ToolConfigError:
        check("校验能发现缺描述的工具", True)

    rendered = catalog.render_prompt("Who is Caroline?")
    for slot in ("{{TOOLS}}", "{{COMMANDS}}", "{{WORKSPACE}}", "{{SCHEMA}}", "{{PROTOCOL}}",
                 "{{WRITING_POLICY}}", "{{QUESTION}}"):
        check(f"渲染后没有残留 {slot}", slot not in rendered)
    check("渲染含问题原文", "Who is Caroline?" in rendered)
    check("渲染含 jq 示例", "jq -c 'select(.metadata.id==" in rendered)
    check("渲染含 search 用法", "search \"QUERY\"" in rendered)
    check("渲染含 timecalc 用法", "timecalc shift" in rendered)
    check("渲染含 schema 字段", "metadata.source" in rendered)
    check("渲染含 source 非空规则", "non-empty" in rendered.lower())
    # 核心：证据是**推导**出来的，不是抄的
    check("渲染含'推导而非抄袭'策略", "DERIVED, NOT COPIED" in rendered)
    check("策略提到时间推理", "timecalc" in rendered and "absolute" in rendered.lower())
    check("策略提到事件/性格推理", "Join facts" in rendered and "Infer" in rendered)
    check("策略允许改写 memory 正文", "REWRITE" in rendered)
    check("schema 说明可以改写 memory", "REWRITE THE LIBRARY TEXT" in rendered)
    check("schema 说明要解析相对时间", "RESOLVE relative times" in rendered)
    check("有 worked example（时间推理）", "timecalc shift 2023-06-17 -1month" in rendered)

    # 响应解析
    check("解析裸 JSON", (toolcfg.parse_tool_call('{"tool":"bash","args":{"command":"ls"}}') or
                          type("x", (), {"tool": ""})()).tool == "bash")
    check("解析围栏包裹", (toolcfg.parse_tool_call('```json\n{"tool":"read","args":{"path":"a"}}\n```')
                          or type("x", (), {"tool": ""})()).tool == "read")
    check("解析前后有说明文字",
          (toolcfg.parse_tool_call('Sure!\n{"tool":"bash","args":{"command":"ls"}}\nDone.')
           or type("x", (), {"tool": ""})()).tool == "bash")
    check("参数平铺在顶层",
          (toolcfg.parse_tool_call('{"tool":"bash","command":"ls"}') or
           type("x", (), {"args": {}})).args.get("command") == "ls")
    check("无对象的纯文本 -> None（= 完成信号）", toolcfg.parse_tool_call("I am done.") is None)
    check("空输出 -> None", toolcfg.parse_tool_call("") is None)
    check("对象但没有 tool 键 -> None", toolcfg.parse_tool_call('{"memory":"x"}') is None)
    check("坏 JSON -> None", toolcfg.parse_tool_call('{"tool": "bash", "args": {') is None)

    # 校验：描述与执行层不一致要报错
    bad = toolcfg.ToolCatalog(protocol={}, tools={"read": {}}, commands={}, workspace={},
                              schema={}, prompt={"system": "{{TOOLS}} {{COMMANDS}} "
                                                          "{{WORKSPACE}} {{SCHEMA}} {{PROTOCOL}}"})
    try:
        bad.validate(["read", "bash"])
        check("描述了未注册的工具要报错", False)
    except toolcfg.ToolConfigError:
        check("描述了未注册的工具要报错", True)


# ---------------------------------------------------------------------------
# 2. tools
# ---------------------------------------------------------------------------

async def test_tools(tmp: Path) -> None:
    section("2. tools：read / write / edit / bash + 路径约束")
    ws = tmp / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    built = tools_mod.build_tools(cwd=ws, bash_timeout=20)
    check("工具名齐全", set(built) == {"read", "write", "edit", "bash"})

    # write / read
    result = await built["write"].run({"path": "a.txt", "content": "hello\nworld\n"})
    check("write 成功", result.ok and (ws / "a.txt").read_text() == "hello\nworld\n",
          result.text)
    result = await built["write"].run({"path": "nested/deep/b.txt", "content": "x"})
    check("write 自动建父目录", (ws / "nested/deep/b.txt").exists())

    result = await built["read"].run({"path": "a.txt"})
    check("read 全文", result.ok and "hello" in result.text and "world" in result.text)
    result = await built["read"].run({"path": "a.txt", "offset": 2, "limit": 1})
    check("read offset/limit", result.ok and result.text.startswith("world"), repr(result.text))
    try:
        await built["read"].run({"path": "missing.txt"})
        check("read 缺文件报错", False)
    except tools_mod.ToolError:
        check("read 缺文件报错", True)
    try:
        await built["read"].run({"path": "nested"})
        check("read 目录报错", False)
    except tools_mod.ToolError:
        check("read 目录报错", True)
    try:
        await built["read"].run({"path": "a.txt", "offset": 99})
        check("read offset 越界报错", False)
    except tools_mod.ToolError:
        check("read offset 越界报错", True)

    # edit
    result = await built["edit"].run(
        {"path": "a.txt", "edits": [{"oldText": "world", "newText": "there"}]}
    )
    check("edit 成功", result.ok and "there" in (ws / "a.txt").read_text())
    try:
        await built["edit"].run({"path": "a.txt", "edits": [{"oldText": "zzz", "newText": "y"}]})
        check("edit oldText 找不到报错", False)
    except tools_mod.ToolError:
        check("edit oldText 找不到报错", True)
    (ws / "dup.txt").write_text("same same\n")
    try:
        await built["edit"].run({"path": "dup.txt", "edits": [{"oldText": "same", "newText": "x"}]})
        check("edit oldText 不唯一报错", False)
    except tools_mod.ToolError:
        check("edit oldText 不唯一报错", True)
    check("edit 失败后文件未变", (ws / "dup.txt").read_text() == "same same\n")
    try:
        await built["edit"].run({"path": "a.txt", "edits": [{"oldText": "there", "newText": "there"}]})
        check("edit 无变化报错", False)
    except tools_mod.ToolError:
        check("edit 无变化报错", True)

    # 路径约束：写到工作目录之外必须被拒
    for name in ("write", "edit"):
        try:
            await built[name].run({"path": "../escape.txt", "content": "x"})
            check(f"{name} 拒绝工作目录外的路径", False)
        except tools_mod.ToolError:
            check(f"{name} 拒绝工作目录外的路径", True)
    check("没有写出 escape.txt", not (tmp / "escape.txt").exists())
    try:
        await built["write"].run({"path": "/tmp/codemem-should-not-exist.txt", "content": "x"})
        check("write 拒绝绝对路径逃逸", not Path("/tmp/codemem-should-not-exist.txt").exists())
    except tools_mod.ToolError:
        check("write 拒绝绝对路径逃逸", True)

    # 软链逃逸：qa/inputs -> 只读目录，写它应被拒
    outside = tmp / "readonly"
    outside.mkdir(exist_ok=True)
    (outside / "lib.jsonl").write_text("{}\n")
    (ws / "link").symlink_to(outside)
    try:
        await built["write"].run({"path": "link/lib.jsonl", "content": "hacked"})
        check("write 拒绝软链逃逸", (outside / "lib.jsonl").read_text() == "{}\n")
    except tools_mod.ToolError:
        check("write 拒绝软链逃逸", True)

    # bash
    result = await built["bash"].run({"command": "echo hi", "description": "Testing"})
    check("bash 执行成功", result.ok and "hi" in result.text, result.text)
    result = await built["bash"].run({"command": "exit 3", "description": "Testing failure"})
    check("bash 非零退出被标记为不 ok", (not result.ok) and result.details["exit_code"] == 3)
    result = await built["bash"].run({"command": "echo err >&2", "description": "Testing stderr"})
    check("bash 合并 stderr", "err" in result.text)
    result = await built["bash"].run(
        {"command": "sleep 5", "description": "Testing timeout", "timeout": 0.5}
    )
    check("bash 超时被杀", result.details["timed_out"] and not result.ok)
    result = await built["bash"].run({"command": "python -c \"print('x'*90000)\"",
                                     "description": "Testing truncation"})
    check("bash 输出被 tail 截断", result.details["truncation"]["truncated"] is True)
    check("bash 截断时给出完整输出路径", bool(result.details["full_output_path"]))
    try:
        await built["bash"].run({"command": "echo x"})
        check("bash 缺 description 不报错（纯说明性参数）", True)
    except tools_mod.ToolError as exc:
        check("bash 缺 description 不报错（纯说明性参数）", False, str(exc))

    # --- shell prefix 必须能安全地承载 JSON ---
    # 回归：JSON 不加引号时 bash 会做花括号展开（{a,b} 语法），把它拆成多个词 ——
    # 实测只有最后一段传进去，于是**每次** search 都吐一行解析错误。
    from codemem.search import runner as V3

    prefix = V3.build_shell_prefix(
        inputs_dir=ws, index_dir=tmp / "idx", lexical=False,
        config={"embedding": {"base_url": "http://x/v1", "api_key": "k", "model": "m"},
                "search": {"rrf_k": 60, "memory_chars": 400,
                           "weights": {"dense": 1.0, "bm25": 1.0, "tag": 0.5}}},
    )
    probe = tools_mod.build_tools(cwd=ws, shell_command_prefix=prefix)
    result = await probe["bash"].run(
        {"command": "echo \"$CODEMEM_SEARCH_WEIGHTS\"", "description": "Checking weights env"}
    )
    echoed = result.text.strip()
    check("weights JSON 原样传到子进程（未被花括号展开）",
          echoed == '{"dense":1.0,"bm25":1.0,"tag":0.5}', echoed)
    import json as _json
    try:
        parsed_weights = _json.loads(echoed)
        check("传进去的 weights 是合法 JSON", isinstance(parsed_weights, dict))
    except ValueError as exc:
        check("传进去的 weights 是合法 JSON", False, str(exc))
    result = await probe["bash"].run(
        {"command": "python -m codemem.search.searchctl --help >/dev/null 2>&1; echo rc=$?",
         "description": "Checking searchctl importable"}
    )
    check("searchctl 在注入环境下可运行", "rc=0" in result.text, result.text)

    # --- 空输出提示：jq 选不到东西时要明确说出来（否则模型会重复重试）---
    from codemem.search.toolcfg import ToolCall

    call = ToolCall(tool="bash", args={"command": "jq -c 'select(.metadata.id==\"nope\")' f.jsonl"})
    notice = agent_mod.no_output_notice(call)
    check("空 jq 输出提示说明 selector 没命中", "matched NOTHING" in notice, notice[:200])
    check("空 jq 输出提示讲清 id 规则", "sequence number" in notice, notice[:300])
    non_jq = agent_mod.no_output_notice(ToolCall(tool="bash", args={"command": "true"}))
    check("非 jq 的空输出不误报 jq 语义", "matched NOTHING" not in non_jq)

    # --- 重复拒绝消息要给出脱困方向 ---
    repeat = agent_mod.repeat_notice(ToolCall(tool="bash", args={"command": "x"}), 2)
    check("重复拒绝消息提到 id 差异", "sequence number" in repeat)
    check("重复拒绝消息提到 timecalc", "timecalc" in repeat)
    check("重复拒绝消息里的 jq 示例花括号未被 f-string 吃掉",
          "jq -c 'select(.metadata.id==" in repeat and "time:.metadata.time}" in repeat)


    # --- 文件完整性：原子写 + 截断保护 ---
    protected = tools_mod.build_tools(cwd=ws, protect_filenames=("evidence.jsonl",))
    await protected["write"].run({"path": "evidence.jsonl", "content": "X" * 500 + "\n"})
    check("write 到受保护文件成功", (ws / "evidence.jsonl").stat().st_size > 500)
    # 空内容重写受保护文件必须被拒（截断的典型表现），否则会静默销毁已有证据
    try:
        await protected["write"].run({"path": "evidence.jsonl", "content": ""})
        check("拒绝用空内容覆盖 evidence.jsonl", False)
    except tools_mod.ToolError as exc:
        check("拒绝用空内容覆盖 evidence.jsonl", "refusing" in str(exc), str(exc))
    check("被拒后文件内容完好",
          "X" * 500 in (ws / "evidence.jsonl").read_text())
    # 大幅缩短时给警告（不拒绝）
    result = await protected["write"].run({"path": "evidence.jsonl", "content": "small\n"})
    check("大幅缩短时给出截断警告", "shrank" in result.text, result.text)
    check("缩短被记为 shrank", result.details.get("shrank") is True)
    check("写操作标记为原子", result.details.get("atomic") is True)
    # 原子写不留下临时文件
    leftovers = [p.name for p in ws.iterdir() if ".evidence.jsonl." in p.name]
    check("原子写不留临时文件", not leftovers, str(leftovers))
    # 不受保护的文件可以用空内容（比如清空草稿）
    result = await protected["write"].run({"path": "scratch.txt", "content": ""})
    check("非受保护文件可用空内容", result.ok)


# ---------------------------------------------------------------------------
# 3. 检索
# ---------------------------------------------------------------------------

def make_memory(memory_id: str, text: str, *, source: list[str] | None = None,
                speaker: str = "Caroline", kind: str = "outer") -> dict:
    return {
        "memory": text,
        "metadata": {
            "id": memory_id, "type": kind, "time": "2023-05-08T13:56:00",
            "tag": [f"speaker:{speaker}"], "source": source or ["session_1_1"],
            "changelog": [],
        },
    }


async def test_search(tmp: Path) -> None:
    section("3. index + searchmem：纯词法检索")
    corpus = [
        make_memory("a1", "Caroline joined an LGBTQ support group"),
        make_memory("a2", "Melanie asked Caroline when she started going"),
        make_memory("a3", "Caroline has been going for about a month"),
        make_memory("a4", "Dave likes pizza and rock climbing"),
    ]

    scored = await searchmem.search_scored(
        "LGBTQ support group", corpus, top_k=2, embed=None,
        weights=searchmem.lexical_weights(),
    )
    check("词法检索命中", len(scored) == 2, str(len(scored)))
    check("最相关的排第一", scored[0].id == "a1", scored[0].id)
    check("带通道明细", "bm25" in scored[0].channels, str(scored[0].channels))

    # 零信号候选要兜底填满 top_k
    scored_all = await searchmem.search_scored(
        "LGBTQ support group", corpus, top_k=4, embed=None,
        weights=searchmem.lexical_weights(),
    )
    check("候选不足时兜底填满 top_k", len(scored_all) == 4, str(len(scored_all)))
    check("兜底项 score=0 无通道", scored_all[3].score == 0.0 and not scored_all[3].channels)

    # extra_channels 注入（searchctl 的用法）
    injected = await searchmem.search_scored(
        "anything", corpus, top_k=4, embed=None,
        weights={"dense": 1.0, "bm25": 0.0, "tag": 0.0},
        extra_channels={"dense": [0.0, 0.9, 0.1, 0.0]},
    )
    check("extra_channels 生效", injected[0].id == "a2", injected[0].id)

    # 索引：建 + 载 + 打分
    fake_vectors = {
        "a1": [1.0, 0.0, 0.0], "a2": [0.9, 0.1, 0.0],
        "a3": [0.0, 1.0, 0.0], "a4": [0.0, 0.0, 1.0],
    }

    async def fake_embed(texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            for key, vector in fake_vectors.items():
                if key in text or text in key:
                    out.append(vector)
                    break
            else:
                out.append(fake_vectors["a1"] if "LGBTQ" in text else [0.0, 0.0, 1.0])
        return out

    index_dir = tmp / "idx"
    built = await index_mod.build_index(
        corpus, embed=fake_embed, directory=index_dir, model="fake", source="test"
    )
    check("索引维度正确", built.dim == 3, str(built.dim))
    loaded = index_mod.load_index(index_dir)
    check("索引可加载且条数一致", len(loaded) == 4)
    row_of = loaded.build_row_index()
    scores = loaded.dense_scores([1.0, 0.0, 0.0], row_of, ["a1", "a3", "unknown"])
    check("dense 打分：同向最高", abs(scores[0] - 1.0) < 1e-6, str(scores))
    check("dense 打分：正交为 0", abs(scores[1]) < 1e-6)
    check("dense 打分：无向量给 0", scores[2] == 0.0)

    # --- 共享索引缓存：同一语料只建一次（存储优化）---

    def mem(i, t):
        return {"memory": t, "metadata": {"id": i, "type": "raw", "tag": [], "source": []}}

    corpus_a = [mem("x1", "one"), mem("x2", "two")]
    corpus_b = [mem("x1", "CHANGED"), mem("x2", "two")]     # 正文改了
    corpus_c = [mem("x1", "one")]                           # 少一条

    key_a = index_mod.cache_key(corpus_a, "modelA")
    check("同语料同模型 key 稳定", key_a == index_mod.cache_key(corpus_a, "modelA"))
    check("正文变了 key 就变", key_a != index_mod.cache_key(corpus_b, "modelA"))
    check("条数变了 key 就变", key_a != index_mod.cache_key(corpus_c, "modelA"))
    check("换模型 key 就变", key_a != index_mod.cache_key(corpus_a, "modelB"))

    cache_root = tmp / "idx_cache"
    shared = index_mod.shared_index_dir(cache_root, corpus_a, "modelA")
    check("共享目录在 cache_root 下", shared.parent == cache_root, str(shared))
    check("未建时 index_is_usable=False",
          index_mod.index_is_usable(shared, corpus_a, "modelA") is False)

    async def fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    await index_mod.build_index(corpus_a, embed=fake_embed, directory=shared,
                                model="modelA", source="test")
    check("建好后 index_is_usable=True",
          index_mod.index_is_usable(shared, corpus_a, "modelA") is True)
    # 关键安全性质：语料变了必须判为**不可用**，否则会静默复用过期向量
    check("语料变了判为不可用（防静默复用过期向量）",
          index_mod.index_is_usable(shared, corpus_b, "modelA") is False)
    check("模型变了判为不可用",
          index_mod.index_is_usable(shared, corpus_a, "modelB") is False)
    check("条数变了判为不可用",
          index_mod.index_is_usable(shared, corpus_c, "modelA") is False)
    check("目录不存在判为不可用",
          index_mod.index_is_usable(tmp / "nope", corpus_a, "modelA") is False)

    # --- 运行目录清理：只删运行目录，永不删共享索引缓存 ---
    runs = tmp / "runs"
    for name in ("search_a_20260101_000000", "search_b_20260102_000000",
                 "full_20260103_000000", "search_c_20260104_000000"):
        d = runs / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text("{}")
        (d / "blob.jsonl").write_text("x" * 5000)
    cache = runs / ".index_cache" / "abc"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "vectors.f32").write_text("v" * 1000)

    plan = index_mod.plan_prune(runs)
    names = {p.name for p, _ in plan}
    check("默认保留含 full 的运行", "full_20260103_000000" not in names, str(names))
    check("默认清理其它运行目录", len(plan) == 3, str(sorted(names)))
    check("共享索引缓存永不出现在清理列表",
          not any(".index_cache" in str(p) for p, _ in plan))
    check("清理列表带大小", all(size > 0 for _, size in plan))

    plan2 = index_mod.plan_prune(runs, keep_last=2)
    check("--keep-last 2 只清 2 个", len(plan2) == 2, str(len(plan2)))
    check("--keep-last 保留最新的",
          "search_c_20260104_000000" not in {p.name for p, _ in plan2})

    check("目录不存在时返回空", index_mod.plan_prune(tmp / "nope") == [])
    check("run_size 能算出大小", index_mod.run_size(runs / "full_20260103_000000") > 5000)

    # 执行删除：缓存必须还在
    removed, freed = index_mod.prune_runs(runs, plan)
    check("删除了计划中的目录", removed == 3, str(removed))
    check("释放字节数 > 0", freed > 0)
    check("共享索引缓存仍在", (cache / "vectors.f32").exists())
    check("保留的运行仍在", (runs / "full_20260103_000000" / "summary.json").exists())

    # 坏索引目录要报错，不要静默
    empty = tmp / "idx_empty"
    empty.mkdir(exist_ok=True)
    try:
        index_mod.load_index(empty)
        check("缺文件的索引报错", False)
    except FileNotFoundError:
        check("缺文件的索引报错", True)


# ---------------------------------------------------------------------------
# 4. evidence
# ---------------------------------------------------------------------------

def test_evidence(tmp: Path) -> None:
    section("4. evidence：逐行校验与规范化")
    path = tmp / "evidence.jsonl"
    good = make_memory("e1", "Caroline joined a support group")
    good["metadata"]["changelog"] = [{"time": "2026-01-01T00:00:00Z", "content": "created"}]
    repaired = make_memory("e2", "Caroline has been going for a month")
    repaired["metadata"]["changelog"] = []          # 缺系统自有字段 -> 补全
    no_source = make_memory("e3", "no provenance")
    no_source["metadata"]["source"] = []
    dup = make_memory("e1", "duplicate id")
    bad_type = make_memory("e5", "bad type")
    bad_type["metadata"]["type"] = "wrong"
    no_speaker = make_memory("e6", "no speaker tag")
    no_speaker["metadata"]["tag"] = ["topic:x"]
    empty_memory = make_memory("e7", "")
    lines = [
        json.dumps(good), json.dumps(repaired), json.dumps(no_source), json.dumps(dup),
        json.dumps(bad_type), json.dumps(no_speaker), json.dumps(empty_memory),
        "{not json", "", json.dumps(good),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = evidence_mod.validate_file(path)
    # total_lines 是**解析出的记录数**（一行一数组会展开、美化 JSON 算一批），
    # 不是原始行数。这里 9 行里有 1 行坏 JSON、1 行空 → 8 条记录。
    check("记录数统计正确", report.total_lines == 8, str(report.total_lines))
    check("空行被计数", report.empty_lines == 1)
    check("解析失败被计数", len(report.parse_errors) == 1)
    check("补全被计数", len(report.repaired) == 1)
    check("补全后 changelog 非空", report.repaired[0]["metadata"]["changelog"] != [])
    check("交付数正确", len(report.valid) == 2, report.summary())
    check("重复 id 被拒", any("duplicate" in item["reason"] for item in report.rejected))
    check("缺 source 被拒", any("source" in item["reason"] for item in report.rejected))
    check("坏 type 被拒", any("type" in item["reason"] for item in report.rejected))
    check("缺 speaker 被拒", any("speaker" in item["reason"] for item in report.rejected))
    check("空 memory 被拒", any("memory" in item["reason"] for item in report.rejected))
    check("拒绝数正确", len(report.rejected) == 6, str(len(report.rejected)))
    check("summary 可读", "交付 2" in report.summary(), report.summary())

    # 规范化写回 + 备份
    evidence_mod.write_normalized(path, report)
    check("规范化后只剩交付行", len(evidence_mod.validate_file(path).valid) == 2)
    check("原文件备份为 .raw", path.with_suffix(".jsonl.raw").exists())

    # 空文件 / 不存在
    missing = evidence_mod.validate_file(tmp / "nope.jsonl")
    check("文件不存在 -> 空报告（不报错）", missing.total_lines == 0 and not missing.rejected)

    # --- 一行是数组（jq -s 的自然输出）必须能被接受 ---
    arr_path = tmp / "evidence_array.jsonl"
    batch = [make_memory("b1", "First fact"), make_memory("b2", "Second fact")]
    arr_path.write_text(json.dumps(batch) + "\n", encoding="utf-8")
    arr_report = evidence_mod.validate_file(arr_path)
    check("一行数组被摊成多条记录", len(arr_report.valid) == 2, arr_report.summary())
    check("数组内 id 正确", [m["metadata"]["id"] for m in arr_report.valid] == ["b1", "b2"])
    check("数组行不产生拒绝", not arr_report.rejected, str(arr_report.rejected))
    check("单行数组不触发整份兜底", arr_report.whole_file_fallback is False)
    # 回归：jq -s 不加 -c 会输出**多行美化** JSON。逐行读会得到一堆"解析失败"碎行，
    # 若不做整体兜底，write_normalized 会把这些碎行全丢掉 —— 等于静默删掉全部证据。
    pretty = tmp / "evidence_pretty.jsonl"
    pretty.write_text(json.dumps([make_memory("p1", "Fact one"),
                                  make_memory("p2", "Fact two")], indent=2) + "\n",
                      encoding="utf-8")
    pretty_report = evidence_mod.validate_file(pretty)
    check("jq -s 美化输出被整体兜底救回", len(pretty_report.valid) == 2, pretty_report.summary())
    check("美化输出被标记为兜底", pretty_report.whole_file_fallback is True)
    evidence_mod.write_normalized(pretty, pretty_report)
    check("兜底后重写为逐行", len(pretty.read_text().strip().splitlines()) == 2)
    check("重写后仍可正常校验", len(evidence_mod.validate_file(pretty).valid) == 2)

    # --- 被 max_tokens 截断的数组必须抢救（实测踩到的真 bug）---
    # 模型一次 write 写出 39 条记录、末尾的 `]` 被截掉。旧实现看到"1 行、解析失败"就
    # 把**全部证据判成空**（坏行数不满足 >= 2 的兜底门槛）。现在按对象边界抢救。
    truncated = tmp / "evidence_truncated.jsonl"
    full = json.dumps([make_memory("t1", "First"), make_memory("t2", "Second"),
                       make_memory("t3", "Third")])
    truncated.write_text(full[:-1], encoding="utf-8")      # 切掉结尾的 `]`
    trunc_report = evidence_mod.validate_file(truncated)
    check("截断的数组被抢救出完整记录", len(trunc_report.valid) == 3,
          trunc_report.summary())
    check("抢救后 id 完整", [m["metadata"]["id"] for m in trunc_report.valid]
          == ["t1", "t2", "t3"])
    evidence_mod.write_normalized(truncated, trunc_report)
    check("抢救后重写为逐行", len(truncated.read_text().strip().splitlines()) == 3)
    check("抢救能被识别（不是静默）",
          evidence_mod.validate_file(truncated).salvaged is False)  # 已规范化，不再需要抢救

    # --- 漏写括号：模型的常见错误，会让整份文件解析失败、证据全丢 ---
    # 实测形态：metadata 少一个 `}`，于是 `...}]}, {...` 非法，**全部记录一起消失**
    missing_brace = tmp / "evidence_missing_brace.jsonl"
    missing_brace.write_text(
        '[{"memory": "First", "metadata": {"id": "m1", "type": "outer", "time": "", '
        '"tag": ["speaker:C"], "source": ["s"], "changelog": [{"time": "t", "content": "c"}]}, '
        '{"memory": "Second", "metadata": {"id": "m2", "type": "outer", "time": "", '
        '"tag": ["speaker:C"], "source": ["s"], "changelog": [{"time": "t", "content": "c"}]}}]',
        encoding="utf-8")
    mb = evidence_mod.validate_file(missing_brace)
    check("漏写一个 } 时仍能救回全部记录", len(mb.valid) == 2, mb.summary())
    check("救回的 id 正确", [m["metadata"]["id"] for m in mb.valid] == ["m1", "m2"])

    # salvage 的基本性质（单元）
    from codemem.io import salvage_json_records
    check("正常数组逐条切分",
          len(salvage_json_records('[{"a":1},{"b":2}]')[0]) == 2)
    check("嵌套对象不丢字段",
          salvage_json_records('[{"a":{"b":1},"c":2}]')[0] == [{"a": {"b": 1}, "c": 2}])
    check("记录写一半时丢掉残片",
          salvage_json_records('[{"a":1},{"b":')[0] == [{"a": 1}])
    check("只缺数组闭括号时保住完整记录",
          len(salvage_json_records('[{"a":1},{"b":2}')[0]) == 2)

    # 截断在**记录中间**：前面的完整记录要保住，残片丢掉
    mid = tmp / "evidence_mid.jsonl"
    mid.write_text(full[:len(full) - 12], encoding="utf-8")
    mid_report = evidence_mod.validate_file(mid)
    check("记录中间被截断时保住前面的", len(mid_report.valid) == 2, mid_report.summary())

    # 完全不是 JSON 的文件不该被误当成记录
    garbage = tmp / "evidence_garbage.jsonl"
    garbage.write_text("this is not json at all\nnor is this\n", encoding="utf-8")
    garbage_report = evidence_mod.validate_file(garbage)
    check("非 JSON 文件不产生记录", not garbage_report.valid, garbage_report.summary())

    # 数组里混入坏项：坏项单独被拒，好项保留
    mixed = tmp / "evidence_mixed.jsonl"
    mixed.write_text(json.dumps([make_memory("c1", "Good"), {"bad": "no metadata"}]) + "\n",
                     encoding="utf-8")
    mixed_report = evidence_mod.validate_file(mixed)
    check("数组内坏项单独被拒", len(mixed_report.valid) == 1 and len(mixed_report.rejected) == 1,
          mixed_report.summary())
    # 规范化的写回也要按条展开（而不是把数组整行写回）
    evidence_mod.write_normalized(arr_path, arr_report)
    check("规范化后数组被展开为逐行",
          len(arr_path.read_text().strip().splitlines()) == 2)


# ---------------------------------------------------------------------------
# 5. verifier
# ---------------------------------------------------------------------------

async def test_verifier() -> None:
    section("5. verifier：解析稳健性 + 保守默认 + 空 evidence 短路")
    verdict = verifier_mod.parse_verdict('{"sufficient": true, "answer": "2018", "missing": []}')
    check("解析 sufficient=true", verdict.sufficient and verdict.answer == "2018")
    verdict = verifier_mod.parse_verdict(
        '```json\n{"sufficient": false, "answer": "", "missing": ["the year"]}\n```'
    )
    check("解析围栏包裹", (not verdict.sufficient) and verdict.missing == ["the year"])
    verdict = verifier_mod.parse_verdict('Here: {"sufficient": false, "missing": ["a","b"]} ok')
    check("解析夹在文字里", (not verdict.sufficient) and len(verdict.missing) == 2)
    verdict = verifier_mod.parse_verdict("not json at all")
    check("坏输出 -> 保守判不足", (not verdict.sufficient) and not verdict.parsed)
    verdict = verifier_mod.parse_verdict('{"answer": "x"}')
    check("缺 sufficient -> 保守判不足", (not verdict.sufficient) and not verdict.parsed)
    verdict = verifier_mod.parse_verdict('{"sufficient": "yes"}')
    check("sufficient 非布尔 -> 保守判不足", (not verdict.sufficient) and not verdict.parsed)
    verdict = verifier_mod.parse_verdict('{"sufficient": false, "missing": "a single gap"}')
    check("missing 是字符串也能收", verdict.missing == ["a single gap"])
    check("describe 可读", "verifier=" in verifier_mod.parse_verdict(
        '{"sufficient": true, "answer": "x"}').describe())

    messages = verifier_mod.build_verifier_messages(
        "When?", [make_memory("a1", "Caroline joined a group")]
    )
    check("verifier 消息含问题", "When?" in messages[1]["content"])
    check("verifier 消息含记录", "Caroline joined a group" in messages[1]["content"])
    check("verifier 看不到参考答案", "reference" not in messages[0]["content"].lower())
    check("verifier 消息数为 2", len(messages) == 2)

    # --- 空 evidence：不问模型，直接判不足并给出**可执行**反馈 ---
    async def must_not_be_called(messages, meta):
        raise AssertionError("evidence 为空时不应该调用模型")

    verdict = await verifier_mod.verify(
        question="When?", memories=[], model_call=must_not_be_called,
        log=Logger(level="error"), evidence_present=False,
    )
    check("空 evidence 不调用模型", verdict.parsed and not verdict.sufficient)
    joined = " ".join(verdict.missing)
    check("空 evidence 反馈指出没写文件", "produced NO records" in joined, joined)
    check("空 evidence 反馈给出可执行动作", "WRITE the records" in joined, joined)
    verdict2 = await verifier_mod.verify(
        question="When?", memories=[], model_call=must_not_be_called,
        log=Logger(level="error"), evidence_present=True,
    )
    check("全部被拒时的反馈不同", "rejected" in " ".join(verdict2.missing),
          " ".join(verdict2.missing))


# ---------------------------------------------------------------------------
# 6. agent loop
# ---------------------------------------------------------------------------

class FakeModel:
    """按脚本返回预设输出，并记录收到的 messages。"""

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.calls: list[list[dict]] = []

    async def __call__(self, messages: list[dict], meta: dict) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self.script:
            return "I am done."
        return self.script.pop(0)


async def test_agent(tmp: Path) -> None:
    section("6. agent：工具循环 / 失败不中断 / 无进展检测 / 重复检测 / 压缩 / 重试")
    ws = tmp / "agent_ws"
    ws.mkdir(parents=True, exist_ok=True)
    # 用**紧凑** JSON 写 fixture：`jq -c` 会重新压掉空格，紧凑形态下 jq 的输出与它逐字节相同，
    # 于是"文件增长了多少"这类断言可以精确比较。
    one_record = json.dumps(make_memory("a1", "Caroline joined a group"),
                            separators=(",", ":")) + "\n"
    (ws / "lib.jsonl").write_text(one_record)
    evidence_path = ws / "evidence.jsonl"
    log = Logger(level="error")
    from codemem.search import toolcfg as tc

    catalog = tc.load_catalog(PROJECT_ROOT / "configs/tool.json")
    built = tools_mod.build_tools(cwd=ws)

    def fresh(script: list[str], **kwargs):
        return agent_mod.run_agent_round(
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "go"}],
            tools=built, model_call=FakeModel(script), evidence_path=evidence_path, cwd=ws,
            max_steps=kwargs.pop("max_steps", 10),
            no_progress_patience=kwargs.pop("no_progress_patience", 5),
            step_offset=0, round_index=1, log=log, **kwargs,
        )

    # --- 正常一轮：jq 拷贝 → 不再调工具 ---
    if evidence_path.exists():
        evidence_path.unlink()
    script = [
        json.dumps({"tool": "bash", "args": {
            "command": "jq -c . lib.jsonl >> evidence.jsonl", "description": "Copying"}}),
        "Done.",
    ]
    outcome, messages = await fresh(script)
    check("一轮跑完 2 步", len(outcome.steps) == 2, str(len(outcome.steps)))
    check("第一次是工具调用", outcome.steps[0].call is not None)
    check("工具调用成功", outcome.steps[0].result.ok)
    check("检测到唯一 id 增加", outcome.steps[0].progressed)
    check("唯一 id 统计正确", outcome.steps[0].state_after.unique_ids == 1,
          str(vars(outcome.steps[0].state_after)))
    check("第二次是完成信号", outcome.steps[1].call is None)
    check("done_reason=no_tool_call", outcome.done_reason == "no_tool_call", outcome.done_reason)
    check("产物已写出", evidence_path.exists() and evidence_path.stat().st_size > 0)
    # P0: 变更型调用后必须附证据摘要（重定向让 stdout 为空，模型否则看不见自己写了什么）
    obs = [m["content"] for m in messages if m["role"] == "user"]
    check("P0：观测里带了 evidence 摘要", any("evidence.jsonl now:" in o for o in obs),
          str(obs[-2:])[:300])
    check("P0：摘要里有唯一 id 数", any("unique ids" in o for o in obs))

    # --- 重复检测：同一轮内同样的调用第二次被拒，且**不执行** ---
    # 语义：repeat_limit = 完全相同的调用**允许执行**几次。默认 1（跑一次，第二次拒）。
    # 新一轮会用新的 tracker，所以跨轮的第一次调用是允许的 —— 这是刻意的。
    if evidence_path.exists():
        evidence_path.unlink()
    evidence_path.write_text(one_record)
    baseline = evidence_path.stat().st_size
    script = [
        json.dumps({"tool": "bash", "args": {
            "command": "jq -c . lib.jsonl >> evidence.jsonl", "description": "Copying"}}),
        json.dumps({"tool": "bash", "args": {
            "command": "jq -c . lib.jsonl >> evidence.jsonl", "description": "Copying"}}),
        "Done.",
    ]
    outcome, messages = await fresh(script, repeat_limit=1)
    repeated = [s for s in outcome.steps if s.repeat_rejected]
    check("重复调用被拒（第二次）", len(repeated) == 1, str(len(repeated)))
    check("重复计数被记账", outcome.repeated_calls == 1, str(outcome.repeated_calls))
    # 第一次执行了、第二次被拒 → 文件只增长一次（而不是两次）
    check("重复被拒时文件只增长一次",
          evidence_path.stat().st_size == baseline + len(one_record),
          f"{baseline} -> {evidence_path.stat().st_size}，期望 +{len(one_record)}")
    check("拒绝消息教它做别的事", any("already ran this exact" in m["content"]
                                    for m in messages if m["role"] == "user"))
    check("重复拒绝不产生新增唯一 id",
          repeated[0].state_before.unique_ids == repeated[0].state_after.unique_ids,
          f"{repeated[0].state_before.unique_ids} -> {repeated[0].state_after.unique_ids}")

    # --- 进展判据是**唯一记录集合**：重复追加同样的行不算进展 ---
    # 用各不相同的命令绕过重复检测，专门验证"行数涨、字节涨，但唯一集合没变 → 不算进展"。
    if evidence_path.exists():
        evidence_path.unlink()
    evidence_path.write_text(one_record)
    script = [
        json.dumps({"tool": "bash", "args": {
            "command": f"jq -c . lib.jsonl >> evidence.jsonl && echo step{i}",
            "description": "Appending duplicates"}})
        for i in range(20)
    ]
    outcome, _ = await fresh(script, no_progress_patience=3, repeat_limit=99)
    check("重复追加不算进展（唯一集合没变）", outcome.done_reason.startswith("no_progress"),
          outcome.done_reason)
    check("按唯一记录判停时行数已很大",
          outcome.steps[-1].state_after.lines > 3, str(outcome.steps[-1].state_after.lines))
    check("唯一 id 数一直是 1", outcome.steps[-1].state_after.unique_ids == 1,
          str(outcome.steps[-1].state_after.unique_ids))
    # 反例：真正新增一条记录 → 算进展
    evidence_path.write_text(one_record)
    other = json.dumps(make_memory("a2", "Another fact")) + "\n"
    (ws / "lib2.jsonl").write_text(other)
    script = [
        json.dumps({"tool": "bash", "args": {
            "command": "jq -c . lib2.jsonl >> evidence.jsonl", "description": "Adding"}}),
        "Done.",
    ]
    outcome, _ = await fresh(script)
    check("新增记录算进展", outcome.steps[0].progressed,
          str(vars(outcome.steps[0].state_after)))
    # 反例：原地改写一条记录（id 不变、正文变）也算进展
    evidence_path.write_text(one_record)
    script = [
        json.dumps({"tool": "bash", "args": {"command": (
            "jq -c '.memory=\"REWRITTEN with the date resolved\"' evidence.jsonl > t "
            "&& mv t evidence.jsonl"), "description": "Rewriting"}}),
        "Done.",
    ]
    outcome, _ = await fresh(script)
    check("原地改写也算进展（唯一集合变了）", outcome.steps[0].progressed,
          str(vars(outcome.steps[0].state_after)))
    check("改写后唯一 id 数不变", outcome.steps[0].state_after.unique_ids == 1)

    # --- 重复检测必须**感知状态**：文件变了之后同一个命令应该被允许再跑 ---
    # 修文件内容 → 指纹变化 → 之前被拒的命令应放行（成熟 code agent 的语义是
    # "重复一次**没推进**的调用"，不是"这个命令一辈子只能跑一次"）。
    if evidence_path.exists():
        evidence_path.unlink()
    evidence_path.write_text(one_record)
    rewrite = json.dumps({"tool": "write", "args": {
        "path": "evidence.jsonl",
        "content": json.dumps([make_memory("a1", "rewritten body")],
                              separators=(",", ":")) + "\n"}})
    dedupe = json.dumps({"tool": "bash", "args": {
        "command": "jq -sc 'unique_by(.metadata.id)[]' evidence.jsonl > t && mv t evidence.jsonl",
        "description": "Deduplicating"}})
    script = [dedupe, rewrite, dedupe, "Done."]
    outcome, _ = await fresh(script, repeat_limit=1)
    rejected = {s.index: s.repeat_rejected for s in outcome.steps}
    check("状态变了之后同一命令被放行", rejected.get(3) is False, str(rejected))
    check("整轮没有误拒", outcome.repeated_calls == 0, str(outcome.repeated_calls))
    check("改写确实生效", "rewritten body" in evidence_path.read_text())

    # 反向：文件**没有**变化时，同一命令第二次被拒
    if evidence_path.exists():
        evidence_path.unlink()
    evidence_path.write_text(one_record)
    dedupe_compact = json.dumps({"tool": "bash", "args": {
        "command": "jq -sc 'unique_by(.metadata.id)[]' evidence.jsonl > t && mv t evidence.jsonl",
        "description": "Deduplicating"}})
    script = [dedupe_compact, dedupe_compact, "Done."]
    outcome, _ = await fresh(script, repeat_limit=1)
    check("状态未变时重复被拒", any(s.repeat_rejected for s in outcome.steps),
          str([(s.index, s.repeat_rejected) for s in outcome.steps]))
    check("只拒绝了一次", outcome.repeated_calls == 1, str(outcome.repeated_calls))

    # --- 错误重试：截断的 JSON 要重试，不当完成 ---
    if evidence_path.exists():
        evidence_path.unlink()
    script = [
        '{"tool": "bash", "args": {"command": "jq -c . lib.jsonl >> evidence.jsonl"',  # 截断
        json.dumps({"tool": "bash", "args": {
            "command": "jq -c . lib.jsonl >> evidence.jsonl", "description": "Fixing"}}),
        "Done.",
    ]
    outcome, messages = await fresh(script)
    check("截断的 JSON 触发解析重试", outcome.parse_retries >= 1, str(outcome.parse_retries))
    check("重试后成功执行", evidence_path.exists() and evidence_path.stat().st_size > 0)
    check("重试不是完成信号", outcome.done_reason == "no_tool_call", outcome.done_reason)
    check("注入了纠正消息", any("not valid JSON" in m["content"]
                              for m in messages if m["role"] == "user"))

    # --- 空回复：重试，仍空才当 empty_reply ---
    class EmptyThenStop:
        def __init__(self): self.n = 0
        async def __call__(self, messages, meta):
            self.n += 1
            return ""
    outcome, _ = await agent_mod.run_agent_round(
        messages=[{"role": "user", "content": "go"}], tools=built, model_call=EmptyThenStop(),
        evidence_path=evidence_path, cwd=ws, max_steps=5, no_progress_patience=5,
        step_offset=0, round_index=1, log=log, max_parse_retries=2,
    )
    check("空回复重试后仍空 → empty_reply", outcome.done_reason == "empty_reply",
          outcome.done_reason)

    # --- 失败不中断 ---
    if evidence_path.exists():
        evidence_path.unlink()
    script = [
        json.dumps({"tool": "nope", "args": {}}),
        json.dumps({"tool": "bash", "args": {"command": "jq 'bad syntax' lib.jsonl",
                                            "description": "Breaking jq"}}),
        json.dumps({"tool": "edit", "args": {"path": "lib.jsonl",
                                            "edits": [{"oldText": "zzz", "newText": "y"}]}}),
        json.dumps({"tool": "bash", "args": {"command": "jq -c . lib.jsonl >> evidence.jsonl",
                                            "description": "Fixing"}}),
        "Done.",
    ]
    outcome, _ = await fresh(script)
    check("未知工具被记账", outcome.tool_errors >= 1, str(outcome.tool_errors))
    check("失败后循环继续并最终成功",
          outcome.steps[-1].call is None and evidence_path.stat().st_size > 0)
    check("失败步带 ERROR 观测",
          any("ERROR" in m["content"] for m in outcome.messages if m["role"] == "user"))
    check("失败步不中断（步数=5）", len(outcome.steps) == 5, str(len(outcome.steps)))
    check("未知工具提示里有 timecalc", any("timecalc" in m["content"]
                                          for m in outcome.messages if m["role"] == "user"))

    # --- 步数用尽 ---
    script = [json.dumps({"tool": "read", "args": {"path": "lib.jsonl"}})] * 20
    outcome, _ = await fresh(script, max_steps=4, no_progress_patience=99)
    check("步数用尽时收尾", outcome.done_reason == "steps_exhausted", outcome.done_reason)
    check("步数上限生效", len(outcome.steps) == 4, str(len(outcome.steps)))

    # --- 上下文压缩 ---
    script = [json.dumps({"tool": "read", "args": {"path": "lib.jsonl"}})] * 20
    outcome, messages = await agent_mod.run_agent_round(
        messages=[{"role": "system", "content": "S" * 500}, {"role": "user", "content": "go"}],
        tools=built, model_call=FakeModel(script), evidence_path=evidence_path, cwd=ws,
        max_steps=10, no_progress_patience=99, step_offset=0, round_index=1, log=log,
        context_max_chars=2000, context_keep_recent=4,
    )
    check("触发了上下文压缩", outcome.compactions >= 1, str(outcome.compactions))
    check("压缩后消息数受控", len(messages) <= 8, str(len(messages)))
    check("压缩保留了 system", messages[0]["content"].startswith("S" * 10))
    check("压缩摘要提到压缩", any("context compacted" in m["content"] for m in messages))
    check("压缩后仍能继续到步数用尽", outcome.done_reason == "steps_exhausted",
          outcome.done_reason)


async def test_compaction_units(tmp: Path) -> None:
    section("6b. 上下文压缩（单元）")
    messages = [{"role": "system", "content": "SYS"}]
    for index in range(12):
        messages.append({"role": "assistant", "content": json.dumps(
            {"tool": "bash", "args": {"command": f"jq 'select(.metadata.id==\"session_{index}_1\")' "
                                                 f"inputs/atommem.jsonl >> evidence.jsonl",
                                      "description": "x"}})})
        messages.append({"role": "user", "content": "X" * 400})
    result = agent_mod.compact_messages(messages, max_chars=1000, keep_recent=4)
    check("压缩丢掉了中段", result.dropped > 0, str(result.dropped))
    check("压缩后条数变少", len(result.messages) < len(messages))
    check("保留 system 在首位", result.messages[0]["content"] == "SYS")
    check("摘要含工具统计", "bash×" in result.digest, result.digest[:200])
    check("摘要保留了见过的 id", "session_0_1" in result.digest, result.digest[:400])
    check("末尾保留了最近的消息", result.messages[-1]["content"] == "X" * 400)
    # 未超预算时不动
    small = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    untouched = agent_mod.compact_messages(small, max_chars=100000, keep_recent=4)
    check("未超预算不压缩", untouched.dropped == 0 and untouched.messages == small)
    # 关掉压缩（max_chars=0）
    off = agent_mod.compact_messages(messages, max_chars=0, keep_recent=4)
    check("max_chars=0 表示不压缩", off.dropped == 0 and len(off.messages) == len(messages))


def test_no_undefined_names() -> None:
    """静态检查：整个 src/codemem 里不能有"调用了但没定义/没导入"的名字。

    重构最容易留下的就是这个 —— 把函数搬到别处、忘了补 import，运行时才在某个分支炸。
    用 AST 做一次全包扫描，比逐个 import 更能覆盖到"只在某个错误分支里才走到"的代码。
    """
    section("0. 静态检查：无未定义引用（重构回归网）")
    import ast
    import builtins

    package = PROJECT_ROOT / "src" / "codemem"
    offenders: list[str] = []
    for path in sorted(package.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            offenders.append(f"{path.name}: SyntaxError {exc}")
            continue
        local: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                local.update(a.asname or a.name for a in node.names)
            elif isinstance(node, ast.Import):
                local.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, ast.Assign):
                local.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                local.add(node.target.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local.add(node.name)
                # 函数参数也算已定义
                args = getattr(node, "args", None)
                if args is not None:
                    for group in (args.posonlyargs, args.args, args.kwonlyargs):
                        local.update(a.arg for a in group)
                    if args.vararg: local.add(args.vararg.arg)
                    if args.kwarg: local.add(args.kwarg.arg)
            elif isinstance(node, (ast.For, ast.comprehension)) and isinstance(node.target, ast.Name):
                local.add(node.target.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                local.add(node.name)
            elif isinstance(node, ast.Lambda):
                args = node.args
                for group in (args.posonlyargs, args.args, args.kwonlyargs):
                    local.update(a.arg for a in group)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None and isinstance(item.optional_vars, ast.Name):
                        local.add(item.optional_vars.id)
            elif isinstance(node, ast.Global):
                local.update(node.names)
        local.update({"self", "cls"})
        missing = sorted(
            {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id not in local and not hasattr(builtins, n.func.id)}
        )
        if missing:
            offenders.append(f"{path.relative_to(package.parent)}: {missing}")
    check("无未定义调用", not offenders, "; ".join(offenders[:6]))


def test_qa_selection() -> None:
    """--qa / --max-qa / 预算闸门的 QA 选择逻辑（纯 CPU）。"""
    section("12. search：QA 选择（--qa / --max-qa / --list-qa）")
    from codemem.search import runner

    def make(index, category=2, question="q"):
        return (index, {"question": question, "category": category})

    questions = [make(0), make(1, category=5), make(2), make(3, question=""), make(4)]

    # 默认：跳过 category5 与空问题
    picked = runner.select_questions(questions)
    check("默认跳过 category5 与空问题", picked.indices == [0, 2, 4], str(picked.indices))
    check("默认 skipped 计数正确", picked.skipped == 2, str(picked.skipped))

    # --qa：精确点名，优先于 max_qa 与预算
    picked = runner.select_questions(questions, qa_indices=[4, 0], max_qa=1,
                                     max_total_steps=10, max_steps=30)
    check("--qa 精确定点（保序、无视 max_qa/预算）", picked.indices == [0, 4], str(picked.indices))
    check("--qa 标记为 explicit", picked.explicit is True)
    check("--qa 的 note 说明了范围", any("只跑指定 QA" in n for n in picked.notes), str(picked.notes))

    # --qa 点名了不存在的下标：明确报出来，不静默变成"跑了别的"
    picked = runner.select_questions(questions, qa_indices=[0, 99])
    check("--qa 不存在的下标被报出", any("99" in n for n in picked.notes), str(picked.notes))
    check("--qa 存在的下标照常跑", picked.indices == [0], str(picked.indices))

    # --qa 点名一条 category5 的：仍然跳过（skip_category5 依然生效）
    picked = runner.select_questions([make(1, category=5)], qa_indices=[1])
    check("--qa 点名 category5 仍被跳过", picked.indices == [], str(picked.indices))

    # --max-qa：截断（在 category5 过滤之前，保持旧行为）
    picked = runner.select_questions(questions, max_qa=2)
    check("--max-qa 在过滤前截断", picked.indices == [0], str(picked.indices))

    # 预算闸门：只在没有 --qa 时生效
    picked = runner.select_questions(questions, max_total_steps=10, max_steps=10)
    check("预算闸门生效（10/10 = 1 条）", picked.indices == [0], str(picked.indices))
    check("预算闸门有 note", any("步数预算" in n for n in picked.notes), str(picked.notes))

    # --qa 值与解析
    check("--qa 逗号分隔", runner.parse_qa_indices("3,7,12") == [3, 7, 12])
    check("--qa 空格分隔", runner.parse_qa_indices("3 7 12") == [3, 7, 12])
    check("--qa 混合分隔", runner.parse_qa_indices("3, 7 12") == [3, 7, 12])
    check("--qa 单个", runner.parse_qa_indices("5") == [5])
    try:
        runner.parse_qa_indices("3,abc")
        check("--qa 非法值报错", False, "没有报错")
    except ValueError as exc:
        check("--qa 非法值报错", "abc" in str(exc), str(exc))

    # --list-qa 输出
    listing = runner.format_qa_listing(questions, "Test_Pair")
    lines = listing.split("\n")
    check("--list-qa 含表头", lines[0].startswith("# Test_Pair"))
    check("--list-qa 每条一行", len(lines) == len(questions) + 2, str(len(lines)))
    check("--list-qa 含下标与问题", "     0" in listing and "q" in listing)

    # --repeat：同一条 QA 多次运行的一致性汇总
    def rec(index, repeat, sufficient, answer, count):
        return runner.QaRecord(qa_index=index, repeat=repeat, question="q",
                               sufficient=sufficient, final_answer=answer,
                               evidence_count=count)

    stable = [rec(0, 1, True, "A", 3), rec(0, 2, True, "A", 3)]
    lines = runner.repeat_agreement(stable)
    check("repeat 一致时报一致", lines and "一致" in lines[0], str(lines))
    unstable = [rec(0, 1, True, "A", 3), rec(0, 2, False, "B", 1)]
    lines = runner.repeat_agreement(unstable)
    check("repeat 不一致时报出来", lines and "不一致" in lines[0], str(lines))
    check("repeat 报告不同答案数", lines and "2 种" in lines[0], str(lines))
    check("单次运行不产生 repeat 报告", runner.repeat_agreement([rec(0, 1, True, "A", 1)]) == [])


def test_agent_guards() -> None:
    """agent 的两条机制提醒（纯 CPU）：该动笔了 / 写太多了。"""
    section("13. agent 机制提醒（write nudge / oversize guard）")
    from codemem.search import agent as A

    nudge = A.write_nudge_notice(7, 0)
    check("空 evidence 的动笔提醒说明了步数", "7 steps" in nudge, nudge[:120])
    check("动笔提醒要求写下来", "WRITE" in nudge)
    check("动笔提醒反对继续读", "re-reading" in nudge or "Stop reading" in nudge)
    partial = A.write_nudge_notice(9, 3)
    check("非空 evidence 的提醒措辞不同", "3 record" in partial and "EMPTY" not in partial,
          partial[:120])

    over = A.oversize_notice(658, 40)
    check("过大提醒报出实际条数与目标区间", "658" in over and "5-15" in over, over[:150])
    check("过大提醒要求精简而非截断", "trimmed" in over)
    check("过大提醒给出集合型问题的替代写法", "ONE record" in over)


def test_judge_temporal() -> None:
    """judge 的相对时间折算（纯 CPU）—— 8B 判不准日期算术，所以由代码算。"""
    section("14. judge：相对时间确定性折算")
    from codemem.eval import judge as J

    cases = [
        ("The week before 6 July 2023", "2023-06-29"),
        ("a week before 6 July 2023", "2023-06-29"),
        ("The Friday before 15 July 2023", "2023-07-08"),
        ("The weekend before 20 October 2023", "2023-10-14"),
        ("two weeks before 1 June 2023", "2023-05-18"),
    ]
    for reference, expected in cases:
        got = J.resolve_relative_date(reference)
        check(f"折算 {reference!r}", got is not None and got.isoformat() == expected,
              f"{got} != {expected}")

    check("非相对时间参考返回 None",
          J.resolve_relative_date("2 July 2023") is None)
    check("无法解析的参考返回 None",
          J.resolve_relative_date("sometime last summer") is None)

    # 日期解析的四种写法
    from datetime import date
    for text, expected in [("2023-06-29", date(2023, 6, 29)),
                           ("6 July 2023", date(2023, 7, 6)),
                           ("July 6, 2023", date(2023, 7, 6)),
                           ("July 2023", date(2023, 7, 1))]:
        check(f"解析 {text!r}", J._parse_date(text) == expected, str(J._parse_date(text)))

    # hint：同一天要明确说 SAME DATE
    hint = J.temporal_hint("The week before 6 July 2023", "2023-06-29")
    check("同一天给 SAME DATE 提示", "SAME date" in hint, hint[:140])
    hint = J.temporal_hint("The week before 6 July 2023", "2023-07-06")
    check("差一周时报告差异", "differ" in hint, hint[:140])
    # 不是时间型参考时不干扰 judge
    check("非时间参考不给提示", J.temporal_hint("2 July 2023", "2 July 2023") == "")
    check("预测无日期时如实说明", "does not state" in J.temporal_hint(
        "The week before 6 July 2023", "some time in June"))

    # hint 要真的进入 judge 的消息
    msgs = J.build_judge_messages("when?", "The week before 6 July 2023", "2023-06-29")
    check("hint 注入到 judge 消息", "PRECOMPUTED" in msgs[1]["content"])


def test_answer_parse() -> None:
    """answer 步骤的输出解析（纯 CPU）。"""
    section("10. answer：输出解析与证据渲染")
    from codemem.answer import answer as A

    cases = [
        ('{"answer":"2023-05-07","unsupported":false,"reasoning":"x"}', "2023-05-07", False),
        ('```json\n{"answer": "May 2023", "unsupported": false}\n```', "May 2023", False),
        ('Sure: {"answer": null, "unsupported": true} done', None, True),
        ('{"prediction":"2022"}', "2022", False),
        ('{"answer":"x","reasoning":{"a":1}}', "x", False),
        ('<answer>2022</answer>', "2022", False),
        ("2022", "2022", False),
        ("", None, True),
        ("not json at all", "not json at all", False),
    ]
    for text, want_answer, want_unsupported in cases:
        result = A.parse_answer(text)
        check(f"parse {text[:32]!r}", result.answer == want_answer
              and result.unsupported == want_unsupported,
              f"answer={result.answer!r} unsupported={result.unsupported}")
    # 显式 unsupported=false + 空答案：尊重模型判断（由 judge 判错，而不是归成证据不足）
    result = A.parse_answer('{"answer": "", "unsupported": false}')
    check("空答案 + 显式 unsupported=false 尊重模型判断",
          result.answer is None and result.unsupported is False,
          f"answer={result.answer!r} unsupported={result.unsupported}")

    memory = {"memory": "X happened", "metadata": {"time": "2023-05-07",
                                                  "tag": ["speaker:Caroline"]}}
    check("渲染带 time + speaker", A.render_evidence([memory]) == "[2023-05-07] Caroline: X happened",
          A.render_evidence([memory]))
    check("无证据时给出占位", "no memories" in A.render_evidence([]))
    check("current_date 取最晚日期",
          A.current_date_hint([memory, {"memory": "y", "metadata": {"time": "2023-01-01"}}])
          == "2023-05-07")
    check("无日期时回退今天", len(A.current_date_hint([{"memory": "y", "metadata": {}}])) == 10)


def test_eval_metrics() -> None:
    """eval 步骤的评分与汇总（纯 CPU，不调 judge）。"""
    section("11. eval：评分与失败归类")
    from codemem.eval import metrics, runner

    check("evidence_to_ids 解析 D1:3",
          metrics.evidence_recall({"session_1_3"}, {"session_1_3", "session_2_1"}) == 1.0)
    check("evidence_recall 部分命中",
          metrics.evidence_recall({"session_1_3", "session_2_1"}, {"session_1_3"}) == 0.5)
    check("evidence_recall 无期望时为 1.0", metrics.evidence_recall(set(), set()) == 1.0)

    items = [
        runner.ScoredItem(qa_index=0, question="q", reference="r", candidate="r",
                          unsupported=False, category=2, judge_label="CORRECT",
                          token_f1=1.0, evidence_recall=1.0, evidence_count=3),
        runner.ScoredItem(qa_index=1, question="q", reference="r", candidate="w",
                          unsupported=False, category=2, judge_label="WRONG",
                          token_f1=0.0, evidence_recall=0.5, evidence_count=3),
        runner.ScoredItem(qa_index=2, question="q", reference="r", candidate=None,
                          unsupported=True, category=1, judge_label="WRONG",
                          token_f1=0.0, evidence_recall=0.0, evidence_count=0),
        runner.ScoredItem(qa_index=3, question="q", reference="r", candidate="x",
                          unsupported=False, category=5, judge_label="", skipped=True),
    ]
    summary = runner.summarize(items)
    overall = summary["overall"]
    check("跳过 category5", overall["skipped"] == 1 and overall["scored"] == 3,
          f"scored={overall['scored']} skipped={overall['skipped']}")
    # summarize 里对均值做了 round(...,4)，所以容差按 1e-4 比
    check("judge_accuracy 只算判过的（2 correct / 3 judged 之外的）",
          abs(overall["judge_accuracy"] - 1 / 3) < 1e-3, str(overall["judge_accuracy"]))
    check("unsupported_rate 正确",
          abs(overall["unsupported_rate"] - 1 / 3) < 1e-3, str(overall["unsupported_rate"]))
    check("evidence_recall 均值正确",
          abs(overall["evidence_recall"] - 0.5) < 1e-6, str(overall["evidence_recall"]))
    check("by_category 含 temporal", "temporal" in summary["by_category"],
          str(list(summary["by_category"])))
    check("category5 不出现在 by_category", "adversarial_skip" not in summary["by_category"])

    breakdown = runner.failure_breakdown(items)
    check("失败归类：证据不足 1 条", breakdown.get("missing_evidence") == 1, str(breakdown))
    check("失败归类：答错 1 条", breakdown.get("wrong_answer") == 1, str(breakdown))

    check("load_evidence_ids 空路径返回空集", runner.load_evidence_ids("") == set())

    # 没跑 judge 时不能报成 0% 准确率（那会被读成"全答错了"）——给 None
    unjudged = [runner.ScoredItem(qa_index=0, question="q", reference="r", candidate="c",
                                  unsupported=False, category=2)]
    plain = runner.summarize(unjudged)["overall"]
    check("未跑 judge 时 accuracy 为 None 而非 0.0", plain["judge_accuracy"] is None,
          str(plain["judge_accuracy"]))
    check("未跑 judge 时 judged=0", plain["judged"] == 0)
    breakdown_unjudged = runner.failure_breakdown(unjudged)
    check("未跑 judge 归到 unjudged 而不是 wrong_answer",
          breakdown_unjudged.get("unjudged") == 1 and "wrong_answer" not in breakdown_unjudged,
          str(breakdown_unjudged))
    payload = runner.ScoredItem(qa_index=0, question="q", reference="r", candidate="r",
                                unsupported=False, category=2, judge_label="CORRECT")
    check("ScoredItem.to_dict 可序列化", isinstance(payload.to_dict(), dict))
    check("correct 只看 judge_label",
          payload.correct and not runner.ScoredItem(
              qa_index=1, question="q", reference="r", candidate="r",
              unsupported=False, category=2, judge_label="WRONG").correct)


def test_timecalc() -> None:
    section("9. timecalc：确定性日期算术")
    from codemem.search import timecalc

    def run(*args: str) -> tuple[int, str]:
        return timecalc.run(list(args))

    check("shift -1month", run("shift", "2023-06-17", "-1month") == (0, "2023-05-17"),
          str(run("shift", "2023-06-17", "-1month")))
    check("shift +2week", run("shift", "2023-05-08", "+2week") == (0, "2023-05-22"),
          str(run("shift", "2023-05-08", "+2week")))
    check("月份越界取月末", run("shift", "2023-01-31", "+1month") == (0, "2023-02-28"),
          str(run("shift", "2023-01-31", "+1month")))
    check("多个偏移叠加", run("shift", "2023-06-17", "-1month", "+3day") == (0, "2023-05-20"),
          str(run("shift", "2023-06-17", "-1month", "+3day")))
    code, out = run("diff", "2023-05-08", "2023-06-17")
    check("diff 天数正确", code == 0 and "+40 days" in out, out)
    code, out = run("diff", "2023-06-17", "2023-05-08")
    check("diff 带符号", code == 0 and "-40 days" in out, out)
    check("range 输出 ISO 区间",
          run("range", "2023-05-01", "2023-05-31") == (0, "2023-05-01/2023-05-31"),
          str(run("range", "2023-05-01", "2023-05-31")))
    code, out = run("info", "2023-05-08T13:56:00")
    check("info 给出星期", code == 0 and "Monday" in out, out)
    check("只给日期时不造出假时分",
          run("shift", "2023-06-17", "-1month")[1] == "2023-05-17")
    check("坏偏移报错", run("shift", "2023-05-08", "x1month")[0] == 1)
    check("坏日期报错", run("diff", "2023-13-01", "2023-01-01")[0] == 1)
    check("参数不足给用法", run("shift", "2023-05-08")[0] == 2)
    check("未知命令报错", run("frobnicate")[0] == 2)
    check("无参数给用法", run()[0] == 2)



# ---------------------------------------------------------------------------
# 7. searchctl CLI 端到端
# ---------------------------------------------------------------------------

async def test_searchctl_cli(tmp: Path) -> None:
    section("7. searchctl：CLI 端到端（词法模式）")
    ws = tmp / "cli_ws"
    (ws / "inputs").mkdir(parents=True, exist_ok=True)
    corpus = [
        make_memory("a1", "Caroline joined an LGBTQ support group"),
        make_memory("a2", "Melanie asked Caroline when she started going"),
        make_memory("a3", "Dave likes pizza"),
    ]
    (ws / "inputs/atommem.jsonl").write_text(
        "".join(json.dumps(m) + "\n" for m in corpus), encoding="utf-8"
    )
    (ws / "inputs/msgmem.jsonl").write_text(
        json.dumps({"memory": "Hey Mel!", "metadata": {
            "id": "session_1_1", "type": "raw", "time": "", "tag": ["speaker:Caroline"],
            "source": [], "changelog": []}}) + "\n", encoding="utf-8"
    )
    (ws / "evidence.jsonl").write_text(
        json.dumps(make_memory("e9", "Caroline started the group in June")) + "\n", encoding="utf-8"
    )

    from codemem.search.searchctl import main as searchctl_main

    # 词法模式**不需要**索引：索引只有 dense 通道要。这正是降级路径的意义。
    code = await asyncio.to_thread(searchctl_main, ["LGBTQ support group", "-k", "2",
                                                    "--lexical", "--atoms", str(ws / "inputs/atommem.jsonl"),
                                                    "--evidence", str(ws / "evidence.jsonl"),
                                                    "--index", str(tmp / "nope_idx")])
    check("词法模式缺索引也能跑（退出码 0）", code == 0, str(code))

    # dense 模式缺索引 -> 明确报错（退出码 2），不静默给出无意义排序
    os.environ["CODEMEM_EMBED_URL"] = "http://127.0.0.1:1/v1"
    try:
        code = await asyncio.to_thread(
            searchctl_main, ["LGBTQ support group", "-k", "2",
                             "--atoms", str(ws / "inputs/atommem.jsonl"),
                             "--evidence", str(ws / "evidence.jsonl"),
                             "--index", str(tmp / "nope_idx")]
        )
        check("dense 模式缺索引明确报错（退出码 2）", code == 2, str(code))
    finally:
        os.environ.pop("CODEMEM_EMBED_URL", None)

    # 无 CODEMEM_INDEX 但给一个空索引也不行；改用模块内函数直接验证 merge_pool
    from codemem.search import searchctl
    raws = [make_memory("r1", "raw message one"), make_memory("r2", "raw message two")]
    pool = searchctl.merge_pool(corpus, [make_memory("e9", "Caroline started the group")], "all", raw=raws)
    check("pool 合并 base+evidence+raw", len(pool) == 6, str(len(pool)))
    pool_ev = searchctl.merge_pool(corpus, [make_memory("a1", "overridden")], "all", raw=raws)
    check("同 id 时 evidence 侧优先", pool_ev[0]["memory"] == "overridden")
    pool_base = searchctl.merge_pool(corpus, [make_memory("e9", "x")], "base", raw=raws)
    check("--source base 排除 evidence 与 raw", len(pool_base) == 3, str(len(pool_base)))
    pool_raw = searchctl.merge_pool(corpus, [], "raw", raw=raws)
    check("--source raw 只搜原话", [m["metadata"]["id"] for m in pool_raw] == ["r1", "r2"],
          str([m["metadata"]["id"] for m in pool_raw]))
    check("--source raw 无 raw 语料时为空", searchctl.merge_pool(corpus, [], "raw") == [])

    # --tag 过滤（枚举型问题的关键工具）
    tagged = [
        make_memory("a1", "Melanie is a big fan of pottery", source=["s1"]),
        make_memory("a2", "Melanie asks Caroline a question", source=["s2"]),
        make_memory("a3", "Melanie went camping", source=["s3"]),
    ]
    tagged[0]["metadata"]["tag"] = ["speaker:Melanie", "action:pottery"]
    tagged[1]["metadata"]["tag"] = ["speaker:Melanie", "action:ask"]
    tagged[2]["metadata"]["tag"] = ["speaker:Melanie", "event:camping trip"]

    picked = searchctl.filter_by_tag(tagged, ["action"])
    check("--tag 子串匹配", [m["metadata"]["id"] for m in picked] == ["a1", "a2"],
          str([m["metadata"]["id"] for m in picked]))
    picked = searchctl.filter_by_tag(tagged, ["action"], exclude=True)
    check("--exclude-tag 取反", [m["metadata"]["id"] for m in picked] == ["a3"],
          str([m["metadata"]["id"] for m in picked]))
    picked = searchctl.filter_by_tag(tagged, ["action:ask"], exclude=True)
    check("--exclude-tag action:ask 精确排除", [m["metadata"]["id"] for m in picked] == ["a1", "a3"],
          str([m["metadata"]["id"] for m in picked]))
    picked = searchctl.filter_by_tag(tagged, ["Action", "EVENT"])
    check("--tag 大小写不敏感且可多值", len(picked) == 3, str(len(picked)))
    check("空 pattern 不过滤", searchctl.filter_by_tag(tagged, []) == tagged)
    check("无命中返回空", searchctl.filter_by_tag(tagged, ["nope"]) == [])
    # 无 tag 字段的记录不该让过滤崩溃
    check("缺 tag 的记录被安全排除",
          searchctl.filter_by_tag([{"memory": "x", "metadata": {"id": "n"}}], ["action"]) == [])

    # --tag-values：列出标签值分布（枚举型问题的第一步）
    counted = searchctl.tag_value_counts(tagged)
    check("tag_value_counts 统计全部", dict(counted).get("speaker:Melanie") == 3, str(counted))
    filtered = searchctl.tag_value_counts(tagged, ["action"])
    check("tag_value_counts 按前缀过滤",
          dict(filtered) == {"action:pottery": 1, "action:ask": 1}, str(filtered))
    check("tag_value_counts 按次数降序",
          [t for t, _ in searchctl.tag_value_counts(tagged, ["speaker"])] == ["speaker:Melanie"])
    check("tag_value_counts 空池返回空", searchctl.tag_value_counts([]) == [])

    # 溯源/发散字段
    atom = make_memory("session_1_11_1", "keen on counseling", source=["session_1_11"])
    prov = searchctl.provenance_fields(atom)
    check("原子的 kind=atom 且 source 指向原话",
          prov["kind"] == "atom" and prov["source"] == ["session_1_11"], str(prov))
    raw_record = make_memory("session_1_11", "original words", source=[], kind="raw")
    raw_prov = searchctl.provenance_fields(raw_record)
    check("原话的 kind=raw 且 source 为空",
          raw_prov["kind"] == "raw" and raw_prov["source"] == [], str(raw_prov))

    # 环境变量（由 harness 注入）不应影响词法检索。searchctl_main 内部会 asyncio.run，
    # 而本函数本身跑在事件循环里，所以直接 await 它的异步实现。
    from codemem.search import searchctl as searchctl_mod

    args = searchctl_mod.parse_args(
        ["x", "-k", "1", "--lexical", "--atoms", str(ws / "inputs/atommem.jsonl"),
         "--evidence", str(ws / "evidence.jsonl"), "--index", str(tmp / "nope_idx")]
    )
    rc = await searchctl_mod.run_search(args)
    check("CLI 在无 CODEMEM_* 环境变量下也能跑", rc == 0, str(rc))


# ---------------------------------------------------------------------------
# 8. evomem_v3 端到端（假模型）
# ---------------------------------------------------------------------------

async def test_end_to_end(tmp: Path) -> None:
    section("8. evomem_v3：端到端（假模型 + 真工具）")
    from codemem.search import runner as evomem_v3

    data_dir = tmp / "data"
    sample_dir = data_dir / "Test_Pair"
    sample_dir.mkdir(parents=True, exist_ok=True)
    corpus = [
        make_memory("a1", "Caroline joined an LGBTQ support group"),
        make_memory("a2", "Caroline has been going for about a month"),
    ]
    (sample_dir / "atommem.jsonl").write_text(
        "".join(json.dumps(m) + "\n" for m in corpus), encoding="utf-8"
    )
    (sample_dir / "msgmem.jsonl").write_text(
        "".join(json.dumps(make_memory(f"s{i}", f"raw message {i}", source=[])) + "\n"
                for i in range(3)), encoding="utf-8"
    )

    run_dir = tmp / "runs"
    config = evomem_v3.load_config(None)
    config["max_steps"] = 6
    config["max_verify_rounds"] = 2
    config["concurrency"] = 1
    config["log"] = {"level": "error", "output": ""}
    catalog = toolcfg.load_catalog(PROJECT_ROOT / "configs/tool.json")
    log = Logger(level="error")

    inputs_dir, index_dir, search_bin, lexical = await evomem_v3.build_inputs(
        sample_dir, run_dir, config, embedder=None, log=log
    )
    check("inputs 已建立", (inputs_dir / "atommem.jsonl").exists())
    check("inputs 是硬链（同一 inode）",
          (inputs_dir / "atommem.jsonl").stat().st_ino == (sample_dir / "atommem.jsonl").stat().st_ino)
    check("search 包装可执行", search_bin.exists() and (search_bin.stat().st_mode & 0o111))
    check("无 embedding -> lexical 降级", lexical is True)

    # 假模型：第一个 QA 先写 evidence 再"完成"；verifier 由 role 区分
    class ScriptedModel:
        def __init__(self) -> None:
            self.agent_turns = 0
            self.verify_turns = 0

        async def __call__(self, messages: list[dict], meta: dict) -> str:
            if meta.get("role") == "verifier":
                self.verify_turns += 1
                # 第一次判不足，第二次判足够 -> 覆盖"缺口回灌 + 再跑一轮"
                if self.verify_turns == 1:
                    return '{"sufficient": false, "answer": "", "missing": ["the year"]}'
                return '{"sufficient": true, "answer": "about a month before June 2023", "missing": []}'
            self.agent_turns += 1
            if self.agent_turns == 1:
                return json.dumps({"tool": "bash", "args": {
                    "command": "jq -c . inputs/atommem.jsonl >> evidence.jsonl && "
                               "jq -s length evidence.jsonl",
                    "description": "Assembling evidence"}})
            if self.agent_turns == 2:
                return json.dumps({"tool": "bash", "args": {
                    "command": "jq -c '{id,memory}' evidence.jsonl",
                    "description": "Verifying evidence"}})
            return "I am done."

    model = ScriptedModel()
    # 直接跑 run_qa（绕开 QA 数据源，注入一条假问题）
    record = await evomem_v3.run_qa(
        qa_index=0, qa={"question": "When did Caroline join the group?", "category": 2,
                        "reference": "June 2023"},
        directory=sample_dir, run_dir=run_dir, inputs_dir=inputs_dir, index_dir=index_dir,
        config=config, catalog=catalog, model=model, embedder=None, lexical=lexical,
        log=log, semaphore=asyncio.Semaphore(1),
    )
    check("QA 记录有 evidence", record.evidence_count == 2, str(record.evidence_count))
    check("QA 记录了 2 轮", len(record.rounds) == 2, str(len(record.rounds)))
    check("第 1 轮注入缺口后续跑", len(record.verifications) == 2)
    check("最终判定为足够", record.sufficient is True, record.stop_reason)
    check("带上最终答案", "June 2023" in record.final_answer, record.final_answer)
    check("stop_reason=sufficient", record.stop_reason == "sufficient", record.stop_reason)
    check("工具调用被记账", record.tool_calls >= 2, str(record.tool_calls))
    check("未篡改只读输入", record.tampered is False)
    check("evidence ids 正确", record.evidence_ids == ["a1", "a2"], str(record.evidence_ids))

    # --- 篡改检测：手写一个 QA，模型去改 inputs/ ---
    class TamperingModel:
        def __init__(self) -> None:
            self.turn = 0

        async def __call__(self, messages: list[dict], meta: dict) -> str:
            if meta.get("role") == "verifier":
                return '{"sufficient": true, "answer": "x", "missing": []}'
            self.turn += 1
            if self.turn == 1:
                # 软链逃逸：写 inputs/atommem.jsonl（resolve 后在外，应被拒）
                return json.dumps({"tool": "write", "args": {
                    "path": "inputs/atommem.jsonl", "content": "{}"}})
            return "done"

    record2 = await evomem_v3.run_qa(
        qa_index=1, qa={"question": "Anything?", "category": 2, "reference": ""},
        directory=sample_dir, run_dir=run_dir, inputs_dir=inputs_dir, index_dir=index_dir,
        config=config, catalog=catalog, model=TamperingModel(), embedder=None, lexical=lexical,
        log=log, semaphore=asyncio.Semaphore(1),
    )
    check("写只读输入被拒（未篡改）", record2.tampered is False)
    check("原 atommem.jsonl 未被改动",
          "LGBTQ" in (inputs_dir / "atommem.jsonl").read_text())
    check("工具错误被记账", record2.rounds[0]["steps"][0]["ok"] is False
          or not record2.rounds[0]["steps"][0]["result_text"] == "")

    # --- summary 生成 ---
    summary = evomem_v3.summarize_dir(
        "Test_Pair", [record, record2], type("M", (), {"stats": staticmethod(
            lambda: {"calls": 9, "failures": 0})})(), 2, 0, lexical
    )
    check("summary 有 qa_done", summary["qa_done"] == 2, str(summary["qa_done"]))
    # 只有第一条被判足够（第二条是篡改测试，verifier 直接返回 sufficient）
    check("summary 统计 sufficient 条数", summary["sufficient"] == sum(
        1 for r in [record, record2] if r.sufficient), str(summary["sufficient"]))
    check("summary 的 sufficient_rate 与条数一致",
          abs(summary["sufficient_rate"] - summary["sufficient"] / 2) < 1e-9,
          str(summary["sufficient_rate"]))
    check("summary 标记 lexical_fallback", summary["lexical_fallback"] is True)
    check("summary 统计 model_calls", summary["model_calls"] == 9, str(summary["model_calls"]))


# ---------------------------------------------------------------------------

def main() -> int:
    with tempfile.TemporaryDirectory(prefix="codemem-v3-test-") as handle:
        tmp = Path(handle)
        test_no_undefined_names()
        test_toolcfg(tmp)
        test_evidence(tmp)
        test_timecalc()
        test_qa_selection()
        test_agent_guards()
        test_judge_temporal()
        test_answer_parse()
        test_eval_metrics()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(test_verifier())
            loop.run_until_complete(test_tools(tmp))
            loop.run_until_complete(test_search(tmp))
            loop.run_until_complete(test_agent(tmp))
            loop.run_until_complete(test_compaction_units(tmp))
            loop.run_until_complete(test_searchctl_cli(tmp))
            loop.run_until_complete(test_end_to_end(tmp))
        finally:
            loop.close()
        # test_tools 里的纯 CPU 部分用到 get_event_loop，跑完补一次
    print(f"\n{'=' * 60}\n通过 {PASS} / 失败 {FAIL}\n{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
