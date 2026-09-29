#!/usr/bin/env python
"""search 步骤的纯 CPU 自测：假模型 + 真工具，零模型成本、零网络。

覆盖：

1. ``toolcfg``：catalog 加载与校验、prompt 渲染（槽位全部替换）、响应解析的稳健性。
2. ``tools``：read/write/edit/bash 的正常路径 + 全部失败路径 + 路径约束。
3. ``evidence``：逐行校验、补全 changelog、拒绝缺 source / 重复 id / 坏类型。
4. ``verifier``：解析稳健性 + 解析失败**保守判不足**。
5. ``agent``：用假模型脚本驱动一轮 agent loop，验证无进展检测与失败不中断。
6. ``runner``：dry-run 渲染，以及用假模型跑通一条 QA 的端到端（含篡改检测）。

检索不需要测 —— 它就是 ``grep``，没有我们的代码。
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
from codemem.search import prune as prune_mod   # noqa: E402
from codemem.search import toolcfg              # noqa: E402
from codemem.search import tools as tools_mod   # noqa: E402
from codemem.search import verifier as verifier_mod  # noqa: E402
from codemem.log import Logger                  # noqa: E402

PASS, FAIL = 0, 0


def symlinks_available() -> bool:
    """本机能否创建符号链接。

    Windows 上普通账户创建符号链接需要开发者模式或管理员权限，否则 ``WinError 1314``；
    而软链逃逸、每 QA 一个 inputs 链接这些路径都依赖它。**跳过而不是失败** ——
    这是环境能力的差异，不是代码的错（生产侧 ``link_dir_or_copy`` 会自动退回硬链）。
    """
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as handle:
        base = Path(handle)
        (base / "d").mkdir()
        try:
            (base / "l").symlink_to(base / "d", target_is_directory=True)
            return True
        except (OSError, NotImplementedError):
            return False


SYMLINKS = symlinks_available()


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
    check("命令含 grep / date（检索与算术都走 bash 原生工具）",
          set(catalog.commands) == {"grep", "time"}, str(set(catalog.commands)))
    catalog.validate(["read", "write", "edit", "bash"])
    try:
        catalog.validate(["read", "write", "edit", "bash", "nope"])
        check("校验能发现缺描述的工具", False, "没有报错")
    except toolcfg.ToolConfigError:
        check("校验能发现缺描述的工具", True)

    rendered = catalog.render_prompt("Who is Caroline?")
    for slot in ("{{TOOLS}}", "{{COMMANDS}}", "{{WORKSPACE}}", "{{SCHEMA}}", "{{PROTOCOL}}",
                 "{{WRITING_POLICY}}", "{{TIME_POLICY}}", "{{QUESTION}}"):
        check(f"渲染后没有残留 {slot}", slot not in rendered)
    check("渲染含问题原文", "Who is Caroline?" in rendered)
    # 检索就是 grep：语料是 JSONL，行号稳定，可以 grep -n 再 read offset
    check("渲染含 grep 用法", "grep -in" in rendered)
    check("渲染给出 grep -n 定位 + read 精读的组合", "offset=" in rendered and "read inputs/sessions.jsonl" in rendered)
    check("渲染含 date -d 做时间算术", "date -d" in rendered)
    check("不再有已删除的检索 CLI", "search \"" not in rendered and "timecalc" not in rendered)
    check("不再有 jq", "jq " not in rendered)
    check("渲染含 schema 字段", "metadata.source" in rendered)
    check("渲染含 source 非空规则", "NON-EMPTY" in rendered)
    # 按问题类型选策略
    check("渲染给出策略 A（关键词多跳）", "KEYWORD HOP" in rendered)
    check("渲染给出策略 B（遍历多会话）", "SWEEP" in rendered)
    check("渲染说明集合型问题走遍历", "the answer is a SET" in rendered)
    check("渲染给出两种策略的判据", "PICK YOUR STRATEGY" in rendered)
    # 核心：证据是**推导**出来的，不是抄的
    check("渲染含'推导而非抄袭'策略", "REASON, do not just copy" in rendered)
    check("策略提到事件/性格推理", "joined into the one" in rendered and "summarised" in rendered)
    check("策略要求消解上下文", "RESOLVE THE CONTEXT" in rendered)
    check("schema 说明可以改写 content", "Rewrite freely" in rendered)
    check("schema 要求 source 非空", "NON-EMPTY array of msg_ids" in rendered)
    # 时间精度：这是这一版的核心规则
    check("时间策略：不得编造更细的精度", "Never invent precision" in rendered)
    check("时间策略：按源语料的单位作答", "answer in that unit" in rendered)
    check("时间策略：保留锚点表述", "the week before 9 June 2023" in rendered)
    check("时间策略：区间是合法答案", "A period is a valid answer" in rendered)
    check("时间策略：记录必须自带锚点", "stands ALONE" in rendered)
    check("时间策略：原文表述要照录", "verbatim" in rendered)
    check("有 worked example（关键词多跳）", "WORKED EXAMPLE (strategy A: keyword hop)" in rendered)
    check("有 worked example（相对时间）", "date -d \"8 May 2023 -1 day\"" in rendered)
    check("有 worked example（遍历）", "WORKED EXAMPLE (strategy B: sweep)" in rendered)
    # 语料只剩两层：不能再出现已删除的 speakers summary
    check("提示里不再有 speakers summary", "speakers_summary" not in rendered)
    check("提示里不再有 atommem/msgmem", "atommem" not in rendered and "msgmem" not in rendered)
    # 上下文预算：prompt 每条 QA 都整体注入一次，涨回去就等于白做这次简化
    check("system prompt 控制在 14000 字符内", len(rendered) < 14000, str(len(rendered)))

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
    if not SYMLINKS:
        print("  skip 软链逃逸检查（本机不允许创建符号链接）")
    else:
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

    prefix = V3.build_shell_prefix(inputs_dir=ws, config={})
    probe = tools_mod.build_tools(cwd=ws, shell_command_prefix=prefix)
    result = await probe["bash"].run(
        {"command": "echo \"$CODEMEM_INPUTS\"", "description": "Checking inputs env"}
    )
    check("语料目录经环境变量传进子进程", ws.as_posix() in result.text.replace("\\", "/"), result.text)

    # 回归：date 的输出必须与 locale 无关。中文环境下 `date +%b` 会给「5月」，写进 evidence
    # 就是一条下游读不了的记录 —— 所以 harness 注入 LC_ALL=C。
    result = await probe["bash"].run({
        "command": "date -d \"8 May 2023 -1 day\" +\"%d %b %Y\"; date -d \"15 Jul 2023 -2 day\" +\"%A\"",
        "description": "Checking date locale",
    })
    check("date 输出月份/星期为英文（LC_ALL=C 生效）",
          "May" in result.text and "Thursday" in result.text, result.text)

    # --- grep 无匹配（退出码 1）是**最容易被误判成失败**的一种观测 ---
    from codemem.search.toolcfg import ToolCall

    notice = agent_mod.grep_no_match_notice()
    check("grep 提示说明退出码 1 = 无匹配而非崩溃", "not a failure" in notice, notice[:200])
    check("grep 提示给出下一步（换文件/换词）", "other file" in notice, notice[:250])
    check("grep 提示给出查看真实内容的命令", "grep -c ." in notice, notice[:300])

    call = ToolCall(tool="bash", args={"command": "sed -n '1p' f.jsonl"})
    empty = agent_mod.no_output_notice(call)
    check("非 grep 的空输出不误报 grep 语义", "grep exited 1" not in empty)

    # --- 重复拒绝消息要给出脱困方向 ---
    repeat = agent_mod.repeat_notice(ToolCall(tool="bash", args={"command": "x"}), 2)
    check("重复拒绝消息提到真实语料文件", "inputs/sessions.jsonl" in repeat)
    check("重复拒绝消息提示多跳检索", "hop" in repeat.lower())
    check("重复拒绝消息给出集合型问题的出路（遍历）", "SET" in repeat)
    # 已删除的语料名不该再出现在给模型的文案里（会诱使它去搜不存在的文件）
    for stale in ("atommem", "msgmem", "speakers_summary", "timecalc", "jq"):
        check(f"重复拒绝消息不含已删除的 {stale}", stale not in repeat)
        check(f"grep 提示不含已删除的 {stale}", stale not in notice)


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


# ---------------------------------------------------------------------------
# 4. evidence
# ---------------------------------------------------------------------------

def test_evidence(tmp: Path) -> None:
    section("4. evidence：逐行校验与规范化（新 evidence_template 格式）")
    path = tmp / "evidence.jsonl"

    def ev(content, source=None, score=None, changelog=None):
        meta = {"source": ["session_1_1"] if source is None else source}
        if score is not None:
            meta["score"] = score
        meta["changelog"] = [{"time": "2026-01-01T00:00:00Z", "content": "created"}] \
            if changelog is None else changelog
        return {"content": content, "metadata": meta}

    good = ev("Caroline joined a support group", score=0.9)
    repaired = ev("Caroline has been going for a month", score=0.8, changelog=[])
    no_source = ev("no provenance", source=[])
    dup = ev("Caroline joined a support group", score=0.9)      # 与 good 同 content
    empty = ev("", score=0.5)
    bad_source = ev("bad source", source=["ok", ""])
    lines = [
        json.dumps(good), json.dumps(repaired), json.dumps(no_source), json.dumps(dup),
        json.dumps(empty), json.dumps(bad_source),
        "{not json", "", json.dumps(good),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = evidence_mod.validate_file(path)
    check("记录数统计正确（9 行 - 1 坏 - 1 空）", report.total_lines == 7,
          str(report.total_lines))
    check("空行被计数", report.empty_lines == 1)
    check("解析失败被计数", len(report.parse_errors) == 1)
    check("补全被计数（score/changelog 缺失）", len(report.repaired) == 1, report.summary())
    check("补全后 changelog 非空", report.repaired[0]["metadata"]["changelog"] != [])
    check("交付数正确（good + repaired）", len(report.valid) == 2, report.summary())
    check("重复 content 被拒",
          any("duplicate" in item["reason"] for item in report.rejected),
          str([i["reason"] for i in report.rejected]))
    check("缺 source 被拒", any("source" in item["reason"] for item in report.rejected))
    check("空 content 被拒", any("content" in item["reason"] for item in report.rejected))
    check("source 里有空串被拒",
          any("source" in item["reason"] for item in report.rejected))
    check("summary 可读", "交付 2" in report.summary(), report.summary())

    # score 的规范化：缺失补 0.5；超范围夹到 [0,1]
    _, reason = evidence_mod.normalize({"content": "x", "metadata": {"source": ["s"]}})
    check("缺 score 被补全", reason == "REPAIRED")
    norm, _ = evidence_mod.normalize(
        {"content": "x", "metadata": {"source": ["s"], "score": 87,
                                      "changelog": [{"time": "t", "content": "c"}]}})
    check("百分比 score 被夹到 0-1", norm["metadata"]["score"] == 0.87, str(norm["metadata"]))
    norm, _ = evidence_mod.normalize(
        {"content": "x", "metadata": {"source": ["s"], "score": 5,
                                      "changelog": [{"time": "t", "content": "c"}]}})
    check("0-10 刻度被折算（5 -> 0.5）", norm["metadata"]["score"] == 0.5, str(norm["metadata"]))
    norm, _ = evidence_mod.normalize(
        {"content": "x", "metadata": {"source": ["s"], "score": 200,
                                      "changelog": [{"time": "t", "content": "c"}]}})
    check("越界 score 被夹到 1.0", norm["metadata"]["score"] == 1.0, str(norm["metadata"]))
    # 旧格式仍可读（历史产物）
    legacy = {"memory": "old atom", "metadata": {"id": "a1", "source": ["s", "t"],
                                                 "changelog": [{"time": "t", "content": "c"}]}}
    # 旧格式仍被接受（缺 score 会被补全，所以是 REPAIRED 而非拒绝）
    check("旧格式仍被接受（补 score）", evidence_mod.normalize(legacy)[1] == "REPAIRED",
          str(evidence_mod.normalize(legacy)[1]))

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
    # 语料是两层：summary（粗筛）+ 原始消息。这里用 summary 形态的 fixture。
    one_record = json.dumps(
        {"msg_id": "session_1_summary", "role": "summary", "time": "1:56 pm on 8 May, 2023",
         "content": "Caroline joined a group."}, separators=(",", ":")
    ) + "\n"
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
    # 注意比的是 len(one_record) + 1：文本模式的 write_text 会把 "\n" 写成 CRLF（+1 字节），
    # 而 jq 追加的是 LF。用 `baseline + len(one_record) + 1` 才对得上（实测 118 -> 236）。
    check("重复被拒时文件只增长一次",
          evidence_path.stat().st_size == baseline + len(one_record) + 1,
          f"{baseline} -> {evidence_path.stat().st_size}，期望 +{len(one_record) + 1}")
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
    other = json.dumps(
        {"msg_id": "session_2_summary", "role": "summary", "time": "1:14 pm on 25 May, 2023",
         "content": "Another fact."}, separators=(",", ":")
    ) + "\n"
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
            "jq -c '.content=\"REWRITTEN with the date resolved\"' evidence.jsonl > t "
            "&& mv t evidence.jsonl"), "description": "Rewriting"}}),
        "Done.",
    ]
    outcome, _ = await fresh(script)
    check("原地改写也算进展（唯一集合变了）", outcome.steps[0].progressed,
          str(vars(outcome.steps[0].state_after)))
    # 语料记录没有 metadata.id，标识取 msg_id；evidence 记录没有 id，标识取 content。
    # 判据是"唯一记录集合"而不是"id 数不变"——所以这里断言的正是这一点。
    check("改写后唯一 id 数不变", outcome.steps[0].state_after.unique_ids == 1)
    check("改写确实改变了内容指纹",
          "REWRITTEN" in evidence_path.read_text(encoding="utf-8"),
          evidence_path.read_text(encoding="utf-8")[:200])

    # --- 重复检测必须**感知状态**：文件变了之后同一个命令应该被允许再跑 ---
    # 修文件内容 → 指纹变化 → 之前被拒的命令应放行（成熟 code agent 的语义是
    # "重复一次**没推进**的调用"，不是"这个命令一辈子只能跑一次"）。
    if evidence_path.exists():
        evidence_path.unlink()
    evidence_path.write_text(one_record)
    rewrite = json.dumps({"tool": "write", "args": {
        "path": "evidence.jsonl",
        "content": json.dumps(
            [{"msg_id": "session_1_summary", "role": "summary", "time": "",
              "content": "rewritten body"}], separators=(",", ":")) + "\n"}})
    dedupe = json.dumps({"tool": "bash", "args": {
        "command": "jq -sc 'unique_by(.msg_id)[]' evidence.jsonl > t && mv t evidence.jsonl",
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
        "command": "jq -sc 'unique_by(.msg_id)[]' evidence.jsonl > t && mv t evidence.jsonl",
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
    check("未知工具提示里点名可用的 bash 命令", any("grep" in m["content"]
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
            {"tool": "bash", "args": {"command": f"jq 'select(.msg_id==\"session_{index}_1\")' "
                                                 f"inputs/sessions.jsonl >> evidence.jsonl",
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


def test_verifier_rendering() -> None:
    """verifier 必须真正"看见"证据 —— 这是最容易静默出错的一环。

    实测：``render_memories`` 硬读 v2 的 ``memory``/``metadata.id`` 字段，而新的 evidence
    是 ``content`` + ``metadata.{source,score}``，于是每条记录都渲染成 ``[] () type= tag=[] null``。
    **verifier 收到的证据是空的**，全量 1540 条 QA 的 sufficient 全是 false ——
    看起来像"模型判得太严"，实际是数据管道坏了。0% 这个数字本身就说明不该往能力上归因。
    """
    section("15. verifier：证据渲染（静默出错的高危环节）")
    from codemem.search import verifier as V

    ev = [{"content": "Caroline joined an LGBTQ support group in May 2023.",
           "metadata": {"source": ["session_5_2"], "score": 0.9}}]
    rendered = V.render_memories(ev)
    check("新格式的正文被渲染出来", "joined an LGBTQ support group" in rendered, rendered)
    check("不再出现 v2 的空占位", "type= tag=[]" not in rendered, rendered)
    check("source 被带上", "session_5_2" in rendered, rendered)
    # evidence 记录没有 msg_id（标识就是 content），所以方括号为空、靠 source 溯源 ——
    # 这是预期的；关键是**正文必须渲染出来**，而不是像 bug 时那样渲染成 null
    check("正文不是 null", "null" not in rendered.split(chr(10))[1], rendered[:80])

    msgs = V.build_verifier_messages("When?", ev)
    check("verifier 消息里真的有证据", "LGBTQ support group" in msgs[1]["content"])
    check("verifier 看不到参考答案", "reference" not in msgs[0]["content"].lower())

    # 旧格式仍能渲染（历史产物）
    legacy = [{"memory": "old atom text", "metadata": {"id": "a1", "source": ["s"]}}]
    check("旧格式仍能渲染", "old atom text" in V.render_memories(legacy))
    check("空证据给出说明", "no records" in V.render_memories([]))


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
    # 新格式：content + metadata.source。渲染必须带上正文与来源（这是修掉的 bug 的核心）
    new_ev = {"content": "X happened", "metadata": {"source": ["session_1_1"], "score": 0.9}}
    rendered_ev = A.render_evidence([new_ev])
    check("渲染带正文", "X happened" in rendered_ev, rendered_ev)
    check("渲染带来源", "session_1_1" in rendered_ev, rendered_ev)
    check("不再渲染出空正文", "unknown:" not in rendered_ev, rendered_ev)
    check("旧格式仍能渲染", "old text" in A.render_evidence(
        [{"memory": "old text", "metadata": {"source": ["s"]}}]))
    check("无证据时给出占位", "no memories" in A.render_evidence([]))
    # evidence 通常没有 time 字段，日期写在正文里 —— 提示要扫正文
    check("current_date 从正文里取日期",
          A.current_date_hint([{"content": "event on 2023-05-07", "metadata": {"source": ["s"]}},
                               {"content": "other on 2023-01-01", "metadata": {"source": ["s"]}}])
          == "2023-05-07")
    check("无日期时回退今天", len(A.current_date_hint(
        [{"content": "no date here", "metadata": {"source": ["s"]}}])) == 10)


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


# ---------------------------------------------------------------------------
# 7. searchctl CLI 端到端
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 8. evomem_v3 端到端（假模型）
# ---------------------------------------------------------------------------

async def test_end_to_end(tmp: Path) -> None:
    section("8. search runner：端到端（假模型 + 真工具）")
    from codemem.search import runner as evomem_v3

    data_dir = tmp / "data"
    sample_dir = data_dir / "Test_Pair"
    sample_dir.mkdir(parents=True, exist_ok=True)
    corpus = [
        make_memory("a1", "Caroline joined an LGBTQ support group"),
        make_memory("a2", "Caroline has been going for about a month"),
    ]
    # 新语料：三层（speakers summary / session summaries / 原始消息）
    def sess(mid, role, content, time=""):
        return {"msg_id": mid, "role": role, "time": time, "content": content}

    (sample_dir / "sessions.jsonl").write_text(
        "".join(json.dumps(sess(f"session_1_{i}", "Caroline" if i % 2 else "Melanie",
                                f"raw message {i}", "1:56 pm on 8 May, 2023")) + "\n"
                for i in range(1, 4)), encoding="utf-8"
    )
    (sample_dir / "session_summaries.jsonl").write_text(
        json.dumps(sess("session_1_summary", "summary", "They discussed the support group.",
                        "1:56 pm on 8 May, 2023")) + "\n", encoding="utf-8"
    )

    run_dir = tmp / "runs"
    config = evomem_v3.load_config(None)
    config["max_steps"] = 6
    config["max_verify_rounds"] = 2
    config["concurrency"] = 1
    config["log"] = {"level": "error", "output": ""}
    catalog = toolcfg.load_catalog(PROJECT_ROOT / "configs/tool.json")
    log = Logger(level="error")

    inputs_dir = await evomem_v3.build_inputs(sample_dir, run_dir, config, log=log)
    check("inputs 已建立", (inputs_dir / "sessions.jsonl").exists())
    check("两层语料都硬链进 inputs",
          all((inputs_dir / n).exists() for n in
              ("sessions.jsonl", "session_summaries.jsonl")))
    check("不再有 speakers_summary.jsonl", not (inputs_dir / "speakers_summary.jsonl").exists())
    check("inputs 是硬链（同一 inode）",
          (inputs_dir / "sessions.jsonl").stat().st_ino == (sample_dir / "sessions.jsonl").stat().st_ino)
    # Windows 上 stat 的权限位不反映可执行性（一律 0o666），所以只断言文件存在 + 内容对。
    check("inputs 里没有多余的东西（检索靠系统 grep）",
          sorted(p.name for p in inputs_dir.iterdir()) == ["session_summaries.jsonl",
                                                           "sessions.jsonl"],
          str(sorted(p.name for p in inputs_dir.iterdir())))

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
            # 按 agent 的真实工作流走一遍：grep 定位 -> read 精读 -> date 算 -> write 落盘。
            # 这样端到端测的不只是"文件写成了"，还有 grep/read/date 这条新路径本身。
            if self.agent_turns == 1:
                return json.dumps({"tool": "bash", "args": {
                    "command": 'grep -in "support group" inputs/session_summaries.jsonl',
                    "description": "Grepping summaries"}})
            if self.agent_turns == 2:
                return json.dumps({"tool": "read", "args": {
                    "path": "inputs/session_summaries.jsonl", "limit": 2}})
            if self.agent_turns == 3:
                return json.dumps({"tool": "bash", "args": {
                    "command": 'date -d "8 May 2023 -1 day" +"%d %b %Y"',
                    "description": "Computing the date"}})
            if self.agent_turns == 4:
                # 写一条**合法 evidence**（新格式：content + metadata.source/score）。
                # 注意不能直接抄 inputs 的记录 —— 那没有 source，会被正确地拒绝。
                rec = {"content": "Caroline joined the group on 7 May 2023.",
                       "metadata": {"source": ["session_1_summary"], "score": 0.9}}
                return json.dumps({"tool": "write", "args": {
                    "path": "evidence.jsonl",
                    "content": json.dumps(rec, ensure_ascii=False) + "\n"}})
            return "I am done."

    model = ScriptedModel()
    # 直接跑 run_qa（绕开 QA 数据源，注入一条假问题）
    record = await evomem_v3.run_qa(
        qa_index=0, qa={"question": "When did Caroline join the group?", "category": 2,
                        "reference": "June 2023"},
        directory=sample_dir, run_dir=run_dir, inputs_dir=inputs_dir,
        config=config, catalog=catalog, model=model,
        log=log, semaphore=asyncio.Semaphore(1),
    )
    check("QA 记录有 evidence", record.evidence_count == 1, str(record.evidence_count))
    check("QA 记录了 2 轮", len(record.rounds) == 2, str(len(record.rounds)))
    check("第 1 轮注入缺口后续跑", len(record.verifications) == 2)
    check("最终判定为足够", record.sufficient is True, record.stop_reason)
    check("带上最终答案", "June 2023" in record.final_answer, record.final_answer)
    check("stop_reason=sufficient", record.stop_reason == "sufficient", record.stop_reason)
    check("工具调用被记账（grep + read + date + write）", record.tool_calls >= 4,
          str(record.tool_calls))
    check("未篡改只读输入", record.tampered is False)
    # evidence 没有 id 字段，标识取 content（见 evidence._id_of）
    check("evidence 标识取自 content",
          record.evidence_ids == ["Caroline joined the group on 7 May 2023."],
          str(record.evidence_ids))

    # --- 篡改检测：手写一个 QA，模型去改 inputs/ ---
    class TamperingModel:
        def __init__(self) -> None:
            self.turn = 0

        async def __call__(self, messages: list[dict], meta: dict) -> str:
            if meta.get("role") == "verifier":
                return '{"sufficient": true, "answer": "x", "missing": []}'
            self.turn += 1
            if self.turn == 1:
                # 软链逃逸：写 inputs/sessions.jsonl（resolve 后在外，应被拒）
                return json.dumps({"tool": "write", "args": {
                    "path": "inputs/sessions.jsonl", "content": "{}"}})
            return "done"

    record2 = await evomem_v3.run_qa(
        qa_index=1, qa={"question": "Anything?", "category": 2, "reference": ""},
        directory=sample_dir, run_dir=run_dir, inputs_dir=inputs_dir, index_dir=index_dir,
        config=config, catalog=catalog, model=TamperingModel(),
        log=log, semaphore=asyncio.Semaphore(1),
    )
    check("写只读输入被拒（未篡改）", record2.tampered is False)
    check("原 sessions.jsonl 未被改动",
          "raw message" in (inputs_dir / "sessions.jsonl").read_text())
    check("工具错误被记账", record2.rounds[0]["steps"][0]["ok"] is False
          or not record2.rounds[0]["steps"][0]["result_text"] == "")

    # --- summary 生成 ---
    summary = evomem_v3.summarize_dir(
        "Test_Pair", [record, record2], type("M", (), {"stats": staticmethod(
            lambda: {"calls": 9, "failures": 0})})(), 2, 0
    )
    check("summary 有 qa_done", summary["qa_done"] == 2, str(summary["qa_done"]))
    # 只有第一条被判足够（第二条是篡改测试，verifier 直接返回 sufficient）
    check("summary 统计 sufficient 条数", summary["sufficient"] == sum(
        1 for r in [record, record2] if r.sufficient), str(summary["sufficient"]))
    check("summary 的 sufficient_rate 与条数一致",
          abs(summary["sufficient_rate"] - summary["sufficient"] / 2) < 1e-9,
          str(summary["sufficient_rate"]))
    check("summary 不再有 embed_available（无向量层）", "embed_available" not in summary)
    check("summary 统计 model_calls", summary["model_calls"] == 9, str(summary["model_calls"]))


# ---------------------------------------------------------------------------

def main() -> int:
    with tempfile.TemporaryDirectory(prefix="codemem-v3-test-") as handle:
        tmp = Path(handle)
        test_no_undefined_names()
        test_toolcfg(tmp)
        test_evidence(tmp)
        test_qa_selection()
        test_verifier_rendering()
        test_agent_guards()
        test_judge_temporal()
        test_answer_parse()
        test_eval_metrics()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(test_verifier())
            loop.run_until_complete(test_tools(tmp))
            loop.run_until_complete(test_agent(tmp))
            loop.run_until_complete(test_compaction_units(tmp))
            loop.run_until_complete(test_end_to_end(tmp))
        finally:
            loop.close()
        # test_tools 里的纯 CPU 部分用到 get_event_loop，跑完补一次
    print(f"\n{'=' * 60}\n通过 {PASS} / 失败 {FAIL}\n{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
