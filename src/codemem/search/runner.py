"""search：code-agent 式证据构建（编排层）。

设计见 ``docs/search.md``。一句话流程：

    for 每条 QA（QA 级并发）:
        建私有 workspace（inputs/ 硬链只读 + 空 evidence.jsonl + env）
        agent loop（read/write/edit/bash，最多 max_steps 步）
        → 停下 → verifier 判"够不够" → 不够就把 missing[] 喂回去再跑一轮
        → 校验 evidence.jsonl（evidence.validate_file）+ 校验 inputs/ 未被篡改
        → 落 trajectory

产物：

    data/search_runs/{experiment}_{ts}/{dir}/
        inputs/{sessions,session_summaries}.jsonl   只读输入（硬链）
        qa_{idx}/evidence.jsonl         **唯一产物**
        {dir}.qa_trajectories.jsonl     每 QA 一行（含完整步记录）
        {dir}.steps.jsonl               每次模型调用的原始输出（排错用）
        summary.json

**语料只有两层**（原始消息 + session summary）。曾经的顶层 speakers summary 已取消：
它是一条没有 msg_id 可回溯的合成文本，`source` 指不到原始消息，破掉溯源不变量。

**检索靠 grep**：agent 的工具箱就是 read/write/edit/bash，检索是 `grep` 一条命令。
没有向量索引、没有 embedding 服务、没有外部检索 CLI —— 语料是 JSONL，一行一条记录，
所以 grep 直接可用、行号稳定，可以 `grep -n` 定位再 `read offset/limit` 精读。
这砍掉了整条链路里唯一的外部依赖。

**为什么 QA 级并发是安全的**：旧版必须目录内串行，只因为所有 QA 共享一份可变的记忆库；
现在 base 库只读、产物 per-QA，QA 之间零共享。

自测：``python scripts/test_search.py``（假模型 + 真工具，纯 CPU，零模型成本）。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import stat
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import openai
import yaml
from tqdm import tqdm

from . import agent as agent_mod
from . import evidence as evidence_mod
from . import toolcfg
from . import verifier as verifier_mod
from .. import dataset
from ..log import Logger, block

# 从 io 导入**唯一**的 PROJECT_ROOT，不要自己 parents[N] 推 ——
# 本文件在子包里深度不同，重算会落到 src/ 而不是项目根。
from ..io import PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_FILE = PROJECT_ROOT / "configs" / "search.yaml"
TOOL_CONFIG_FILE = PROJECT_ROOT / "configs" / "tool.json"

#: 两层语料文件名。硬链、篡改校验、来源描述都用它。
CORPUS_FILES = ("sessions.jsonl", "session_summaries.jsonl")

DEFAULT_CONFIG: dict[str, Any] = {
    "generator": {"base_url": "http://127.0.0.1:30000/v1", "api_key": "sglang", "model": "local",
                  "temperature": 0.2, "max_tokens": 4096, "enable_thinking": False},
    "max_steps": 30,
    "no_progress_patience": 6,
    "max_verify_rounds": 2,
    "observation_max_chars": 8000,
    # 已经烧了这么多步、evidence 还是空的时候，由 harness 插一句"该动笔了"（0 = 关）。
    # 为什么需要它：prompt 里写了"第 4 步前必须写"，但 8B 会无视 —— 实测 12 步全在读、
    # 一次都没写。软约束不可靠，机制可靠。
    "write_nudge_after": 5,
    # evidence 条数软上限：超过就提醒模型精简（**不硬截断** —— 截断会静默丢证据）。
    # 实测踩到：pump"集合型问题要收全成员"后，模型单事实问题也写 658 条，顶爆上下文。
    "evidence_soft_limit": 12,
    # 轨迹文件里每步观测与模型输出的保留字符数（0 = 不截断）。
    # 实测全量跑的轨迹 4.8MB，其中 95% 在 rounds[].steps[]，而观测一项占 55%。
    "trajectory_observation_chars": 600,
    # 模型调用落盘是否带上完整 messages（默认否：相邻记录 95% 重复，体积放大 ~15 倍）
    "log_full_messages": False,
    # 模型调用失败 / 输出不是合法工具调用时，注入纠正消息再试几次（**不**当作完成信号）
    "max_parse_retries": 2,
    # 同一个工具+参数重复调用几次后**拒绝执行**（实测模型会用 >> 重复追加很多遍）
    "repeat_limit": 1,
    # evidence 里重复行达到这个数量就给一次去重警告
    "duplicate_warn_at": 10,
    # 上下文压缩：messages 总字符超过这个数就把中段坍缩成摘要（0 = 不压缩）
    "context_max_chars": 160000,
    "context_keep_recent": 8,
    "bash_timeout": 60,
    "concurrency": 4,
    "concurrency_dirs": 1,
    "max_total_steps": 0,
    "max_qa": 0,
    "skip_category5": True,
    "timeout": 300,
    "max_retries": 3,
    "backoff_base": 2.0,
    "rate_limit_backoff": 10.0,
    "max_backoff": 60.0,
    "backoff_jitter": 0.2,
    "log": {"level": "info", "output": "data/search_runs"},
}


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def load_config(path: Path | None = None) -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    target = path or CONFIG_FILE
    if target.exists():
        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                    config[key].update(value)
                else:
                    config[key] = value
    return config


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            items.append(parsed)
    return items


# ``llm.chat_completion`` 需要的键。**重试参数在顶层**（timeout / max_retries /
# backoff_*），模型参数在 ``generator`` 段里 —— 所以要把两处合起来给它，否则 KeyError。
CALL_KEYS = (
    "timeout", "max_retries", "backoff_base", "rate_limit_backoff", "max_backoff",
    "backoff_jitter",
)


def gen_call_config(config: dict[str, Any]) -> dict[str, Any]:
    """把 ``generator`` 段与顶层的重试参数合成一份 ``chat_completion`` 能吃的配置。"""
    merged = dict(config.get("generator") or {})
    for key in CALL_KEYS:
        if key in config:
            merged[key] = config[key]
    merged.setdefault("max_retries", 3)
    merged.setdefault("timeout", 300)
    merged.setdefault("temperature", 0.2)
    merged.setdefault("max_tokens", 4096)
    return merged


# ---------------------------------------------------------------------------
# 模型调用
# ---------------------------------------------------------------------------

class ModelRunner:
    """封装生成模型：带重试、统计调用数、可选地把每次调用落盘。

    ``chat_completion`` 已有 ``finish_reason=length`` 重试与退避，这里不再重复实现，
    只做统计与可选的 raw 落盘。

    **落盘默认不写完整 ``messages``**（``log_full_messages=False``）。原因：每次调用都要
    带上**整个不断增长的上下文**，所以 ``messages`` 字段占了落盘体积的绝大多数，而相邻两次
    记录里有 95% 是重复的 —— 实测单条 21KB、一个 QA 16 次调用就是 **800KB**；全量
    10 目录 × 200 QA 约 **1.6GB**。

    默认只记 ``raw``（模型的原始输出）+ 元信息，排错需要的正是这些；要复现"模型当时
    看到了什么"时再开 ``--log-full-messages``（或直接看 debug 日志 —— 它本来就打印完整
    prompt 与观测）。
    """

    def __init__(self, client: openai.AsyncOpenAI, config: dict[str, Any],
                 log: Logger, raw_log_path: Path | None = None,
                 full_messages: bool = False) -> None:
        self.client = client
        self.config = config
        self.log = log
        self.raw_log_path = raw_log_path
        self.full_messages = full_messages
        self.calls = 0
        self.failures = 0
        self._raw_handle = raw_log_path.open("w", encoding="utf-8") if raw_log_path else None
        self._lock = asyncio.Lock()

    async def __call__(self, messages: list[dict[str, str]], meta: dict[str, Any]) -> str:
        from ..llm import chat_completion

        async with self._lock:
            self.calls += 1
            call_index = self.calls
        role = meta.get("role", "agent")
        try:
            raw = await chat_completion(self.client, self.config, messages)
        except Exception as exc:  # noqa: BLE001 - 模型调用失败不炸整条 QA
            async with self._lock:
                self.failures += 1
            self.log.warn(
                f"模型调用 #{call_index}（{role} step {meta.get('step')}）失败："
                f"{type(exc).__name__}: {exc}"
            )
            return ""
        async with self._lock:
            if self._raw_handle is not None:
                record: dict[str, Any] = {
                    "call": call_index,
                    "role": role,
                    "round": meta.get("round"),
                    "step": meta.get("step"),
                    "raw": raw,
                    "chars": len(raw),
                }
                if self.full_messages:
                    record["messages"] = messages
                else:
                    # 不写 messages，但留一个体积/规模线索，便于判断上下文涨到多大
                    record["context_chars"] = sum(len(m.get("content", "")) for m in messages)
                    record["context_messages"] = len(messages)
                self._raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._raw_handle.flush()
        return raw

    def stats(self) -> dict[str, Any]:
        return {"calls": self.calls, "failures": self.failures}

    def close(self) -> None:
        if self._raw_handle is not None:
            try:
                self._raw_handle.close()
            finally:
                self._raw_handle = None


# ---------------------------------------------------------------------------
# workspace 准备
# ---------------------------------------------------------------------------

@dataclass
class Workspace:
    """一条 QA 的私有工作目录。"""

    root: Path                     # = qa_{idx}/
    inputs: Path                   # = qa_{idx}/inputs（软链到 <run>/{dir}/inputs）
    evidence: Path                 # = qa_{idx}/evidence.jsonl
    shell_prefix: str

    @property
    def empty_evidence(self) -> bool:
        return not self.evidence.exists() or self.evidence.stat().st_size == 0


def link_or_copy(source: Path, target: Path) -> None:
    """优先硬链（省空间、写不穿）；跨设备时退回拷贝。"""
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def link_dir_or_copy(source: Path, target: Path) -> str:
    """让 ``target`` 指向目录 ``source``。返回实际用的方式（``symlink`` / ``hardlink``）。

    优先**符号链接**，因为它顺带提供了只读保护：工具层的路径约束会 ``resolve()``，
    解析后落在工作目录之外，所以 agent 写 ``inputs/...`` 会被直接拒绝。

    退路是**逐个硬链语料文件**。为什么不是整目录拷贝：拷贝出来的文件在 workspace 内，
    `resolve()` 之后仍落在里面，路径约束同样失效，还白占一份磁盘。硬链保留了两件事：

    ① ``bash >>`` 写穿链接会改到**源文件**，而收尾的 sha256 篡改校验能抓到它（这一条实测
       验证过：bash 追加后源文件指纹确实变了）；
    ② 磁盘上不产生重复数据。

    丢掉的只有 ``write`` / ``edit`` 那一层的前置拦截：它们的**原子替换**（临时文件 +
    ``os.replace``）会先 unlink 再 rename，结果是悄悄把链接换成新 inode，既不报错也不影响
    源文件 —— agent 只是看坏了自己那份 inputs 视图，基座语料是安全的。

    之所以需要退路：Windows 上创建符号链接需要开发者模式或管理员权限，普通账户会拿到
    ``WinError 1314``。让整条链路因为一个目录链接就用不了，代价太大。
    """
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    try:
        target.symlink_to(source, target_is_directory=True)
        return "symlink"
    except OSError:
        pass
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        if item.is_file():
            destination = target / item.name
            try:
                os.link(item, destination)
            except OSError:
                shutil.copy2(item, destination)
    return "hardlink"


def build_shell_prefix(
    *,
    inputs_dir: Path,
    config: dict[str, Any],
) -> str:
    """注入给每条 bash 命令的环境。

    三条：

    - ``PYTHONPATH`` —— 让 agent 自己的代码在子进程里可导入。
    - ``CODEMEM_INPUTS`` —— 语料目录的绝对路径（用相对路径也行，但绝对路径写进示例
      更省一次试错）。
    - **``LC_ALL=C``** —— 这条不是小事。agent 要用 ``date -d`` 算日期，而 ``date`` 的
      输出格式（月份名、星期名）**跟随系统 locale**：中文环境下
      ``date -d "8 May 2023 -1 day" +"%d %b %Y"`` 会输出 ``07 5月 2023``，写进 evidence
      就是一条下游读不了的记录。固定成 C，输出永远是 ``07 May 2023``。
    """
    return "\n".join([
        'export LC_ALL=C',
        f'export PYTHONPATH="{PROJECT_ROOT / "src"}:$PYTHONPATH"',
        f'export CODEMEM_INPUTS="{inputs_dir}"',
    ])


# ---------------------------------------------------------------------------
# QA 记录
# ---------------------------------------------------------------------------

@dataclass
class QaRecord:
    """一条 QA 的完整 trajectory。``to_dict`` 是落盘的形状。"""

    qa_index: int
    question: str
    # 这是该 QA 的第几次重复（`--repeat N` 时 >1）。默认 1，即没开重复。
    repeat: int = 1
    category: Any = None
    reference: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    evidence_count: int = 0
    steps: int = 0
    tool_calls: int = 0
    counts_by_tool: dict[str, int] = field(default_factory=dict)
    verifications: list[dict[str, Any]] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    evidence_report: dict[str, Any] = field(default_factory=dict)
    tampered: bool = False
    inputs_sha256: dict[str, str] = field(default_factory=dict)
    stop_reason: str = ""
    final_answer: str = ""
    sufficient: bool = False
    duration_seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "qa_index": self.qa_index,
            "repeat": self.repeat,
            "question": self.question,
            "category": self.category,
            "reference": self.reference,
            "evidence_ids": self.evidence_ids,
            "evidence_count": self.evidence_count,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "counts_by_tool": self.counts_by_tool,
            "sufficient": self.sufficient,
            "final_answer": self.final_answer,
            "verifications": self.verifications,
            "evidence_report": self.evidence_report,
            "tampered": self.tampered,
            "inputs_sha256": self.inputs_sha256,
            "stop_reason": self.stop_reason,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "rounds": self.rounds,
        }


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 单条 QA
# ---------------------------------------------------------------------------

async def run_qa(
    *,
    qa_index: int,
    qa: dict[str, Any],
    directory: Path,
    run_dir: Path,
    inputs_dir: Path,
    config: dict[str, Any],
    catalog: toolcfg.ToolCatalog,
    model: ModelRunner,
    log: Logger,
    semaphore: asyncio.Semaphore,
    repeat: int = 1,
) -> QaRecord:
    """跑一条 QA：建 workspace → agent loop（可多轮）→ 校验 → 落 trajectory。

    ``repeat`` 是这条 QA 的第几次重复（``--repeat N`` 时 1..N）。每次重复用**独立的工作
    目录**（``qa_{下标}_r{次数}``），否则同一条 QA 的多次运行会互相覆盖证据文件，
    也就没法对比"同样的 QA 为什么这次跑出了不同结果"。
    """
    import time

    started = time.monotonic()
    question = str(qa.get("question") or "").strip()
    record = QaRecord(
        qa_index=qa_index,
        repeat=repeat,
        question=question,
        category=qa.get("category"),
        reference=str(qa.get("reference") or qa.get("answer") or ""),
    )
    qa_log = log.bind(f"{directory.name}:qa{qa_index}" + (f"r{repeat}" if repeat > 1 else ""))

    # 开重复时，工作目录带上 rN；只跑一次时保持 `qa_{idx}`（向后兼容既有产物路径）
    dir_name = f"qa_{qa_index}" if repeat <= 1 else f"qa_{qa_index}_r{repeat}"
    workspace_root = run_dir / directory.name / dir_name
    evidence_path = workspace_root / "evidence.jsonl"
    # 篡改校验覆盖**全部语料**（不只原始消息）—— 只查一个是自欺欺人
    inputs_sha_before = {name: sha256_of(inputs_dir / name) for name in CORPUS_FILES}

    async with semaphore:
        qa_log.info(
            (f"开始第 {repeat} 次（category {record.category}）" if repeat > 1
             else f"开始（category {record.category}）")
            + f"question={question!r}"
        )
        workspace_root.mkdir(parents=True, exist_ok=True)

        # qa_{idx}/inputs 是**一个**指向公共 inputs/ 的链接（不是每 QA 建一整套软链）。
        #
        # 为什么不把每个文件逐个软链进来：那些文件在**一个运行目录内对每个 QA 都完全相同**，
        # 逐个建链等于把同一组链接复制 152 遍。每个目录项占一个 4K 块，实测 152 个 QA
        # 就浪费 4.2MB（目录 458 个 + 软链 608 个），而真实证据总共才 45KB。
        # 整目录一个链接把每 QA 的 6 个条目压到 1 个。
        #
        # 优先符号链接（顺带提供只读保护），不支持时退回逐文件硬链 —— 见 link_dir_or_copy。
        qa_inputs = workspace_root / "inputs"
        link_dir_or_copy(inputs_dir, qa_inputs)

        shell_prefix = build_shell_prefix(inputs_dir=inputs_dir, config=config)
        workspace = Workspace(root=workspace_root, inputs=qa_inputs,
                              evidence=evidence_path, shell_prefix=shell_prefix)

        from . import tools as tools_mod

        tools = tools_mod.build_tools(
            cwd=workspace_root,
            shell_command_prefix=shell_prefix,
            bash_timeout=float(config.get("bash_timeout", 60)),
        )
        catalog.apply(tools)
        scripted_prompt = catalog.render_prompt(question)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": scripted_prompt},
            {"role": "user", "content": catalog.prompt_text("user_begin")},
        ]
        qa_log.debug(block("system prompt", scripted_prompt, limit=20000))

        max_steps = int(config.get("max_steps", 30))
        patience = int(config.get("no_progress_patience", 6))
        max_verify_rounds = max(1, int(config.get("max_verify_rounds", 2)))
        observation_max_chars = int(config.get("observation_max_chars", 8000))
        write_nudge_after = int(config.get("write_nudge_after", 5))
        evidence_soft_limit = int(config.get("evidence_soft_limit", 12))
        # 轨迹里每步观测/模型输出的保留长度（0 = 全文）。见 Step.to_dict 的说明。
        observation_chars = int(config.get("trajectory_observation_chars", 600))
        max_parse_retries = max(0, int(config.get("max_parse_retries", 2)))
        repeat_limit = max(0, int(config.get("repeat_limit", 1)))
        duplicate_warn_at = max(1, int(config.get("duplicate_warn_at", 10)))
        context_max_chars = max(0, int(config.get("context_max_chars", 0)))
        context_keep_recent = max(2, int(config.get("context_keep_recent", 8)))
        step_offset = 0

        try:
            for round_index in range(1, max_verify_rounds + 1):
                outcome, messages = await agent_mod.run_agent_round(
                    messages=messages,
                    tools=tools,
                    model_call=model,
                    evidence_path=evidence_path,
                    cwd=workspace_root,
                    max_steps=max_steps,
                    no_progress_patience=patience,
                    step_offset=step_offset,
                    round_index=round_index,
                    log=qa_log,
                    observation_max_chars=observation_max_chars,
                    write_nudge_after=write_nudge_after,
                    evidence_soft_limit=evidence_soft_limit,
                    max_parse_retries=max_parse_retries,
                    repeat_limit=repeat_limit,
                    duplicate_warn_at=duplicate_warn_at,
                    context_max_chars=context_max_chars,
                    context_keep_recent=context_keep_recent,
                )
                step_offset += len(outcome.steps)
                record.rounds.append({
                    "round": round_index,
                    "done_reason": outcome.done_reason,
                    "summary": outcome.summary(),
                    "tool_calls": outcome.tool_calls,
                    "counts_by_tool": outcome.counts_by_tool(),
                    "steps": [step.to_dict(observation_chars) for step in outcome.steps],
                })

                # ---- 校验产物 → 交给 verifier ----
                report = evidence_mod.validate_file(evidence_path)
                qa_log.info(
                    f"round {round_index} 收尾：{outcome.summary()} | evidence "
                    f"{report.summary()}"
                )
                qa_log.debug(block(
                    f"qa {qa_index} round {round_index} evidence.jsonl",
                    evidence_path.read_text(encoding="utf-8", errors="replace") if evidence_path.exists()
                    else "(不存在)",
                    limit=12000,
                ))

                verdict = await verifier_mod.verify(
                    question=question,
                    memories=report.valid,
                    model_call=model,
                    log=qa_log,
                    round_index=round_index,
                    evidence_present=(
                        evidence_path.exists() and evidence_path.stat().st_size > 0
                    ),
                )
                record.verifications.append(verdict.to_dict())
                qa_log.info(f"round {round_index} {verdict.describe()}")

                if verdict.sufficient:
                    record.sufficient = True
                    record.final_answer = verdict.answer
                    record.stop_reason = "sufficient"
                    break

                record.final_answer = verdict.answer or record.final_answer
                if round_index >= max_verify_rounds:
                    record.stop_reason = "verify_rounds_exhausted"
                    break

                # ---- 把缺口喂回去，再跑一轮 ----
                missing = verdict.missing or ["(the auditor gave no specific gap; re-read the question and your evidence)"]
                messages.append({
                    "role": "user",
                    "content": catalog.prompt_text(
                        "user_gaps",
                        missing="\n".join(f"- {item}" for item in missing),
                        attempted_answer=verdict.answer or "(no answer could be produced)",
                    ),
                })
                qa_log.info(f"round {round_index} 注入 {len(missing)} 项缺口，继续 round {round_index + 1}")

        except Exception as exc:  # noqa: BLE001 - 单条 QA 失败不该拖垮整个目录
            record.error = f"{type(exc).__name__}: {exc}"
            record.stop_reason = record.stop_reason or "error"
            qa_log.error(f"QA 失败：{record.error}")
            import traceback
            qa_log.debug(traceback.format_exc())

        # ---- 收尾：规范化产物 + 只读输入的篡改检测 ----
        # 先记下**规范化之前**的报告：`rejected_reasons` 正是排错最需要的部分，
        # 原来把规范化之后的报告写进 trajectory，导致恰好在出问题的那几条 QA 上
        # 丢掉了全部拒绝原因（QA2 的 119 条重复 id 就是这么消失的）。
        raw_report = evidence_mod.validate_file(evidence_path)
        if (raw_report.rejected or raw_report.parse_errors or raw_report.empty_lines
                or raw_report.multi_record_lines or raw_report.salvaged):
            evidence_mod.write_normalized(evidence_path, raw_report)
            note = ""
            if raw_report.salvaged:
                # 被 max_tokens 截断：**必须说出来**。静默抢救会让人以为模型写全了，
                # 而实际上尾部记录已经丢了 —— 那会影响评测的解释。
                note = (
                    f"（注意：产物被 max_tokens 截断，按对象边界抢救出 "
                    f"{len(raw_report.valid)} 条，丢弃 {raw_report.dropped_fragments} 个残片）"
                )
            qa_log.warn(
                f"产物有问题，已规范化为 evidence.jsonl（原文件存 evidence.jsonl.raw）："
                f"{raw_report.summary()}{note}"
            )
        report = evidence_mod.validate_file(evidence_path)
        record.evidence_report = report.to_dict()
        record.evidence_report["before_normalization"] = {
            "lines": raw_report.total_lines,
            "rejected": len(raw_report.rejected),
            "parse_errors": len(raw_report.parse_errors),
            "empty_lines": raw_report.empty_lines,
            "rejected_reasons": [
                {"line": item.get("line"), "reason": item.get("reason")}
                for item in raw_report.rejected
            ],
        }
        record.evidence_ids = [_id_of(m) for m in report.valid]
        record.evidence_count = len(report.valid)
        record.inputs_sha256 = {
            **{name: {"before": digest} for name, digest in inputs_sha_before.items()},
        }
        for name in CORPUS_FILES:
            after = sha256_of(inputs_dir / name)
            record.inputs_sha256[name]["after"] = after
            if inputs_sha_before[name] != after:
                record.tampered = True
                qa_log.error(f"inputs/{name} 被改动了！该 QA 结果作废（tampered=true）")

        record.steps = step_offset
        record.tool_calls = sum(item["tool_calls"] for item in record.rounds)
        counts: dict[str, int] = {}
        for item in record.rounds:
            for name, value in item["counts_by_tool"].items():
                counts[name] = counts.get(name, 0) + value
        record.counts_by_tool = counts
        record.duration_seconds = round(time.monotonic() - started, 2)

        qa_log.info(
            f"完成：evidence {record.evidence_count} 条 / 步 {record.steps} / "
            f"工具 {record.tool_calls}"
            + (f" / {record.stop_reason}" if record.stop_reason else "")
            + (f" / {record.duration_seconds}s" if record.duration_seconds else "")
        )
    return record


def _id_of(memory: dict[str, Any]) -> str:
    """记录标识。**委托给 ``evidence._id_of``**，别在这里再写一份 ——
    新格式的 evidence 没有 ``metadata.id``（用 content 当标识），两处各写一份就会不一致
    （实测：这里只读 metadata.id，于是所有新格式记录的标识全是空串）。"""
    return evidence_mod._id_of(memory)


# ---------------------------------------------------------------------------
# 单个目录
# ---------------------------------------------------------------------------

@dataclass
class SelectedQuestions:
    """一次运行实际要跑的 QA 列表 + 为什么少了若干条。

    ``usable`` 是 ``[(原始下标, qa), ...]``；``notes`` 记录"跳过了什么、为什么"，
    直接拼进启动日志 —— 复现一次运行最需要的就这两件事。
    """

    usable: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    skipped: int = 0
    notes: list[str] = field(default_factory=list)
    explicit: bool = False

    @property
    def indices(self) -> list[int]:
        return [index for index, _ in self.usable]


def select_questions(
    questions: list[tuple[int, dict[str, Any]]],
    *,
    qa_indices: Sequence[int] = (),
    max_qa: int = 0,
    skip_category5: bool = True,
    max_total_steps: int = 0,
    max_steps: int = 30,
) -> SelectedQuestions:
    """从目录的全部 QA 里选出这次要跑的。

    **``qa_indices`` 是调试用的精确选择，优先于一切截断**：明确点名了某几条 QA，
    就把它们全跑完 —— ``max_qa`` 与步数预算都不再适用（否则"想看这条 QA 到底哪里出问题"
    会被预算静默砍掉，白等一场）。点名了下标但该目录没有时，会在 ``notes`` 里说清楚，
    不会静默变成"跑了别的 QA"。

    选择顺序（显式点名时跳过 3、4）：

    1. 只保留问题文本非空的；
    2. 按 ``skip_category5`` 过滤 category 5（adversarial，与评测口径一致）；
    3. ``max_qa`` 截断到前 N 条（**在过滤前截断**，保持与旧行为一致：成本闸门按原始条数算）；
    4. 步数预算 ``max_total_steps`` 换算成最多几条 QA（按每条最坏 ``max_steps`` 估）。
    """
    selected = list(questions)
    notes: list[str] = []

    explicit = bool(qa_indices)
    if explicit:
        wanted = list(dict.fromkeys(int(i) for i in qa_indices))   # 去重且保序
        wanted_set = set(wanted)
        available = {index for index, _ in questions}
        missing = [index for index in wanted if index not in available]
        selected = [(index, qa) for index, qa in questions if index in wanted_set]
        if missing:
            notes.append(
                f"指定的 QA 下标在本目录不存在：{missing}"
                f"（可用范围 {min(available) if available else '-'}"
                f"..{max(available) if available else '-'}）"
            )
        if selected:
            notes.append(f"只跑指定 QA {sorted(index for index, _ in selected)}")
    elif max_qa:
        selected = selected[:max_qa]
        notes.append(f"只跑前 {max_qa} 条 QA")

    before = len(selected)
    usable = [
        (index, qa) for index, qa in selected
        if str(qa.get("question") or "").strip()
        and not (skip_category5 and qa.get("category") == 5)
    ]
    skipped = before - len(usable)
    if skipped:
        notes.append(f"跳过 {skipped} 条（空问题/category5）")

    # 预算闸门只在**没有显式点名**时生效
    if not explicit and max_total_steps > 0:
        per_qa = max(1, int(max_steps))
        allowed = max(1, max_total_steps // per_qa)
        if allowed < len(usable):
            notes.append(
                f"步数预算 {max_total_steps} → 只跑前 {allowed} 条"
                f"（按每条最坏 {per_qa} 步估）"
            )
            skipped += len(usable) - allowed
            usable = usable[:allowed]

    return SelectedQuestions(usable=usable, skipped=skipped, notes=notes, explicit=explicit)


def parse_qa_indices(raw: str) -> list[int]:
    """解析 ``--qa`` 的值：``"3"`` / ``"3,7,12"`` / ``"3 7"`` 都接受。

    给逗号也接受空格是刻意的 —— 命令行里两种写法都会下意识用，为这点差异报错很烦人。
    非法片段直接抛 ``ValueError``，由调用方转成 argparse 的报错（**不静默忽略**：
    静默忽略会让你以为跑了 QA 12，实际跑的是别的）。
    """
    indices: list[int] = []
    for chunk in raw.replace(",", " ").split():
        try:
            indices.append(int(chunk))
        except ValueError:
            raise ValueError(
                f"--qa 的取值必须是整数（逗号或空格分隔），收到 {chunk!r}"
            ) from None
    return indices


def format_qa_listing(questions: list[tuple[int, dict[str, Any]]], label: str) -> str:
    """``--list-qa`` 的输出：一行一条 ``下标  category  问题``。

    存在的理由：``--qa`` 要的是**原始下标**，而下标在数据集里并不直观。先列一遍
    再点名，比让人自己数更靠谱。
    """
    lines = [f"# {label}（{len(questions)} 条 QA）", "# index  category  question"]
    for index, qa in questions:
        category = qa.get("category")
        question = str(qa.get("question") or "").replace("\n", " ").strip()
        lines.append(f"{index:>6}  {str(category):>8}  {question}")
    return "\n".join(lines)


async def run_dir(
    directory: Path,
    config: dict[str, Any],
    catalog: toolcfg.ToolCatalog,
    generator: openai.AsyncOpenAI,
    run_dir: Path,
    log: Logger,
) -> dict[str, Any]:
    """对一个 speaker 目录跑完整流程：建输入 → 并发跑全部 QA → 落盘。"""

    label = directory.name
    log = log.bind(label)
    # 需要的输入：原始消息 + session summary。缺 summary 时**只警告不跳过** ——
    # 原始消息本身是可检索的，退化成"没有概览"总比整个目录跑不了好。
    sessions_path = directory / "sessions.jsonl"
    if not sessions_path.exists():
        log.warn("跳过：缺少 sessions.jsonl（先跑 `python -m codemem.add --stage session`）")
        return {"dir": label, "status": "SKIPPED", "reason": "missing sessions.jsonl"}
    missing_summaries = [
        name for name in ("session_summaries.jsonl",) if not (directory / name).exists()
    ]
    if missing_summaries:
        log.warn(
            f"缺少 {missing_summaries}：检索将只有原始消息，没有概览层"
            f"（建议先跑 `python -m codemem.add --stage summary`）"
        )

    out_dir = run_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = await build_inputs(directory, run_dir, config, log)

    max_qa = max(0, int(config.get("max_qa", 0) or 0))
    skip_category5 = bool(config.get("skip_category5", True))
    # QA 选择：--qa 精确点名 > max_qa 截断 > 步数预算（见 select_questions）
    qa_indices = list(config.get("qa_indices") or [])
    questions = dataset.questions_for_dir(label, skip_category5=False)
    if not questions:
        log.warn(f"跳过：QA 数据源里找不到目录 {label}")
        return {"dir": label, "status": "SKIPPED", "reason": "no qa for dir"}

    selection = select_questions(
        questions,
        qa_indices=qa_indices,
        max_qa=max_qa,
        skip_category5=skip_category5,
        max_total_steps=max(0, int(config.get("max_total_steps", 0) or 0)),
        max_steps=int(config.get("max_steps", 30)),
    )
    usable, skipped = selection.usable, selection.skipped

    log.info(
        f"开始：QA {len(usable)} 条"
        + (f"（跳过 {skipped} 条）" if skipped else "")
        + f" / 步上限 {config.get('max_steps')} / verifier≤{config.get('max_verify_rounds')} "
        f"/ 并发 {config.get('concurrency')}"

        + ("".join(f" / {note}" for note in selection.notes))
    )

    # 每次模型调用落盘：默认只记原始输出（完整 messages 会把体积放大十几倍，见 ModelRunner）
    raw_log_path = out_dir / f"{label}.model_calls.jsonl"
    model = ModelRunner(
        generator, gen_call_config(config), log, raw_log_path,
        full_messages=bool(config.get("log_full_messages", False)),
    )
    semaphore = asyncio.Semaphore(max(1, int(config.get("concurrency", 4))))
    qa_path = out_dir / f"{label}.qa_trajectories.jsonl"

    # --repeat N：把每条 QA 展开成 N 个独立任务（各自的工作目录 / trajectory）
    repeats = max(1, int(config.get("repeat", 1) or 1))
    jobs = [(index, qa, run) for run in range(1, repeats + 1) for index, qa in usable]
    if repeats > 1:
        log.info(f"每条 QA 重复 {repeats} 次 → 共 {len(jobs)} 个任务")

    records: list[QaRecord] = []
    pbar = tqdm(
        total=len(jobs),
        desc=f"[{label}] evidence",
        unit="qa",
        disable=log.threshold > 20,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    try:
        tasks = [
            asyncio.create_task(
                run_qa(
                    qa_index=index, qa=qa, directory=directory, run_dir=run_dir,
                    inputs_dir=inputs_dir, config=config,
                    catalog=catalog, model=model,
                    log=log, semaphore=semaphore, repeat=run,
                )
            )
            for index, qa, run in jobs
        ]
        with qa_path.open("w", encoding="utf-8") as handle:
            for future in asyncio.as_completed(tasks):
                try:
                    record = await future
                except Exception as exc:  # noqa: BLE001 - 兜住 run_qa 之外的意外
                    log.error(f"QA 任务异常：{type(exc).__name__}: {exc}")
                    pbar.update(1)
                    continue
                records.append(record)
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False, default=str) + "\n")
                handle.flush()
                pbar.update(1)
    finally:
        pbar.close()
        model.close()

    # 按 (QA 下标, 重复次数) 稳定排序，同一条 QA 的多次运行排在一起便于对比
    records.sort(key=lambda item: (item.qa_index, item.repeat))
    summary = summarize_dir(label, records, model, len(usable), skipped)
    log.info(
        f"完成：QA {summary['qa_done']}/{len(usable)}"
        + (f"×{repeats}" if repeats > 1 else "")
        + f" / evidence 共 "
        f"{summary['evidence_total']} 条 / 步 {summary['steps_total']} / "
        f"模型调用 {summary['model_calls']}"
        + f" / 充分 {summary['sufficient_rate']:.0%}"
        + (f" / 篡改 {summary['tampered']}" if summary["tampered"] else "")
    )
    # --repeat：报告同一条 QA 多次运行的结果是否一致 —— 这正是重复跑要看的东西
    if repeats > 1:
        for line in repeat_agreement(records):
            log.info(line)
    return summary


def repeat_agreement(records: list[QaRecord]) -> list[str]:
    """按 QA 汇总多次重复的**一致性**（``--repeat`` 的产出）。

    每条重复报三件事：判足够与否是否一致、最终答案是否一致、evidence 条数分布。
    不一致就说明这条 QA 的结果**不稳定** —— 要么 prompt 有歧义，要么链路里有随机性，
    要么这题本身就处在能力边界上。这比只看单次结果有用得多。
    """
    groups: dict[int, list[QaRecord]] = {}
    for record in records:
        if record.repeat >= 1:
            groups.setdefault(record.qa_index, []).append(record)

    lines: list[str] = []
    for index in sorted(groups):
        runs = sorted(groups[index], key=lambda item: item.repeat)
        if len(runs) < 2:
            continue
        sufficient = [bool(run.sufficient) for run in runs]
        answers = [(run.final_answer or "").strip() for run in runs]
        counts = [run.evidence_count for run in runs]
        stable_sufficient = "一致" if len(set(sufficient)) == 1 else f"**不一致** {sufficient}"
        unique_answers = {answer for answer in answers if answer}
        stable_answer = (
            "一致" if len(unique_answers) <= 1 else f"**{len(unique_answers)} 种**"
        )
        lines.append(
            f"repeat qa{index} n={len(runs)} 充分={stable_sufficient} "
            f"答案={stable_answer} evidence条数={counts}"
            + (f" 答案样例={list(unique_answers)[:3]}" if len(unique_answers) > 1 else "")
        )
    return lines


async def build_inputs(
    directory: Path,
    run_dir: Path,
    config: dict[str, Any],
    log: Logger,
) -> Path:
    """建 ``<run>/{dir}/inputs/``：把两层语料硬链进去。返回 ``inputs_dir``。

    没有别的东西了 —— 没有索引、没有命令包装、没有环境文件。语料是 JSONL，agent 用
    ``grep`` 检索、用 ``read`` 精读，这两样都在系统里。于是：

    - 一次运行的可再生中间产物只剩 evidence.jsonl 与轨迹；
    - 同一目录反复跑不再需要"只嵌一次"的缓存机制（那套是为向量索引存在的）；
    - 硬链保证运行目录不占额外空间，也保证收尾的 sha256 篡改校验有意义。
    """
    inputs_dir = run_dir / directory.name / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    # 两层语料：
    #   session_summaries    每次会话一份（粗筛：定位到哪次会话）
    #   sessions             原始消息（精确措辞、时间锚点）
    for name in CORPUS_FILES:
        source = directory / name
        if source.exists():
            link_or_copy(source, inputs_dir / name)
    return inputs_dir


def summarize_dir(
    label: str,
    records: list[QaRecord],
    model: ModelRunner,
    qa_total: int,
    skipped: int,
) -> dict[str, Any]:
    done = [r for r in records if not r.error]
    sufficient = [r for r in records if r.sufficient]
    steps = sum(r.steps for r in records)
    return {
        "dir": label,
        "status": "OK",
        "qa_total": qa_total,
        "qa_skipped": skipped,
        "qa_done": len(done),
        "qa_error": len(records) - len(done),
        "sufficient": len(sufficient),
        "sufficient_rate": (len(sufficient) / len(records)) if records else 0.0,
        "evidence_total": sum(r.evidence_count for r in records),
        "evidence_max": max((r.evidence_count for r in records), default=0),
        "steps_total": steps,
        "steps_mean": round(steps / len(records), 2) if records else 0.0,
        "tool_calls_total": sum(r.tool_calls for r in records),
        "tampered": sum(1 for r in records if r.tampered),
        "model_calls": model.stats()["calls"],
        "model_failures": model.stats()["failures"],
        # --repeat：逐条 QA 的多次运行是否一致（单次跑时为 1，没有意义）
        "repeats": max((r.repeat for r in records), default=1),
        "repeat_agreement": repeat_agreement(records),
        # 实际跑的 QA 下标（--qa / --max-qa 之后），复现一次运行靠它
        "qa_indices": sorted({r.qa_index for r in records}),
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="search：搜索 agent 的证据构建"
                    "（read/write/edit/bash；检索靠 grep，时间靠 date -d）"
    )
    parser.add_argument("dirs", nargs="*", help="speaker 目录名或路径；缺省处理全部")
    parser.add_argument("--config", default=str(CONFIG_FILE), help="搜索配置（默认 configs/search.yaml）")
    parser.add_argument("--tool-config", default=str(TOOL_CONFIG_FILE), help="工具配置（默认 configs/tool.json）")
    parser.add_argument("--sample", default="", help="只处理指定目录")
    parser.add_argument("--qa", default="",
                        help="只跑指定的 QA 原始下标，逗号/空格分隔，如 `--qa 3,7,12`。"
                             "优先于 --max-qa 与步数预算（点名了就一定跑完）。"
                             "配 --list-qa 查看下标")
    parser.add_argument("--repeat", type=int, default=1,
                        help="每条 QA 重复跑 N 次（默认 1）。用来反复测同一条 QA "
                             "看结果是否稳定 / 定位偶发问题；结果按 repeat 分组落盘")
    parser.add_argument("--tag", default="",
                        help="给这次运行打个标记，写进输出目录名与 summary（如 `--tag promptA`）")
    parser.add_argument("--list-qa", action="store_true",
                        help="只列出该目录的 QA 下标就退出，不跑演化")
    parser.add_argument("--max-qa", type=int, default=None, help="每个目录最多处理前 N 条 QA（0=全部）")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖 max_steps")
    parser.add_argument("--max-verify-rounds", type=int, default=None, help="覆盖 max_verify_rounds")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖 QA 级并发")
    parser.add_argument("--experiment", default="search", help="实验名，作为输出子目录前缀")
    parser.add_argument("--output-dir", default=str(DATA_DIR / "search_runs"), help="输出根目录")
    parser.add_argument("--prune-runs", action="store_true",
                        help="清理历史运行目录（默认保留含 'full' 的那次），然后退出。"
                             "不传 --yes 时只列出将要删除的内容")
    parser.add_argument("--keep-last", type=int, default=0,
                        help="配合 --prune-runs：保留最近 N 次运行（默认 0 = 只保留含 full 的那次）")
    parser.add_argument("--yes", action="store_true", help="配合 --prune-runs 真正执行删除")
    parser.add_argument("--log-full-messages", action="store_true",
                        help="模型调用落盘时带上完整 messages（默认只记原始输出；"
                             "打开后体积约放大 15 倍，仅在需要复现上下文时用）")
    parser.add_argument("--log-level", default="", help="debug/info/warn/error/silent")
    parser.add_argument("--log-output", default=None, help="日志保存目录（空串=只打终端）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只建 inputs/ + 渲染 prompt 后打印，不调模型")
    return parser.parse_args(argv)


def build_logger(config: dict[str, Any], args: argparse.Namespace) -> Logger:
    section = dict(config.get("log") or {})
    if args.log_level:
        section["level"] = args.log_level
    if args.log_output is not None:
        section["output"] = args.log_output
    if section.get("output"):
        directory = Path(str(section["output"]))
        if not directory.is_absolute():
            directory = PROJECT_ROOT / directory
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        level = str(section.get("level", "info")).lower()
        path = directory / f"search_{level}_{stamp}.log"
        handle = path.open("w", encoding="utf-8")
        logger = Logger(level=level, path=path, _handle=handle)
        logger.info(f"日志落盘 {path}（级别 {level}）")
        return logger
    return Logger(level=str(section.get("level", "info")))


def resolve_dirs(config: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    if args.dirs:
        targets = []
        for item in args.dirs:
            path = Path(item)
            if not path.is_absolute():
                candidate = DATA_DIR / item
                path = candidate if candidate.exists() else PROJECT_ROOT / item
            targets.append(path)
        return targets
    if args.sample:
        return [DATA_DIR / args.sample]
    return sorted(
        path for path in DATA_DIR.iterdir()
        if path.is_dir() and (path / "sessions.jsonl").exists()
    )


async def async_main(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    for key, override in (
        ("max_qa", args.max_qa), ("max_steps", args.max_steps),
        ("max_verify_rounds", args.max_verify_rounds), ("concurrency", args.concurrency),
    ):
        if override is not None:
            config[key] = override
    if args.log_full_messages:
        config["log_full_messages"] = True

    # --qa：精确点名要跑的 QA 下标（调试用）。解析失败直接报错退出，不静默忽略 ——
    # 静默忽略会让你以为跑了 QA 12，实际跑的是前 N 条。
    try:
        qa_indices = parse_qa_indices(args.qa) if args.qa else []
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2) from None
    config["qa_indices"] = qa_indices
    config["repeat"] = max(1, int(args.repeat or 1))

    log = build_logger(config, args)

    # --prune-runs：清理历史运行目录
    if args.prune_runs:
        from . import prune as prune_mod

        runs_root = Path(args.output_dir)
        plan = prune_mod.plan_prune(runs_root, keep_last=args.keep_last)
        if not plan:
            print(f"没有可清理的运行目录（{runs_root}）")
            return
        total = sum(size for _, size in plan)
        print(f"可清理 {len(plan)} 个运行目录，共 {total / 1024 / 1024:.1f} MB：")
        for path, size in plan[:10]:
            print(f"  {size / 1024:>8.0f} KB  {path.name}")
        if len(plan) > 10:
            print(f"  … 另外 {len(plan) - 10} 个")
        print("\n保留：" + (f"最近 {args.keep_last} 次运行" if args.keep_last
                            else "含 'full' 的运行（没有则保留最近一次）"))
        if not args.yes:
            print("\n加 --yes 才会真正删除")
            return
        removed, freed = prune_mod.prune_runs(runs_root, plan)
        print(f"已删除 {removed} 个目录，释放 {freed / 1024 / 1024:.1f} MB")
        return

    catalog = toolcfg.load_catalog(args.tool_config)
    catalog.validate(["read", "write", "edit", "bash"])

    directories = resolve_dirs(config, args)
    if not directories:
        log.error("没有找到任何含 sessions.jsonl 的目录"
                  "（先跑 `python -m codemem.add --stage session`）")
        return

    # --list-qa：只列下标就退出（run 之前先看下标，比让人自己数靠谱）
    if args.list_qa:
        for directory in directories:
            questions = dataset.questions_for_dir(directory.name, skip_category5=False)
            if not questions:
                print(f"# {directory.name}：QA 数据源里找不到该目录", file=sys.stderr)
                continue
            print(format_qa_listing(questions, directory.name))
        return

    # 输出目录名带上 tag / qa 标记，便于区分同一批 QA 的多次运行
    suffix = args.experiment
    if args.tag:
        suffix = f"{suffix}_{args.tag}"
    if qa_indices:
        suffix = f"{suffix}_qa{'-'.join(str(i) for i in qa_indices[:4])}" + (
            f"+{len(qa_indices) - 4}" if len(qa_indices) > 4 else ""
        )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_dir) / f"{suffix}_{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    gen_cfg = config.get("generator") or {}
    gen_log = log.bind("main")
    gen_log.info(
        f"search 启动：目录 {len(directories)} 个 / 模型 {gen_cfg.get('model')} "
        f"@ {gen_cfg.get('base_url')} / 输出 {run_root}"
    )
    gen_log.info(
        f"工具配置 {catalog.source_path} / 工具 {sorted(catalog.tools)} / "
        f"命令 {sorted(catalog.commands)}"
    )
    if qa_indices:
        gen_log.info(f"只跑指定 QA 下标 {qa_indices}（--qa）")
    if config["repeat"] > 1:
        gen_log.info(f"每条 QA 重复 {config['repeat']} 次（--repeat）")
    if args.dry_run:
        sample_question = "When did Caroline go to the LGBTQ support group?"
        rendered = catalog.render_prompt(sample_question)
        gen_log.info(f"[dry-run] system prompt {len(rendered)} 字符，渲染如下：")
        print(rendered)
        return

    generator = openai.AsyncOpenAI(
        base_url=gen_cfg.get("base_url"), api_key=gen_cfg.get("api_key") or "none"
    )

    dir_semaphore = asyncio.Semaphore(max(1, int(config.get("concurrency_dirs", 1))))
    summaries: list[dict[str, Any]] = []

    async def worker(directory: Path) -> dict[str, Any]:
        async with dir_semaphore:
            try:
                return await run_dir(directory, config, catalog, generator, run_root, log)
            except Exception as exc:  # noqa: BLE001 - 一个目录失败不影响其它目录
                log.error(f"[{directory.name}] 目录失败：{type(exc).__name__}: {exc}")
                import traceback
                log.debug(traceback.format_exc())
                return {"dir": directory.name, "status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"}

    summaries = list(await asyncio.gather(*(worker(item) for item in directories)))

    payload = {
        "experiment": args.experiment,
        "tag": args.tag,
        "run_root": str(run_root),
        "started_at": stamp,
        # 这次跑的是哪些 QA + 重复几次 —— 复现一次运行只需要这两个字段
        "selection": {
            "qa_indices": qa_indices,
            "repeat": config.get("repeat", 1),
            "max_qa": config.get("max_qa", 0),
            "skip_category5": config.get("skip_category5", True),
        },
        "config": {k: v for k, v in config.items() if k != "log"},
        "tool_config": str(catalog.source_path),
        "log_path": str(log.path) if log.path else "",
        "dirs": summaries,
    }
    (run_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    ok = sum(1 for item in summaries if item.get("status") == "OK")
    err = sum(1 for item in summaries if item.get("status") == "ERROR")
    skip = sum(1 for item in summaries if item.get("status") == "SKIPPED")
    log.info(f"全部结束：成功 {ok} / 失败 {err} / 跳过 {skip}")
    for item in summaries:
        if item.get("status") == "OK":
            log.info(
                f"  OK       {item['dir']:20s} qa={item.get('qa_done')}/{item.get('qa_total')} "
                f"ev={item.get('evidence_total')} 步={item.get('steps_total')} "
                f"调用={item.get('model_calls')} 充分={item.get('sufficient_rate', 0):.0%}"
            )
        else:
            log.info(f"  {item.get('status'):8s} {item['dir']:20s} {item.get('reason', '')}")
    log.info(f"summary -> {run_root / 'summary.json'}")
    log.info(log.summary())
    log.close()


def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(parse_args(argv)))


if __name__ == "__main__":
    main()
