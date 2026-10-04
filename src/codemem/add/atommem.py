"""add · 第三步：原子事实抽取（atommem.jsonl）—— **逐 message 的 add agent**。

     msgmem.jsonl / summem.jsonl / atommem.jsonl  ─►  atommem.jsonl（本 message 的 atom）

**核心原理**：逐个 message 处理。后面的 message 的信息可能受**前面内容**影响、可能**跨
session 影响**，所以处理一条 message 时，agent 需要看到前面**相关的**信息才能把指代消解对、
把隐含事实补全。于是这里不是一个单次 LLM 调用，而是一个**朴素的 agent loop**（与 search
同形）：给 agent 三个工具 `read` / `edit` / `bash`、一个装着三层记忆的工作目录，让它自己
`grep` 前文、`read` 相关 session，把原子事实写进 `atommem.jsonl`。默认 `max_step=8`。

**下笔前先查重、能合并就合并**（"合并演化"）：agent 写一条事实前先 `grep` atommem 看是否已有；
若已有就从 `edit append` 改走 `edit replace`，把两条**合并成一条**（`source` 记下所有依据的
message id），而不是再追加一条近似重复。跨 message 的同一条事实由此收敛成单条。

**agent 只填四个字段**（``memory`` + ``metadata.{time, role, tag}``）；``id`` / ``source`` /
``type`` / ``changelog`` 由系统在收尾时补上 —— 见 ``finalize_file``。这既省了 agent 的
力气，也保证这些字段永远正确（id 唯一、type 恒为 atom）。**时间信息全部进 ``time``**：
时点用 ISO，时长/年龄算成起点或区间，算不出就保留带锚点的原表述。

**串行**：同一目录内按 message 顺序**串行**（后面的 message 要看到前面已生成的 atommem）；
**不同目录并行**。

**增量/幂等**：重跑一条 message 时，先把它**最初产出**的 atom（primary source = 本 message）去掉，
再重新生成 —— 于是重复跑不会累积重复 atom（"就地覆盖是幂等的"的 agent 版）。

复用 search 的组件（``search/tools.py`` 的 ``build_tools``、``search/toolcfg.py`` 的
``ToolCatalog``、``search/agent.py`` 的 ``run_agent_round``），配置在 ``configs/add_tool.json``。

用法::

    python -m codemem.add --stage atommem                      # 全部目录
    python -m codemem.add --stage atommem Caroline_Melanie     # 指定目录
    python -m codemem.add --stage atommem --limit-messages 20  # 每目录只跑前 20 条（调试）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import openai
import yaml

from .. import llm
from ..io import DATA_DIR, PROJECT_ROOT, memory_id, memory_role, memory_text, now_iso, read_jsonl
from ..log import Logger
from ..prompts import ATOM_AGENT_STANDARD, ATOM_AGENT_SYSTEM_PROMPT, ATOM_AGENT_USER_PROMPT
from ..search import agent as agent_mod
from ..search import toolcfg
from ..search import tools as tools_mod
from .memory import (
    ATOMMEM_NAME,
    MSGMEM_NAME,
    SUMMEM_NAME,
    group_by_session,
    msg_session_number,
    parse_dir_label,
    write_memories,
)

CONFIG_FILE = PROJECT_ROOT / "configs" / "add.yaml"
TOOL_CONFIG_FILE = PROJECT_ROOT / "configs" / "add_tool.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "model": "local",
    "base_url": "http://127.0.0.1:30000/v1",
    "api_key": "sglang",
    "temperature": 0.2,
    "max_tokens": 4096,
    "concurrency": 8,
    "timeout": 300,
    "max_retries": 3,
    "backoff_base": 2.0,
    "rate_limit_backoff": 10.0,
    "max_backoff": 60.0,
    "backoff_jitter": 0.2,
    "enable_thinking": False,
    # agent loop
    "max_steps": 8,
    "no_progress_patience": 4,
    "observation_max_chars": 8000,
    "max_parse_retries": 2,
    "repeat_limit": 1,
    "context_max_chars": 120000,
    "context_keep_recent": 6,
    "bash_timeout": 60,
    # 同一条 message 的 prompt 里附上前 N 条同会话消息作为消解指代的上下文窗口（原始措辞）。
    "context_window": 12,
}

CALL_KEYS = (
    "timeout", "max_retries", "backoff_base", "rate_limit_backoff", "max_backoff",
    "backoff_jitter",
)


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
                elif key in config:
                    config[key] = value
    return config


def gen_call_config(config: dict[str, Any], section: str = "atommem") -> dict[str, Any]:
    """把某个模型子段（可选）与顶层重试参数合成 ``chat_completion`` 能吃的配置。"""
    merged = dict(config.get(section) or {})
    for key in CALL_KEYS + ("model", "base_url", "api_key", "temperature", "max_tokens",
                            "enable_thinking"):
        if key not in merged and key in config:
            merged[key] = config[key]
    merged.setdefault("max_retries", 3)
    merged.setdefault("timeout", 300)
    merged.setdefault("temperature", 0.2)
    merged.setdefault("max_tokens", 4096)
    return merged


# ---------------------------------------------------------------------------
# 目录 / 输入
# ---------------------------------------------------------------------------

def directory_files(directory: Path) -> dict[str, Path]:
    return {
        "msgmem": directory / MSGMEM_NAME,
        "summem": directory / SUMMEM_NAME,
        "atommem": directory / ATOMMEM_NAME,
    }


def targets_with_msgmem(dirs: list[str]) -> list[Path]:
    """把目录名/路径解析成实际目录（只保留含 msgmem.jsonl 的）。"""
    if dirs:
        resolved: list[Path] = []
        for item in dirs:
            path = Path(item)
            if not path.is_absolute():
                candidate = DATA_DIR / item
                path = candidate if candidate.exists() else PROJECT_ROOT / item
            resolved.append(path)
        return resolved
    return sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_dir() and (p / MSGMEM_NAME).exists()
    )


# ---------------------------------------------------------------------------
# 渲染：会话清单 / 会话内窗口 / 目标 message
# ---------------------------------------------------------------------------

def render_sessions(summaries: dict[int, dict[str, Any]]) -> str:
    """会话清单：每行 ``session_N (时间)``，**不含事实内容**（克制，防止跳过检索）。"""
    if not summaries:
        return "(no sessions found -- check msgmem.jsonl)"
    lines = ["The conversation is split into these sessions (each has a summary in summem.jsonl):"]
    for number in sorted(summaries):
        when = str((summaries[number].get("metadata") or {}).get("time") or "")
        lines.append(f"  session_{number}" + (f"  ({when})" if when else ""))
    return "\n".join(lines)


def _render_msg_line(record: dict[str, Any], *, anchor: str) -> str:
    mid = memory_id(record)
    role = memory_role(record) or "unknown"
    when = str((record.get("metadata") or {}).get("time") or "")
    stamp = f" ({when})" if when else ""
    return f"[{mid}]{stamp} {role}: {memory_text(record)}"


def render_window(prior: list[dict[str, Any]], window: int) -> str:
    if window <= 0 or not prior:
        return "(this is the first message of the session)"
    return "\n".join(_render_msg_line(r, anchor="") for r in prior[-window:])


def build_messages(
    *,
    target: dict[str, Any],
    prior: list[dict[str, Any]],
    summaries: dict[int, dict[str, Any]],
    config: dict[str, Any],
    catalog: toolcfg.ToolCatalog,
    speaker_a: str,
    speaker_b: str,
) -> list[dict[str, str]]:
    when = str((target.get("metadata") or {}).get("time") or "")
    target_line = _render_msg_line(target, anchor="")
    system = catalog.render_prompt("").replace("{speaker_a}", speaker_a).replace(
        "{speaker_b}", speaker_b
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": ATOM_AGENT_USER_PROMPT
            .replace("{sessions}", render_sessions(summaries))
            .replace("{msg_id}", memory_id(target))
            .replace("{window}", render_window(prior, int(config.get("context_window", 12))))
            .replace("{target}", target_line)},
    ]


# ---------------------------------------------------------------------------
# agent 驱动的单条 message
# ---------------------------------------------------------------------------

class _Runner:
    """把整体的重试参数与模型配置合成一份，供 ``ModelRunner`` 使用。"""

    def __init__(self, client: openai.AsyncOpenAI, config: dict[str, Any], log: Logger,
                 raw_path: Path | None = None) -> None:
        from ..search.runner import ModelRunner

        self.model = ModelRunner(client, gen_call_config(config, "atommem"), log, raw_path)


async def run_message(
    *,
    target: dict[str, Any],
    prior: list[dict[str, Any]],
    summaries: dict[int, dict[str, Any]],
    workspace: Path,
    atommem_path: Path,
    config: dict[str, Any],
    catalog: toolcfg.ToolCatalog,
    runner: _Runner,
    log: Logger,
    speaker_a: str = "speaker_a",
    speaker_b: str = "speaker_b",
) -> list[dict[str, Any]]:
    """处理一条 message：跑一轮 agent loop，返回处理后的**整份** atommem（含系统字段）。

    合并会改动/吸收别的 message 的 atom，所以返回的是完整视图（``finalize_file`` 的产物），
    而不只是本条 message 碰过的那些。
    """
    # 处理前先给整份 atommem 拍快照（id -> 记录）。收尾时用它来判断 agent 到底
    # **新增**了哪些行、**改写**了哪些已有行、**删掉**了哪些行 —— 合并/演化要靠这个，
    # 不能再像旧版那样只保留"前缀 + 追加"（那会把 agent 对已有 atom 的编辑整个丢掉）。
    before = _snapshot(atommem_path)

    shell_prefix = _shell_prefix(workspace)
    tools = tools_mod.build_tools(
        cwd=workspace, shell_command_prefix=shell_prefix,
        bash_timeout=float(config.get("bash_timeout", 60)),
        default_write_file=ATOMMEM_NAME,
    )
    catalog.apply(tools)

    messages = build_messages(
        target=target, prior=prior, summaries=summaries, config=config,
        catalog=catalog, speaker_a=speaker_a, speaker_b=speaker_b,
    )
    outcome, _ = await agent_mod.run_agent_round(
        messages=messages,
        tools=tools,
        model_call=runner.model,
        evidence_path=atommem_path,
        cwd=workspace,
        max_steps=int(config.get("max_steps", 8)),
        no_progress_patience=int(config.get("no_progress_patience", 4)),
        step_offset=0,
        round_index=1,
        log=log,
        observation_max_chars=int(config.get("observation_max_chars", 8000)),
        write_nudge_after=max(2, int(config.get("max_steps", 8)) // 2),
        evidence_soft_limit=0,          # 不提醒"太多"：一条 message 的 atom 数由内容决定
        max_parse_retries=int(config.get("max_parse_retries", 2)),
        repeat_limit=int(config.get("repeat_limit", 1)),
        duplicate_warn_at=10,
        context_max_chars=int(config.get("context_max_chars", 120000)),
        context_keep_recent=int(config.get("context_keep_recent", 6)),
    )

    # 读取 agent 处理后的整份文件，与快照对比，做归并 + 补系统字段，原子写回。
    after = read_jsonl(atommem_path)
    atoms = finalize_file(
        before=before,
        after=after,
        source_id=memory_id(target),
        default_time=str((target.get("metadata") or {}).get("time") or ""),
        default_role=memory_role(target),
    )
    _safe_write(atommem_path, atoms)
    # 返回**整份** atommem（而不只是本条 message 碰过的那些）—— 因为合并会改动/吸收别的
    # message 的 atom，调用方需要拿到处理后的完整视图来维护"当前语料"。
    return atoms


def _snapshot(path: Path) -> dict[str, dict[str, Any]]:
    """把整份 atommem 读成 ``id -> record``（只收有 ``metadata.id`` 的完整记录）。"""
    out: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        mem_id = memory_id(record)
        if mem_id:
            out[mem_id] = record
    return out


def _safe_write(path: Path, records: list[dict[str, Any]]) -> None:
    """原子写回整份 atommem（临时文件 + os.replace），避免中途损坏。"""
    from ..io import write_jsonl

    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, records)


def _shell_prefix(workspace: Path) -> str:
    from ..search.runner import build_shell_prefix

    return build_shell_prefix(inputs_dir=workspace, config={})


# ---------------------------------------------------------------------------
# 系统字段补全
# ---------------------------------------------------------------------------

def _normalize_tags(value: Any) -> list[str]:
    """把 agent 写的 ``tag`` 归一成 ``list[str]``。

    模型常把 ``tag`` 写成**字符串**（``"activity:swimming"``）而不是数组 —— 直接迭代字符串
    会把每个字符拆成一个 tag（``["a","c","t",...]``）。所以：字符串按整体当一个 tag（逗号
    分隔时再拆），列表逐项保留，非字符串一律丢弃。空白 tag 丢掉。
    """
    if isinstance(value, str):
        raw_items = value.replace("，", ",").split(",")
    elif isinstance(value, list):
        raw_items = value
    else:
        return []
    tags: list[str] = []
    for item in raw_items:
        text = item.strip() if isinstance(item, str) else ""
        if text and text not in tags:
            tags.append(text)
    return tags


def finalize_file(
    *,
    before: dict[str, dict[str, Any]],
    after: list[dict[str, Any]],
    source_id: str,
    default_time: str,
    default_role: str,
) -> list[dict[str, Any]]:
    """把 agent 处理后的整份 atommem 归并成**带系统字段、无重复**的最终记录。

    与只认 append 的旧版相比，这里尊重 agent 对文件的**全部**改动（用户要求的"合并演化"）：

    - **新增行**（agent 只写了 ``memory`` + ``metadata.{time, role, tag}``，没有系统 ``id``）：
      补 ``id = {source_id}_{k}``（k 在本 message 既有 atom 之后续号）、``type=atom``、
      ``source=[source_id]``、``changelog=[created]``。
    - **改写行**（``id`` 在 ``before`` 里出现过，或带合法旧 id）：**保留原 id**（
      ``source`` 补上本 message id），``changelog`` 追加一条记录**改动前的 memory** ——
      这就是"演化"。合并时 agent 把两条的来源 id 都写进 ``source``，这里原样收下。
    - **消失的行**（``before`` 里有、``after`` 里没有）：尊重删除（合并时"吸收进另一条"）。

    最后按 **memory 归一化**做一次 CPU 侧去重兜底：同一事实只留一条，``source`` 取并集、
    ``changelog`` 取最长的一条。空 memory 一律丢弃。
    """
    existing_source_ids = {
        memory_id(rec) for rec in before.values() if memory_id(rec).startswith(f"{source_id}_")
    }
    # 本 message 已有 id 的续号起点：{source_id}_{k} 的最大 k + 1。
    counter = _max_suffix(existing_source_ids, source_id)
    # 辅助索引：memory 归一化 -> 已有记录的 id。当 agent 用 replace 改写一行却丢了 ``id``
    # 时，用它把改后的行认回原来那一条（否则会被误当成新 atom，丢掉演化与去重）。
    before_by_memory: dict[str, str] = {}
    for mem_id, rec in before.items():
        key = _dedup_key(memory_text(rec))
        if key:
            before_by_memory.setdefault(key, mem_id)

    finalized: list[dict[str, Any]] = []
    for raw in after:
        text = memory_text(raw).strip()
        if not text:
            continue
        meta = dict(raw.get("metadata") or {})
        old_id = memory_id(raw)
        if old_id not in before:
            # 没有系统 id（或 id 不认识）—— 可能是一次丢掉了 id 的 replace；按 memory 认回。
            old_id = before_by_memory.get(_dedup_key(text), "")
        time_value = meta.get("time")
        if not isinstance(time_value, str) or not time_value.strip():
            time_value = default_time
        role_value = meta.get("role")
        if not isinstance(role_value, str) or not role_value.strip():
            role_value = default_role
        tags = _normalize_tags(meta.get("tag"))

        if old_id and old_id in before:
            # ---- 已有 atom：agent 可能改写（合并/演化）也可能原样没动 ----
            prev = before[old_id]
            old_memory = memory_text(prev).strip()
            if old_memory == text:
                # 内容没变 = agent 没碰这一行，只是它一直在文件里 —— **原样保留**，
                # 绝不能把本 message 的 id 塞进它的 source（否则每处理一条 message，
                # 整份文里所有旧 atom 的 source 都会多一个 id，几步就炸成几十个）。
                finalized.append(prev)
                continue
            # 内容变了 = 真正的改写/合并：保留 id、累加 source、changelog 记下改动前的值。
            sources = _merge_ids(meta.get("source"), prev.get("metadata", {}).get("source"))
            sources = _merge_ids(sources, [source_id])
            changelog = list((prev.get("metadata") or {}).get("changelog") or [])
            changelog.append({"time": now_iso(), "content": old_memory})
            finalized.append(_atom(old_id, text, time_value, role_value, tags, sources, changelog))
        else:
            # ---- 新 atom：系统分配 id / source / changelog（**忽略模型自带的 source**）----
            counter += 1
            finalized.append(_atom(
                f"{source_id}_{counter}", text, time_value, role_value, tags,
                [source_id], [{"time": now_iso(), "content": "created"}],
            ))
    return _dedup_atoms(finalized)


def _atom(
    mem_id: str, text: str, time_value: str, role_value: str,
    tags: list[str], sources: list[str], changelog: list[dict[str, Any]],
) -> dict[str, Any]:
    return {"memory": text, "metadata": {
        "id": mem_id, "type": "atom", "time": time_value, "role": role_value,
        "tag": tags, "source": sources, "changelog": changelog,
    }}


def _max_suffix(ids: set[str], source_id: str) -> int:
    """从 ``{source_id}_{k}`` 这类 id 里取出最大的 k（用于新 atom 续号，避免撞已有 id）。"""
    top = 0
    prefix = f"{source_id}_"
    for mem_id in ids:
        if mem_id.startswith(prefix):
            tail = mem_id[len(prefix):]
            if tail.isdigit():
                top = max(top, int(tail))
    return top


def _merge_ids(*groups: Any) -> list[str]:
    """把多个 id 来源（字符串或列表）合成一个**去重、保序**的 id 列表。"""
    out: list[str] = []
    for group in groups:
        if isinstance(group, str):
            group = [group]
        if not isinstance(group, list):
            continue
        for item in group:
            if isinstance(item, str) and item.strip() and item not in out:
                out.append(item)
    return out


def _dedup_key(text: str) -> str:
    """memory 归一化后的去重键：小写、去标点、压空白（容忍大小写/标点/尾句号差异）。"""
    import re

    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


# 近义去重的相似度阈值：token 集合 Jaccard ≥ 此值即视为"同一事实的换说法"。
# 0.6 是精挑的：真换说法（"values X" / "appreciates X"）通常 ≥0.6，而不同事实即便共享
# 主语也多在 0.3 以下。调低会误杀，调高会漏并。
NEAR_DUP_JACCARD = 0.6
# 参与相似度比较的实词长度下限：太短的 token（a / is / in）不承载"是不是同一事实"的信息。
_TOKEN_MIN_LEN = 3
# 常见虚词，比较前先丢掉，避免它们把 Jaccard 抬到 0.6 以上误判。
_STOPWORDS = frozenset(
    "the a an and or but of to in on at for with her his their its it is are was were be been "
    "being that this these those as by from into over under about".split()
)


def _tokens(text: str) -> set[str]:
    """把 memory 拆成**实词**集合（小写、≥3 字母、去虚词），用于近义相似度比较。"""
    import re

    return {
        tok for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) >= _TOKEN_MIN_LEN and tok not in _STOPWORDS
    }


def _subject(text: str) -> str:
    """取 memory 的**主语**（句首第一个专有名词 / 实词），用于挡住"A 欣赏 B" vs "B 欣赏 A"
    这类**主客体颠倒**的假合并 —— 它们的 token 集合几乎一样，但主语不同，是两条不同事实。"""
    import re

    match = re.search(r"[A-Z][a-z]+", text)
    return match.group(0).lower() if match else ""


def _is_same_fact(a: str, b: str) -> bool:
    """两条 memory 是否描述**同一事实**。

    三个都满足才算 —— 单看 token 集合会把主客体颠倒的句子（"Caroline 欣赏 Melanie" /
    "Melanie 欣赏 Caroline"）误判成同一事实。
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if _subject(a) and _subject(b) and _subject(a) != _subject(b):
        return False
    # 长度差太远（一方是另一方两倍以上）多半不是换说法，而是长句里捎带提了一下。
    if max(len(ta), len(tb)) > 2 * min(len(ta), len(tb)):
        return False
    return len(ta & tb) / len(ta | tb) >= NEAR_DUP_JACCARD


def _dedup_atoms(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """CPU 侧去重兜底：同一事实只留一条，``source`` 取并集、``changelog`` 取更全的。

    两遍扫描 ——
    1. **精确去重**（归一化后完全相同）：抓大小写/标点/尾句号差异，O(n)。
    2. **近义去重**：模型仍会用**不同措辞**重写同一事实（"values self-expression" /
       "appreciates self-expression" / 同一 turn 里一口气吐 7 条 community/inclusivity）。
       prompt 拦不全，所以这里用 token 集合 Jaccard（≥ ``NEAR_DUP_JACCARD``）再并一轮。
    保留**第一次**出现的那条（它带最原始的措辞与时间）。合并是 O(n²)，但 n 是"逐 message
    增量"的单条 message 产出 + 全库，实测几十~几百条，可接受。
    """
    # 第一遍：精确去重（同时按归一化键归并 source/changelog）。
    exact: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for atom in atoms:
        key = _dedup_key(memory_text(atom))
        if not key:
            continue
        kept = exact.get(key)
        if kept is None:
            exact[key] = atom
            order.append(key)
            continue
        _absorb(kept, atom)

    # 第二遍：近义去重 —— 每条与已保留的比，命中就并进去（source 并集）。
    merged: list[dict[str, Any]] = []
    for key in order:
        atom = exact[key]
        text = memory_text(atom)
        target = None
        for kept in merged:
            if _is_same_fact(text, memory_text(kept)):
                target = kept
                break
        if target is None:
            merged.append(atom)
        else:
            _absorb(target, atom)
    return merged


def _absorb(kept: dict[str, Any], atom: dict[str, Any]) -> None:
    """把 ``atom`` 并进 ``kept``：``source`` 取并集，``changelog`` 取更长的一份。"""
    kept_meta = kept.setdefault("metadata", {})
    kept_meta["source"] = _merge_ids(kept_meta.get("source"), atom["metadata"].get("source"))
    if len(atom["metadata"].get("changelog") or []) > len(kept_meta.get("changelog") or []):
        kept_meta["changelog"] = atom["metadata"].get("changelog")


# ---------------------------------------------------------------------------
# 单目录
# ---------------------------------------------------------------------------
async def run_dir(
    directory: Path,
    config: dict[str, Any],
    client: openai.AsyncOpenAI,
    catalog: toolcfg.ToolCatalog,
    log: Logger,
    *,
    limit_messages: int = 0,
) -> dict[str, Any]:
    """对一个目录的 msgmem **逐 message 串行**跑 atom 抽取，就地写 atommem.jsonl。

    串行的意义：处理第 N 条 message 时，`atommem.jsonl` 里已经有前 N-1 条产出的 atom，
    agent 能 grep 到它们（"跨 session 影响"就靠这个）。所以每处理一条就**增量落盘**一次。
    """
    label = directory.name
    files = directory_files(directory)
    msg_path = files["msgmem"]
    if not msg_path.exists():
        return {"dir": label, "status": "SKIPPED",
                "reason": f"no {MSGMEM_NAME}（先跑 add --stage msgmem）"}
    messages = read_jsonl(msg_path)
    summaries_by_session = group_by_session(read_jsonl(files["summem"]))
    summaries = {number: records[0] for number, records in summaries_by_session.items() if records}

    atommem_path = files["atommem"]
    # 单一有序的"当前语料"列表。合并会让一条 atom 带多个 source id，所以**不能**再按
    # source 分桶（那会把合并后的 atom 复制到多个桶里）；这里直接维护整份列表，
    # 每处理一条 message 就把它交给 agent、再取回处理后的完整列表。
    known = {memory_id(m) for m in messages}
    current: list[dict[str, Any]] = read_jsonl(atommem_path)

    speaker_a, speaker_b = parse_dir_label(label)
    workspace = directory
    runner = _Runner(client, config, log, directory / ".atommem.model_calls.jsonl")

    targets = messages if not limit_messages else messages[:limit_messages]
    for position, target in enumerate(targets):
        session = msg_session_number(memory_id(target))
        prior = _session_prior(messages, position, session)
        node_log = log.bind(f"{label}:{memory_id(target)}")

        # 幂等：重跑一条 message 时，先移除它**最初产出**的 atom（primary source = 本 message），
        # 再让 agent 重新生成/合并。历史里 source 不在 msgmem 的 atom（手动标的）原样保留。
        current = [
            a for a in current
            if _primary_source(a) != memory_id(target)
            or not (set((a.get("metadata") or {}).get("source") or []) & known)
        ]
        # 处理前把当前视图落盘（增量：第 N 条看见前 N-1 条产出的 atom）。
        write_memories(atommem_path, current)
        try:
            current = await run_message(
                target=target, prior=prior, summaries=summaries, workspace=workspace,
                atommem_path=atommem_path, config=config, catalog=catalog, runner=runner,
                log=node_log, speaker_a=speaker_a, speaker_b=speaker_b,
            )
        except Exception as exc:  # noqa: BLE001 - 单条 message 失败不该让整目录没有输出
            node_log.warn(f"处理失败：{type(exc).__name__}: {exc}")
        # 处理后再落一次盘（run_message 内部已写过，这里保证 current 与磁盘一致）。
        write_memories(atommem_path, current)

    runner.model.close()

    return {
        "dir": label, "status": "OK",
        "messages": len(targets),
        "atoms": len(current),
    }


def _primary_source(record: dict[str, Any]) -> str:
    """atom 的**第一个** source id（= 它最初由哪条 message 产出）。合并时保持不变。"""
    sources = (record.get("metadata") or {}).get("source") or []
    return str(sources[0]) if sources else ""


def _session_prior(
    messages: list[dict[str, Any]], position: int, session: int | None
) -> list[dict[str, Any]]:
    """取 target 之前、**同一会话**的消息作为消解指代的窗口。"""
    if session is None:
        return messages[:position]
    prior: list[dict[str, Any]] = []
    for record in messages[:position]:
        if msg_session_number(memory_id(record)) == session:
            prior.append(record)
    return prior


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.add.atommem",
        description="逐 message 用 agent 抽原子事实，写 atommem.jsonl",
    )
    parser.add_argument("dirs", nargs="*", help="speaker 目录名；缺省处理全部")
    parser.add_argument("--config", default=str(CONFIG_FILE), help="配置（默认 configs/add.yaml）")
    parser.add_argument("--tool-config", default=str(TOOL_CONFIG_FILE),
                        help="add agent 工具配置（默认 configs/add_tool.json）")
    parser.add_argument("--limit-messages", type=int, default=0,
                        help="每个目录只处理前 N 条 message（调试用）")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖目录级并发数")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖 agent 步数上限")
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    if args.concurrency is not None:
        config["concurrency"] = args.concurrency
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps

    targets = targets_with_msgmem(args.dirs)
    if not targets:
        print(f"没有找到含 {MSGMEM_NAME} 的目录（先跑 `python -m codemem.add --stage msgmem`）",
              file=sys.stderr)
        return

    catalog = toolcfg.load_catalog(Path(args.tool_config))
    catalog.validate(["read", "edit", "bash"])
    log = Logger(level="info")
    client = llm.make_client(gen_call_config(config, "atommem"))
    semaphore = asyncio.Semaphore(max(1, int(config.get("concurrency", 8))))

    async def one(directory: Path) -> dict[str, Any]:
        async with semaphore:
            return await run_dir(
                directory, config, client, catalog, log, limit_messages=args.limit_messages,
            )

    try:
        results = list(await asyncio.gather(*(one(d) for d in targets)))
    finally:
        await client.close()

    for item in results:
        if item.get("status") == "OK":
            print(f"atommem   {item['dir']}: {item['messages']} 条 message / "
                  f"{item['atoms']} 条 atom")
        else:
            print(f"atommem   {item['dir']}: {item.get('status')} — {item.get('reason', '')}")


def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(parse_args(argv)))


if __name__ == "__main__":
    main()
