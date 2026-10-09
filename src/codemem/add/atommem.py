"""add · 第三步：原子事实抽取（atommem.jsonl）—— **pipeline（整目录并行）**。

     msgmem.jsonl ─► atommem.jsonl（逐 message 的原子事实）

**三阶段**（实现见 ``atommem_pipeline.py``）：召回（同会话前文）→ 单次 LLM 抽
``{memory, time_expr, role, tag}`` → 代码级后处理（时间解析 + 补字段 + 去重）。

**整目录并行**：每条 message 的抽取互不依赖（召回只读 ``msgmem``、id 按各自 source 分配、
去重是收尾的一次全局 pass），所以全部 message 并发抽取后一次性去重落盘。实测 419 条
message：串行 ~45 分钟 → 并行 ~1 分钟。

本模块保留**共享**的配置加载、目录解析、tag 归一化（``_normalize_tags``）与近义去重
（``_dedup_atoms`` 及其 token helper）。逐 message 的 **code agent** 变体已迁至 ``add-agent``
分支（``git switch add-agent``）。

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
from ..io import DATA_DIR, PROJECT_ROOT, memory_id, memory_text, read_jsonl
from ..log import Logger
from .memory import (
    ATOMMEM_NAME,
    MSGMEM_NAME,
    SUMMEM_NAME,
    parse_dir_label,
    write_memories,
)

CONFIG_FILE = PROJECT_ROOT / "configs" / "add.yaml"

# atom 的 tag key 固定集合（固定 12 个 key）。校验/收敛在 _normalize_tags；模型实测会自造
# 100+ 个自由 key，故由 harness 兜底，而非只靠 prompt。
TAG_KEYS: frozenset[str] = frozenset({
    "speaker", "topic", "activity", "action", "event", "state",
    "sentiment", "relation", "entity", "possession", "time", "other",
})
# 常见近义词 → 规范 key（值保留，只收敛 key）；其余未知 key 一律落 other。
# 覆盖两类：通用近义词 + 实测里模型最爱自造的 key（见 docs 变更记录）。
_TAG_ALIASES: dict[str, str] = {
    "hobby": "activity", "hobbies": "activity", "interest": "topic", "interests": "topic",
    "emotion": "sentiment", "feeling": "sentiment", "feelings": "sentiment", "mood": "sentiment",
    "art": "activity", "music": "activity", "sport": "activity", "sports": "activity",
    "person": "entity", "people": "entity", "place": "entity", "object": "entity",
    "item": "possession", "items": "possession", "owner": "possession",
    "plan": "action", "goal": "action", "opinion": "state", "belief": "state", "value": "state",
    "values": "state", "relationhip": "relation", "relationship": "relation", "family": "relation",
    "friend": "relation", "friendship": "relation", "date": "time", "when": "time",
    # —— 实测高频自造 key ——
    "support": "relation", "adoption": "topic", "appreciation": "sentiment",
    "gratitude": "sentiment", "pride": "sentiment", "excitement": "sentiment",
    "curiosity": "sentiment", "motivation": "state", "determination": "state",
    "commitment": "state", "resilience": "state", "career": "topic", "job": "topic",
    "activism": "activity", "volunteering": "activity", "self-care": "activity",
    "experience": "state", "memory": "event", "growth": "state", "identity": "topic",
    "greeting": "other", "question": "other", "apology": "action", "agreement": "action",
    "transition": "event", "books": "entity", "book": "entity", "pet": "entity",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "model": "local",
    "base_url": "http://127.0.0.1:30000/v1",
    "api_key": "sglang",
    "temperature": 0.2,
    "max_tokens": 4096,
    # pipeline：同目录内并行抽取的 message 上限（也用作目录级并发）
    "concurrency": 8,
    "timeout": 300,
    "max_retries": 3,
    "backoff_base": 2.0,
    "rate_limit_backoff": 10.0,
    "max_backoff": 60.0,
    "backoff_jitter": 0.2,
    "enable_thinking": False,
    # 逐消息抽取时附上的同会话前 N 条消息（原始措辞），供消解指代
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
# tag 归一化（pipeline 与上层都依赖）
# ---------------------------------------------------------------------------

def _normalize_tags(value: Any) -> list[str]:
    """把模型写的 ``tag`` 归一成 ``list[str]``，并**强制 key 落在固定集合内**。

    模型常把 ``tag`` 写成**字符串**（``"activity:swimming"``）而不是数组 —— 直接迭代字符串
    会把每个字符拆成一个 tag（``["a","c","t",...]``）。所以：字符串按整体当一个 tag（逗号
    分隔时再拆），列表逐项保留，非字符串一律丢弃。空白 tag 丢掉。

    **key 纪律由 harness 兜底**：模型实测会自造 100+ 个自由 key（``art:`` / ``emotion:`` /
    ``support:`` ……），prompt 里的"key 必须从固定集合里选"这条软约束靠不住。所以这里把
    key 不在允许集合内的 tag **改写成 ``other:<原 value>``**（值保留、只收敛 key），
    常用近义词顺带映射到规范 key。这样落盘的 tag key 永远 ≤ 固定集合大小。
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
        if not text:
            continue
        key, sep, val = text.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if not sep or not val:
            # 没有 ``key:value`` 形状 —— 模型只给了一个裸词（实测常见：直接写 "art" / "emotion"
            # / "support"）。若是已知 key 或别名，收敛成 ``<norm_key>:<原词>``；否则落 other。
            bare = key or text.strip().lower()
            norm_key = _TAG_ALIASES.get(bare, bare)
            if norm_key in TAG_KEYS:
                norm = f"{norm_key}:{bare}"
            else:
                norm = f"other:{text}"
        else:
            norm_key = _TAG_ALIASES.get(key, key)
            if norm_key not in TAG_KEYS:
                norm_key = "other"
            norm = f"{norm_key}:{val}"
        if norm not in tags:
            tags.append(norm)
    return tags


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


# ---------------------------------------------------------------------------
# 近义去重（CPU 侧兜底）
# ---------------------------------------------------------------------------

def _dedup_key(text: str) -> str:
    """memory 归一化后的去重键：小写、去标点、压空白（容忍大小写/标点/尾句号差异）。"""
    import re

    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


# 近义去重的相似度阈值：token 集合 Jaccard ≥ 此值即视为"同一事实的换说法"。
# 0.6 是精挑的：真换说法（"values X" / "appreciates X"）**有时**能到 0.6，但模型更常见的
# 是同一条事实换一串**情绪动词**（praises / supports / values / appreciates / recognizes /
# highlights + 同一宾语），这类两两 Jaccard 只有 0.2~0.5，纯 Jaccard 抓不住。所以补一条
# **去动词核心**规则：剥掉情绪动词后，若两句话共享 ≥3 个实词且共享率 ≥0.25，判为同一事实
# —— 但**仅当两句话本身都含情绪动词**时才适用（见 `_is_same_fact` 的前提判断）。
#
# **实测事故（范式1 根因之一）**：这条前提最初缺失，导致 "Caroline passed adoption agency
# interviews"（动作动词 passed）与 "Caroline chose an adoption agency"（动作动词 chose）
# 剥词后共享 {caroline, adoption, agency} 3 个实词、ratio=0.5 ≥ 0.25，被误判成"同一事实
# 换说法"而合并，核心事实（通过面试）整条丢失，QA 答案跟着错。problem 不在阈值本身——
# 真正的误判案例 ratio 落在 0.33~0.5，恰好与真换说法案例（情绪动词剥完后 ratio 0.27~1.0，
# 因宾语措辞不同也会低到 0.27）**区间重叠**，调阈值两头不能兼顾。真正的区分信号是：
# 情绪动词换说法的两句话必须**都在表达情绪/态度**（praises/supports/values/...），而
# "passed"/"chose" 是具体动作，根本不该走这条"换个态度词、事实不变"的捷径。
NEAR_DUP_JACCARD = 0.6
_SHARED_CORE_MIN = 3      # 去动词后至少要共享几个实词
_SHARED_CORE_RATIO = 0.25  # 且共享率（交集/并集）不低于这个数（仅情绪动词句对生效）
# 情绪/态度动词：只表达"对同一件事的态度"，本身不构成"不同事实"的区分度。合并比较前剥掉。
_SENTIMENT_VERBS = frozenset(
    "praise praises praised support supports supported value values valued appreciate appreciates "
    "appreciated recognize recognizes recognized highlight highlights highlighted acknowledge "
    "acknowledges acknowledged believe believes believed think thinks thought find finds found "
    "consider considers considered regard regards regarded encourage encourages encouraged admire "
    "admires admired approve approves approved endorse endorses endorsed celebrate celebrates "
    "celebrated feel feels felt".split()
)
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


def _core_tokens(text: str) -> set[str]:
    """剥掉情绪/态度动词后的实词集合 —— 用来识别"同一事实换了个态度动词"的重复。"""
    return {tok for tok in _tokens(text) if tok not in _SENTIMENT_VERBS}


def _has_sentiment_verb(text: str) -> bool:
    """这句话里是不是真的出现了情绪/态度动词 —— "去动词核心"合并规则的生效前提。

    没有这条前提，规则会把任何"剥掉情绪动词后共享几个实词"的句子都判成重复 —— 但
    "Caroline passed interviews" 和 "Caroline chose an agency" 里根本没有情绪动词，
    "passed"/"chose" 是具体动作，剥不掉也不该剥。只有两句话都在表达情绪/态度时，
    "换了个态度词、事实没变"这个假设才成立。"""
    return bool(_SENTIMENT_VERBS & _tokens(text))


def _subject(text: str) -> str:
    """取 memory 的**主语**（句首第一个专有名词 / 实词），用于挡住"A 欣赏 B" vs "B 欣赏 A"
    这类**主客体颠倒**的假合并 —— 它们的 token 集合几乎一样，但主语不同，是两条不同事实。"""
    import re

    match = re.search(r"[A-Z][a-z]+", text)
    return match.group(0).lower() if match else ""


# 句首大写但**不是**专有名词的功能词 —— 不参与"专有名词"集合（否则 "The kids..." 里的
# "The" 会被当成实体名）。只留真正的实体名（人名/地名/书名/宠物名…）。
_PROPER_STOP = frozenset(
    "the a an this that these those her his their its it he she they we i my our your there then "
    "when if and or but in on at for with from by as of to after before during last next every "
    "each some all both no not".split()
)


def _proper_nouns(text: str) -> set[str]:
    """句子里的**专有名词**集合（首字母大写的实词，小写化；去句首功能词）。

    用来挡住"模板相同、只有名字不同"的误合并 —— 实测 `_is_same_fact` 把
    "Melanie has a pet named **pup**" 与 "Melanie has a pet named **Luna**" 判成同一事实
    （token Jaccard 0.67），于是把 Luna / Oliver / Bailey 三条**并进了 pup 那条**，枚举成员
    名全丢（qa52 / qa124 的 pet 题就错在这）。**不同的专有名词 = 不同的实体 = 不同的事实。**
    """
    import re

    return {
        word.lower()
        for word in re.findall(r"\b[A-Z][a-zA-Z]+\b", text)
        if word.lower() not in _PROPER_STOP
    }


def _is_same_fact(a: str, b: str) -> bool:
    """两条 memory 是否描述**同一事实**。

    主语必须一致 —— 单看 token 集合会把主客体颠倒的句子（"Caroline 欣赏 Melanie" /
    "Melanie 欣赏 Caroline"）误判成同一事实。主语一致后再看两条路径任一命中：

    1. **整体 Jaccard ≥ 0.6**：措辞几乎一致（大小写/冠词差异）。
    2. **两句话都含情绪动词，且去动词核心后 ≥ 3 个实词、共享率 ≥ 0.25**：模型反复犯的
       "同一事实换情绪动词"（praises/supports/values/appreciates/recognizes/highlights +
       同一宾语）。**情绪动词前提是硬性的** —— 没有它，"passed interviews" 与 "chose an
       agency" 这类具体动作不同、仅主宾语偶然重叠的句子也会被判成重复（见上面的事故记录）。
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if _subject(a) and _subject(b) and _subject(a) != _subject(b):
        return False
    # **专有名词不同 = 不同事实。** 挡住 "has a pet named pup/Luna/Oliver" 这类
    # "同一模板、仅名字不同"的误合并 —— 名字正是区分枚举成员的关键，措辞再像也不能并。
    if _proper_nouns(a) != _proper_nouns(b):
        return False
    # 路径 1：措辞近乎一致。
    if len(ta & tb) / len(ta | tb) >= NEAR_DUP_JACCARD:
        return True
    # 路径 2：仅当两句话都在表达情绪/态度时才适用（换个态度词、事实不变 —— 具体动作不适用）。
    if not (_has_sentiment_verb(a) and _has_sentiment_verb(b)):
        return False
    ca, cb = _core_tokens(a), _core_tokens(b)
    if not ca or not cb:
        return False
    shared = ca & cb
    return len(shared) >= _SHARED_CORE_MIN and len(shared) / len(ca | cb) >= _SHARED_CORE_RATIO


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
# per-directory（pipeline）
# ---------------------------------------------------------------------------

async def run_dir(
    directory: Path,
    config: dict[str, Any],
    client: openai.AsyncOpenAI,
    log: Logger,
    *,
    limit_messages: int = 0,
    semaphore: "asyncio.Semaphore | None" = None,
) -> dict[str, Any]:
    """对一个目录的 msgmem 抽 atom，就地写 atommem.jsonl（**pipeline，整目录并行**）。

    每条 message 的抽取互不依赖（召回只读 ``msgmem``、id 按各自 source 分配、去重是收尾的
    一次全局 pass），所以全部 message 并发抽取后一次性去重落盘。实测 419 条 message：串行
    ~45 分钟 → 并行 ~1 分钟。
    """
    label = directory.name
    files = directory_files(directory)
    msg_path = files["msgmem"]
    if not msg_path.exists():
        return {"dir": label, "status": "SKIPPED",
                "reason": f"no {MSGMEM_NAME}（先跑 add --stage msgmem）"}
    messages = read_jsonl(msg_path)

    atommem_path = files["atommem"]
    known = {memory_id(m) for m in messages}
    current: list[dict[str, Any]] = read_jsonl(atommem_path)

    speaker_a, speaker_b = parse_dir_label(label)
    targets = messages if not limit_messages else messages[:limit_messages]

    from .atommem_pipeline import run_message_pipeline

    # 幂等：先移除**本次要处理的每个 message** 最初产出的 atom（重跑不累积）。
    target_ids = {memory_id(t) for t in targets}
    kept = [
        a for a in current
        if _primary_source(a) not in target_ids
        or not (set((a.get("metadata") or {}).get("source") or []) & known)
    ]

    # 并发闸门：多目录一起跑时共用调用方传进来的**全局** semaphore（否则每个目录各开
    # concurrency 个任务，10 个目录就是 10×concurrency 个并发请求，会把模型端点打爆）。
    gate = semaphore or asyncio.Semaphore(max(1, int(config.get("concurrency", 8))))

    async def extract(target: dict[str, Any]) -> list[dict[str, Any]]:
        async with gate:
            node_log = log.bind(f"{label}:{memory_id(target)}")
            try:
                return await run_message_pipeline(
                    target=target, msgmem=messages, existing_atoms=[], client=client,
                    config=config, log=node_log, speaker_a=speaker_a, speaker_b=speaker_b,
                )
            except Exception as exc:  # noqa: BLE001 - 单条失败不该让整目录没有输出
                node_log.warn(f"处理失败：{type(exc).__name__}: {exc}")
                return []

    groups = await asyncio.gather(*(extract(t) for t in targets))
    merged = _dedup_atoms(kept + [a for group in groups for a in group])
    write_memories(atommem_path, merged)
    return {"dir": label, "status": "OK", "messages": len(targets), "atoms": len(merged)}


def _primary_source(record: dict[str, Any]) -> str:
    """atom 的**第一个** source id（= 它最初由哪条 message 产出）。合并时保持不变。"""
    sources = (record.get("metadata") or {}).get("source") or []
    return str(sources[0]) if sources else ""


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.add.atommem",
        description="逐 message 抽原子事实，写 atommem.jsonl（pipeline）",
    )
    parser.add_argument("dirs", nargs="*", help="speaker 目录名；缺省处理全部")
    parser.add_argument("--config", default=str(CONFIG_FILE), help="配置（默认 configs/add.yaml）")
    parser.add_argument("--limit-messages", type=int, default=0,
                        help="每个目录只处理前 N 条 message（调试用）")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖并发数")
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    if args.concurrency is not None:
        config["concurrency"] = args.concurrency

    targets = targets_with_msgmem(args.dirs)
    if not targets:
        print(f"没有找到含 {MSGMEM_NAME} 的目录（先跑 `python -m codemem.add --stage msgmem`）",
              file=sys.stderr)
        return

    log = Logger(level="info")
    client = llm.make_client(gen_call_config(config, "atommem"))
    # 并发由所有 message 任务共享的**全局** sem 统一约束（跨目录也收敛到 concurrency）。
    semaphore = asyncio.Semaphore(max(1, int(config.get("concurrency", 8))))

    async def one(directory: Path) -> dict[str, Any]:
        return await run_dir(directory, config, client, log,
                             limit_messages=args.limit_messages, semaphore=semaphore)

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
