"""确定性证据审计：拿语料原文核对 evidence 里每条结论**有没有出处**。

为什么需要它：``configs/tool.json`` 已经写明"不得编造 source""会话时间只是锚点不是事件
日期"，但模型不遵守。实测 756 条 evidence 里有 103 条（13.6%）与其 cite 的来源对不上 ——
典型如把 "Caroline attended the LGBTQ support group **on 25 May 2023**" 挂在 ``session_2_5``
上，而 "support group" 全语料只出现在 session 1 和 4，``session_2_5`` 讲的是 self-care。
另有把 "Matt Patterson" 当成音乐家的凭空捏造（qa112/113）。

**prompt 已经说尽了，模型不听** —— 所以这里由 harness 做机器能**确定判定**的两件事：

1. **引用的 source id 必须真实存在于语料**（不存在的 id = 凭空编的）—— 零误报。
2. **正文里的专有名词必须能在被引来源的原文里找到**。

第 2 条的关键是**怎么只抓真问题**。实测三种朴素做法都会淹没在噪声里：

- 整句 token 覆盖率：evidence 是改写过的长句、来源是短聊天句，正常改写的中位覆盖率只有
  0.385，阈值放到 0.34 会命中 46% —— 全在误报。
- 所有首字母大写词：会把句首的普通词（"Through"、"Seasons"、"Overall"）当成实体名。
- 纯句中外的大写词：仍会把 "The **Four** Seasons" 里的 "Four"、"Mel"（Melanie 的昵称）算进去。

最终判据（本模块的实现）：**句中外的大写词** ∩ **语料里从未以小写出现过** ∩ **不是月份/
星期/说话人名（或其前缀）**。这样只剩真正的专名。在 969 条真实 evidence 上标出 6.8%，
且命中已知的造假样本（``lgbtq`` 挂错会话、``patterson``/``matt`` 凭空捏造）。

**只提示、绝不删改** agent 写的内容 —— 与 ``agent.py`` 里其它 notice（nudge / oversize /
grep-no-match / evidence-digest）同一风格。删改语义问题是 ``evidence.py`` 明确拒绝做的事：
那会让评测数字好看，而 agent 没学会。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from .io import memory_id, memory_text, read_jsonl

# 与 ``search/runner.CORPUS_FILES`` 一致。这里另起一份而不是 import：runner 会 import
# agent，而 agent 要 import 本模块 —— 反向 import 会成环。
CORPUS_FILES = ("msgmem.jsonl", "summem.jsonl", "atommem.jsonl")

# 月份/星期：正文里出现月份是**正常的时间表达**（"25 May 2023"），不该当成"编造的实体"。
_DATE_WORDS = frozenset(
    "january february march april may june july august september october november december "
    "jan feb mar apr jun jul aug sep sept oct nov dec "
    "monday tuesday wednesday thursday friday saturday sunday "
    "mon tue tues wed thu thur thurs fri sat sun".split()
)
# 非说话人的 role 值（这些不是人名）。
_NON_SPEAKER_ROLES = frozenset({"summary", "atom", "unknown", "", "none"})
# 一个词的"较短前缀"至少要这么长才算昵称匹配（"mel" ⊂ "melanie"）。
_NICKNAME_MIN = 3

_MID_PROPER = re.compile(r"[A-Z][a-zA-Z]{2,}")
_LOWERCASE_WORD = re.compile(r"\b[a-z]{3,}\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class CorpusIndex:
    """语料索引：``id -> 原文``，外加"哪些词是普通词""哪些是说话人名"。"""

    texts: dict[str, str]
    speakers: frozenset[str]
    lowercase_words: frozenset[str]

    def text_for(self, source_id: str) -> str:
        """取某个 source id 的原文。atom id（``session_1_3_1``）回退到它的消息 id。"""
        if source_id in self.texts:
            return self.texts[source_id]
        parts = source_id.split("_")
        if len(parts) >= 3:
            return self.texts.get("_".join(parts[:3]), "")
        return ""

    def is_speaker_like(self, word: str) -> bool:
        """这个词像不像说话人的名字（含昵称前缀，如 "mel" vs "melanie"）。"""
        if word in self.speakers:
            return True
        return any(
            len(word) >= _NICKNAME_MIN and (speaker.startswith(word) or word.startswith(speaker))
            for speaker in self.speakers
        )


def build_corpus_index(inputs_dir: Path) -> CorpusIndex:
    """读 ``inputs/`` 下的三个记忆文件，建一份索引（每 QA 建一次即可）。"""
    texts: dict[str, str] = {}
    speakers: set[str] = set()
    lowercase: set[str] = set()
    for name in CORPUS_FILES:
        path = inputs_dir / name
        if not path.exists():
            continue
        for record in read_jsonl(path):
            record_id = memory_id(record)
            if not record_id:
                continue
            body = memory_text(record)
            when = str((record.get("metadata") or {}).get("time") or "")
            texts[record_id] = f"{body} {when}".strip()
            role = str((record.get("metadata") or {}).get("role") or "").strip().lower()
            if role and role not in _NON_SPEAKER_ROLES:
                speakers.add(role)
            lowercase.update(_LOWERCASE_WORD.findall(body))
    return CorpusIndex(
        texts=texts,
        speakers=frozenset(speakers),
        lowercase_words=frozenset(lowercase),
    )


@lru_cache(maxsize=8)
def _cached_index(inputs_dir_str: str) -> CorpusIndex:
    return build_corpus_index(Path(inputs_dir_str))


def corpus_index_for(inputs_dir: Path) -> CorpusIndex:
    """带缓存的入口。

    一次跑里所有 QA 指向的是**同一份** ``inputs/``（每条 QA 目录下的 ``inputs`` 是软链接到
    同一目标），文件在整轮运行期间不会变 —— 所以**先 resolve 解析掉软链接**再当缓存键，
    否则 138 条 QA 会各建一次索引（每条都要重读 1.5MB 语料）。
    """
    try:
        key = str(Path(inputs_dir).resolve())
    except OSError:
        key = str(inputs_dir)
    return _cached_index(key)


def _named_entities(text: str) -> set[str]:
    """正文里"像专名"的词：**句中外**的首字母大写词（小写化）。

    句首词不算 —— "Through"/"Seasons"/"Overall" 这类句首大写是英文语法，不是实体名。
    """
    found: set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(text):
        for word in sentence.split()[1:]:
            if _MID_PROPER.fullmatch(word):
                found.add(word.lower())
    return found


@dataclass
class AuditFinding:
    """一条证据的审计结论。"""

    body: str
    unknown_sources: tuple[str, ...] = ()
    missing_entities: tuple[str, ...] = ()

    def describe(self) -> str:
        return " ".join(self.missing_entities + self.unknown_sources)


@dataclass
class AuditReport:
    findings: list[AuditFinding] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings

    def count(self) -> int:
        return len(self.findings)


def audit_facts(
    facts: Iterable[tuple[str, Sequence[str]]],
    index: CorpusIndex,
) -> AuditReport:
    """审计 ``(正文, source id 列表)`` 序列。

    只报两类问题：**引用了语料里不存在的 id**，**正文里的专名在被引来源里找不到**。
    """
    report = AuditReport()
    for body, sources in facts:
        if not body:
            continue
        unknown = tuple(s for s in sources if not index.text_for(s))
        supported = " ".join(index.text_for(s) for s in sources).lower()
        missing = tuple(sorted(
            word for word in _named_entities(body)
            if word not in supported
            and word not in _DATE_WORDS
            and word not in index.lowercase_words      # 语料里当普通词出现过 → 不是专名
            and not index.is_speaker_like(word)
        ))
        if unknown or missing:
            report.findings.append(AuditFinding(body=body, unknown_sources=unknown,
                                                missing_entities=missing))
    return report


def audit_notice(report: AuditReport, limit: int = 2) -> str:
    """把审计结论渲染成一条**建议性**观测（最多列 ``limit`` 条）。"""
    if report.clean:
        return ""
    lines = [
        "\n\n[SELF-CHECK: some evidence cites things its sources do not contain. "
        "A downstream reader sees ONLY evidence.jsonl, so an unsupported claim is a wrong "
        "answer waiting to happen. For each item below, either (a) point it at the source id "
        "that actually states it, (b) drop the unsupported detail, or (c) if you already read "
        "it in a source, ignore this note.]"
    ]
    for finding in report.findings[:limit]:
        snippet = finding.body[:160] + ("…" if len(finding.body) > 160 else "")
        reasons = []
        if finding.missing_entities:
            reasons.append("these names/entities do not appear in the cited source: "
                           + ", ".join(finding.missing_entities))
        if finding.unknown_sources:
            reasons.append("these source ids do not exist in inputs/: "
                           + ", ".join(finding.unknown_sources))
        lines.append(f"  - {snippet}\n      -> " + "; ".join(reasons))
    if report.count() > limit:
        lines.append(f"  (+{report.count() - limit} more)")
    return "\n".join(lines)
