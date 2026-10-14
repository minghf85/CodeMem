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
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from . import adapters
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
# 非说话人的 role 值（这些不是人名）。来自 ``adapters`` 的单一事实源 —— 适配层会产出
# ``corpus`` / ``document`` / ``narration`` / ``feedback`` 这些**记录类型**角色，它们一样不是人名，
# 混进来的话"专名核对"会把 ``document`` 当成一个说话人，白白放宽对它的检查。
_NON_SPEAKER_ROLES = adapters.NON_PERSON_ROLES
# 一个词的"较短前缀"至少要这么长才算昵称匹配（"mel" ⊂ "melanie"）。
_NICKNAME_MIN = 3

_MID_PROPER = re.compile(r"[A-Z][a-zA-Z]{2,}")
# 词首/词尾的非字母字符：`'Stardew`、`Valley'`、`(Seraphim)`、`Sunset,"` 里的引号与括号。
# **实测漏检**：evidence 按证据标准本来就**鼓励引用源话**（`recommending 'Stardew Valley'`），
# 于是句中的专名常常被引号包住 —— 而 `fullmatch` 要求整词恰好是大写开头的字母串，带引号就
# 匹配不上，**恰好把最该查的"引号里的专名"整类漏掉**（qa125 编造的 'Stardew Valley' 就这样
# 逃过了审计）。所以先把两端的非字母字符剥掉再匹配。
_TRIM_NON_LETTER = re.compile(r"^[^A-Za-z]+|[^A-Za-z]+$")
_LOWERCASE_WORD = re.compile(r"\b[a-z]{3,}\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class CorpusIndex:
    """语料索引：``id -> 原文``，外加"哪些词是普通词""哪些是说话人名"。"""

    texts: dict[str, str]
    speakers: frozenset[str]
    lowercase_words: frozenset[str]
    # 消息层的原文与会话时间戳。用来抓"把消息自己的时间戳当成事件日期"这一类错
    # （见 ``_copied_anchor``）—— 那是本项目里反复出现、也最贵的一类时间错。
    msg_bodies: dict[str, str] = field(default_factory=dict)
    msg_times: dict[str, str] = field(default_factory=dict)

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
    msg_bodies: dict[str, str] = {}
    msg_times: dict[str, str] = {}
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
            if name == "msgmem.jsonl":
                msg_bodies[record_id] = body
                msg_times[record_id] = when
    return CorpusIndex(
        texts=texts,
        speakers=frozenset(speakers),
        lowercase_words=frozenset(lowercase),
        msg_bodies=msg_bodies,
        msg_times=msg_times,
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
    匹配前剥掉词首/词尾的非字母字符，好让引号或括号里的专名（``'Stardew Valley'``、
    ``(Seraphim)``）也算进来 —— 见 ``_TRIM_NON_LETTER`` 的注释。
    """
    found: set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(text):
        for word in sentence.split()[1:]:
            word = _TRIM_NON_LETTER.sub("", word)
            if _MID_PROPER.fullmatch(word):
                found.add(word.lower())
    return found


@dataclass
class AuditFinding:
    """一条证据的审计结论。"""

    body: str
    unknown_sources: tuple[str, ...] = ()
    missing_entities: tuple[str, ...] = ()
    # 把消息自己的时间戳当成事件日期（见 ``_copied_anchor``）：
    # (被引消息 id, 该消息的会话日期, 更正后的日期, 消息里的相对时间词)
    anchor_copied: tuple[str, str, str, str] | None = None

    def describe(self) -> str:
        """这条结论的指纹（``agent`` 用它做"同一个问题只提示一次"的去重键）。

        **必须非空** —— 调用方写的是 ``key = tuple(sorted(...)); if key and key not in seen``，
        所以纯"抄了锚点"的结论如果在这里返回空串，那条提示**永远不会被发出去**（实测就是这么
        踩到的：13 条 anchor 命中一条都没进观测）。
        """
        parts = list(self.missing_entities) + list(self.unknown_sources)
        if self.anchor_copied:
            source_id, said_on, corrected, _ = self.anchor_copied
            parts.append(f"anchor:{source_id}:{said_on}->{corrected}")
        return " ".join(parts)


@dataclass
class AuditReport:
    findings: list[AuditFinding] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings

    def count(self) -> int:
        return len(self.findings)


# ---------------------------------------------------------------------------
# "把消息自己的时间戳当成事件日期"
# ---------------------------------------------------------------------------

# 只有**精确偏移**才判定 —— 这些表达带的是一天的粒度，算得出来，而且算出来的日期与
# 会话日期**必然不同**，所以"记录里写的是会话日期"就等于确定写错了。
# （"today"/"tonight" 偏移为 0 —— 写会话日期反而是对的，所以不在表里。）
_MONTHS = ("january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december")
_WORD_DAYS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
              "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_EXACT_OFFSETS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bthe day before yesterday\b", re.IGNORECASE), -2),
    (re.compile(r"\bthe day after tomorrow\b", re.IGNORECASE), 2),
    (re.compile(r"\byesterday\b", re.IGNORECASE), -1),
    (re.compile(r"\btomorrow\b", re.IGNORECASE), 1),
    (re.compile(r"\b(last|this)\s+night\b", re.IGNORECASE), -1),
    (re.compile(r"\b(\d+)\s+days?\s+ago\b", re.IGNORECASE), 0),      # 0 = 取捕获组取负
)
_WORD_DAYS_AGO = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+ago\b", re.IGNORECASE)
# 会话时间戳形如 "9:17 am on 26 June, 2023"；也容忍已是 ISO 的写法。
_ANCHOR_DATE = re.compile(
    r"(?:on\s+)?(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})|(\d{4})-(\d{2})-(\d{2})")


def _anchor_date(text: str) -> "date | None":
    """从一条消息的 ``metadata.time`` 里抠出会话日期（抠不到返回 ``None``）。"""
    match = _ANCHOR_DATE.search(text or "")
    if not match:
        return None
    try:
        if match.group(4):                       # ISO
            return date(int(match.group(4)), int(match.group(5)), int(match.group(6)))
        month = _month_index(match.group(2))
        if month is None:
            return None
        return date(int(match.group(3)), month, int(match.group(1)))
    except (ValueError, IndexError):
        return None


def _month_index(name: str) -> int | None:
    """月份名（全称或三字母缩写）-> 1-12，认不出返回 ``None``。"""
    lowered = (name or "").strip().lower()
    for index, month in enumerate(_MONTHS, start=1):
        if lowered == month or (len(lowered) >= 3 and month.startswith(lowered[:3])):
            return index
    return None


def _exact_offset(text: str) -> int | None:
    """消息原文里的**精确天偏移**（"yesterday" -> -1）；没有这类表达返回 ``None``。"""
    for pattern, fixed in _EXACT_OFFSETS:
        match = pattern.search(text or "")
        if not match:
            continue
        if fixed == 0 and match.groups():        # "N days ago"
            return -int(match.group(1))
        return fixed
    word_match = _WORD_DAYS_AGO.search(text or "")
    if word_match:
        return -_WORD_DAYS[word_match.group(1).lower()]
    return None


def _date_forms(moment: "date") -> tuple[str, ...]:
    """一个日期在 evidence 正文里可能出现的写法（证据标准要求 ``"7 May 2023"``）。"""
    return moment.strftime("%-d %B %Y"), moment.strftime("%d %B %Y"), moment.isoformat()


def _copied_anchor(
    body: str,
    sources: Sequence[str],
    index: CorpusIndex,
) -> tuple[str, str, str, str] | None:
    """记录是不是**把被引消息自己的时间戳写成了事件日期**。

    判据全部是机器能确定判定的，所以零猜测：被引消息的原文里有**精确天偏移**（yesterday /
    "N days ago" / last night …），而被写成的事件日期**正好等于那条消息自己的会话日期** ——
    这时事件必然发生了偏移，记录里那个日期只能是"说这话的时间"。

    为什么值得单列一条检查：这是本项目里**反复出现、也最贵**的一类错（§2.4 整节都在讲它），
    prompt 从 S16 起写到现在，8B 与 gpt-4o-mini 都仍会犯 —— 实测 Deborah_Jolene 全量里
    每跑一次仍有 **6~10 条** 是这个形态（qa62 "Yesterday ... got back" 写成 30 August、
    qa47 "the day before yesterday" 写成 26 June）。既然偏移和锚点都是确定的，就在这里
    **把正确日期直接算出来**告诉 agent，而不是再叮嘱一遍"别把锚点当事件日期"。
    """
    lowered = (body or "").lower()
    for source_id in sources:
        raw = index.msg_bodies.get(source_id)
        if not raw:
            continue
        offset = _exact_offset(raw)
        if not offset:
            continue
        moment = _anchor_date(index.msg_times.get(source_id, ""))
        if moment is None:
            continue
        if not any(form.lower() in lowered for form in _date_forms(moment)):
            continue                              # 正文没写这条消息自己的日期 -> 没事
        corrected = moment + timedelta(days=offset)
        if any(form.lower() in lowered for form in _date_forms(corrected)):
            continue                              # 已经写了正确的日期 -> 没事
        relative = ""
        for pattern, _ in _EXACT_OFFSETS:
            match = pattern.search(raw)
            if match:
                relative = match.group(0)
                break
        if not relative:
            word_match = _WORD_DAYS_AGO.search(raw)
            relative = word_match.group(0) if word_match else ""
        return source_id, moment.strftime("%-d %B %Y"), corrected.strftime("%-d %B %Y"), relative
    return None


def audit_facts(
    facts: Iterable[tuple[str, Sequence[str]]],
    index: CorpusIndex,
) -> AuditReport:
    """审计 ``(正文, source id 列表)`` 序列。

    报三类问题，每类都是机器能**确定判定**的：**引用了语料里不存在的 id**、
    **正文里的专名在被引来源里找不到**、**把被引消息自己的时间戳当成了事件日期**
    （见 ``_copied_anchor``）。
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
        anchor = _copied_anchor(body, sources, index)
        if unknown or missing or anchor:
            report.findings.append(AuditFinding(body=body, unknown_sources=unknown,
                                                missing_entities=missing, anchor_copied=anchor))
    return report


def audit_notice(report: AuditReport, limit: int = 2) -> str:
    """把审计结论渲染成一条**建议性**观测（最多列 ``limit`` 条）。"""
    if report.clean:
        return ""
    anchors = [finding for finding in report.findings if finding.anchor_copied]
    others = [finding for finding in report.findings if not finding.anchor_copied]
    lines: list[str] = []
    if anchors:
        lines.append(
            "\n\n[SELF-CHECK: some records use the source message's OWN timestamp as the date of "
            "the event. A message dated <D> that says \"yesterday\" / \"N days ago\" / \"last "
            "night\" means the event happened at that offset from <D> -- <D> is when it was SAID. "
            "The correct date is computed for you below; EDIT the record to use it.]"
        )
        for finding in anchors[:limit]:
            source_id, said_on, corrected, relative = finding.anchor_copied or ("", "", "", "")
            snippet = finding.body[:160] + ("…" if len(finding.body) > 160 else "")
            lines.append(
                f"  - {snippet}\n      -> cites {source_id}, dated {said_on}, which says "
                f"\"{relative}\": the event happened on {corrected}, NOT {said_on}."
            )
    if others:
        lines.append(
            "\n\n[SELF-CHECK: some evidence cites things its sources do not contain. "
            "A downstream reader sees ONLY evidence.jsonl, so an unsupported claim is a wrong "
            "answer waiting to happen. For each item below, either (a) point it at the source id "
            "that actually states it, (b) drop the unsupported detail, or (c) if you already read "
            "it in a source, ignore this note.]"
        )
        for finding in others[:limit]:
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
