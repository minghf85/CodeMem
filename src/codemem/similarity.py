"""文本相似度原语：判断两条记忆是不是**同一事实**。

为什么单独成模块：这套判据现在有两个用户，而且各自的用途不同 ——

- **add/atommem**：抽取后去重，把"同一事实换了个说法"的 atom 并成一条（``_dedup_atoms``）；
- **search**：判断 agent 这一步有没有**真正新增事实**（还是只在同一行上来回改措辞），
  以及拒绝"近重复的 evidence 追加"。

一开始它们长在 ``atommem.py`` 里。search 要用时若直接 import add 的私有函数，就形成
``search → add`` 的层级倒挂（两个阶段本该只通过数据文件耦合）。所以把**判据**下沉到这里，
两边都只依赖它。

判据本身的设计依据、以及一次真实误合并事故的记录，见 ``is_same_fact`` 的注释。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 阈值与词表
# ---------------------------------------------------------------------------

# 近义去重的相似度阈值：token 集合 Jaccard ≥ 此值即视为"同一事实的换说法"。
# 0.6 是精挑的：真换说法（"values X" / "appreciates X"）**有时**能到 0.6，但更常见的
# 是同一条事实换一串**情绪动词**（praises / supports / values / appreciates / recognizes /
# highlights + 同一宾语），这类两两 Jaccard 只有 0.2~0.5，纯 Jaccard 抓不住。所以补一条
# **去动词核心**规则：剥掉情绪动词后，若两句话共享 ≥3 个实词且共享率 ≥0.25，判为同一事实
# —— 但**仅当两句话本身都含情绪动词**时才适用（见 `is_same_fact` 的前提判断）。
#
# **实测事故**：这条前提最初缺失，导致 "Caroline passed adoption agency interviews" 与
# "Caroline chose an adoption agency" 剥词后共享 {caroline, adoption, agency}、ratio=0.5 ≥ 0.25，
# 被误判成"同一事实换说法"而合并，核心事实（通过面试）整条丢失，QA 答案跟着错。问题不在
# 阈值本身 —— 真正的误判案例 ratio 落在 0.33~0.5，恰好与真换说法案例（情绪动词剥完后
# ratio 0.27~1.0）**区间重叠**，调阈值两头不能兼顾。真正的区分信号是：情绪动词换说法的
# 两句话必须**都在表达情绪/态度**，而 "passed"/"chose" 是具体动作，不该走这条捷径。
NEAR_DUP_JACCARD = 0.6
SHARED_CORE_MIN = 3        # 去动词后至少要共享几个实词
SHARED_CORE_RATIO = 0.25   # 且共享率（交集/并集）不低于这个数（仅情绪动词句对生效）

# 情绪/态度动词：只表达"对同一件事的态度"，本身不构成"不同事实"的区分度。合并比较前剥掉。
SENTIMENT_VERBS = frozenset(
    "praise praises praised support supports supported value values valued appreciate appreciates "
    "appreciated recognize recognizes recognized highlight highlights highlighted acknowledge "
    "acknowledges acknowledged believe believes believed think thinks thought find finds found "
    "consider considers considered regard regards regarded encourage encourages encouraged admire "
    "admires admired approve approves approved endorse endorses endorsed celebrate celebrates "
    "celebrated feel feels felt".split()
)

# 参与相似度比较的实词长度下限：太短的 token（a / is / in）不承载"是不是同一事实"的信息。
TOKEN_MIN_LEN = 3

# 常见虚词，比较前先丢掉，避免它们把 Jaccard 抬到阈值以上误判。
STOPWORDS = frozenset(
    "the a an and or but of to in on at for with her his their its it is are was were be been "
    "being that this these those as by from into over under about".split()
)

# 句首大写但**不是**专有名词的功能词 —— 不参与"专有名词"集合（否则 "The kids..." 里的
# "The" 会被当成实体名）。只留真正的实体名（人名/地名/书名/宠物名…）。
PROPER_STOP = frozenset(
    "the a an this that these those her his their its it he she they we i my our your there then "
    "when if and or but in on at for with from by as of to after before during last next every "
    "each some all both no not".split()
)


# ---------------------------------------------------------------------------
# 原语
# ---------------------------------------------------------------------------

def dedup_key(text: str) -> str:
    """归一化去重键：小写、去标点、压空白（容忍大小写/标点/尾句号差异）。"""
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def tokens(text: str) -> set[str]:
    """拆成**实词**集合（小写、≥3 字母、去虚词），用于近义相似度比较。"""
    return {
        tok for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) >= TOKEN_MIN_LEN and tok not in STOPWORDS
    }


def core_tokens(text: str) -> set[str]:
    """剥掉情绪/态度动词后的实词集合 —— 用来识别"同一事实换了个态度动词"的重复。"""
    return {tok for tok in tokens(text) if tok not in SENTIMENT_VERBS}


def has_sentiment_verb(text: str) -> bool:
    """这句话里是不是真的出现了情绪/态度动词 —— "去动词核心"合并规则的生效前提。

    没有这条前提，规则会把任何"剥掉情绪动词后共享几个实词"的句子都判成重复 —— 但
    "Caroline passed interviews" 和 "Caroline chose an agency" 里根本没有情绪动词，
    "passed"/"chose" 是具体动作，剥不掉也不该剥。只有两句话都在表达情绪/态度时，
    "换了个态度词、事实没变"这个假设才成立。
    """
    return bool(SENTIMENT_VERBS & tokens(text))


def subject(text: str) -> str:
    """取**主语**（句首第一个专有名词 / 实词），用于挡住"A 欣赏 B" vs "B 欣赏 A"
    这类**主客体颠倒**的假合并 —— 它们的 token 集合几乎一样，但主语不同，是两条不同事实。"""
    match = re.search(r"[A-Z][a-z]+", text)
    return match.group(0).lower() if match else ""


def proper_nouns(text: str) -> set[str]:
    """句子里的**专有名词**集合（首字母大写的实词，小写化；去句首功能词）。

    用来挡住"模板相同、只有名字不同"的误合并 —— 实测把
    "Melanie has a pet named **pup**" 与 "Melanie has a pet named **Luna**" 判成同一事实
    （token Jaccard 0.67），于是 Luna / Oliver / Bailey 三条**并进了 pup 那条**，枚举成员
    名全丢。**不同的专有名词 = 不同的实体 = 不同的事实。**
    """
    return {
        word.lower()
        for word in re.findall(r"\b[A-Z][a-zA-Z]+\b", text)
        if word.lower() not in PROPER_STOP
    }


def jaccard(a: set[str], b: set[str]) -> float:
    """两个集合的 Jaccard 相似度；任一为空返回 0.0。"""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_same_fact(a: str, b: str) -> bool:
    """两条文本是否描述**同一事实**。

    主语必须一致 —— 单看 token 集合会把主客体颠倒的句子误判成同一事实。主语一致后再看
    两条路径任一命中：

    1. **整体 Jaccard ≥ ``NEAR_DUP_JACCARD``**：措辞几乎一致（大小写/冠词差异）。
    2. **两句话都含情绪动词，且去动词核心后 ≥ 3 个实词、共享率 ≥ 0.25**：反复犯的
       "同一事实换情绪动词"。**情绪动词前提是硬性的**（事故记录见模块顶部）。
    """
    return same_fact(signature(a), signature(b))


@dataclass(frozen=True)
class FactSignature:
    """一句话的**预计算特征**（分词 / 主语 / 专名 / 情绪动词 / 核心词各算一次）。

    为什么要它：``is_same_fact`` 每比一对就要把这些都算一遍（4 次正则），而批量去重时
    **同一句话会被拿出去比很多次**（它要跟库里的候选逐一比）。API 每次 Add 的合并就是这种
    形状：几条新的 × 几千条已有的。把"每句算一次"提出来之后，一对的比较只剩集合运算。
    """

    tokens: frozenset[str]
    subject: str
    proper: frozenset[str]
    has_sentiment: bool
    core: frozenset[str]


def signature(text: str) -> FactSignature:
    """算一句文本的特征（见 ``FactSignature``）。"""
    return FactSignature(
        tokens=frozenset(tokens(text)),
        subject=subject(text),
        proper=frozenset(proper_nouns(text)),
        has_sentiment=has_sentiment_verb(text),
        core=frozenset(core_tokens(text)),
    )


def same_fact(a: FactSignature, b: FactSignature) -> bool:
    """两个**特征**是否同一事实 —— 与 ``is_same_fact`` 判据一字不差，只是不重复分词。"""
    if not a.tokens or not b.tokens:
        return False
    if a.subject and b.subject and a.subject != b.subject:
        return False
    # **专有名词不同 = 不同事实。** 名字正是区分枚举成员的关键，措辞再像也不能并。
    if a.proper != b.proper:
        return False
    # 路径 1：措辞近乎一致。
    if jaccard(set(a.tokens), set(b.tokens)) >= NEAR_DUP_JACCARD:
        return True
    # 路径 2：仅当两句话都在表达情绪/态度时才适用（具体动作不适用）。
    if not (a.has_sentiment and b.has_sentiment):
        return False
    if not a.core or not b.core:
        return False
    shared = a.core & b.core
    return len(shared) >= SHARED_CORE_MIN and len(shared) / len(a.core | b.core) >= SHARED_CORE_RATIO
