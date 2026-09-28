"""混合检索：dense embedding + BM25 词法 + tag 通道，用 RRF 融合。

设计动机（实测）：纯 embedding 检索是当前瓶颈 —— 多证据 QA 中 81% 的答案本来就在
记忆池里、只是没被检索到；纯 dense 的天花板约 78%，LoCoMo category 3 几乎为 0。
`memory` 文本之外其实还有**大量没被用上**的结构化信息：`metadata.tag`（`speaker:X` /
`topic:Y` / `entity:Z`）等。本模块把这些信号各列为一路，用 RRF 融合。

**两种查询模式共用同一套打分**，差别只在 query 文本与 query 侧 tag：

    atom → atom      evomem 演化时：query = 某条原子的 memory 文本（+ 它自己的 tag）
    question → mem   QA 回答时：   query = 问题的文本（+ 从问题里抽出的关键词）

**为什么用 RRF 而不是加权分数**：dense 的余弦（-1..1）、BM25 的分数（无上界）、tag 的
匹配计数（小整数）三者量纲完全不同。直接加权求和要先做分数归一化，而归一化对分布很敏感
（换库、换 query 长度就得重调）。RRF 只看**名次**：

    score(d) = Σ_r  w_r / (k + rank_r(d))        k = 60（缺省）

名次天然可比，也不需要调参；以后要加 time 通道就是多一路 `r`，不影响其它路。

**为什么 tag 单列一路、而不是拼进被嵌入的文本**：`speaker:Caroline` 这种值对 embedding
几乎不携带信息（会被整句话的语义平均掉），但对词法/标签匹配很关键。单列一路才能给它
独立权重。

本模块是**纯 CPU** 的（除了调用方注入的 embed 函数）：不打模型、不读盘。`embed=None`
时只跑 BM25 + tag 两路，供离线自测（`scripts/test_searchmem.py`）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Sequence

# 英文停用词（LoCoMo 是英文对话语料）。只去高频虚词，不做过度的语言学处理 ——
# 目的是让 BM25 不被 "the/a/is" 这类词主导，而不是追求完美的分词。
STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does doing for from had has have
having he her here hers him his how i if in into is it its just me my no nor not of
off on or our out over own said she should so some such than that the their them then
there these they this those to too up us was we were what when where which while who
whom why will with would you your yours
""".split())

# 词元：连续字母数字，且至少含一个字母（避免把纯数字年份/编号全过滤掉 —— 年份是有用的
# 时间线索，所以保留纯数字 token，只是单独在处理时可选降权）。
_TOKEN_RE = re.compile(r"[a-z0-9]+")

DEFAULT_WEIGHTS: dict[str, float] = {"dense": 1.0, "bm25": 1.0, "tag": 0.5}
DEFAULT_RRF_K = 60

# BM25 的经典参数（Okapi BM25 / rank_bm25 的默认值）
BM25_K1 = 1.2
BM25_B = 0.75


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ScoredMemory:
    """一条被检索到的记忆 + 它为什么被选中的明细。

    ``score`` 是 RRF 融合分；``channels`` 记录它在**每一路**里的名次（未命中该路的
    不出现）。名次从 1 开始。调试"为什么这条被召回、那条没有"就看这个。
    """

    memory: dict[str, Any]
    score: float
    channels: dict[str, int] = field(default_factory=dict)

    @property
    def id(self) -> str:
        value = (self.memory.get("metadata") or {}).get("id")
        return value if isinstance(value, str) else ""

    @property
    def text(self) -> str:
        value = self.memory.get("memory")
        return value if isinstance(value, str) else ""

    def describe(self) -> str:
        ranks = " ".join(f"{name}={rank}" for name, rank in sorted(self.channels.items()))
        return f"[{self.id}] rrf={self.score:.4f} {ranks}".rstrip()


# ---------------------------------------------------------------------------
# 词法：分词 + BM25（纯 stdlib，无新依赖）
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """小写 + 抽字母数字词元 + 去停用词。"""
    if not text:
        return []
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


class BM25:
    """Okapi BM25。语料在构造时给定（= 候选池的 memory 文本，一篇文档一条候选）。

    公式（与 rank_bm25 一致）：

        idf(t)     = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
        score(d,t) = idf(t) · f(t,d)·(k1+1) / (f(t,d) + k1·(1 - b + b·|d|/avgdl))

    刻意自己实现而不是引 ``rank_bm25``：项目只有 openai/pyyaml/tqdm 三个依赖，
    为了这个几十行的公式不值得加新依赖。
    """

    def __init__(self, corpus: Sequence[str], k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = [tokenize(text) for text in corpus]
        self.doc_len: list[int] = [len(tokens) for tokens in self.doc_tokens]
        self.n = len(self.doc_tokens)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self._freqs: list[dict[str, int]] = []
        self._df: dict[str, int] = {}
        for tokens in self.doc_tokens:
            freq: dict[str, int] = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            self._freqs.append(freq)
            for token in freq:
                self._df[token] = self._df.get(token, 0) + 1

    def _idf(self, token: str) -> float:
        df = self._df.get(token, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        """返回与语料等长的 BM25 分数。未知词（df=0）的 idf 取 0，即不贡献分数。"""
        result = [0.0] * self.n
        if not self.n:
            return result
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return result
        avgdl = self.avgdl or 1.0
        for token in query_tokens:
            idf = self._idf(token)
            if idf == 0.0:
                continue
            for index, freq in enumerate(self._freqs):
                count = freq.get(token)
                if not count:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[index] / avgdl)
                result[index] += idf * count * (self.k1 + 1) / (count + norm)
        return result


# ---------------------------------------------------------------------------
# tag 通道
# ---------------------------------------------------------------------------

def tag_terms(memory: dict[str, Any]) -> set[str]:
    """取一条记忆的 tag 词元集合：``key:value`` 的 value 与 key 各自作为一个词元。

    例：``["speaker:Caroline", "topic:travel"]`` → ``{"caroline", "speaker", "travel", "topic"}``。
    刻意**不去停用词**（"topic"/"speaker" 这些 key 本身是有效信号），只做小写规范化。
    """
    terms: set[str] = set()
    for tag in (memory.get("metadata") or {}).get("tag") or []:
        if not isinstance(tag, str):
            continue
        text = tag.strip().lower()
        if not text:
            continue
        if ":" in text:
            key, value = text.split(":", 1)
            if key.strip():
                terms.add(key.strip())
            if value.strip():
                terms.add(value.strip())
        else:
            terms.add(text)
    return terms


def extract_query_tags(question: str) -> set[str]:
    """从问题文本里抽出可用来匹配 tag 的词元。

    刻意**不做 entity linking / NER**：问题里出现的名词（"Caroline"、"pottery"）本身
    就是 tag value 里会出现的字符串，直接取分词结果即可命中。停用词已在 tokenize 里去掉，
    所以不会把 "when"/"did" 这类虚词当 tag 词元。
    """
    return set(tokenize(question))


# ---------------------------------------------------------------------------
# 融合
# ---------------------------------------------------------------------------

class _Tail:
    """某个通道里"零信号"的候选：按库中顺序兜底，**不参与 RRF 计分**。

    存在的理由：RRF 只对"有信号的候选"排名，所以只跑 BM25 + tag 时，一个在词法和标签上
    都不命中的候选会**完全消失**。但调用方（evomem 的候选集）要的是**恰好 K 条**候选去
    演化 —— 一条没有词法信号的原子仍然可能语义相关（那正是 dense 通道存在的意义）。
    如果让结果少于 top_k，演化集合会静默缩水。

    所以把零信号候选按库中顺序接在有效候选之后：它们拿不到 RRF 分（不污染排序），
    但在有效候选不足 top_k 时被用来**补足**。有 dense 通道时这条几乎用不到（dense 很少
    给全 0），纯词法模式（自测 / 离线）下它保证 top_k 语义一致。
    """

    __slots__ = ("indices",)

    def __init__(self, indices: list[int]) -> None:
        self.indices = indices


def _split_head_tail(scores: Sequence[float]) -> tuple[list[int], list[int]]:
    """拆成 (有信号的降序下标, 零信号的库中顺序下标)。"""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    head = [i for i in order if scores[i] != 0]
    tail = [i for i in order if scores[i] == 0]
    return head, tail


def rrf_fuse(
    rankings: dict[str, Any],
    weights: dict[str, float],
    k: int = DEFAULT_RRF_K,
) -> tuple[dict[int, tuple[float, dict[str, int]]], list[int]]:
    """倒数排名融合。

    ``rankings`` 是 通道名 -> 有信号候选的下标列表（各自最好在前）；值也可以是
    ``(head, tail)`` 二元组（见 ``_split_head_tail``），此时 ``tail`` 会被合并进返回的
    兜底列表。

    返回 ``(fused, fallback)``：``fused`` 是 下标 -> (融合分, {通道: 名次})，名次从 1
    开始；``fallback`` 是没拿到任何分的候选（按库中顺序），供调用方在结果不足 top_k 时补足。
    权重为 0 的通道直接跳过（用作 dense-only 兼容开关）。
    """
    fused: dict[int, tuple[float, dict[str, int]]] = {}
    fallback: list[int] = []
    for name, value in rankings.items():
        weight = float(weights.get(name, 0.0))
        if isinstance(value, _Tail):
            fallback.extend(value.indices)
            continue
        if weight == 0.0:
            continue
        for position, index in enumerate(value, start=1):
            score, channels = fused.get(index, (0.0, {}))
            channels = dict(channels)
            channels[name] = position
            fused[index] = (score + weight / (k + position), channels)
    # 去重且不重复 fused 里已有的（理论上 head/tail 不交，防御性处理）
    seen: set[int] = set()
    ordered_fallback = [
        i for i in fallback if not (i in fused or i in seen or seen.add(i))
    ]
    return fused, ordered_fallback



# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def _memory_id(memory: dict[str, Any]) -> str:
    value = (memory.get("metadata") or {}).get("id")
    return value if isinstance(value, str) else ""


def dedupe_by_id(memories: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 id 去重（保留首次出现）。没有 id 的条目一律保留 —— 它们无法互相判定重复。

    为什么需要：演化过程中 ADD 可能在库里落下同样文本/同样 id 的条目；重复条目会让
    RRF 把同一个事实算两遍，也会让 prompt 里的证据块出现两次。
    """
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for memory in memories:
        mem_id = _memory_id(memory)
        if mem_id and mem_id in seen:
            continue
        if mem_id:
            seen.add(mem_id)
        result.append(memory)
    return result


async def search_scored(
    query: str,
    memories: list[dict[str, Any]],
    *,
    top_k: int,
    embed: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
    query_tags: Iterable[str] | None = None,
    weights: dict[str, float] | None = None,
    k: int = DEFAULT_RRF_K,
    extra_channels: dict[str, Sequence[float]] | None = None,
) -> list[ScoredMemory]:
    """混合检索，返回带通道明细的结果（按 RRF 分降序，最多 top_k 条）。

    ``embed`` 是**注入的异步嵌入函数**（``texts -> vectors``），不是 client：evomem 传
    它的 ``Embedder.embed``，eval 传 ``EmbeddingCache.embed`` —— 两者签名一致。为 None 时
    跳过 dense 通道（只跑 BM25 + tag），供纯 CPU 自测。

    ``query_tags`` 是 query 侧用于 tag 匹配的词元；缺省时从 query 文本自动抽。
    传空集合可以显式关掉 tag 通道（此时该路无候选，等价于不带 tag 权重）。

    ``extra_channels`` 是**调用方算好的通道分数**（与 ``memories`` 等长），用来避免重复
    请求 embedding：``searchctl`` 已经从预建索引里拿到了 dense 余弦分，直接作为 ``dense``
    一路传进来，而不是让这里再嵌一次。同名通道以 ``extra_channels`` 为准（覆盖内置通道）。
    """
    memories = dedupe_by_id(memories)
    if not memories or top_k <= 0:
        return []

    active_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        active_weights.update(weights)
    tag_query = set(tokenize(query)) if query_tags is None else set(query_tags)

    texts: list[str] = []
    for memory in memories:
        value = memory.get("memory")
        texts.append(value if isinstance(value, str) else "")
    rankings: dict[str, Any] = {}

    def add_channel(name: str, scores: Sequence[float]) -> None:
        """把一个通道的分数拆成 head（计分）/ tail（兜底），权重为 0 则不注册。"""
        if active_weights.get(name, 0.0) == 0.0:
            return
        head, tail = _split_head_tail(scores)
        if head:
            rankings[name] = head
        if tail:
            rankings[f"__tail_{name}"] = _Tail(tail)

    # ---- 通道 1：BM25 词法 ----
    add_channel("bm25", BM25(texts).scores(query))

    # ---- 通道 2：tag 匹配计数 ----
    if tag_query:
        add_channel("tag", [float(len(tag_query & tag_terms(m))) for m in memories])

    # ---- 通道 3：dense 余弦 ----
    # 一次请求把 query + 全部候选文本一起嵌入（与 eval_utils.retrieve_by_embedding_async
    # 的做法一致）。调用方的 embed 自带去重与缓存，所以这里不必自己分批。
    if active_weights.get("dense", 0.0) != 0.0 and embed is not None:
        vectors = await embed([query, *texts])
        if len(vectors) == len(texts) + 1:
            from .embedder import cosine_similarity

            add_channel(
                "dense",
                [cosine_similarity(vectors[0], vector) for vector in vectors[1:]],
            )

    # ---- 调用方注入的通道（覆盖内置同名通道，避免重复嵌入）----
    for name, scores in (extra_channels or {}).items():
        if len(scores) != len(memories):
            raise ValueError(
                f"extra_channels[{name!r}] 长度 {len(scores)} 与候选数 {len(memories)} 不一致"
            )
        rankings.pop(name, None)
        rankings.pop(f"__tail_{name}", None)
        add_channel(name, scores)

    # ---- 融合 ----
    fused, fallback = rrf_fuse(rankings, active_weights, k=k)

    ordered = sorted(
        fused.items(),
        key=lambda pair: (-pair[1][0], pair[0]),  # 分数降序；同分按库中位置稳定排序
    )

    results = [
        ScoredMemory(memory=memories[index], score=score, channels=channels)
        for index, (score, channels) in ordered[:top_k]
    ]
    # 有效候选不足 top_k 时用零信号候选补足（见 _Tail 的说明）——保证调用方拿到的候选数
    # 与 top_k 一致，而不是被"没有词法信号"静默截短。这些条目 score=0、channels 为空，
    # 明确表示"没被任何一路命中，只是凑数"。
    if len(results) < top_k:
        for index in fallback[: top_k - len(results)]:
            results.append(ScoredMemory(memory=memories[index], score=0.0, channels={}))
    return results



async def search(
    query: str,
    memories: list[dict[str, Any]],
    *,
    top_k: int,
    embed: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
    query_tags: Iterable[str] | None = None,
    weights: dict[str, float] | None = None,
    k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    """混合检索的**主要对外接口**：返回 top_k 条 memory（不带通道明细）。

    返回 memory 列表（而非 ScoredMemory）是为了直接兼容既有调用方
    （``build_recalled_block`` 等期望的就是 memory 列表）。要看通道明细用
    ``search_scored``。
    """
    scored = await search_scored(
        query, memories, top_k=top_k, embed=embed, query_tags=query_tags,
        weights=weights, k=k,
    )
    return [item.memory for item in scored]


# ---------------------------------------------------------------------------
# 权重预设
# ---------------------------------------------------------------------------

def dense_only_weights() -> dict[str, float]:
    """dense-only 兼容开关：用于和 join 之前（纯 embedding）的检索结果做对照。"""
    return {"dense": 1.0, "bm25": 0.0, "tag": 0.0}


def lexical_weights() -> dict[str, float]:
    """纯 CPU 可复现的权重（不依赖 embed），供自测与离线分析。"""
    return {"dense": 0.0, "bm25": 1.0, "tag": 0.5}
