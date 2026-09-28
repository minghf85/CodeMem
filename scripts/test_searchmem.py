"""searchmem 混合检索的快速自测（纯 CPU，不联网、不打模型）。

用法：
    python scripts/test_searchmem.py

覆盖三路各自的排序行为 + RRF 融合的单调性 + 边界情况（空池 / 空 query / 重复 id）。
dense 通道用**确定性假 embedder**（按字符重合度构造向量）测，所以不需要真的 embeddings 端点。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codemem.search import searchmem as S  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    if not condition:
        FAILURES.append(f"{name}: {detail}")
    print(f"  {mark} {name}" + (f"  [{detail}]" if detail and not condition else ""))


def mem(memory: str, mem_id: str, tags: list[str] | None = None) -> dict:
    return {
        "memory": memory,
        "metadata": {
            "id": mem_id, "type": "outer", "time": "",
            "tag": tags or ["speaker:A"], "source": [], "changelog": [],
        },
    }


def ids(items: list[dict]) -> list[str]:
    return [i["metadata"]["id"] for i in items]


# ---------------------------------------------------------------------------
# 假 embedder：确定性、纯 CPU。按 query 与文档的字符 3-gram 重合度构造向量，
# 足够让"文本更像 query 的候选"拿到更高余弦。
# ---------------------------------------------------------------------------

def _grams(text: str, n: int = 3) -> dict[str, int]:
    text = text.lower()
    grams: dict[str, int] = {}
    for i in range(max(0, len(text) - n + 1)):
        gram = text[i: i + n]
        grams[gram] = grams.get(gram, 0) + 1
    return grams


def _vector(text: str, dims: list[str]) -> list[float]:
    grams = _grams(text)
    return [float(grams.get(d, 0)) for d in dims]


def make_fake_embedder(corpus: list[str]):
    """返回一个 (texts) -> vectors 的异步函数。词表 = corpus 的全部 3-gram。"""
    vocab: set[str] = set()
    for text in corpus:
        vocab.update(_grams(text))
    dims = sorted(vocab)

    async def embed(texts: list[str]) -> list[list[float]]:
        return [_vector(t, dims) for t in texts]

    return embed


# ---------------------------------------------------------------------------
# 通道测试
# ---------------------------------------------------------------------------

def test_tokenize() -> None:
    print("tokenize")
    check("小写 + 去停用词", S.tokenize("When did Caroline go to the support group?")
          == ["caroline", "go", "support", "group"], str(S.tokenize("When did Caroline go to the support group?")))
    check("保留年份数字", "2023" in S.tokenize("painted a sunrise in 2023"))
    check("空串", S.tokenize("") == [])


def test_bm25() -> None:
    print("bm25")
    corpus = [
        "Caroline went to the LGBTQ support group",
        "Melanie painted a sunrise",
        "Caroline researches adoption agencies",
    ]
    bm25 = S.BM25(corpus)
    scores = bm25.scores("LGBTQ support group")
    check("命中文档排第一", scores.index(max(scores)) == 0, str(scores))
    check("无关文档低分", scores[1] < scores[0], str(scores))
    check("OOV query 全 0 不崩", bm25.scores("zzzzqqqq") == [0.0, 0.0, 0.0])
    check("空语料不崩", S.BM25([]).scores("x") == [])


def test_tag_terms() -> None:
    print("tag channel")
    m = mem("x", "a", ["speaker:Caroline", "topic:travel"])
    check("key 与 value 都进词元", S.tag_terms(m) == {"speaker", "caroline", "topic", "travel"},
          str(S.tag_terms(m)))
    check("无 tag", S.tag_terms({"metadata": {}}) == set())
    check("容忍非字符串 tag", S.tag_terms({"metadata": {"tag": [123, "topic:x"]}}) == {"topic", "x"})


def test_rrf_fuse() -> None:
    print("rrf fuse")
    rankings = {"dense": [0, 1, 2], "bm25": [2, 0, 1]}
    fused, fallback = S.rrf_fuse(rankings, {"dense": 1.0, "bm25": 1.0}, k=60)
    check("三路都进入结果", len(fused) == 3, str(len(fused)))
    check("无兜底项", fallback == [], str(fallback))
    check("名次从 1 开始", fused[0][1] == {"dense": 1, "bm25": 2}, str(fused[0][1]))
    # 下标 0：1/61 + 1/62；下标 2：1/63 + 1/61 —— 0 应略高
    check("两路都靠前的不下于只靠一路的", fused[0][0] > fused[2][0], f"{fused[0][0]} vs {fused[2][0]}")
    fused_zero, _ = S.rrf_fuse(rankings, {"dense": 1.0, "bm25": 0.0}, k=60)
    check("权重 0 的通道被跳过", fused_zero[2][1] == {"dense": 3})
    check("全 0 权重 → 空", S.rrf_fuse(rankings, {"dense": 0.0, "bm25": 0.0})[0] == {})
    # k 变大 → 不同名次之间的分差被压平（单调性）
    narrow, _ = S.rrf_fuse({"d": [0, 1]}, {"d": 1.0}, k=1)
    wide, _ = S.rrf_fuse({"d": [0, 1]}, {"d": 1.0}, k=1000)
    check("k 变大压平分差", (narrow[0][0] - narrow[1][0]) > (wide[0][0] - wide[1][0]))
    # 零信号候选进兜底列表、不参与计分
    fused_tail, tail = S.rrf_fuse({"bm25": [0], "__tail_bm25": S._Tail([1, 2])}, {"bm25": 1.0}, k=60)
    check("零信号进兜底", tail == [1, 2], str(tail))
    check("兜底不计分", set(fused_tail.keys()) == {0}, str(list(fused_tail)))


def test_search_lexical_only() -> None:
    print("search: lexical only (embed=None)")
    memories = [
        mem("Caroline went to the LGBTQ support group", "s1", ["speaker:Caroline", "topic:support"]),
        mem("Melanie painted a sunrise", "s2", ["speaker:Melanie", "topic:art"]),
        mem("Caroline researches adoption agencies", "s3", ["speaker:Caroline", "topic:research"]),
    ]
    got = asyncio.run(S.search("LGBTQ support group", memories, top_k=2, embed=None))
    check("返回 2 条（零信号候选补足）", len(got) == 2, str(len(got)))
    check("BM25 命中排第一", ids(got)[0] == "s1", str(ids(got)))
    check("top_k 可覆盖全库", len(asyncio.run(
        S.search("LGBTQ support group", memories, top_k=99, embed=None))) == 3)
    # 零信号候选带 score=0 明细，可辨认
    scored = asyncio.run(S.search_scored("LGBTQ support group", memories, top_k=3, embed=None))
    check("零信号候选 score=0 且无通道", scored[-1].score == 0.0 and scored[-1].channels == {},
          scored[-1].describe())


def test_search_tag_boost() -> None:
    print("search: tag channel moves rank")
    # 两条文本对 BM25 而言都不含 query 词，只有 tag 能区分
    memories = [
        mem("something entirely unrelated here", "no_tag", ["speaker:A", "topic:other"]),
        mem("another totally different sentence", "has_tag", ["speaker:A", "topic:pottery"]),
    ]
    got = asyncio.run(S.search("pottery", memories, top_k=1, embed=None))
    check("tag 命中的候选胜出", ids(got) == ["has_tag"], str(ids(got)))


def test_search_dense_channel() -> None:
    print("search: dense channel with fake embedder")
    memories = [
        mem("Melanie painted a sunrise", "art", ["speaker:Melanie"]),
        mem("Caroline went to the support group", "support", ["speaker:Caroline"]),
    ]
    embed = make_fake_embedder([m["memory"] for m in memories])
    got = asyncio.run(S.search(
        "Melanie painted a sunrise", memories, top_k=1, embed=embed,
        weights={"dense": 1.0, "bm25": 0.0, "tag": 0.0},
    ))
    check("dense-only 命中正确", ids(got) == ["art"], str(ids(got)))


def test_search_scored_channels() -> None:
    print("search: scored detail carries channel ranks")
    memories = [
        mem("Caroline went to the LGBTQ support group", "s1", ["speaker:Caroline"]),
        mem("Melanie painted a sunrise", "s2", ["speaker:Melanie"]),
    ]
    embed = make_fake_embedder([m["memory"] for m in memories])
    scored = asyncio.run(S.search_scored("LGBTQ support group", memories, top_k=2, embed=embed))
    check("每条都有明细", len(scored) == 2)
    check("第一名带 bm25 名次", "bm25" in scored[0].channels, str(scored[0].channels))
    check("describe 有 id 与分数", scored[0].id in scored[0].describe(), scored[0].describe())


def test_boundaries() -> None:
    print("search: boundaries")
    check("空候选池", asyncio.run(S.search("q", [], top_k=5, embed=None)) == [])
    memories = [mem("aaa", "x"), mem("bbb", "y")]
    check("top_k=0", asyncio.run(S.search("aaa", memories, top_k=0, embed=None)) == [])
    check("空 query 不崩", isinstance(asyncio.run(S.search("", memories, top_k=2, embed=None)), list))
    # 重复 id 去重
    dupes = [mem("aaa", "dup"), mem("aaa", "dup"), mem("bbb", "other")]
    got = asyncio.run(S.search("aaa", dupes, top_k=5, embed=None))
    check("重复 id 已去重", len(ids(got)) == len(set(ids(got))), str(ids(got)))
    check("无 id 条目保留", len(S.dedupe_by_id([mem("a", ""), mem("b", "")])) == 2)


def test_weight_presets() -> None:
    print("weight presets")
    check("dense-only 关掉另两路", S.dense_only_weights() == {"dense": 1.0, "bm25": 0.0, "tag": 0.0})
    check("lexical 关掉 dense", S.lexical_weights()["dense"] == 0.0)


def main() -> None:
    for test in (
        test_tokenize, test_bm25, test_tag_terms, test_rrf_fuse,
        test_search_lexical_only, test_search_tag_boost, test_search_dense_channel,
        test_search_scored_channels, test_boundaries, test_weight_presets,
    ):
        test()
    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}:")
        for failure in FAILURES:
            print(f"  - {failure}")
        sys.exit(1)
    print("all ok")


if __name__ == "__main__":
    main()
