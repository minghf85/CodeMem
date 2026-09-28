"""``search`` —— 把混合检索做成一条终端命令。

    search "when did Caroline go to the LGBTQ support group" -k 8
    search "support group" -k 5 --source evidence
    search "support group" -k 8 --lexical          # 嵌入服务挂了时的词法回退

输出的每一行是一条 JSON（按 RRF 分降序），直接喂给 ``jq``：

    {"id":"session_5_2_1","score":0.0328,"rank_dense":1,"rank_bm25":4,"rank_tag":null,
     "memory":"Caroline joined an LGBTQ support group","type":"outer"}

**为什么做成 CLI 而不是一个"检索工具"**：让它能和 ``jq`` / ``grep`` / ``head`` 自由组合，
而这正是这个 agent 唯一需要学的思维模型 —— 一切都是一条对 JSONL 文件的命令。代价有两条，
都可接受：① 每次调用要 load 一次预建索引（float32，1273×4096 ≈ 20MB，约 50ms）；
② 检索失败以子进程失败的形式暴露 —— 这反而**比旧版更好**：旧版 embedding 挂了就直接终止
整个目录，现在 agent 看得见 ``search: embedding service unreachable``，可以改用
``--lexical`` 绕过去。

**检索池 = base 库 ∪ 当前 evidence.jsonl**（按 id 去重，evidence 侧优先）。所以 agent 自己
写出的新记忆立刻可被检索到，不需要重建索引 —— evidence 通常只有几条，实时嵌入即可。

本模块也可用来**建索引**（``--index-only``）：

    python -m codemem.searchctl --index-only \\
        --atoms data/Caroline_Melanie/atommem.jsonl --index /tmp/idx

环境变量（由 harness 通过 bash 的 ``shell_command_prefix`` 注入）：

    CODEMEM_INDEX      索引目录（``inputs/index``）
    CODEMEM_EMBED_URL  嵌入服务地址（缺省则只用词法两路）
    CODEMEM_EMBED_KEY  嵌入服务 api key
    CODEMEM_EMBED_MODEL 嵌入模型名
    CODEMEM_SEARCH_WEIGHTS  JSON，形如 {"dense":1.0,"bm25":1.0,"tag":0.5}
    CODEMEM_RRF_K      RRF 平滑常数（缺省 60）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# 从 io 导入**唯一**的 PROJECT_ROOT，不要自己 parents[N] 推 ——
# 本文件在子包里深度不同，重算会落到 src/ 而不是项目根。
from ..io import PROJECT_ROOT

# 默认输出里带上 kind / source：**溯源是 agent 最常需要的下一步**（这条原子来自哪条原话），
# 直接给出来能省掉一轮 jq。
DEFAULT_FIELDS = ("id", "kind", "score", "rank_dense", "rank_bm25", "rank_tag", "memory", "source")
ALL_FIELDS = ("id", "kind", "score", "rank_dense", "rank_bm25", "rank_tag", "memory",
              "type", "time", "tag", "source", "derived_from")


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or "").strip()


def _weights_from_env() -> dict[str, float] | None:
    raw = _env("CODEMEM_SEARCH_WEIGHTS")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"search: CODEMEM_SEARCH_WEIGHTS 不是合法 JSON: {raw!r}", file=sys.stderr)
        return None
    if isinstance(parsed, dict):
        return {str(k): float(v) for k, v in parsed.items()}
    return None


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


def _memory_id(memory: dict[str, Any]) -> str:
    value = (memory.get("metadata") or {}).get("id")
    return value if isinstance(value, str) else ""


def merge_pool(
    base: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    source: str,
    raw: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """检索池。优先级：evidence > base（atommem）> raw（msgmem）。

    为什么要能搜到 ``msgmem``：实测失败的典型形态是"答案在原话里，但**没有任何原子**
    正面说出来"。例如 Caroline 被问到"想读什么方向"，原文是
    ``I'm keen on counseling or working in mental health``（D1:11），而库里最强的原子只有
    一句泛泛的 "continue her education and check out career options"。此时
    agent 必须能**按原话关键词**找到那条消息 —— 否则它只能靠猜 id，实测就猜到了别的 session
    然后空转到预算耗尽。

    同 id 时保留高优先级那一份（evidence 侧是 agent 改过的，raw 侧是原话，不重叠）。
    """
    include_evidence = source in ("evidence", "all")
    include_base = source in ("base", "all")
    include_raw = source in ("raw", "all") and raw is not None

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def add(memories: list[dict[str, Any]], allow_override: bool) -> None:
        for memory in memories:
            memory_id = _memory_id(memory) or f"__noid_{len(order)}"
            if memory_id in merged and not allow_override:
                continue
            if memory_id not in merged:
                order.append(memory_id)
            merged[memory_id] = memory

    if include_evidence:
        add(evidence, allow_override=True)
    if include_base:
        add(base, allow_override=False)          # evidence 侧优先
    if include_raw:
        add(raw, allow_override=False)           # 已被前面收进来的不动
    return [merged[key] for key in order]


def filter_by_tag(
    pool: list[dict[str, Any]],
    patterns: list[str],
    *,
    exclude: bool = False,
) -> list[dict[str, Any]]:
    """按 tag 过滤（子串匹配，大小写不敏感）。``--tag action: --exclude-tag topic:`` 之类。

    **为什么需要它**：有一类问题是"某人做过哪些 X"——答案是一个**集合**，散在几十条
    记录里。稠密检索对这类问题结构性无能为力：它按与 query 的相似度排序，而"游泳"
    这条记录与"Melanie 参加什么活动"的相似度并不高（实测 dense 排名 101），于是永远
    进不了 top-k —— 把 k 调到 100 也救不回来，因为相关度天梯本身就不指向它。

    但库里**已经有结构化标签**（``action:Swimming``、``topic:activities``）。按标签枚举
    是这类问题的正确工具：先把候选收全，再由 agent 判断哪些算"活动"。所以给 agent 一个
    按 tag 过滤的开关，而不是指望它用关键词把集合"检索"出来。
    """
    if not patterns:
        return pool
    wanted = [pattern.strip().lower() for pattern in patterns if pattern.strip()]
    if not wanted:
        return pool

    def matches(memory: dict[str, Any]) -> bool:
        tags = [str(tag).lower() for tag in (memory.get("metadata") or {}).get("tag") or []]
        return any(any(pattern in tag for tag in tags) for pattern in wanted)

    return [memory for memory in pool if matches(memory) != exclude]


def tag_value_counts(
    pool: list[dict[str, Any]],
    prefixes: list[str] | None = None,
) -> list[tuple[str, int]]:
    """统计 pool 里的 tag 值及出现次数，按次数降序。

    **为什么需要**：集合型问题（"某人做过哪些 X"）的正解是**按标签值枚举**，而不是把
    100 条检索结果逐条读一遍 —— 实测 agent 拿到 `-k 100` 后只读了前十几条就动笔，
    因为读完全部对 8B 来说太重了（20KB 输出）。

    先看"池子里到底有哪些 activity/topic 值、各有多少条"，再按值逐个取，枚举才收得齐。
    ``prefixes`` 为空时统计全部 tag；给定 ``["activity", "topic:activ"]`` 时只统计匹配的。
    """
    counts: dict[str, int] = {}
    wanted = [p.strip().lower() for p in (prefixes or []) if p.strip()]
    for memory in pool:
        for tag in (memory.get("metadata") or {}).get("tag") or []:
            text = str(tag).strip()
            if not text:
                continue
            lowered = text.lower()
            if wanted and not any(p in lowered for p in wanted):
                continue
            counts[text] = counts.get(text, 0) + 1
    return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))


def provenance_fields(memory: dict[str, Any]) -> dict[str, Any]:
    """给检索结果补上**溯源/发散**所需的字段。

    agent 拿到一条命中的原子后，最常问的下一句是"这条是从哪条原话来的？"——
    ``source`` 就是答案（原子 → msgmem 消息）。反过来，``--source raw`` 让 agent 从原话
    出发去找"哪些原子是从这句话抽出来的"（msgmem → 原子），即发散。
    把这两个方向需要的字段直接放进结果里，省掉一轮 jq 查询。
    """
    meta = memory.get("metadata") or {}
    kind = meta.get("type")
    if kind == "raw":
        return {
            "kind": "raw",           # msgmem：原话，可以直接引为证据
            "source": [],            # raw 没有出处
            "derived_from": [],      # 谁从这条抽出来的由 --derive 列
            "time": meta.get("time", ""),
        }
    return {
        "kind": "atom",              # atommem：抽取出的原子，source 指向原话
        "source": meta.get("source", []),
        "derived_from": [],
        "time": meta.get("time", ""),
    }


async def _embed_query_evidence(
    texts: list[str],
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> list[list[float]]:
    """实时嵌入 query（+ evidence 里没有向量的条目）。失败抛异常，由调用方处理。"""
    import openai

    from .embedder import Embedder

    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key or "none")
    embedder = Embedder(client, model, chunk_size=int(_env("CODEMEM_EMBED_CHUNK", "32")))
    return await embedder.embed(texts)


def _render(scored: Any, fields: tuple[str, ...], derived: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """把一条 ScoredMemory 渲染成输出行。``memory`` 默认截断到 400 字符，避免刷屏。

    ``kind``/``source`` 是**溯源**用的：``kind=atom`` 表示这是抽出来的原子，``source``
    指向它的原话消息；``kind=raw`` 表示这本身就是一条原话，可以直接引为证据。
    ``derived_from`` 是**反向发散**（这条原话抽出了哪些原子），由 ``--derive`` 填。
    """
    memory = scored.memory
    meta = memory.get("metadata") or {}
    channels = scored.channels
    provenance = provenance_fields(memory)
    row: dict[str, Any] = {
        "id": scored.id,
        "kind": provenance["kind"],
        "score": round(float(scored.score), 6),
        "rank_dense": channels.get("dense"),
        "rank_bm25": channels.get("bm25"),
        "rank_tag": channels.get("tag"),
        "memory": scored.text,
        "type": meta.get("type", ""),
        "time": provenance["time"],
        "tag": meta.get("tag", []),
        "source": provenance["source"],
        "derived_from": (derived or {}).get(scored.id, []),
    }
    if "memory" in fields:
        limit = int(_env("CODEMEM_MEMORY_CHARS", "400"))
        if limit > 0 and len(row["memory"]) > limit:
            row["memory"] = row["memory"][:limit] + "…"
    return {key: row[key] for key in fields if key in row}


async def run_search(args: argparse.Namespace) -> int:
    from . import index as index_mod
    from . import searchmem

    workspace = Path.cwd()
    index_dir = Path(args.index) if args.index else Path(_env("CODEMEM_INDEX", "inputs/index"))
    if not index_dir.is_absolute():
        index_dir = workspace / index_dir

    atoms_path = Path(args.atoms) if args.atoms else workspace / "inputs/atommem.jsonl"
    evidence_path = Path(args.evidence) if args.evidence else workspace / "evidence.jsonl"
    # msgmem 也要能搜：答案常常只在原话里（没有任何原子正面说出来），见 merge_pool 的说明
    raw_path = Path(args.raw) if args.raw else workspace / "inputs/msgmem.jsonl"

    # 索引只有 dense 通道需要。纯词法模式（--lexical，或没配嵌入服务）**不该**因为缺索引
    # 就退出 —— 那正是降级路径存在的意义（索引没建成时 harness 会切到词法模式）。
    weights = searchmem.DEFAULT_WEIGHTS if args.lexical else _weights_from_env()
    needs_dense = not args.lexical and float((weights or {}).get("dense", 1.0) or 0.0) != 0.0
    embed_url = _env("CODEMEM_EMBED_URL")
    loaded: Any = None
    if needs_dense and embed_url:
        try:
            loaded = index_mod.load_index(index_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"search: 无法加载索引 {index_dir}: {exc}", file=sys.stderr)
            return 2

    base = read_jsonl(atoms_path)
    evidence = read_jsonl(evidence_path)
    raw = read_jsonl(raw_path)
    pool = merge_pool(base, evidence, args.source, raw=raw)
    if not pool:
        print("search: 检索池为空（atommem / evidence / msgmem 都没有记录）",
              file=sys.stderr)
        return 3

    # --tag-values：只列标签值分布就退出（枚举型问题的第一步：先看清词表）
    if getattr(args, "tag_values", False):
        for tag, count in tag_value_counts(pool, args.tag):
            print(f"{count:>6}  {tag}")
        return 0

    # --tag / --exclude-tag：先按标签收窄候选，再检索。枚举型问题靠这一步收齐集合。
    if args.tag or args.exclude_tag:
        before = len(pool)
        pool = filter_by_tag(pool, args.tag, exclude=False)
        if args.exclude_tag:
            pool = filter_by_tag(pool, args.exclude_tag, exclude=True)
        if not pool:
            print(
                f"search: --tag 过滤后没有候选（原 {before} 条；"
                f"--tag {args.tag} --exclude-tag {args.exclude_tag}）。"
                f"先看看有哪些 tag：jq -r '.metadata.tag[]' inputs/atommem.jsonl | sort | uniq -c",
                file=sys.stderr,
            )
            return 3

    ids = [_memory_id(m) for m in pool]

    # ---- dense：索引里已有的直接用；没有的（新写进 evidence 的）实时嵌入 ----
    dense_scores: list[float] | None = None
    if loaded is not None:
        row_of = loaded.build_row_index()
        missing_positions = [i for i, memory_id in enumerate(ids) if memory_id not in row_of]
        try:
            vectors = await _embed_query_evidence(
                [args.query, *[(pool[i].get("memory") or "") for i in missing_positions]],
                base_url=embed_url,
                api_key=_env("CODEMEM_EMBED_KEY"),
                model=_env("CODEMEM_EMBED_MODEL"),
            )
        except Exception as exc:  # noqa: BLE001 - 明确告诉 agent 哪里坏了，它会改用 --lexical
            print(
                f"search: embedding service unreachable ({type(exc).__name__}: {exc}). "
                f"Retry, or use `--lexical` for keyword-only search.",
                file=sys.stderr,
            )
            return 4
        query_vector = vectors[0]
        dense_scores = loaded.dense_scores(query_vector, row_of, ids)
        # 索引里没有的条目（新写进 evidence 的）用刚实时嵌入的向量覆盖 0 分。
        query_norm = sum(v * v for v in query_vector) ** 0.5 or 1.0
        for position, vector in zip(missing_positions, vectors[1:]):
            vector_norm = sum(v * v for v in vector) ** 0.5 or 1.0
            dense_scores[position] = (
                sum(a * b for a, b in zip(query_vector, vector)) / (query_norm * vector_norm)
            )
    elif needs_dense:
        print(
            "search: CODEMEM_EMBED_URL 未设置，退化为词法检索（BM25 + tag）",
            file=sys.stderr,
        )

    # ---- 用 searchmem 的三路 RRF 融合。search_scored 会把 cosine 当"分数"重排名次，
    # 这里传 dense=None 并把已知的 dense 分数作为一路显式注入，避免重复请求 embedding。
    scored = await searchmem.search_scored(
        args.query,
        pool,
        top_k=args.k,
        embed=None,
        weights=weights,
        k=args.rrf_k,
        extra_channels={"dense": dense_scores} if dense_scores is not None else None,
    )
    # --derive：给命中的**原话**补上"这句话抽出了哪些原子"（msgmem → atommem 发散）。
    # 只在需要时算（要扫一遍全库 atommem，没必要每次检索都付这个成本）。
    derived: dict[str, list[str]] = {}
    if args.derive:
        for atom in base:
            atom_meta = atom.get("metadata") or {}
            atom_id = _memory_id(atom)
            if not atom_id:
                continue
            for src in atom_meta.get("source") or []:
                if isinstance(src, str) and src:
                    derived.setdefault(src, []).append(atom_id)

    fields = tuple(args.fields.split(",")) if args.fields else DEFAULT_FIELDS
    for item in scored:
        print(json.dumps(_render(item, fields, derived), ensure_ascii=False, default=str))
    if not scored:
        print("search: 没有命中（试试别的措辞，或用 --source all 同时搜原话与原子）",
              file=sys.stderr)
        return 1
    return 0


async def run_index(args: argparse.Namespace) -> int:
    import openai

    from . import index as index_mod
    from .embedder import Embedder

    index_dir = Path(args.index)
    atoms_path = Path(args.atoms)
    memories = read_jsonl(atoms_path)
    if not memories:
        print(f"searchctl: {atoms_path} 里没有记录", file=sys.stderr)
        return 2

    base_url = args.embed_url or _env("CODEMEM_EMBED_URL")
    model = args.embed_model or _env("CODEMEM_EMBED_MODEL")
    if not base_url or not model:
        print("searchctl: 建索引需要 --embed-url / --embed-model（或对应环境变量）",
              file=sys.stderr)
        return 2

    client = openai.AsyncOpenAI(base_url=base_url, api_key=args.embed_key or _env("CODEMEM_EMBED_KEY") or "none")
    embedder = Embedder(client, model, chunk_size=int(_env("CODEMEM_EMBED_CHUNK", "32")))

    def on_batch(done: int, total: int) -> None:
        if done == 0:
            print(f"searchctl: 嵌入 {total} 条 …", file=sys.stderr)

    built = await index_mod.build_index(
        memories,
        embed=embedder.embed,
        directory=index_dir,
        model=model,
        source=str(atoms_path),
        on_batch=on_batch,
    )
    print(
        f"searchctl: 索引已写入 {index_dir}（{len(built)} 条 × {built.dim} 维，"
        f"{embedder.stats()}）",
        file=sys.stderr,
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="search",
        description="混合检索（dense + BM25 + tag，RRF 融合），每行输出一条 JSON",
    )
    parser.add_argument("query", nargs="?", default="", help="检索 query（--tag-values 时可省略）")
    parser.add_argument("-k", type=int, default=8, help="返回条数（缺省 8）")
    parser.add_argument("--source", choices=("base", "evidence", "raw", "all"), default="all",
                        help="检索池：base=atommem（原子）/ raw=msgmem（原话）/ evidence=已写出的证据"
                             "/ all=三者并集（缺省）")
    parser.add_argument("--raw", default="", help="msgmem 路径（缺省 inputs/msgmem.jsonl）")
    parser.add_argument("--derive", action="store_true",
                        help="给结果补 derived_from：这条原话抽出了哪些原子（msgmem → atommem 发散）")
    parser.add_argument("--tag", action="append", default=[],
                        help="按 tag 子串过滤（可重复），如 `--tag action: --tag topic:activ`。"
                             "用于**枚举型**问题（'某人做过哪些 X'）——这类问题的答案是一个"
                             "集合，稠密检索捞不全，按标签枚举才收得齐")
    parser.add_argument("--exclude-tag", action="append", default=[],
                        help="排除含这些 tag 子串的记录（可重复），如 `--exclude-tag speaker:Caroline`")
    parser.add_argument("--tag-values", action="store_true",
                        help="不检索，只列出池子里的 tag 值及出现次数（按 --tag 前缀过滤）。"
                             "枚举型问题先用它看清有哪些值，再逐个取值")
    parser.add_argument("--fields", default="", help="逗号分隔的输出字段（缺省常用集）")
    parser.add_argument("--atoms", default="", help="base 库路径（缺省 inputs/atommem.jsonl）")
    parser.add_argument("--evidence", default="", help="evidence 路径（缺省 evidence.jsonl）")
    parser.add_argument("--index", default="", help="索引目录（缺省 $CODEMEM_INDEX）")
    parser.add_argument("--lexical", action="store_true", help="只用 BM25 + tag，不调 embedding")
    parser.add_argument("--rrf-k", type=int, default=int(_env("CODEMEM_RRF_K", "60") or 60))
    parser.add_argument("--all-fields", action="store_true", help="输出全部字段（默认集之上再加 time/tag/source）")

    index_group = parser.add_argument_group("建索引（离线跑一次）")
    index_group.add_argument("--index-only", action="store_true", help="只建索引、不检索")
    index_group.add_argument("--embed-url", default="")
    index_group.add_argument("--embed-key", default="")
    index_group.add_argument("--embed-model", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.all_fields and not args.fields:
        args.fields = ",".join(ALL_FIELDS)
    if args.index_only:
        if not args.index or not args.atoms:
            print("searchctl: --index-only 需要 --index 与 --atoms", file=sys.stderr)
            return 2
        return asyncio.run(run_index(args))
    # --tag-values 不需要 query（它只统计标签词表）
    if not args.query.strip() and not args.tag_values:
        print('search: 缺少 query。用法：search "QUERY" [-k N] [--source base|raw|evidence|all]',
              file=sys.stderr)
        return 2
    return asyncio.run(run_search(args))


if __name__ == "__main__":
    if str(PROJECT_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.exit(main())
