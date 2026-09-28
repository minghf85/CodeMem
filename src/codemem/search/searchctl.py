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
from ..io import (
    PROJECT_ROOT,
    is_summary,
    msg_content,
    msg_id,
    msg_role,
    msg_time,
)

# 默认输出里带上 kind / source：**溯源是 agent 最常需要的下一步**（这条原子来自哪条原话），
# 直接给出来能省掉一轮 jq。
DEFAULT_FIELDS = ("id", "kind", "score", "rank_dense", "rank_bm25", "rank_tag", "memory", "source")
ALL_FIELDS = ("id", "kind", "score", "rank_dense", "rank_bm25", "rank_tag", "memory",
              "type", "time", "tag", "source")


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
    """记录 id。统一走 ``io.msg_id``（兼容新的 msg_id 与旧的 metadata.id）。"""
    return msg_id(memory)


def _library_layers(source: str) -> tuple[str, ...]:
    """``--source`` 决定读哪几层语料。

    - ``all``（缺省）：三层全读 —— 概览与细节一起搜，按相关度统一排序。
    - ``summary``：只读两层 summary。适合"先定位到哪次会话"的粗检索。
    - ``sessions`` / ``raw``：只读原始消息。适合关键词检索具体原话
      （summary 会改写措辞，精确的词不一定在里面）。
    """
    if source in ("summary", "summaries"):
        return (LIBRARY_FILES[0], LIBRARY_FILES[1])
    if source in ("sessions", "raw"):
        return (LIBRARY_FILES[2],)
    return LIBRARY_FILES


def filter_by_tag(
    pool: list[dict[str, Any]],
    patterns: list[str],
    *,
    exclude: bool = False,
) -> list[dict[str, Any]]:
    """按 tag 过滤（子串匹配，大小写不敏感）。

    **注意**：新的 session 记录格式**没有 tag 字段**（只有 msg_id/role/time/content），
    所以这个过滤器在新语料上默认不命中任何东西。保留它是为了兼容旧产物，以及将来
    summary 里若加上结构化标签时可以直接用。
    """
    if not patterns:
        return pool
    wanted = [pattern.strip().lower() for pattern in patterns if pattern.strip()]
    if not wanted:
        return pool

    def matches(memory: dict[str, Any]) -> bool:
        tags = [str(tag).lower() for tag in (memory.get("metadata") or {}).get("tag") or []]
        # role 也算一路可匹配的标签（说话人名），与 searchmem.tag_terms 的处理一致
        role = msg_role(memory)
        if role:
            tags.append(role.strip().lower())
        return any(any(pattern in tag for tag in tags) for pattern in wanted)

    return [memory for memory in pool if matches(memory) != exclude]


def tag_value_counts(
    pool: list[dict[str, Any]],
    prefixes: list[str] | None = None,
) -> list[tuple[str, int]]:
    """统计 pool 里的 tag 值及出现次数，按次数降序。

    集合型问题（"某人做过哪些 X"）的正解是**按标签值枚举**，而不是把 100 条检索结果
    逐条读一遍 —— 后者对 8B 太重（实测 20KB 输出，它只读了前十几条就动笔）。
    """
    counts: dict[str, int] = {}
    wanted = [p.strip().lower() for p in (prefixes or []) if p.strip()]
    for memory in pool:
        values: list[str] = []
        role = msg_role(memory)
        if role:
            values.append(f"role:{role}")
        values.extend(str(tag).strip() for tag in (memory.get("metadata") or {}).get("tag") or [])
        for text in values:
            if not text:
                continue
            lowered = text.lower()
            if wanted and not any(p in lowered for p in wanted):
                continue
            counts[text] = counts.get(text, 0) + 1
    return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))


def merge_pool(
    library: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    """检索池。优先级：evidence > library（三层语料并集）。

    **library 是三层语料合并后的结果**（speakers summary / session summaries / 原始消息），
    由 ``load_library`` 读取并按粒度从粗到细排列。为什么要一次全搜而不是分层分别搜：
    agent 不该被迫记住"这个问题该查哪一层" —— 让检索把三层的命中混在一起排好序，
    粗粒度的 summary 因为语义密度高天然排在前面（它是概览），细节需要时再往下看。

    evidence 侧优先：那是 agent 自己写出来的、已经过推理的结论，比原始语料更贴题。
    """
    include_evidence = source in ("evidence", "all")
    include_library = source in ("library", "base", "summary", "raw", "all")

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
    if include_library:
        add(library, allow_override=False)      # evidence 侧优先
    return [merged[key] for key in order]


#: 三层语料文件名，**按粒度从粗到细**（概览 → 细节）。顺序有意义：
#: 索引与检索都按这个顺序，粗粒度先出现，便于 agent 先看到概览。
LIBRARY_FILES = (
    "speakers_summary.jsonl",
    "session_summaries.jsonl",
    "sessions.jsonl",
)


def load_library(workspace: Path, names: tuple[str, ...] = LIBRARY_FILES) -> list[dict[str, Any]]:
    """读三层语料并合并成检索库（不存在的文件跳过）。

    合并进**一个**库而不是让 agent 选层：``--source`` 的粒度选择留给"只看原话"
    （``--source raw``）这类明确需要，默认行为是三层一起搜、按相关度统一排序。
    """
    merged: list[dict[str, Any]] = []
    for name in names:
        merged.extend(read_jsonl(workspace / "inputs" / name))
    return merged


def layer_of(memory: dict[str, Any]) -> str:
    """判断一条记录属于哪一层（给检索结果标注，agent 据此决定下一步去哪）。"""
    if is_summary(memory):
        return "speakers_summary" if msg_id(memory) == "speakers_summary" else "session_summary"
    return "session"


def provenance_fields(memory: dict[str, Any]) -> dict[str, Any]:
    """给检索结果补上**层级与溯源**字段。

    agent 拿到命中后最常问的下一句是"这是概览还是原话？我要不要往下钻？"——
    ``kind`` 与 ``source`` 直接回答它：

    - ``kind=speakers_summary`` —— 整段关系的概览。适合定位"哪次会话"，细节要往下钻。
    - ``kind=session_summary``  —— 某次会话的概览。``source`` 是该会话的消息 id 列表，
      可以直接读原文。
    - ``kind=session``          —— 原始消息。``source`` 为空（它就是源头）。
    """
    meta = memory.get("metadata") or {}
    kind = layer_of(memory)
    # summary 的 source 从正文里推不出来，靠 msg_id 的前缀给"这次会话包含哪些消息"
    source = meta.get("source")
    if not isinstance(source, list):
        source = []
    if kind == "session":
        source = []
    return {
        "kind": kind,
        "source": source,
        "time": str(memory.get("time") or meta.get("time") or ""),
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


def _render(scored: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    """把一条 ScoredMemory 渲染成输出行。``memory`` 默认截断到 400 字符，避免刷屏。

    ``kind`` / ``source`` 是**层级与溯源**：``kind`` 说明这是哪一层
    （speakers_summary / session_summary / session），``source`` 给出往下钻的入口。
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

    evidence_path = Path(args.evidence) if args.evidence else workspace / "evidence.jsonl"
    # library = 三层语料的并集（speakers summary / session summaries / 原始消息），
    # 按粒度从粗到细排列。默认三层的命中混在一起统一排序（见 merge_pool 的说明）。

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

    if args.library:
        library = read_jsonl(Path(args.library))
    else:
        library = load_library(workspace, _library_layers(args.source))
    evidence = read_jsonl(evidence_path)
    pool = merge_pool(library, evidence, args.source)
    if not pool:
        print(
            "search: 检索池为空（三层语料与 evidence 都没有记录）。"
            "先确认 inputs/ 下有 sessions.jsonl / session_summaries.jsonl / speakers_summary.jsonl",
            file=sys.stderr,
        )
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
    fields = tuple(args.fields.split(",")) if args.fields else DEFAULT_FIELDS
    for item in scored:
        print(json.dumps(_render(item, fields), ensure_ascii=False, default=str))
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
    atoms_path = Path(args.library or args.atoms)
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
    parser.add_argument("--source", choices=("all", "library", "summary", "sessions", "raw", "evidence"),
                        default="all",
                        help="检索池：all=三层语料+evidence（缺省）/ library=只三层语料 / "
                             "summary=只两层 summary（粗检索，用于定位哪次会话）/ "
                             "sessions(或 raw)=只原始消息（按原话关键词搜）/ evidence=只自己写出的证据")
    parser.add_argument("--library", default="",
                        help="直接指定语料文件路径（缺省按 --source 从 inputs/ 读三层）")
    parser.add_argument("--tag", action="append", default=[],
                        help="按 tag 子串过滤（可重复）。记录里没有 tag 字段时该过滤不命中任何东西")
    parser.add_argument("--exclude-tag", action="append", default=[],
                        help="排除含这些 tag 子串的记录（可重复）")
    parser.add_argument("--tag-values", action="store_true",
                        help="不检索，只列出池子里的 tag 值及出现次数（按 --tag 前缀过滤）")
    parser.add_argument("--fields", default="", help="逗号分隔的输出字段（缺省常用集）")
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
