"""检索索引的**构建**与**加载**。

**为什么必须先建索引**：旧版 ``searchmem.search`` 每轮都把 ``[query, *全库文本]`` 一起嵌入
——实测一个目录跑出 21342 条文本 / 698 批请求。本方案的 agent 会反复调 ``search``，如果每次
都重嵌全库，成本立刻不可接受。所以改成：**每个运行目录预建一次**，之后每次检索只嵌 1 条
query（+ 少量 evidence 条目）。

两种模式共用 ``searchmem`` 的三路打分（dense + BM25 + tag → RRF），本模块只负责 dense 那一路
的向量来源。

格式（``inputs/index/``）：

    ids.json        ["session_1_1_1", ...]           行序与 vectors.f32 一致
    vectors.f32     float32[N, D] 行主序（stdlib array）
    meta.json       {model, dim, n, corpus_sha256, built_at, source}

``corpus_sha256`` 是构建时语料的指纹。加载时如果拿到的语料与之不符，只告警 —— 因为
``searchctl`` 的检索池是"base ∪ 实时 evidence"，本来就与构建语料不同，这不是错误。
"""

from __future__ import annotations

import array
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

PRIMARY = "vectors.f32"

# 纯 Python 的点积（无 numpy 依赖）。1273 × 4096 实测约 0.75s —— 每次 search 一次，可接受。
# 需要更快时给 index 加一个 .npy 分支即可，不必改调用方。


def _dot(left: array.array, right: list[float]) -> float:
    total = 0.0
    for a, b in zip(left, right):
        total += a * b
    return total


def _norm(values: Any) -> float:
    total = 0.0
    for value in values:
        total += value * value
    return total ** 0.5


def corpus_sha256(memories: list[dict[str, Any]]) -> str:
    """语料指纹：按 id + memory 文本算，与文件顺序无关。"""
    digest = hashlib.sha256()
    for memory in memories:
        meta = memory.get("metadata") or {}
        digest.update(str(meta.get("id", "")).encode())
        digest.update(b"\x00")
        digest.update(str(memory.get("memory", "")).encode())
        digest.update(b"\x01")
    return digest.hexdigest()[:16]


@dataclass
class MemoryIndex:
    """已加载的向量索引。``ids[i]`` 对应 ``matrix[i*D:(i+1)*D]``。"""

    ids: list[str]
    matrix: array.array
    dim: int
    meta: dict[str, Any]

    def __len__(self) -> int:
        return len(self.ids)

    def vector(self, row: int) -> array.array:
        start = row * self.dim
        return self.matrix[start:start + self.dim]

    def dense_scores(
        self,
        query_vector: list[float],
        id_to_row: dict[str, int],
        ids: list[str],
    ) -> list[float]:
        """对 ``ids`` 里**有向量**的那些算余弦相似度，缺向量的给 0.0。

        返回与 ``ids`` 等长的分数列表（顺序一致），直接喂给 ``searchmem`` 的 dense 通道。
        """
        query_norm = _norm(query_vector) or 1.0
        scores: list[float] = []
        for memory_id in ids:
            row = id_to_row.get(memory_id)
            if row is None:
                scores.append(0.0)
                continue
            vector = self.vector(row)
            scores.append(_dot(vector, query_vector) / (query_norm * (_norm(vector) or 1.0)))
        return scores

    def build_row_index(self) -> dict[str, int]:
        return {memory_id: row for row, memory_id in enumerate(self.ids)}


async def build_index(
    memories: list[dict[str, Any]],
    *,
    embed: Callable[[list[str]], Awaitable[list[list[float]]]],
    directory: Path,
    model: str = "",
    source: str = "",
    on_batch: Callable[[int, int], None] | None = None,
) -> MemoryIndex:
    """把整库嵌入一次并落盘。

    ``embed`` 是注入的异步嵌入函数（``Embedder.embed`` 或适配器），内部自带分批与重试。
    ``on_batch`` 用于把进度打到日志里 —— 建索引是启动阶段最慢的一步，必须可见。
    """
    directory.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    texts: list[str] = []
    for memory in memories:
        meta = memory.get("metadata") or {}
        memory_id = meta.get("id")
        if not isinstance(memory_id, str) or not memory_id:
            continue
        ids.append(memory_id)
        text = memory.get("memory")
        texts.append(text if isinstance(text, str) else "")

    if not ids:
        raise ValueError("语料为空，无法建索引")
    if on_batch is not None:
        on_batch(0, len(ids))
    vectors = await embed(texts)
    if len(vectors) != len(ids):
        raise RuntimeError(f"嵌入返回 {len(vectors)} 条，期望 {len(ids)} 条")
    if on_batch is not None:
        on_batch(len(ids), len(ids))

    dim = len(vectors[0])
    matrix = array.array("f")
    for vector in vectors:
        if len(vector) != dim:
            raise RuntimeError(f"向量维度不一致：{len(vector)} vs {dim}")
        matrix.extend(vector)

    (directory / "ids.json").write_text(
        json.dumps(ids, ensure_ascii=False), encoding="utf-8"
    )
    with (directory / PRIMARY).open("wb") as handle:
        matrix.tofile(handle)
    meta = {
        "model": model,
        "dim": dim,
        "n": len(ids),
        "corpus_sha256": corpus_sha256(memories),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "dtype": "float32",
    }
    (directory / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return MemoryIndex(ids=ids, matrix=matrix, dim=dim, meta=meta)


# ---------------------------------------------------------------------------
# 共享索引缓存
# ---------------------------------------------------------------------------

def cache_key(memories: list[dict[str, Any]], model: str) -> str:
    """共享索引的目录名 = 语料指纹 + 嵌入模型指纹。

    带上 ``model`` 是因为换了嵌入模型，向量维度与语义空间都变了，旧索引必须作废 ——
    只按语料哈希命名会让两个不同模型的索引互相覆盖。
    """
    model_tag = hashlib.sha256(str(model).encode()).hexdigest()[:8]
    return f"{corpus_sha256(memories)}-{model_tag}"


def shared_index_dir(cache_root: Path, memories: list[dict[str, Any]], model: str) -> Path:
    """共享索引应放的位置：``{cache_root}/{corpus}-{model}/``。"""
    return cache_root / cache_key(memories, model)


def index_is_usable(directory: Path, memories: list[dict[str, Any]], model: str) -> bool:
    """已有索引是否**正好**对应这份语料与模型。

    只检查文件存在是不够的：``atommem.jsonl`` 一旦重新抽取（``add`` 重跑），语料就变了，
    旧向量会**静默**给出错误的排序。所以必须比对 ``corpus_sha256``。
    """
    meta_path = directory / "meta.json"
    if not (directory / PRIMARY).exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        meta.get("corpus_sha256") == corpus_sha256(memories)
        and meta.get("model") == model
        and int(meta.get("n", -1)) == len(memories)
    )


def load_index(directory: Path) -> MemoryIndex:
    """加载索引。任何不一致都抛错 —— 拿坏索引去检索会静默给出无意义的排序。"""
    ids_path = directory / "ids.json"
    vectors_path = directory / PRIMARY
    meta_path = directory / "meta.json"
    for path in (ids_path, vectors_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(
                f"索引文件缺失：{path}（先用 `python -m codemem.searchctl --index-only "
                f"--atoms <atommem.jsonl> --index {directory}` 建一次）"
            )
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dim = int(meta.get("dim") or 0)
    if dim <= 0:
        raise ValueError(f"{meta_path} 里的 dim 非法：{meta.get('dim')!r}")
    matrix = array.array("f")
    with vectors_path.open("rb") as handle:
        matrix.fromfile(handle, len(ids) * dim)
    if len(ids) != int(meta.get("n", -1)):
        raise ValueError(f"索引不一致：ids.json 有 {len(ids)} 条，meta.json 记 {meta.get('n')}")
    return MemoryIndex(ids=list(ids), matrix=matrix, dim=dim, meta=meta)


def find_legacy_indexes(runs_root: Path) -> list[Path]:
    """找出**内嵌在运行目录里**的旧索引（存储优化之前建的）。

    早期实现把索引建在 ``{run}/{dir}/inputs/index/``，而它只取决于语料与模型 —— 于是
    每跑一次就多一份完全相同的 17MB。现在索引放 ``.index_cache``（见 ``runner.build_inputs``），
    运行目录里只有符号链接。

    这个函数供 ``--clean-index-cache`` 使用：列出**真实目录**形态的旧索引（跳过符号链接），
    它们与共享缓存重复，可以安全删除。
    """
    found: list[Path] = []
    if not runs_root.exists():
        return found
    for path in runs_root.glob("*/*/inputs/index"):
        if path.is_symlink() or not path.is_dir():
            continue
        if (path / PRIMARY).exists():
            found.append(path)
    return found


def remove_paths(paths: list[Path]) -> int:
    """删除给定路径，返回释放的字节数。目录用 rmtree，文件直接 unlink。"""
    import shutil

    freed = 0
    for path in paths:
        try:
            if path.is_dir() and not path.is_symlink():
                freed += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                freed += path.stat().st_size
                path.unlink()
        except OSError:
            continue
    return freed


# ---------------------------------------------------------------------------
# 运行目录清理
# ---------------------------------------------------------------------------

#: 一次运行里"值得长期保留"的东西（其余都是可再生的中间产物）
KEEP_SUFFIXES = (
    "summary.json",        # 配置、选中的 QA、指标摘要
    "answers.jsonl",       # answer 步骤的产物
    "eval.jsonl",          # eval 步骤的产物
)


def run_size(path: Path) -> int:
    """目录的**实际占用**（按块算，与 ``du`` 一致），用于报告释放了多少。"""
    total = 0
    for entry in path.rglob("*"):
        try:
            stat = entry.lstat()
        except OSError:
            continue
        if entry.is_dir() and not entry.is_symlink():
            total += stat.st_size
        elif entry.is_file():
            total += stat.st_size
    return total


def plan_prune(
    runs_root: Path,
    keep_last: int = 0,
    include_full: bool = False,
) -> list[tuple[Path, int]]:
    """列出可删除的运行目录及各自大小。

    ``keep_last > 0`` 时保留最近的 N 次运行（按目录名里的时间戳排序）；
    否则只保留"最后一次全量运行"（目录名含 **full**）与 ``index_cache``。

    **只删运行目录，永不删 ``.index_cache``**：索引是跨运行共享的，删了下次要重嵌
    （一个目录 17MB、几十秒），而它不属于任何单次运行。
    """
    if not runs_root.exists():
        return []
    candidates = sorted(
        (p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=lambda p: p.name,
    )
    if keep_last > 0:
        keep = set(candidates[-keep_last:])
    else:
        # 没有任何全量运行时，保留最近一次，免得把唯一结果删了
        full_runs = [p for p in candidates if "full" in p.name]
        keep = {full_runs[-1]} if full_runs else ({candidates[-1]} if candidates else set())

    plan: list[tuple[Path, int]] = []
    for path in candidates:
        if path in keep:
            continue
        size = run_size(path)
        plan.append((path, size))
    return plan


def prune_runs(runs_root: Path, plan: list[tuple[Path, int]]) -> tuple[int, int]:
    """执行清理，返回 ``(删除的目录数, 释放的字节数)``。"""
    import shutil as _shutil

    removed = freed = 0
    for path, size in plan:
        try:
            _shutil.rmtree(path)
        except OSError:
            continue
        removed += 1
        freed += size
    return removed, freed
