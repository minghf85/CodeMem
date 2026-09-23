"""EvoMem：用 prompt 直接做原子的逐步演化（无强化学习）。

设计见 ``docs/evomem_plan.md``。这个模块是**第一阶段**实现：先不训练，纯粹靠 prompt
驱动一个强模型反复演化记忆库，看效果如何。

流程（**QA 驱动**）：

    for 每对 speaker 目录（**目录之间并行**，目录之内串行）:
        base = atommem.jsonl（按 atom_key 排好）        # 每条 QA 都从这份 base 出发
        anchor = 该目录最后一条 raw 消息的 id            # 新记忆 id 挂在这个 anchor 下
        for 每条 QA（跳过 category 5）:
            候选集 = searchmem.search(question, base)    # 混合检索 dense+BM25+tag
            for current in 候选集内每条原子:              # 中层
                召回 = 严格早于 current 的存活原子 → 附 source msgmem + 上下文
                while inner < max_evomem_turn:           # 内层 agent loop
                    构建 prompt（question + current + 证据 + HISTORY）→ 调模型 → 一个动作
                    执行动作（纯 CPU）→ 更新 memory / active / trace
                    动作结果写进 HISTORY，作为下一轮的反馈
                    NOOP → 结束**这一条原子**，候选集里下一条继续
            落一条 QA trajectory（question / 候选 / 演化后集合 / 逐轮轨迹）

**三层循环（重要）**：外层是 **QA 问题**，中层是候选集里的原子，内层是 agent loop。
``max_evomem_turn`` 是**每条原子**的内层轮数上限，不是全目录总轮数。

**为什么 QA 驱动**（旧版是"按 id 顺序走遍全库逐条精修"）：成本从
``原子数 × 轮数``（Tim_John 1771 条原子 ≈ 7000 次强模型调用）降到
``QA 数 × 候选数 × 轮数``，且调用全部花在**问题相关**的原子上；演化目标从"把这条原子
改得更自足"变成"让这组候选足以回答这个问题"（问题经 ``{{QUESTION}}`` 传给模型）；
并且一条 QA 一条 trajectory，便于后续做 RL。

**每条 QA 从同一份 base 库出发**：QA 之间互不影响 —— 这是"一条 QA 一条轨迹"的前提，
也让 base ``atommem.jsonl`` 始终不被修改。代价是不同 QA 可能演化出重复的新原子
（都挂在同一个 anchor 下），第一版接受这一点，重复度只在目录级并集快照里体现。

**每条原子只召回一次**：内层各轮共享同一份证据，而不是每轮重新检索。这样上一轮改了什么
只能通过 HISTORY 传给模型，避免把模型自己刚写出来的措辞当成"独立证据"又召回来，绕成
自我指涉。顺带省掉了每轮重嵌入整个候选池的开销。

**并发模型（重要）**：一行只处理一对 speaker，一对 speaker 内部的 turn 必须串行 ——
第 N 轮的动作依赖前 N-1 轮改完的记忆库，并行会读到同一份旧快照、互相覆盖。
所以 ``concurrency`` 是**目录级**并发：同时跑多对 speaker，各自串行走自己的 turn。

**不用 embedding 缓存**：记忆库每一轮都会被改写，同一段文本下一轮可能已经不存在或者
已经被 UPDATE 成别的措辞，缓存既命中率低又容易读到过期向量。所以每轮直接实时调
embedding（``qwen3-embedding:4b``）。``scripts/eval_atommem.py`` 的静态缓存不适用于这里。

关键设计点：

- **只看得到 current 之前的记忆**：处理到第 i 条时只允许看到它之前的内容（严格
  ``key < key(current)``，见 ``recall_pool``）。拿未来信息改过去的记忆是最容易出的脏数据。
- **动作只作用于当前召回子区**：模型只看到 top_k 条，所以 ADD/UPDATE/DELETE 也只会
  命中这些 id（或者接到 anchor 上）。库里其他记忆原样保留。
- **context budget 是输出的硬约束**：ADD+UPDATE+DELETE 的输出合并进记忆库后不能超过
  ``library_max_tokens``，否则从尾部（最旧的）开始按软删除摘掉。
- 全程不用 judge，也不打分 —— 这一阶段只看演化出的记忆库长什么样。

**原子性**：演化只写到 ``data/evomem_runs/`` 下的输出文件，**绝不修改** ``atommem.jsonl``。
所以跑测试不需要提前备份、事后恢复这一步（原文件本来就不动）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import openai
import yaml
from tqdm import tqdm

from . import evolog
from .evolog import Logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_FILE = PROJECT_ROOT / "configs" / "evomem.yaml"
# 演化任务自己的字段模板。**必须显式传给 load_template**：它的缺省值是抽取模板
# （``memory_template_init_extract.json``），措辞是"抽取一条新事实"，与这里
# "精修一条已有记忆"的任务不符，而且抽取模板没有 source 字段 —— 而 ADD/UPDATE 都要求
# 模型给 source。这里是用错模板导致 schema 与任务错配的地方，别再退回缺省值。
EVOMEM_TEMPLATE_FILE = PROJECT_ROOT / "memory_template_evomem.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "embedding": {"base_url": "http://127.0.0.1:11434/v1", "api_key": "ollama", "model": "qwen3-embedding:4b"},
    "generator": {"base_url": "http://127.0.0.1:30000/v1", "api_key": "sglang", "model": "local"},
    "top_k": 30,
    "max_evomem_turn": 2,
    # QA 驱动演化的闸门：
    #   max_qa        每个目录最多处理前 N 条 QA（0 = 全部）。全库 1986 条 QA 真跑很贵，
    #                 先用这个限制条数试跑。见 --max-qa。
    #   qa_candidates 每条 QA 用混合检索取多少个候选原子来演化（K）。见 --candidates。
    #   skip_category5 category 5 是 adversarial（故意不可回答），与评测口径一致地跳过。
    "max_qa": 0,
    "qa_candidates": 10,
    "skip_category5": True,
    # 混合检索（searchmem）的配置：rrf_k 是 RRF 的平滑常数，weights 是三路通道的权重。
    # 想和旧的纯 embedding 检索对照，把 weights 设成 {dense: 1.0, bm25: 0.0, tag: 0.0}。
    "search": {"rrf_k": 60, "weights": {"dense": 1.0, "bm25": 1.0, "tag": 0.5}},
    "context_window": 2,
    "concurrency": 2,
    "embedding_concurrency": 8,
    "library_max_tokens": 40000,
    "timeout": 300,
    "max_retries": 3,
    "backoff_base": 2.0,
    "rate_limit_backoff": 10.0,
    "max_backoff": 60.0,
    "backoff_jitter": 0.2,
    "enable_thinking": False,
    "log": {"level": "info", "output": ""},
}

LIBRARY_ANCHOR_TURNS = 3  # 给 query 拼上最近几轮的摘要


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------

def load_config() -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # 深拷贝
    if CONFIG_FILE.exists():
        loaded = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                    config[key].update(value)
                else:
                    config[key] = value
    return config


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] {path}:{line_number}: {exc}", file=sys.stderr)
    return items


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def resolve_dir(target: str) -> Path | None:
    """把参数解析成 speaker 目录，兼容裸目录名、相对路径与绝对路径。"""
    path = Path(target)
    if path.is_dir():
        return path
    candidate = DATA_DIR / target
    if candidate.is_dir():
        return candidate
    return None


def memory_id(memory: dict[str, Any]) -> str:
    value = (memory.get("metadata") or {}).get("id")
    return value if isinstance(value, str) else ""


def memory_text(memory: dict[str, Any]) -> str:
    text = memory.get("memory")
    return text if isinstance(text, str) else ""


# ---------------------------------------------------------------------------
# 顺序：按 id 先后处理原子
# ---------------------------------------------------------------------------
# atommem 的 id 是 ``session_{session}_{message}_{n}``，n 是该原子在源消息内的序号。
#
# **注意：文件里的 id 顺序不可靠**。重跑会把新原子以新计数器追加进同一源消息，于是会出现
# ``session_2_1_10`` 紧挨着 ``session_2_1_2``；按 id 字符串排序有 43 处逆序，按数值元组排序
# 仍有 27 处。项目自己的 ``atommem.sort_memory_records`` 也修不掉（它的 tiebreaker 是字符串
# 比较）。所以这里**不依赖任何全局排序**，而是给每条原子算一个可比较的 key：
#
#     key(atom) = (该原子源消息在 msgmem 里的位置, n)
#
# 实测（10 个目录）：所有 id 都符合上面的形状；同一 (session,message) 下的 n 恰好是
# 1..max 无空洞无重复；因此该 key **零 tie**，是一个严格全序。
#
# 为什么从 id 解析而不是用 ``metadata.source``：ADD 允许把 source 写成 atommem id，
# 而 id 形状里已经带了 session/message，直接解析更稳、也少一次查表。

ATOM_ID_RE = re.compile(r"^session_(\d+)_(\d+)_(\d+)$")


def atom_key(atom_id: str, msgmem_pos: dict[str, int]) -> tuple[int, int] | None:
    """算原子的比较 key。``None`` 表示这不是一个规范的原子 id（调用方需兜底）。

    返回 ``(源消息在 msgmem 中的位置, 消息内序号)``。源消息找不到时位置取一个很大的值
    （排到队尾），而不是失败 —— 数据里偶有 source 指向已不存在的消息。
    """
    match = ATOM_ID_RE.match(atom_id or "")
    if match is None:
        return None
    session, message, counter = match.group(1), match.group(2), match.group(3)
    position = msgmem_pos.get(f"session_{session}_{message}", len(msgmem_pos))
    return position, int(counter)


def order_atoms(
    atoms: list[dict[str, Any]], msgmem_pos: dict[str, int]
) -> list[dict[str, Any]]:
    """按 ``atom_key`` 排序，得到"按 id 先后处理"的顺序。

    不受文件顺序影响。非规范 id 的原子（理论上不该有）排到队尾，按 id 字符串兜底。
    """
    def sort_key(atom: dict[str, Any]) -> tuple[int, int, str]:
        key = atom_key(memory_id(atom), msgmem_pos)
        if key is None:
            return (len(msgmem_pos) + 1, 0, memory_id(atom))
        return (key[0], key[1], "")

    return sorted(atoms, key=sort_key)


def recall_pool(
    current: dict[str, Any],
    memories: list[dict[str, Any]],
    active: set[str],
    touched: set[str],
    msgmem_pos: dict[str, int],
) -> list[dict[str, Any]]:
    """must 严格在 ``current`` **之前**的存活原子，作为召回的候选池。

    三个过滤条件缺一不可（写在一处，避免散落各处漏掉）：
    - ``active``：软删除的原子不参与召回；
    - ``touched``：本次运行已经动过的原子不再反复改；
    - ``key < key(current)``：只允许看到"此 id 之前"的内容 —— 拿未来信息改过去的记忆
      是最容易出的脏数据。
    """
    current_key = atom_key(memory_id(current), msgmem_pos)
    if current_key is None:
        return []
    pool: list[dict[str, Any]] = []
    for memory in memories:
        memory_id_value = memory_id(memory)
        if memory_id_value not in active or memory_id_value in touched:
            continue
        key = atom_key(memory_id_value, msgmem_pos)
        if key is not None and key < current_key:
            pool.append(memory)
    return pool


def source_msgmem_id(atom: dict[str, Any], msgmem_pos: dict[str, int]) -> str | None:
    """取原子对应的源消息 id（ADD 新原子的 id 要挂在它下面）。

    优先从原子 id 解析（``session_1_1_2`` → ``session_1_1``）；解析不出就退回
    ``metadata.source`` 里第一个像 msgmem 的元素。
    """
    match = ATOM_ID_RE.match(memory_id(atom))
    if match is not None:
        candidate = f"session_{match.group(1)}_{match.group(2)}"
        if not msgmem_pos or candidate in msgmem_pos:
            return candidate
    for source in (atom.get("metadata") or {}).get("source") or []:
        if isinstance(source, str) and source in msgmem_pos:
            return source
    return None


# ---------------------------------------------------------------------------
# 召回
# ---------------------------------------------------------------------------
# 这里刻意**不做 embedding 缓存**：记忆库每一轮都会被改写，同一段文本下一轮可能已经
# 被 UPDATE 成别的措辞、或者已被 DELETE。用 (model, text) 做 key 的缓存既命中率低，
# 又容易在"改完之后再召回"时读到过期向量，反而引入难以排查的错误。每轮实时嵌入。
# （scripts/eval_atommem.py 的 embedding_cache.jsonl 是静态场景的优化，不适用于 evomem。）


class Embedder:
    """把文本实时嵌入成向量。没有缓存 —— 见上面注释。

    两个必要的稳健性措施（都是实测踩出来的）：

    1. **分批**：一次把整个记忆库（上千条）塞进单个请求时，ollama 的 OpenAI 兼容端点
       会偶发 400 —— 报错内容是关于内部 tokenizer 服务（另一个端口）连不上，属于误导性
       信息，实际是并发压力下的瞬时故障。批越小越稳（批 64 实测 3/3 成功，批 512 只有
       1/3）。而且分批之后单批失败只影响那一批，不会废掉整轮召回。
    2. **单批重试**：上一条说明失败是瞬时的，所以每批失败后短暂退避重试，而不是直接
       把整轮演化打断。
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        chunk_size: int = 64,
        max_retries: int = 4,
        backoff: float = 1.0,
        log: "Logger | None" = None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.chunk_size = max(1, int(chunk_size))
        self.max_retries = max(1, int(max_retries))
        self.backoff = float(backoff)
        self.log = log
        self.requests = 0
        self.texts = 0
        self.retries = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """返回与 texts 等长的向量列表。同一批里先去重，减少请求条数。"""
        if not texts:
            return []
        unique = list(dict.fromkeys(texts))
        if self.log is not None:
            self.log.debug(
                f"embedding 输入 {len(texts)} 条（去重 {len(unique)}），"
                f"分 {(len(unique) + self.chunk_size - 1) // self.chunk_size} 批"
            )
        vectors: dict[str, list[float]] = {}
        for start in range(0, len(unique), self.chunk_size):
            vectors.update(await self._embed_chunk(unique[start: start + self.chunk_size]))
        return [vectors[text] for text in texts]

    async def _embed_chunk(self, chunk: list[str]) -> dict[str, list[float]]:
        """嵌入一批，失败时退避重试。全部重试失败才抛异常。"""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self.requests += 1
            self.texts += len(chunk)
            try:
                response = await self.client.embeddings.create(
                    model=self.model_name, input=chunk
                )
                if len(response.data) != len(chunk):
                    raise RuntimeError(
                        f"embedding 返回数量不匹配：请求 {len(chunk)} 返回 {len(response.data)}"
                    )
                return {
                    text: item.embedding
                    for text, item in zip(chunk, response.data)
                }
            except Exception as exc:  # noqa: BLE001 - 实测为瞬时故障（ollama 内部 tokenizer）
                last_error = exc
                if attempt < self.max_retries:
                    self.retries += 1
                    if self.log is not None:
                        self.log.warn(
                            f"embedding 第 {attempt} 次失败（{len(chunk)} 条），"
                            f"{self.backoff * attempt:.1f}s 后重试：{type(exc).__name__}"
                        )
                    await asyncio.sleep(self.backoff * attempt)
        raise RuntimeError(
            f"embedding 批次失败（{len(chunk)} 条，重试 {self.max_retries} 次）: {last_error}"
        )

    def stats(self) -> str:
        return (
            f"embedding 请求 {self.requests} 批 / {self.texts} 条文本"
            f"（无缓存，批大小 {self.chunk_size}，重试 {self.retries} 次）"
        )


class EmbeddingStore:
    """按 id 维护向量，随记忆库变化**动态调整**。

    为什么不是"一次性嵌入、接受过期"：evomem 会 UPDATE 已存在的原子，旧向量对应的文本
    已经不存在了；也会 ADD 新原子、DELETE 旧原子。所以这里做成一个**随库增删改同步更新**
    的向量表：

    - 初始化：把全部原子文本嵌入一次；
    - ADD   → 嵌入新文本，加入表；
    - UPDATE→ 文本变了，**重嵌**并替换该 id 的向量（旧向量立刻作废）；
    - DELETE→ 从表里摘掉该 id。

    检索时只在**当前有效的索引范围**（``key < current`` 的前缀）上打分，而不是全表扫 ——
    范围随游标推进而增长，位置由 ``order_atoms`` 的顺序给出，所以只要记录"当前游标在有序
    列表里的下标"，前缀就是 ``[:cursor]``，天然是连续的。

    正确性依赖一条不变量：**UPDATE 不改 id**（否则 key 会变、前缀边界失效）。见
    ``evoactions.apply_update`` 里的 id 校验。
    """

    def __init__(self, embedder: "Embedder | None", log: Logger | None = None) -> None:
        self.embedder = embedder
        self.log = log
        self.vectors: dict[str, list[float]] = {}
        self.embedded = 0
        self.reembedded = 0

    async def seed(self, atoms: list[dict[str, Any]]) -> None:
        """初始化：把全部原子文本嵌入一次（一次批量，不是每条一次）。"""
        if self.embedder is None:
            raise RuntimeError("没有可用的 embedder，无法初始化向量表")
        ids = [memory_id(a) for a in atoms]
        texts = [memory_text(a) for a in atoms]
        vectors = await self.embedder.embed(texts)
        self.vectors = dict(zip(ids, vectors))
        self.embedded = len(ids)
        if self.log is not None:
            self.log.debug(f"向量表初始化 {len(ids)} 条（一次批量）")

    async def add(self, atoms: list[dict[str, Any]]) -> None:
        """ADD 出来的新原子：嵌入并入库，供**后续** current 召回。"""
        if not atoms or self.embedder is None:
            return
        vectors = await self.embedder.embed([memory_text(a) for a in atoms])
        for atom, vector in zip(atoms, vectors):
            self.vectors[memory_id(atom)] = vector
        self.embedded += len(atoms)

    async def refresh(self, atoms: list[dict[str, Any]]) -> None:
        """UPDATE 过的原子：文本变了，重嵌以替换过期向量。"""
        if not atoms or self.embedder is None:
            return
        vectors = await self.embedder.embed([memory_text(a) for a in atoms])
        for atom, vector in zip(atoms, vectors):
            self.vectors[memory_id(atom)] = vector
        self.reembedded += len(atoms)

    def remove(self, ids: list[str]) -> None:
        """DELETE 过的原子：从向量表摘掉，后续不会再被检索到。"""
        for atom_id in ids:
            self.vectors.pop(atom_id, None)

    def rank(
        self,
        query_vector: list[float],
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """在给定的候选（= 当前有效索引范围）里按余弦相似度取 top_k。

        只对 candidates 打分，不扫全表 —— 范围由调用方按游标前缀给定。
        """
        scored = [
            (cosine_similarity(query_vector, self.vectors[memory_id(m)]), m)
            for m in candidates
            if memory_id(m) in self.vectors
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [memory for _, memory in scored[:top_k]]
    def stats(self) -> str:
        return f"向量表 {len(self.vectors)} 条（初始 {self.embedded}，重嵌 {self.reembedded}）"


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def build_recall_query(
    current: dict[str, Any],
    traces: list[dict[str, Any]],
    fallback_text: str = "",
) -> str:
    """召回 query：**当前要演化的那条 atommem** + 最近几轮动作的摘要。

    为什么用 ``current`` 而不是 anchor 的源消息：本阶段是"按 id 顺序逐条精修 atommem"，
    每轮的目标不同，query 就该跟着目标走。拿 anchor（最后一条 raw 消息）当 query 会让
    query 从头到尾固定不变，无论处理到哪条原子都召回同一批记忆 —— 前面几轮还能蒙对，
    一旦 current 挪到别的主题上，召回的证据就与目标无关了。

    用 current 的文本还有两个好处：它天然是"当前库里的措辞"（可能已被 UPDATE 过），
    且与候选池同源，相似度分数在同一语义空间里可比。

    把最近动作的摘要拼进去，是为了让连续几轮的 query 互相区分开：同一批候选里，上一轮
    动过的区域会被压低，后几轮能移到邻近但不同的区域。

    ``current`` 为空（理论上不该发生）时退回 ``fallback_text``。
    """
    parts = [memory_text(current).strip() or fallback_text.strip()]
    for trace in traces[-LIBRARY_ANCHOR_TURNS:]:
        summary = trace.get("summary") or trace.get("action", "")
        if summary:
            parts.append(str(summary))
    return "\n".join(part for part in parts if part)


def build_recalled_block(
    recalled: list[dict[str, Any]],
    raw_order: list[dict[str, Any]],
    window: int,
) -> str:
    """召回块的文本：每条原子 + 它的源消息 + 前后 window 条上下文消息。"""
    order = {memory_id(raw): index for index, raw in enumerate(raw_order)}
    speaker_by_id = {
        memory_id(raw): next(
            (t.split(":", 1)[1] for t in (raw.get("metadata") or {}).get("tag", [])
             if str(t).startswith("speaker:")),
            "unknown",
        )
        for raw in raw_order
    }

    blocks: list[str] = []
    for memory in recalled:
        meta = memory.get("metadata") or {}
        speaker = next(
            (t.split(":", 1)[1] for t in meta.get("tag", []) if str(t).startswith("speaker:")),
            "unknown",
        )
        lines = [
            f"[{meta.get('id', '')}] ({meta.get('time', '')}) {speaker}: {memory_text(memory)}",
            f"    type={meta.get('type', '')} | tag={meta.get('tag', [])} | source={meta.get('source', [])}",
        ]
        for source_id in meta.get("source", []):
            index = order.get(source_id)
            if index is None:
                lines.append(f"    (source message {source_id} not found)")
                continue
            start = max(0, index - window)
            end = min(len(raw_order), index + window + 1)
            lines.append(f"    --- source message {source_id} (with context) ---")
            for position in range(start, end):
                raw = raw_order[position]
                raw_id = memory_id(raw)
                marker = ">>" if raw_id == source_id else "  "
                lines.append(
                    f"    {marker} [{raw_id}] ({(raw.get('metadata') or {}).get('time', '')}) "
                    f"{speaker_by_id.get(raw_id, '?')}: {memory_text(raw)}"
                )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(no recalled memories)"


def retrieve_for_evolution(
    query_vector: list[float],
    candidates: list[dict[str, Any]],
    candidate_vectors: list[list[float]],
    top_k: int,
) -> list[dict[str, Any]]:
    """按余弦相似度取 top_k；embedding 不可用时退化为按顺序取前 top_k 条。"""
    if candidate_vectors and len(candidate_vectors) == len(candidates):
        scored = [
            (cosine_similarity(query_vector, vector), memory)
            for vector, memory in zip(candidate_vectors, candidates)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [memory for _, memory in scored[:top_k]]
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# prompt 构建
# ---------------------------------------------------------------------------

def format_current(memory: dict[str, Any]) -> str:
    """渲染"当前要细化的那一条原子"的**值**（不含标题 —— 标题在 prompt 模板里）。

    prompt 里**只有这一条**是目标；其余的 recalled 只是证据。不要把整个库放进来 ——
    模型会以为它看见了全部，然后对着看不见的全集去删（实测过的乱删就是这么来的）。

    调用方必须传**单条** memory。传整个 list 会让 ``memory.get`` 直接 ``AttributeError``
    —— 这里显式挡住，而不是让它在中途以一个看不懂的报错炸掉（这个 bug 真出现过）。
    """
    from .prompts import EVOMEM_CURRENT_HEADER

    if isinstance(memory, list) or not isinstance(memory, dict):
        raise TypeError(
            f"format_current 需要单条 memory（dict），收到 {type(memory).__name__}；"
            "调用方是不是把整个记忆库传进来了？"
        )
    if not memory:
        return f"{EVOMEM_CURRENT_HEADER}\n(empty)"
    meta = memory.get("metadata") or {}
    speaker = next(
        (t.split(":", 1)[1] for t in meta.get("tag", []) if str(t).startswith("speaker:")),
        "unknown",
    )
    return "\n".join([
        f"[{meta.get('id', '')}] ({meta.get('time', '')}) {speaker} | "
        f"{meta.get('type', '')} | {memory_text(memory)}",
        f"tag: {meta.get('tag', [])} | source: {meta.get('source', [])}",
    ])


def format_history(traces: list[dict[str, Any]]) -> str:
    from .prompts import EVOMEM_HISTORY_EMPTY

    if not traces:
        return EVOMEM_HISTORY_EMPTY
    lines: list[str] = []
    for trace in traces:
        lines.append(f"Turn {trace.get('turn')}: {trace.get('summary', trace.get('action', ''))}")
        for detail in trace.get("details", []):
            lines.append(f"  - {detail}")
    return "\n".join(lines)


def count_tokens(text: str) -> int:
    """粗略估算 token 数（4 字符 ≈ 1 token），与 eval_utils 的口径一致。"""
    return int(len(text) / 4) + 1


def library_token_estimate(memories: list[dict[str, Any]]) -> int:
    """整库的 token 估算（只用于 trace 观测，不再用于 prompt 渲染）。

    保留它是为了让新旧 run 的 `library_tokens` 字段可比 —— prompt 现在只含 1 条
    current + recalled，但库的真实规模仍值得记录。
    """
    text = "\n".join(
        f"[{memory_id(m)}] {memory_text(m)}" for m in memories
    )
    return count_tokens(text)


def build_messages(
    current: dict[str, Any],
    recalled_block: str,
    traces: list[dict[str, Any]],
    schema: str,
    question: str = "",
) -> list[dict[str, str]]:
    """构造一次调用的 messages：**1 条 current** + recalled 证据 + 要回答的问题。

    注意占位符是 ``{{CURRENT}}``（不是 ``{{LIBRARY}}``）—— 名字很重要，叫 LIBRARY 会让
    模型以为它看见了整个库。

    ``current`` 必须是单条 memory：传整个 list 会让 prompt 里塞进上千条目标、并且
    ``format_current`` 立刻 AttributeError。历史上这里被误传成 ``memories``（全库），
    所以显式断言，别再犯。

    ``question`` 是 QA 驱动演化的目标（为什么这条原子此刻被演化）。为空时渲染成
    ``EVOMEM_QUESTION_EMPTY``，兼容"纯原子精修"的旧调用方式。
    """
    from .prompts import (
        EVOMEM_PROMPT, EVOMEM_QUESTION_EMPTY, EVOMEM_USER_SUFFIX,
    )

    if not isinstance(current, dict):
        raise TypeError(
            f"build_messages 的 current 必须是单条 memory（dict），"
            f"收到 {type(current).__name__}；调用方是不是把整个记忆库传进来了？"
        )
    system = (
        EVOMEM_PROMPT.replace("{{QUESTION}}", question.strip() or EVOMEM_QUESTION_EMPTY)
        .replace("{{CURRENT}}", format_current(current))
        .replace("{{RECALLED}}", recalled_block)
        .replace("{{HISTORY}}", format_history(traces))
        .replace("{{SCHEMA}}", schema)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": EVOMEM_USER_SUFFIX},
    ]



# ---------------------------------------------------------------------------
# 一轮演化
# ---------------------------------------------------------------------------

def action_summary(action: Any, result: Any) -> str:
    """给下一轮 prompt 用的一句话摘要。

    **关键**：当 UPDATE 被判为 unchanged 时，用非常强的信号告诉模型停止尝试。
    """
    if action.kind == "NOOP":
        return "NOOP - library left unchanged"
    details = []
    if result.added_ids:
        details.append(f"added {result.added_ids}")
    if result.updated_ids:
        details.append(f"updated {result.updated_ids}")
    if result.deleted_ids:
        details.append(f"deleted {result.deleted_ids}")

    # 检测 "unchanged" 错误：UPDATE 被提交但没有改变任何内容
    unchanged_errors = [e for e in result.errors if "unchanged" in e.lower()]

    if not details:
        if unchanged_errors:
            # 用非常强烈的警告信号，告诉模型停止尝试
            return (f"⚠️ {action.kind} REJECTED - Your output is IDENTICAL to the current memory. "
                   "This means the memory is already optimal. "
                   "⚠️ OUTPUT NOOP IMMEDIATELY to avoid wasting resources.")
        details.append("no change applied")
    if result.errors and not unchanged_errors:
        details.append(f"rejected: {result.errors[:3]}")
    return f"{action.kind}: " + "; ".join(details)


def shrink_library(
    memories: list[dict[str, Any]],
    active: set[str],
    max_tokens: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """把记忆库压到 max_tokens 以内：从尾部（最旧）开始软删除。"""
    removed: list[str] = []
    while library_token_estimate(memories) > max_tokens:
        victim = next(
            (m for m in reversed(memories) if memory_id(m) in active),
            None,
        )
        if victim is None:
            break
        victim_id = memory_id(victim)
        active.discard(victim_id)
        removed.append(victim_id)
        # 记忆列表保持完整（软删除），所以必须真的缩短它才能让循环收敛
        memories = [m for m in memories if memory_id(m) != victim_id]
    return memories, removed


def describe_turn(trace: dict[str, Any]) -> str:
    """一轮的**结果**概要：有没有报错、是否成功、改了几条。

    每轮结束都返回这样一行，方便"看到结果再决定下一步"。status 取值：
    - ``OK``    动作正常执行并改动了库
    - ``NOOP``  模型明确说不用改（正常收敛）
    - ``EMPTY`` 动作解析出来但没有一条能执行（内容非法/全部被拒）
    - ``ERROR`` 模型调用失败或用了无效动作
    """
    status = trace.get("status", "?")
    bits = [f"turn {trace.get('turn')} {status}"]
    if trace.get("current_id"):
        bits.append(f"current={trace['current_id']}")
    action = trace.get("action")
    if action:
        bits.append(f"action={action}")
    if trace.get("added"):
        bits.append(f"+{len(trace['added'])}")
    if trace.get("updated"):
        bits.append(f"~{len(trace['updated'])}")
    if trace.get("deleted"):
        bits.append(f"-{len(trace['deleted'])}")
    if trace.get("errors"):
        bits.append(f"errors={len(trace['errors'])}")
    if trace.get("warnings"):
        bits.append(f"warns={len(trace['warnings'])}")
    return " ".join(bits) + f" | library={trace.get('library_size')} active={trace.get('active_size')}"


async def run_dir(
    directory: Path,
    config: dict[str, Any],
    gen_config: dict[str, Any],
    generator: openai.AsyncOpenAI,
    embedder: Embedder | None,
    output_dir: Path,
    log: Logger,
) -> dict[str, Any]:
    """对一个 speaker 目录跑完整演化，返回 summary。

    **QA 驱动的三层循环**（见 ``docs/evomem_plan.md``）：

    - 外层：按 QA 问题走（每条 QA 独立，从同一份 base 库出发）—— 用 ``searchmem`` 的
      混合检索（dense + BM25 + tag，RRF 融合）从库里取出**这条问题相关的候选原子集**；
    - 中层：候选集内逐条原子（``current``）；
    - 内层（agent loop）：针对当前这一条，反复"调模型 → 执行动作 → 把结果写进
      HISTORY"，直到模型输出 **NOOP**（结束这一条）或到 ``max_evomem_turn``。

    **为什么要按 QA 驱动**（旧版是"按 id 顺序走遍全库逐条精修"）：

    1. **成本**：旧版成本 = ``原子数 × max_evomem_turn``（Tim_John 1771 条原子 ≈ 7000 次
       强模型调用），且大部分调用花在与任何问题都无关的原子精修上。新版只演化候选集，
       成本 = ``QA 数 × 候选数 × 轮数``，全部花在**问题相关**的原子上。
    2. **目标更明确**：旧版让模型"把这条原子改得更自足"，没有一个具体的成功判据；新版把
       **问题**传给模型（``{{QUESTION}}``），演化目标变成"让这组候选足以回答这个问题"。
    3. **对 RL 友好**：一条 QA 一条 trajectory，奖励信号（答案对不对）直接挂在这条轨迹上。

    **每条 QA 从同一份 base 库出发**（``memories`` 在每条 QA 开始时重置），所以 QA 之间
    互不影响 —— 这正是"一条 QA 一条轨迹"的前提，也让 base ``atommem.jsonl`` 始终不被修改。

    每条原子**只召回一次**，内层各轮共享同一份证据；上一轮改了什么靠 HISTORY 传递，
    而不是重新检索（避免把自己刚写的措辞当成独立证据）。

    每次模型调用都会把**本轮结果**（status / 动作 / 增删改 / 错误）记进 trace 并在 info
    级别打印一行，debug 级别另外打印完整的真实数据流（召回内容、完整 prompt、动作解析）。
    """
    from . import atommem, evoactions, searchmem

    label = directory.name
    log = log.bind(label)
    atommem_path = directory / "atommem.jsonl"
    msgmem_path = directory / "msgmem.jsonl"
    if not atommem_path.exists():
        log.warn("跳过：没有 atommem.jsonl")
        return {"dir": label, "skipped": "no atommem.jsonl", "status": "SKIPPED"}
    if not msgmem_path.exists():
        log.warn("跳过：没有 msgmem.jsonl")
        return {"dir": label, "skipped": "no msgmem.jsonl", "status": "SKIPPED"}

    base_memories = read_jsonl(atommem_path)
    raw_order = read_jsonl(msgmem_path)
    if not base_memories or not raw_order:
        log.warn("跳过：atommem/msgmem 为空")
        return {"dir": label, "skipped": "empty atommem/msgmem", "status": "SKIPPED"}

    # msgmem 的 id -> 它在文件里的位置。原子 id 形如 ``session_{s}_{m}_{n}``，解析出的
    # ``session_{s}_{m}`` 到这里查表就得到该原子在对话里的先后位置，``atom_key`` 用它排序。
    # 顺序从 msgmem 的真实顺序取（而不信 atommem 文件顺序）——见文件顶部关于顺序的注释。
    msgmem_pos = {memory_id(raw): index for index, raw in enumerate(raw_order)}

    # **把库按 atom_key 排好**。这一步不能省：``recall_pool`` 用 atom_key 判"是否早于
    # current"，两者顺序不一致时（文件顺序 ≠ id 顺序，实测真实数据里有 9~43 处逆序）
    # 前缀语义会失效。排序后两套顺序统一。
    before_sort = [memory_id(m) for m in base_memories]
    base_memories = order_atoms(base_memories, msgmem_pos)
    if [memory_id(m) for m in base_memories] != before_sort:
        log.debug("库按 atom_key 重排（文件顺序 != id 顺序）")

    # 用**演化模板**（不是抽取模板）：字段措辞是"精修已有记忆 + 必须给 source"。
    schema = atommem.build_schema_prompt(atommem.load_template(EVOMEM_TEMPLATE_FILE))
    window = int(config.get("context_window", 2))
    max_turns = max(1, int(config.get("max_evomem_turn", 2)))
    max_library_tokens = int(config.get("library_max_tokens", 40000))
    # QA 驱动的闸门：每个目录最多处理前 N 条 QA（0 = 全部），每条 QA 取 K 个候选。
    max_qa = max(0, int(config.get("max_qa", 0) or 0))
    qa_candidates = max(1, int(config.get("qa_candidates", 10) or 10))
    skip_category5 = bool(config.get("skip_category5", True))
    search_cfg = config.get("search") or {}
    search_weights = search_cfg.get("weights") or None
    search_k = int(search_cfg.get("rrf_k", searchmem.DEFAULT_RRF_K))

    # anchor：新记忆的 id 挂在最后一条 raw 消息下，与 atommem 的 id 约定一致。
    anchor_raw = raw_order[-1]
    anchor_text = memory_text(anchor_raw)
    anchor_atom_id = memory_id(anchor_raw)

    # QA 列表：目录 -> sample -> qa。找不到就当这个目录没有可演化的问题（SKIPPED）。
    questions = load_dir_questions(label, limit=max_qa)
    if not questions:
        log.warn(f"跳过：QA 数据源里找不到目录 {label} 的问题")
        return {"dir": label, "skipped": "no qa for dir", "status": "SKIPPED"}

    traces: list[dict[str, Any]] = []
    qa_records: list[dict[str, Any]] = []
    state_path = output_dir / f"{label}.evolution.jsonl"
    qa_path = output_dir / f"{label}.qa_trajectories.jsonl"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        f"开始演化：库 {len(base_memories)} 条 / msgmem {len(raw_order)} 条 / "
        f"QA {len(questions)} 条"
        + ("（跳过 category5）" if skip_category5 else "")
        + f" / 每条 QA 取 {qa_candidates} 个候选 × 每条最多 {max_turns} 轮 / "
        f"window={window} / anchor={anchor_atom_id}"
        + (f" / 只跑前 {max_qa} 条 QA（--max-qa 调试）" if max_qa else "")
    )

    stop_reason = "已走完全部 QA"
    evolved_candidates = 0   # 演化过的候选原子数（跨 QA 累计）
    total_calls = 0          # 模型调用总次数
    qa_done = 0
    qa_skipped = 0

    # 目录内所有 QA 的演化结果并集（按 id 去重），最后落成快照。
    union: dict[str, dict[str, Any]] = {}

    with state_path.open("w", encoding="utf-8") as state_file, \
            qa_path.open("w", encoding="utf-8") as qa_file:
        pbar = tqdm(
            total=len(questions),
            desc=f"[{label}] 演化 QA",
            unit="qa",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            disable=log.threshold > 20,  # 仅 info/warn/error 时显示进度条（debug 下日志已够详细）
        )

        for qa_index, qa in questions:
            question = str(qa.get("question") or "").strip()
            category = qa.get("category")
            if skip_category5 and category == 5:
                qa_skipped += 1
                pbar.update(1)
                log.debug(f"QA {qa_index}：category 5（adversarial），跳过演化")
                continue
            if not question:
                qa_skipped += 1
                pbar.update(1)
                log.debug(f"QA {qa_index}：question 为空，跳过")
                continue

            # ---- 每条 QA 从同一份 base 库出发（QA 之间互不影响）----
            memories = [dict(m) for m in base_memories]
            active: set[str] = {memory_id(m) for m in memories if memory_id(m)}
            qa_trace_start = len(traces)

            # ---- 外层：混合检索出这条问题的候选原子集 ----
            pool = [m for m in memories if memory_id(m) in active]
            try:
                scored = await searchmem.search_scored(
                    question, pool,
                    top_k=qa_candidates,
                    embed=(embedder.embed if embedder is not None else None),
                    query_tags=searchmem.extract_query_tags(question),
                    weights=search_weights,
                    k=search_k,
                )
            except Exception as exc:  # noqa: BLE001
                log.error(f"QA {qa_index}: 候选检索失败，终止该目录：{type(exc).__name__}: {exc}")
                stop_reason = f"候选检索失败：{type(exc).__name__}"
                break

            candidates = [item.memory for item in scored]
            log.info(
                f"QA {qa_index}/{len(questions)}：候选 {len(candidates)}/{len(pool)}"
                f"（category {category}）question={question[:80]!r}"
            )
            log.debug(
                evolog.block(
                    f"QA {qa_index} 混合检索候选（RRF: dense+bm25+tag）",
                    "\n".join(f"  {item.describe()} | {item.text[:80]}" for item in scored)
                    or "(空)",
                )
            )

            # ---- 中层：候选集内逐条演化（保留现有 per-atom 内层 loop）----
            # 按 atom_key 排序后走，保证"只看得到更早的原子"这个前缀语义成立。
            for position, current in enumerate(order_atoms(candidates, msgmem_pos)):
                current_id = memory_id(current)
                if current_id not in active:
                    continue  # 已被软删除（前面的候选演化时删掉的）

                # 候选池：**库里位置早于 current 的存活原子**（严格 key < key(current)）。
                candidates_pool = recall_pool(current, memories, active, set(), msgmem_pos)

                # ---- 召回：每条原子只做一次（内层多轮共享同一份证据）----
                query = build_recall_query(current, traces, fallback_text=anchor_text)
                try:
                    recalled = await recall(query, candidates_pool, config, embedder)
                except Exception as exc:  # noqa: BLE001
                    log.error(f"current {current_id}: 召回失败，终止该目录：{type(exc).__name__}: {exc}")
                    traces.append(
                        {
                            "turn": total_calls + 1,
                            "qa_index": qa_index,
                            "current_id": current_id,
                            "position": position + 1,
                            "status": "ERROR",
                            "action": None,
                            "summary": f"recall failed: {type(exc).__name__}: {exc}",
                            "details": [],
                            "touched": [],
                            "added": [], "updated": [], "deleted": [],
                            "errors": [f"recall: {type(exc).__name__}: {exc}"],
                            "warnings": [], "library_size": len(memories),
                            "active_size": len(active),
                        }
                    )
                    stop_reason = f"召回失败：{type(exc).__name__}"
                    break

                recalled_block = build_recalled_block(recalled, raw_order, window)
                log.debug(
                    evolog.block(
                        f"QA {qa_index} current {current_id} 召回 {len(recalled)}/{len(candidates_pool)} 条",
                        "\n".join(f"  {memory_id(m)} | {memory_text(m)}" for m in recalled) or "(空)",
                    )
                )

                # ---- 内层（agent loop）：同一条原子反复动作，直到 NOOP 或到 max_turns ----
                inner = 0
                while inner < max_turns:
                    inner += 1
                    total_calls += 1

                    # **每轮按 id 重新取一次 current**：上一轮可能 UPDATE 了它，文本已经变了。
                    live = next((m for m in memories if memory_id(m) == current_id), None)
                    if live is None or current_id not in active:
                        log.warn(f"current {current_id}: 已被删除，结束该原子的内层循环")
                        break

                    # 目标只有 live 这一条；HISTORY 也只含**当前原子**的历史。
                    current_history = [t for t in traces if t.get("current_id") == current_id]
                    messages = build_messages(live, recalled_block, current_history, schema, question)
                    log.info(
                        f"QA {qa_index} 候选 {position + 1}/{len(candidates)} current={current_id} "
                        f"第 {inner}/{max_turns} 轮：候选 {len(candidates_pool)} 召回 {len(recalled)} "
                        f"prompt≈{count_tokens(messages[0]['content'])}tok 调用模型…"
                    )

                    try:
                        content = await atommem.chat_completion(generator, gen_config, messages)
                    except Exception as exc:  # noqa: BLE001
                        log.error(f"current {current_id}: 模型调用失败：{type(exc).__name__}: {exc}")
                        traces.append(
                            {
                                "turn": total_calls,
                                "qa_index": qa_index,
                                "current_id": current_id,
                                "position": position + 1,
                                "inner": inner,
                                "status": "ERROR",
                                "action": "MODEL_ERROR",
                                "summary": f"model call failed: {type(exc).__name__}: {exc}",
                                "details": [],
                                "touched": [],
                                "added": [], "updated": [], "deleted": [],
                                "errors": [f"model: {type(exc).__name__}: {exc}"],
                                "warnings": [], "library_size": len(memories),
                                "active_size": len(active),
                            }
                        )
                        stop_reason = f"模型调用失败：{type(exc).__name__}"
                        log.info(describe_turn(traces[-1]))
                        break

                    log.debug(evolog.block(
                        f"QA {qa_index} current {current_id} 第 {inner} 轮 模型原始输出", content
                    ))

                    action = evoactions.parse_action(content)
                    mismatches = evoactions.verify_payload_matches(action.payload, memories, action.kind)
                    result = evoactions.apply_action(action, memories, active, anchor_atom_id)
                    memories = result.memories
                    touched = result.added_ids + result.updated_ids + result.deleted_ids
                    memories, shrunk = shrink_library(memories, active, max_library_tokens)

                    log.debug(
                        evolog.block(
                            f"QA {qa_index} current {current_id} 第 {inner} 轮 动作解析/执行",
                            f"kind={action.kind} supplied={action.supplied} "
                            f"payload={len(action.payload)}\n"
                            f"repairs={action.repairs}\n"
                            f"parse_errors={action.errors}\n"
                            f"apply_errors={result.errors}\n"
                            f"content_mismatch={mismatches}\n"
                            + "\n".join(
                                f"  {op}: {ids}"
                                for op, ids in (
                                    ("added", result.added_ids),
                                    ("updated", result.updated_ids),
                                    ("deleted", result.deleted_ids),
                                    ("shrunk", shrunk),
                                ) if ids
                            ),
                        )
                    )

                    summary_text = action_summary(action, result)
                    trace = {
                        "turn": total_calls,
                        "qa_index": qa_index,
                        "current_id": current_id,
                        "position": position + 1,
                        "inner": inner,
                        "status": "NOOP" if action.is_noop else ("OK" if touched else "EMPTY"),
                        "action": action.kind,
                        "supplied": action.supplied,
                        "summary": summary_text,
                        "details": [f"warn: {w}" for w in mismatches],
                        "added": result.added_ids,
                        "updated": result.updated_ids,
                        "deleted": result.deleted_ids,
                        "touched": touched,
                        "errors": action.errors + result.errors,
                        "warnings": mismatches,
                        "repaired": action.repairs,
                        "shrunk": shrunk,
                        "library_size": len(memories),
                        "active_size": len(active),
                        "library_tokens": library_token_estimate(memories),
                        "recalled_ids": [memory_id(m) for m in recalled],
                        "candidates": len(candidates_pool),
                        "prompt_tokens": count_tokens(messages[0]["content"]),
                        "raw_output": content,
                    }
                    traces.append(trace)
                    state_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    state_file.flush()

                    for warning in mismatches:
                        log.warn(f"current {current_id}: {warning}")
                    for error in action.errors + result.errors:
                        log.warn(f"current {current_id}: {error}")
                    if shrunk:
                        log.warn(f"current {current_id}: 超出 token 预算，软删除 {shrunk}")
                    log.info(describe_turn(trace))

                    # NOOP 结束**这一条原子**的内层循环；候选集里的下一条继续。
                    if action.is_noop:
                        break
                    if not result.memories:
                        break
                else:
                    # while 正常结束 ＝ 用满 max_turns，这条原子到此为止。
                    log.warn(f"current {current_id}: 内层循环用满 {max_turns} 轮，转下一条")

                evolved_candidates += 1

            # ---- 落一条 QA trajectory ----
            qa_traces = traces[qa_trace_start:]
            survivor_ids = [memory_id(m) for m in candidates if memory_id(m) in active]
            record = {
                "dir": label,
                "qa_index": qa_index,
                "question": question,
                "category": category,
                "reference": qa.get("answer", qa.get("adversarial_answer")),
                "evidence": qa.get("evidence") or [],
                "candidate_ids": [memory_id(m) for m in candidates],
                "evolved_ids": survivor_ids,
                "library_size": len(memories),
                "active_size": len(active),
                "turns": len(qa_traces),
                "actions": [t.get("action") for t in qa_traces],
                "turn_status": [t.get("status") for t in qa_traces],
                "added": sum(len(t.get("added", [])) for t in qa_traces),
                "updated": sum(len(t.get("updated", [])) for t in qa_traces),
                "deleted": sum(len(t.get("deleted", [])) for t in qa_traces),
                "errors": sum(len(t.get("errors", [])) for t in qa_traces),
            }
            qa_records.append(record)
            qa_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            qa_file.flush()

            # 并集快照：这条 QA 演化后仍然存活的候选原子（供目录级快照汇总）
            for memory in memories:
                mem_id = memory_id(memory)
                if mem_id in active and mem_id in record["evolved_ids"]:
                    union[mem_id] = memory

            qa_done += 1
            log.info(
                f"QA {qa_index} 完成：候选 {len(candidates)} 演化 {len(survivor_ids)}"
                f"（+{record['added']} ~{record['updated']} -{record['deleted']}）"
                f"turns={record['turns']}"
            )

            pbar.update(1)
            pbar.set_postfix({
                "turns": total_calls,
                "active": len(active),
                "qa": f"{qa_done}/{len(questions)}",
            })

            # 模型调用失败会 break 内层循环；这里同步结束外层（避免在坏状态上继续跑）
            if stop_reason.startswith(("模型调用失败", "召回失败", "候选检索失败")):
                break

        pbar.close()

    # 演化后快照：base 库 + 各 QA 演化出的新原子（并集）。QA 演化的结果都基于同一份 base
    # 库，所以这里按 id 合并 —— 同名 id 取演化后的版本（union 里的），其余保留 base。
    merged: dict[str, dict[str, Any]] = {memory_id(m): m for m in base_memories}
    merged.update(union)
    out_path = output_dir / f"{label}.atommem.evolved.jsonl"
    write_jsonl(out_path, list(merged.values()))

    errors = sum(len(t.get("errors", [])) for t in traces)
    status = "ERROR" if any(t.get("status") == "ERROR" for t in traces) else "OK"
    summary = {
        "dir": label,
        "status": status,
        "turns": len(traces),                    # 模型调用总次数
        "atoms_done": evolved_candidates,        # 演化过的**候选**原子数（跨 QA 累计）
        "qa_total": len(questions),
        "qa_done": qa_done,
        "qa_skipped": qa_skipped,
        "qa_status": [r["turn_status"][-1] if r["turn_status"] else "EMPTY" for r in qa_records],
        "candidates_total": sum(len(r["candidate_ids"]) for r in qa_records),
        "library_size": len(merged),
        "stop_reason": stop_reason,
        "max_qa": max_qa,
        "qa_candidates": qa_candidates,
        "initial_size": len(base_memories),
        "final_size": len(merged),
        "added": sum(len(t.get("added", [])) for t in traces),
        "updated": sum(len(t.get("updated", [])) for t in traces),
        "deleted": sum(len(t.get("deleted", [])) for t in traces),
        "actions": [t.get("action") for t in traces],
        "turn_status": [t.get("status") for t in traces],
        "errors": errors,
        "warnings": sum(len(t.get("warnings", [])) for t in traces),
        "traces": traces,
        "qa_records": qa_records,
        "output": str(out_path),
        "trace": str(state_path),
        "qa_trajectories": str(qa_path),
    }
    log.info(
        f"完成 {status}：{len(traces)} 次模型调用 / {qa_done} 条 QA（跳过 {qa_skipped}）"
        f" / 共演化 {evolved_candidates} 个候选（停止：{stop_reason}）"
        f"库 {summary['initial_size']}->{summary['final_size']} "
        f"+{summary['added']} ~{summary['updated']} -{summary['deleted']} "
        f"errors={errors} warns={summary['warnings']} -> {out_path}"
    )
    return summary



# ---------------------------------------------------------------------------
# QA 数据源：evomem 现在按问题驱动，所以要知道每个目录有哪些问题
# ---------------------------------------------------------------------------

def load_dir_questions(label: str, limit: int = 0) -> list[tuple[int, dict[str, Any]]]:
    """取某个 speaker 目录的 QA 列表，保留**原始下标**：``[(qa_index, qa), ...]``。

    为什么保留下标而不是用问题文本当键：**问题文本不唯一**（实测 conv-48 有 11 组重复
    question、conv-30 有 1 组，全库 1974/1986 唯一）。用文本做键会把同一条 QA 的两份
    数据当成一条，resume 时静默丢掉一条。下标才是唯一标识。

    目录名 -> sample 靠 ``eval_utils.sample_dir``（``{speaker_a}_{speaker_b}``）。找不到
    对应 sample 时返回空列表 —— 调用方据此把该目录标成 SKIPPED，而不是凭空造问题。

    ``limit > 0`` 时只取前 limit 条（调试闸门）。
    """
    from . import eval_utils

    try:
        samples = eval_utils.load_samples()
    except (OSError, ValueError) as exc:  # 数据文件缺失或坏掉
        print(f"[warn] 无法读取 QA 数据源：{type(exc).__name__}: {exc}", file=sys.stderr)
        return []
    for sample in samples:
        if eval_utils.sample_dir(sample) != label:
            continue
        qas = sample.get("qa") or []
        items = list(enumerate(qas))
        return items[:limit] if limit > 0 else items
    return []


# ---------------------------------------------------------------------------
# 召回
# ---------------------------------------------------------------------------

async def recall(
    query: str,
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    embedder: Embedder | None,
) -> list[dict[str, Any]]:
    """按 embedding 相似度召回 top_k。

    embedding 失败时**抛出异常**，不退化。原因：退化路径只能是"按文件顺序取前 top_k"，
    那不是语义召回，模型拿到的证据与 query 无关，于是多半回一个 NOOP —— 看起来像"库已经
    很好不需要改"，实际是嵌入挂了。这种假成功比直接失败危险得多（实测就踩过：GPU OOM
    导致只嵌入了 256/1265 条，然后安静地报 NOOP）。
    """
    top_k = int(config.get("top_k", 30))
    if embedder is None:
        raise RuntimeError("没有可用的 embedder，无法召回")
    texts = [memory_text(m) for m in candidates]
    vectors = await embedder.embed([query, *texts])
    return retrieve_for_evolution(vectors[0], candidates, vectors[1:], top_k)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 prompt 直接演化原子记忆（无 RL）")
    parser.add_argument("dirs", nargs="*", help="speaker 目录名或路径；缺省处理全部")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE, help="配置文件路径")
    parser.add_argument("--sample", type=str, default=None, help="只处理指定目录（等价于只传一个 dirs）")
    parser.add_argument("--max-turns", type=int, default=None, help="覆盖 max_evomem_turn（每条原子的内层轮数上限）")
    parser.add_argument(
        "--max-qa", type=int, default=None,
        help="每个目录最多处理前 N 条 QA（调试用；0 或不传=全部）",
    )
    parser.add_argument(
        "--candidates", type=int, default=None,
        help="覆盖 qa_candidates：每条 QA 用混合检索取几个候选原子来演化",
    )
    parser.add_argument("--top-k", type=int, default=None, help="覆盖召回条数")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖目录级并发")
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录（默认 data/evomem_runs/）")
    parser.add_argument("--experiment", default=None, help="实验名，作为输出子目录前缀")
    parser.add_argument(
        "--log-level", default=None,
        choices=sorted(evolog.LEVELS),
        help="覆盖 log.level（debug=打印完整数据流：query/召回/prompt/模型输出/动作解析）",
    )
    parser.add_argument("--log-output", type=Path, default=None, help="覆盖 log.output（日志保存目录）")
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> None:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.config.exists():
        loaded = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                    config[key].update(value)
                else:
                    config[key] = value
    for key, value in (
        ("max_evomem_turn", args.max_turns),
        ("max_qa", args.max_qa),
        ("qa_candidates", args.candidates),
        ("top_k", args.top_k),
        ("concurrency", args.concurrency),
    ):
        if value is not None:
            config[key] = value
    if args.log_level is not None:
        config.setdefault("log", {})["level"] = args.log_level
    if args.log_output is not None:
        config.setdefault("log", {})["output"] = str(args.log_output)

    dirs: list[Path] = []
    if args.sample:
        resolved = resolve_dir(args.sample)
        if resolved is None:
            print(f"[error] 找不到目录 {args.sample}", file=sys.stderr)
            return
        dirs = [resolved]
    elif args.dirs:
        for target in args.dirs:
            resolved = resolve_dir(target)
            if resolved is None:
                print(f"[error] 找不到目录 {target}", file=sys.stderr)
                return
            dirs.append(resolved)
    else:
        dirs = sorted(
            path for path in DATA_DIR.iterdir()
            if path.is_dir() and (path / "atommem.jsonl").exists()
        )

    output_dir = args.output_dir or (DATA_DIR / "evomem_runs")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.experiment}_" if args.experiment else ""
    output_dir = Path(output_dir) / f"{prefix}{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # logger 在 output_dir 建好之后初始化：日志文件名里也带时间戳
    log = evolog.Logger.from_config(config)

    # 配置里的 generator 段直接驱动 chat_completion
    gen_config = dict(DEFAULT_CONFIG)
    gen_config.update(config.get("generator") or {})
    for key in ("timeout", "max_retries", "backoff_base", "rate_limit_backoff",
                "max_backoff", "backoff_jitter", "enable_thinking"):
        gen_config[key] = config.get(key, gen_config.get(key))

    generator = openai.AsyncOpenAI(
        base_url=gen_config["base_url"],
        api_key=gen_config.get("api_key") or "not-needed",
        timeout=float(gen_config.get("timeout", 300)),
        max_retries=0,
    )

    embedding_config = config.get("embedding") or {}
    embedding_model = embedding_config.get("model", "qwen3-embedding:4b")
    embedding_client = openai.AsyncOpenAI(
        base_url=embedding_config.get("base_url", "http://127.0.0.1:11434/v1"),
        api_key=embedding_config.get("api_key") or "not-needed",
        timeout=float(gen_config.get("timeout", 300)),
        max_retries=0,
    )
    embedder = Embedder(
        embedding_client, embedding_model,
        int(config.get("embedding_chunk_size", 32)), log=log,
    )

    log.info(
        f"evomem 启动（QA 驱动）：{len(dirs)} 个目录 | max_turns={config['max_evomem_turn']} "
        f"candidates={config['qa_candidates']} max_qa={config['max_qa'] or '全部'} "
        f"top_k={config['top_k']} concurrency={config.get('concurrency')} "
        f"log={log.level}"
    )
    log.info(
        f"混合检索权重 {(config.get('search') or {}).get('weights')} "
        f"rrf_k={(config.get('search') or {}).get('rrf_k')}"
    )
    log.info(f"generator={gen_config['model']} @ {gen_config['base_url']}")
    log.info(f"embedding={embedding_model} @ {embedding_config.get('base_url')}")
    log.info(f"输出目录 {output_dir}")

    # 并发只发生在**目录之间**：一对 speaker 内部 turn 必须串行（第 N 轮召回依赖前
    # N-1 轮改完的记忆库）。所以这里用 asyncio.gather + 信号量扇出，多个 speaker 对
    # 同时推进各自的串行 turn。
    semaphore = asyncio.Semaphore(max(1, int(config.get("concurrency", 2))))
    summaries: dict[str, dict[str, Any]] = {}

    async def worker(directory: Path) -> None:
        label = directory.name
        async with semaphore:
            log.info(f"开始 {label}（当前并发 {len(dirs) - len(summaries)} 待处理）")
            try:
                summary = await run_dir(
                    directory, config, gen_config, generator, embedder, output_dir, log
                )
            except Exception as exc:  # noqa: BLE001
                summary = {
                    "dir": label,
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                    "turns": 0,
                }
                log.error(f"{label} 未捕获异常：{type(exc).__name__}: {exc}")
            summaries[label] = summary

    await asyncio.gather(*(worker(directory) for directory in dirs))

    log.info(embedder.stats())
    await embedding_client.close()
    await generator.close()

    # 按命令行给的顺序输出（并发完成顺序不定）。traces 体积大，从 summary.json 里
    # 摘出去（逐轮明细已经在各自 {dir}.evolution.jsonl 里了）。
    ordered: list[dict[str, Any]] = []
    for directory in dirs:
        item = dict(summaries.get(directory.name, {"dir": directory.name, "status": "SKIPPED"}))
        item.pop("traces", None)
        ordered.append(item)

    ok = [s for s in ordered if s.get("status") == "OK"]
    failed = [s for s in ordered if s.get("status") == "ERROR"]
    skipped = [s for s in ordered if s.get("status") == "SKIPPED"]

    log.info(
        f"全部结束：成功 {len(ok)} / 失败 {len(failed)} / 跳过 {len(skipped)}"
    )
    for item in ordered:
        log.info(
            f"  {item.get('status'):8s} {item.get('dir'):18s} "
            f"qa={item.get('qa_done', 0)}/{item.get('qa_total', 0)} "
            f"cands={item.get('candidates_total', 0)} turns={item.get('turns', 0)} "
            f"+{item.get('added', 0)} ~{item.get('updated', 0)} -{item.get('deleted', 0)} "
            f"errors={item.get('errors', 0)}"
            + (f" | {item['error']}" if item.get("error") else "")
        )
    log.info(log.summary())

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": {k: v for k, v in config.items() if k != "log"},
                "log_level": log.level,
                "log_path": str(log.path) if log.path else None,
                "embedding_stats": embedder.stats(),
                "counts": {"ok": len(ok), "failed": len(failed), "skipped": len(skipped)},
                "dirs": ordered,
            },
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    if not failed:
        log.info(f"summary -> {summary_path}")
    log.close()


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
