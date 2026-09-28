"""嵌入：把文本转成向量，供检索与索引构建使用。

从 search 的各个模块（索引构建 / searchctl / runner）共用。**必须分批**：一次把整个记忆库
（上千条）塞进单个请求时，OpenAI 兼容端点会偶发 400，报错内容关于内部 tokenizer 端口
（误导性信息，实际是并发压力下的瞬时故障）。批越小越稳（批 64 实测 3/3 成功，批 512 只有
1/3），且分批后单批失败只影响那一批。

**单批重试**：上一条说明失败是瞬时的，所以每批失败后短暂退避重试，而不是直接打断整轮检索。
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .log import Logger


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


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """两个向量的余弦相似度。任一为零向量时返回 0.0（而不是 NaN 或除零崩溃）。

    放在这里而不是 index.py：搜索、索引构建、answer 的检索都要用，属于"向量运算"的
    基础工具，跟嵌入本身同源。
    """
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0
