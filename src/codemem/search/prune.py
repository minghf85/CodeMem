"""运行目录清理：``--prune-runs``。

只做一件事 —— 找出可以删掉的**历史运行目录**并报告大小。保留策略见 ``plan_prune``。

> 这里曾经还装着整套向量索引的读写（``MemoryIndex`` / ``build_index`` / 共享缓存 /
> ``--clean-index-cache``）。改用 grep 检索后 search 不再嵌入任何东西，那一层整个删掉了；
> 需要时从 git 历史取回。
"""

from __future__ import annotations

import shutil
from pathlib import Path

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
) -> list[tuple[Path, int]]:
    """列出可删除的运行目录及各自大小。

    ``keep_last > 0`` 时保留最近的 N 次运行（按目录名里的时间戳排序）；
    否则只保留"最后一次全量运行"（目录名含 **full**）。以 ``.`` 开头的目录
    （如编辑器/工具的临时目录）不在候选里。
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
        plan.append((path, run_size(path)))
    return plan


def prune_runs(runs_root: Path, plan: list[tuple[Path, int]]) -> tuple[int, int]:
    """执行删除。返回 ``(删除的目录数, 释放的字节数)``。"""
    removed = 0
    freed = 0
    for path, size in plan:
        try:
            shutil.rmtree(path)
        except OSError:
            continue
        removed += 1
        freed += size
    return removed, freed
