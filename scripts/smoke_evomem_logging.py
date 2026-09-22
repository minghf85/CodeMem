"""用假模型验证 evomem 的日志与逐轮结果上报（纯 CPU，不碰网络）。

跑法：python scripts/smoke_evomem_logging.py
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codemem import atommem, evolog, evomem as V  # noqa: E402

WORK = PROJECT_ROOT / ".smoke_logging"

RAW = [
    ("session_1_1", "Caroline", "Hey Mel! Good to see you!"),
    ("session_1_2", "Melanie", "I'm swamped with the kids & work."),
    ("session_2_1", "Melanie", "I got a new job at Google last month."),
]
ATOMS = [
    ("session_1_2_1", "Melanie has kids and a job", "session_1_2"),
    ("session_2_1_1", "Melanie got a new job last month", "session_2_1"),
    ("session_1_1_1", "Caroline greeted Melanie", "session_1_1"),
]

# 假模型的脚本化回复。**按"每条原子的内层轮数"编排**，顺序 = 游标顺序（按 id 排，
# 不是文件顺序）：
#   session_1_1_1  库里最早的一条 → 候选池为空（召回 0 条），但**照常调模型**（1 次）
#   session_1_2_1  第 1 轮 UPDATE，第 2 轮 NOOP → 内层结束，游标前进（2 次）
#   session_2_1_1  第 1 轮 ADD，第 2 轮坏输出（EMPTY），第 3 轮 NOOP（3 次）
# 两个要盯住的行为：
#   - 候选池为空**不做特殊处理**：第一条照常走完整流程（prompt 的 {{RECALLED}}
#     渲染成 "(no recalled memories)"），不为省一次调用而跳过它；
#   - 坏输出只结束**这一条原子**的内层循环，游标继续走（旧实现会终止整个目录）。
REPLIES = [
    # --- 第 1 条（库里最早）：候选池为空，仍然调用模型 ---
    "<NOOP></NOOP>",
    # --- 第 2 条：session_1_2_1 ---
    '<UPDATE>[{"memory":"Melanie has kids and a job",'
    '"metadata":{"id":"session_1_2_1","type":"outer","time":"2023-05-08",'
    '"tag":["speaker:Melanie"],"source":["session_1_2"],"changelog":[]}}]</UPDATE>',
    "<NOOP></NOOP>",
    # --- 第 3 条：session_2_1_1 ---
    '<ADD>[{"memory":"Melanie works at Google","metadata":{"id":"","type":"outer",'
    '"time":"2023-05-01","tag":["speaker:Melanie"],"source":["session_2_1_1"],'
    '"changelog":[]}}]</ADD>',
    # 坏输出：解析不出任何可执行动作 → 这一轮 EMPTY，但**不终止目录**
    "<ADD>[{\"memory\": \"broken,</ADD>",
    "<NOOP></NOOP>",
]


def seed() -> Path:
    shutil.rmtree(WORK, ignore_errors=True)
    d = WORK / "Fake_Pair"
    d.mkdir(parents=True)
    with (d / "msgmem.jsonl").open("w", encoding="utf-8") as f:
        for rid, spk, text in RAW:
            f.write(json.dumps({
                "memory": text,
                "metadata": {"id": rid, "type": "raw", "time": "2023-05-08T13:56:00",
                             "tag": [f"speaker:{spk}"], "source": [], "changelog": []},
            }, ensure_ascii=False) + "\n")
    with (d / "atommem.jsonl").open("w", encoding="utf-8") as f:
        for aid, text, src in ATOMS:
            f.write(json.dumps({
                "memory": text,
                "metadata": {"id": aid, "type": "outer", "time": "last month",
                             "tag": ["speaker:Melanie"], "source": [src], "changelog": []},
            }, ensure_ascii=False) + "\n")
    return d


class FakeEmbedder:
    """固定向量，避免任何网络调用。"""

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return [[float(len(t) % 7), 1.0, 0.0] for t in texts]

    def stats(self):
        return f"fake embedder 调用 {self.calls} 次"


def main() -> None:
    directory = seed()
    out = WORK / "out"
    out.mkdir(parents=True)

    calls = {"n": 0}

    async def fake_chat(client, config, messages):
        i = min(calls["n"], len(REPLIES) - 1)
        calls["n"] += 1
        return REPLIES[i]

    atommem.chat_completion = fake_chat  # type: ignore[assignment]

    # 统计召回次数：必须是**每条原子一次**，而不是内层每轮一次。
    recall_calls = {"n": 0}
    real_recall = V.recall

    async def counting_recall(query, candidates, config, embedder):
        recall_calls["n"] += 1
        return await real_recall(query, candidates, config, embedder)

    V.recall = counting_recall  # type: ignore[assignment]

    # ---- info 级别 ----
    print("=" * 72)
    print("INFO 级别（应看到每条原子每轮结果一行）")
    print("=" * 72)
    cfg = dict(V.DEFAULT_CONFIG)
    cfg.update({"top_k": 3, "max_evomem_turn": 4, "context_window": 1,
                "log": {"level": "info", "output": str(out)}})
    log = evolog.Logger.from_config(cfg)
    summary = asyncio.run(
        V.run_dir(directory, cfg, cfg["generator"], None, FakeEmbedder(), out, log)
    )
    log.close()

    print()
    print("--- 返回的 summary ---")
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "traces"},
        ensure_ascii=False, indent=2,
    ))
    print("--- 逐轮明细 ---")
    for t in summary["traces"]:
        print(f"  turn {t['turn']}: current={t.get('current_id')} "
              f"inner={t.get('inner')} status={t['status']} action={t['action']} "
              f"errors={len(t['errors'])} warnings={len(t['warnings'])}")

    print("--- 断言（嵌套循环语义） ---")
    failed = 0

    def check(label: str, got, expect) -> None:
        nonlocal failed
        ok = got == expect
        if not ok:
            failed += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: 期望 {expect!r} / 实际 {got!r}")

    statuses = [t["status"] for t in summary["traces"]]
    check("逐轮 status", statuses, ["NOOP", "OK", "NOOP", "OK", "EMPTY", "NOOP"])
    check("逐轮 current_id",
          [t["current_id"] for t in summary["traces"]],
          ["session_1_1_1", "session_1_2_1", "session_1_2_1",
           "session_2_1_1", "session_2_1_1", "session_2_1_1"])
    check("逐轮 inner",
          [t["inner"] for t in summary["traces"]], [1, 1, 2, 1, 2, 3])
    # 三条原子都被走了一遍（含库里最早、候选池为空的那条）
    check("走完内层循环的原子数", summary["atoms_done"], 3)
    # 核心性质：召回每条原子一次（3 条原子），而非每轮一次（6 轮）
    check("召回次数 = 原子数（不是轮数）", recall_calls["n"], 3)
    check("模型调用次数 = 总轮数", summary["turns"], 6)
    # 核心性质：第一条原子候选池为空，但**照常调用模型**（没有跳过分支）
    first = summary["traces"][0]
    check("第一条原子是库里最早的那条", first["current_id"], "session_1_1_1")
    check("第一条原子候选池为空", first["candidates"], 0)
    check("第一条原子召回为空", first["recalled_ids"], [])
    check("候选为空仍然调用了模型", first["action"], "NOOP")
    # 核心性质：坏输出未终止目录
    check("坏输出后游标仍走完最后一条原子",
          summary["traces"][-1]["current_id"], "session_2_1_1")

    print()
    print(f"  改动了 {summary['updated']} 条、新增 {summary['added']} 条、"
          f"错误 {summary['errors']} 个")
    print(f"  {'全部通过' if not failed else f'{failed} 项失败'}")

    # ---- debug 级别：检查数据流是否落盘 ----
    print()
    print("=" * 72)
    print("DEBUG 级别（应看到 query/召回/prompt/模型输出/动作解析）")
    print("=" * 72)
    calls["n"] = 0
    cfg["log"] = {"level": "debug", "output": str(out)}
    log = evolog.Logger.from_config(cfg)
    asyncio.run(V.run_dir(directory, cfg, cfg["generator"], None, FakeEmbedder(), out, log))
    path = log.path
    log.close()

    text = Path(path).read_text(encoding="utf-8") if path else ""
    for marker in ("召回", "prompt (system)", "动作解析/执行"):
        print(f"  {'ok  ' if marker in text else 'FAIL'} 日志含 {marker!r}")
    print(f"  日志行数 {len(text.splitlines())}，文件 {path}")

    shutil.rmtree(WORK, ignore_errors=True)
    print()
    print("done")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
