"""用假模型验证 evomem 的日志与逐轮结果上报（纯 CPU，不碰网络）。

跑法：python scripts/smoke_evomem_logging.py

**QA 驱动**：run_dir 的外层现在是"每条 QA 问题 → 混合检索候选集 → 候选集内逐条演化"，
所以这个自测要注入一个**假的 QA 列表**（真目录名对不上 correct_locomo10.json 的 sample）。
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

# 假 QA 列表。第一条 QA 命中"Melanie ... job"这类原子；第二条 QA 只命中"greeted"那条，
# 用来验证不同 QA 会检索到**不同**候选集（而不是所有 QA 都用同一批）。
FAKE_QAS = [
    (0, {"question": "What job did Melanie get?", "answer": "Google",
         "evidence": ["D2:1"], "category": 1}),
    (1, {"question": "How did Caroline greet Melanie?", "answer": "Hey Mel",
         "evidence": ["D1:1"], "category": 1}),
    (2, {"question": "Unanswerable adversarial question about taxes",
         "answer": "no", "evidence": [], "category": 5}),
]

# 假模型的脚本化回复，**按"每条 QA 的候选集 × 内层轮数"编排**。候选集由 searchmem
# 混合检索决定，顺序按 atom_key。这里给足回复，多余的调用一律回落成 NOOP。
# 要盯住的行为：
#   - 候选集内**逐条**演化，每条直到 NOOP 或 max_evomem_turn；
#   - 坏输出只结束**这一条候选**的内层循环，候选集里下一条继续；
#   - category 5 的 QA **直接跳过**（不检索、不调模型）。
REPLIES = [
    # QA 0（"What job did Melanie get?"）的候选集：先 UPDATE 一条，再 NOOP；
    # 然后第二条候选：ADD 一条（补充信息），下一轮 NOOP。
    '<UPDATE>[{"memory":"Melanie has kids and a job at Google",'
    '"metadata":{"id":"session_2_1_1","type":"outer","time":"2023-05-01",'
    '"tag":["speaker:Melanie"],"source":["session_2_1"],"changelog":[]}}]</UPDATE>',
    "<NOOP></NOOP>",
    '<ADD>[{"memory":"Melanie works at Google","metadata":{"id":"","type":"outer",'
    '"time":"2023-05-01","tag":["speaker:Melanie"],"source":["session_2_1_1"],'
    '"changelog":[]}}]</ADD>',
    # 坏输出：解析不出任何可执行动作 → 这一轮 EMPTY，但**不终止该 QA**
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

    def __init__(self, *a, **kw) -> None:
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

    # 注入假 QA 列表：真目录名 "Fake_Pair" 不在 correct_locomo10.json 里。
    V.load_dir_questions = (  # type: ignore[assignment]
        lambda label, limit=0: FAKE_QAS[:limit] if limit > 0 else list(FAKE_QAS)
    )

    # 统计召回次数：必须是**每条候选原子一次**，而不是内层每轮一次。
    recall_calls = {"n": 0}
    real_recall = V.recall

    async def counting_recall(query, candidates, config, embedder):
        recall_calls["n"] += 1
        return await real_recall(query, candidates, config, embedder)

    V.recall = counting_recall  # type: ignore[assignment]

    # ---- info 级别 ----
    print("=" * 72)
    print("INFO 级别（应看到每条候选原子每轮结果一行 + 每条 QA 一行汇总）")
    print("=" * 72)
    cfg = dict(V.DEFAULT_CONFIG)
    cfg.update({"top_k": 3, "max_evomem_turn": 4, "context_window": 1,
                "qa_candidates": 5,
                "log": {"level": "info", "output": str(out)}})
    log = evolog.Logger.from_config(cfg)
    summary = asyncio.run(
        V.run_dir(directory, cfg, cfg["generator"], None, FakeEmbedder(), out, log)
    )
    log.close()

    print()
    print("--- 返回的 summary ---")
    print(json.dumps(
        {k: v for k, v in summary.items() if k not in ("traces", "qa_records")},
        ensure_ascii=False, indent=2,
    ))
    print("--- 逐轮明细 ---")
    for t in summary["traces"]:
        print(f"  turn {t['turn']}: qa={t.get('qa_index')} current={t.get('current_id')} "
              f"inner={t.get('inner')} status={t['status']} action={t['action']} "
              f"errors={len(t['errors'])} warnings={len(t['warnings'])}")

    print("--- 断言（QA 驱动语义） ---")
    failed = 0

    def check(label: str, got, expect) -> None:
        nonlocal failed
        ok = got == expect
        if not ok:
            failed += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: 期望 {expect!r} / 实际 {got!r}")

    check("QA 总数", summary["qa_total"], len(FAKE_QAS))
    check("QA 完成数（category 5 被跳过）", summary["qa_done"], 2)
    check("QA 跳过数", summary["qa_skipped"], 1)
    check("traces 都带 qa_index",
          all(t.get("qa_index") in (0, 1) for t in summary["traces"]), True)
    check("category 5 没有产生任何轮次",
          any(t.get("qa_index") == 2 for t in summary["traces"]), False)
    # 每条 QA trajectory 一行
    check("QA trajectory 条数", len(summary["qa_records"]), 2)
    check("每条 trajectory 有候选集与问题",
          all(r["question"] and r["candidate_ids"] for r in summary["qa_records"]), True)
    # 两条 QA 的候选集**不完全相同**（不同问题检索到不同原子）
    check("不同 QA 的候选集可区分",
          summary["qa_records"][0]["candidate_ids"] != summary["qa_records"][1]["candidate_ids"],
          True)
    # 核心性质：候选集内**逐条**演化 —— 同一 QA 下会看到多个不同的 current_id
    qa0_ids = [t["current_id"] for t in summary["traces"] if t.get("qa_index") == 0]
    check("QA 0 有轮次", len(qa0_ids) > 0, True)
    check("召回次数 = 演化过的候选数", recall_calls["n"], summary["atoms_done"])
    check("模型调用次数 = 总轮数", summary["turns"], len(summary["traces"]))
    # 核心性质：坏输出未终止该 QA（该 QA 的轮次里出现 EMPTY 后仍有后续轮次）
    statuses = [t["status"] for t in summary["traces"]]
    check("出现过 EMPTY（坏输出）", "EMPTY" in statuses, True)
    check("坏输出后仍有轮次", statuses[-1] in ("NOOP", "OK"), True)

    print()
    print(f"  改动了 {summary['updated']} 条、新增 {summary['added']} 条、"
          f"错误 {summary['errors']} 个")
    print(f"  {'全部通过' if not failed else f'{failed} 项失败'}")

    # ---- debug 级别：检查数据流是否落盘 ----
    print()
    print("=" * 72)
    print("DEBUG 级别（应看到检索候选/召回/prompt/模型输出/动作解析）")
    print("=" * 72)
    calls["n"] = 0
    cfg["log"] = {"level": "debug", "output": str(out)}
    log = evolog.Logger.from_config(cfg)
    asyncio.run(V.run_dir(directory, cfg, cfg["generator"], None, FakeEmbedder(), out, log))
    path = log.path
    log.close()

    text = Path(path).read_text(encoding="utf-8") if path else ""
    for marker in ("混合检索候选", "召回", "模型原始输出", "动作解析/执行"):
        print(f"  {'ok  ' if marker in text else 'FAIL'} 日志含 {marker!r}")
    print(f"  日志行数 {len(text.splitlines())}，文件 {path}")

    shutil.rmtree(WORK, ignore_errors=True)
    print()
    print("done")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
