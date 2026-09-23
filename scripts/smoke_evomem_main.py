"""验证 evomem 的 async_main：多轮上报、汇总、并发、log 配置。纯 CPU。

跑法：python scripts/smoke_evomem_main.py
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codemem import atommem, evomem as V  # noqa: E402

WORK = PROJECT_ROOT / ".smoke_main"
OUT = WORK / "runs"

PAIRS = {
    "Pair_One": [
        ("session_1_1", "A", "Hello there my friend."),
        ("session_1_2", "B", "I got a new job last month."),
    ],
    "Pair_Two": [
        ("session_1_1", "C", "Nice to meet you today."),
        ("session_1_2", "D", "I moved to Berlin last week."),
    ],
}


def seed() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    OUT.mkdir(parents=True)
    for name, raws in PAIRS.items():
        d = WORK / name
        d.mkdir(parents=True)
        with (d / "msgmem.jsonl").open("w", encoding="utf-8") as f:
            for rid, spk, text in raws:
                f.write(json.dumps({
                    "memory": text,
                    "metadata": {"id": rid, "type": "raw", "time": "2023-05-08T13:56:00",
                                 "tag": [f"speaker:{spk}"], "source": [], "changelog": []},
                }, ensure_ascii=False) + "\n")
        with (d / "atommem.jsonl").open("w", encoding="utf-8") as f:
            # 两条原子：第一条在库里最早，没有更早的证据 → 会被跳过（不调模型）；
            # 第二条才有可召回的更早记忆，是真正会走内层循环的那条。
            # 只有一条原子的话整库都会被跳过，测不到任何模型调用/轮次上报。
            for idx, (rid, memory) in enumerate([
                (raws[0][0], f"{raws[0][1]} greeted {raws[1][1]}"),
                (raws[1][0], f"{spk} got a new job last month"),
            ]):
                f.write(json.dumps({
                    "memory": memory,
                    "metadata": {"id": f"{rid}_1", "type": "outer", "time": "last month",
                                 "tag": [f"speaker:{spk}"], "source": [rid], "changelog": []},
                }, ensure_ascii=False) + "\n")


class FakeEmbedder:
    def __init__(self, *a, **kw) -> None:
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        if not texts:
            return []
        return [[float(len(t) % 5), 1.0, 0.0] for t in texts]

    def stats(self):
        return f"fake embedder 调用 {self.calls} 次"


def main() -> None:
    seed()
    calls = {"n": 0}

    async def fake_chat(client, config, messages):
        """每条的奇数轮 UPDATE（精确命中 ## Target entry 的 id），偶数轮 NOOP。"""
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            import re
            # 严格取 "## Target entry" 下面那一行的 id —— 不能拿 prompt 里出现的第一个
            # id，那可能是证据条目；也不能拿 ids[1]，空候选池时它根本不存在。
            m = re.search(r"## Target entry\s*\n\s*\[(\S+?)\]", messages[0]["content"])
            target = m.group(1) if m else "session_1_2_1"
            return (
                f'<UPDATE>[{{"memory":"X started a new job in May 2023",'
                f'"metadata":{{"id":"{target}","type":"outer","time":"2023-05-01",'
                f'"tag":["speaker:X"],"source":[],"changelog":[]}}}}]</UPDATE>'
            )
        return "<NOOP></NOOP>"

    atommem.chat_completion = fake_chat  # type: ignore[assignment]
    V.Embedder = FakeEmbedder  # type: ignore[assignment]

    # QA 驱动后 run_dir 需要问题数据源。假目录没有对应的 sample，所以注入一个假的
    # QA 列表 —— 否则 run_dir 会直接把这个目录标成 SKIPPED，什么都测不到。
    # 每条 QA 只放一个问题；候选检索（FakeEmbedder + BM25/tag）会从两条原子里取候选。
    fake_qas = {
        "Pair_One": [(0, {"question": "Where did A get a new job?", "answer": "Berlin", "evidence": ["D1:2"], "category": 1})],
        "Pair_Two": [(0, {"question": "Where did C move?", "answer": "Berlin", "evidence": ["D1:2"], "category": 1})],
    }
    V.load_dir_questions = (  # type: ignore[assignment]
        lambda label, limit=0: (fake_qas.get(label) or [])[:limit] if limit > 0 else (fake_qas.get(label) or [])
    )

    args = V.parse_args([
        "--sample", str(WORK / "Pair_One"), "--output-dir", str(OUT),
        "--experiment", "smoke", "--concurrency", "1",
        "--log-level", "info", "--log-output", str(WORK / "logs"),
    ])
    asyncio.run(V.async_main(args))

    runs = sorted(OUT.glob("smoke_*"))
    run_dir = runs[-1]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    print()
    print("=" * 72)
    print("检查")
    print("=" * 72)
    checks = []
    checks.append(("summary.json 生成", (run_dir / "summary.json").exists()))
    checks.append(("counts 字段", "counts" in summary))
    checks.append(("log_path 记录", summary.get("log_path") is not None))
    d = summary["dirs"][0]
    checks.append(("目录状态 OK", d.get("status") == "OK", d.get("status")))
    # QA 驱动语义：1 条 QA 的候选集里有原子（FakeEmbedder 对任何 query 都给分，
    # BM25/tag 也会命中），每条候选各走一遍内层循环（UPDATE → NOOP）。
    checks.append(("QA 总数记录", d.get("qa_total") == 1, d.get("qa_total")))
    checks.append(("QA 完成数", d.get("qa_done") == 1, d.get("qa_done")))
    checks.append(("逐轮 status 交错(OK/NOOP 成对)", len(d.get("turn_status") or []) % 2 == 0,
                   d.get("turn_status")))
    checks.append(("actions 与 status 等长", len(d.get("actions") or []) == len(d.get("turn_status") or [])))
    checks.append(("演化过至少一个候选", (d.get("atoms_done") or 0) >= 1, d.get("atoms_done")))
    checks.append(("模型调用数 = 轮数", d.get("turns") == len(d.get("turn_status") or []),
                   d.get("turns")))
    checks.append(("traces 未写进 summary（体积）", "traces" not in d))
    checks.append(("演化输出存在", Path(d["output"]).exists()))
    checks.append(("逐轮 trace 存在", Path(d["trace"]).exists()))
    checks.append(("QA trajectory 存在", Path(d["qa_trajectories"]).exists()))
    logs = list((WORK / "logs").glob("evomem_*.log"))
    checks.append(("日志落盘", bool(logs), str(logs)))
    if logs:
        body = logs[0].read_text(encoding="utf-8")
        # 日志格式随 QA 驱动改成 "QA <idx> 候选 i/K current=<id> 第 N/M 轮"
        checks.append(("日志含每轮结果", "轮：" in body and "turn 1 OK" in body))
        checks.append(("日志含 QA 行", "QA 0/" in body))
        checks.append(("日志含汇总", "全部结束" in body))

    bad = 0
    for item in checks:
        name, ok, *rest = item
        detail = f"  [{rest[0]}]" if rest and not ok else ""
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{detail}")
        bad += 0 if ok else 1

    print()
    print("--- 终端看到的汇总行 ---")
    for line in (logs[0].read_text(encoding="utf-8").splitlines() if logs else []):
        if "全部结束" in line or line.strip().startswith("[") and "  OK" in line:
            print("   ", line)

    shutil.rmtree(WORK, ignore_errors=True)
    print()
    print(f"{bad} failure(s)" if bad else "all checks passed")


if __name__ == "__main__":
    main()
