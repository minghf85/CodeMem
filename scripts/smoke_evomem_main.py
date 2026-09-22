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
    # 嵌套语义：两条原子各走一遍内层循环（UPDATE → NOOP）。库里最早的那条候选池为空，
    # 但照常调模型（不再跳过），所以两条都会被处理。
    checks.append(("逐轮 status 记录", d.get("turn_status") == ["OK", "NOOP", "OK", "NOOP"],
                   d.get("turn_status")))
    checks.append(("actions 记录", d.get("actions") == ["UPDATE", "NOOP", "UPDATE", "NOOP"],
                   d.get("actions")))
    checks.append(("走完内层循环的原子数", d.get("atoms_done") == 2, d.get("atoms_done")))
    checks.append(("模型调用数 = 轮数", d.get("turns") == len(d.get("turn_status") or []),
                   d.get("turns")))
    checks.append(("traces 未写进 summary（体积）", "traces" not in d))
    checks.append(("演化输出存在", Path(d["output"]).exists()))
    checks.append(("逐轮 trace 存在", Path(d["trace"]).exists()))
    logs = list((WORK / "logs").glob("evomem_*.log"))
    checks.append(("日志落盘", bool(logs), str(logs)))
    if logs:
        body = logs[0].read_text(encoding="utf-8")
        # 日志格式随嵌套循环改成 "current <id>（位置/总数）第 N/M 轮"，
        # 断言按新格式走（旧格式的 "turn 1 OK" 已不存在）。
        checks.append(("日志含每轮结果", "轮：" in body and "turn 1 OK" in body))
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
