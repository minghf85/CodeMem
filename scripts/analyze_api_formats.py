"""统计 ``data/api_store/*/requests.jsonl`` 里各数据集的报文形态，并**验证适配层认不认**。

用法::

    python scripts/analyze_api_formats.py                     # 统计 + 抽样
    python scripts/analyze_api_formats.py --samples 3          # 每类多看几条
    python scripts/analyze_api_formats.py --json out.json      # 落一份机器可读的统计

它回答的是"平台到底喂了些什么形状"：

1. 每个 user 的**第一个** ``/add`` 的前两条 message 长什么样（协议 role、正文前缀、标签）；
2. 按 ``codemem.adapters`` 判出来的**形态**分类统计（对话 / 剧本 / 文档 / 三元组 / 流式记录 …）；
3. 每类抽几条，**前后对照**打印（原始正文 → 适配后的 role + 正文），人眼可核。

第 3 条最重要：适配层改的是"能不能 grep 到人名/时间"，不是"数字好不好看"。**抽样的前后
对照就是它的证据**，所以默认一定会打印，而不是只在 ``--verbose`` 时打印。
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codemem.adapters import adapt_messages, split_lead_tags, split_speaker  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = PROJECT_ROOT / "data" / "api_store"


def load_requests(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def head_prefix(content: str) -> str:
    """正文开头那个"说话人前缀"——用**适配层自己的**判断，统计与预处理看到的才是同一件事。"""
    _, body = split_lead_tags(content)
    speaker, _ = split_speaker(body.lstrip())
    if speaker is not None:
        return f"{speaker}:"
    return "(no speaker) " + body.lstrip()[:30].replace("\n", " | ")


def collect(store: Path) -> dict:
    users = sorted(p for p in store.iterdir() if p.is_dir())
    first_adds: dict[str, dict] = {}
    all_adds: list[tuple[str, dict]] = []
    searches: list[tuple[str, dict]] = []
    for user in users:
        records = load_requests(user / "requests.jsonl")
        for record in records:
            if record.get("endpoint") == "add":
                all_adds.append((user.name, record))
                first_adds.setdefault(user.name, record)
            elif record.get("endpoint") == "search":
                searches.append((user.name, record))
    return {
        "users": users,
        "first_adds": first_adds,
        "all_adds": all_adds,
        "searches": searches,
    }


def report(data: dict, samples: int, as_json: Path | None) -> None:
    first_adds = data["first_adds"]
    print(f"user 目录总数        : {len(data['users'])}")
    print(f"有 requests.jsonl 的 : {len({u for u, _ in data['all_adds']})}")
    print(f"add 请求总数         : {len(data['all_adds'])}")
    print(f"search 请求总数      : {len(data['searches'])}")

    # ---- 1. 第一个 add 的前两条：协议 role × 正文前缀 ----
    roles = collections.Counter()
    heads = collections.Counter()
    nmsg = collections.Counter()
    for record in first_adds.values():
        messages = record.get("messages") or []
        nmsg[len(messages)] += 1
        for message in messages[:2]:
            roles[str(message.get("role"))] += 1
            content = message.get("content")
            if not isinstance(content, str):
                continue
            heads[head_prefix(content)] += 1

    print("\n— 第一个 add 的前两条 message —")
    print("  协议 role 分布 :", dict(roles.most_common()))
    print("  首条 add 的消息条数 :", dict(nmsg.most_common(8)))
    print("  正文前缀 top 24 :")
    for head, count in heads.most_common(24):
        print(f"    {count:5d}  {head!r}")

    # ---- 2. 形态分类 ----
    fam_first = collections.Counter()
    fam_all = collections.Counter()
    fam_users: dict[str, set[str]] = collections.defaultdict(set)
    for user, record in first_adds.items():
        batch = adapt_messages(record.get("messages") or [])
        fam_first[batch.family] += 1
        fam_users[batch.family].add(user)
    for _, record in data["all_adds"]:
        fam_all[adapt_messages(record.get("messages") or []).family] += 1

    print("\n— 适配层判出的形态 —")
    print(f"  {'形态':<13}{'user 数':>8}{'首个 add':>10}{'全部 add':>10}")
    for family, users in sorted(fam_users.items(), key=lambda kv: -len(kv[1])):
        print(f"  {family:<13}{len(users):>8}{fam_first[family]:>10}{fam_all[family]:>10}")

    # ---- 3. 前后对照抽样 ----
    for family in sorted(fam_users, key=lambda f: -len(fam_users[f])):
        shown = 0
        print(f"\n{'=' * 96}\n### {family}  ({len(fam_users[family])} users)")
        for user, record in first_adds.items():
            if adapt_messages(record.get("messages") or []).family != family:
                continue
            if shown >= samples:
                break
            shown += 1
            batch = adapt_messages(record.get("messages") or [])
            print(f"-- {user[:34]}  speakers={batch.speakers or '(protocol roles)'}  "
                  f"nmsgs={len(batch.messages)}")
            for index, (raw, adapted) in enumerate(
                zip(record.get("messages") or [], batch.messages)
            ):
                if index >= 2:
                    break
                raw_content = raw.get("content")
                if not isinstance(raw_content, str):
                    raw_content = json.dumps(raw_content, ensure_ascii=False)
                print(f"   [{index}] 原始 role={raw.get('role')!r} ts={raw.get('timestamp')}")
                print(f"        {raw_content[:190]!r}")
                print(f"       → role={adapted.role!r} ts={adapted.timestamp}")
                print(f"        {adapted.content[:190]!r}")

    if as_json is not None:
        payload = {
            "families_first_add": dict(fam_first),
            "families_all_adds": dict(fam_all),
            "family_users": {k: sorted(v) for k, v in fam_users.items()},
            "protocol_roles": dict(roles),
            "content_heads": dict(heads),
        }
        as_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n统计已写入 {as_json}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 api_store 的报文形态")
    parser.add_argument("--store", default=str(DEFAULT_STORE), help="api_store 目录")
    parser.add_argument("--samples", type=int, default=1, help="每类抽样几个 user")
    parser.add_argument("--json", default=None, help="把统计写进这个 JSON 文件")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    store = Path(args.store)
    if not store.exists():
        print(f"找不到 {store}", file=sys.stderr)
        return
    data = collect(store)
    if not data["all_adds"]:
        print(f"{store} 下没有任何 add 请求", file=sys.stderr)
        return
    report(data, max(1, args.samples), Path(args.json) if args.json else None)


if __name__ == "__main__":
    main()
