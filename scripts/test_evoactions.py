"""解析与执行的快速自测（纯 CPU，不打模型）。

用法：
    python scripts/test_evoactions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codemem import evoactions as E  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    if not condition:
        FAILURES.append(f"{name}: {detail}")
    print(f"  {mark} {name}" + (f"  [{detail}]" if detail and not condition else ""))


def item(memory: str, **meta) -> dict:
    base = {"id": "", "type": "outer", "time": "", "tag": ["speaker:A"], "source": [], "changelog": []}
    base.update(meta)
    return {"memory": memory, "metadata": base}


def payload(*items: dict) -> str:
    return json.dumps(list(items), ensure_ascii=False)


def test_parse_happy_paths() -> None:
    print("parse: happy paths")
    for kind in ("ADD", "UPDATE", "DELETE"):
        text = f"<{kind}>{payload(item('m1'))}</{kind}>"
        action = E.parse_action(text)
        check(f"{kind} kind", action.kind == kind, action.kind)
        check(f"{kind} supplied", action.supplied)
        check(f"{kind} payload size", len(action.payload) == 1, str(len(action.payload)))
        check(f"{kind} no errors", action.errors == [], str(action.errors))
        check(f"{kind} memory kept", action.payload[0]["memory"] == "m1")

    action = E.parse_action("<NOOP></NOOP>")
    check("NOOP kind", action.kind == "NOOP")
    check("NOOP is_noop", action.is_noop)
    check("NOOP empty payload", action.payload == [])
    check("NOOP no errors", action.errors == [])


def test_parse_robustness() -> None:
    print("parse: robustness")
    body = payload(item("m1"))
    cases: list[tuple[str, str, str]] = [
        ("fenced", f"```json\n<ADD>{body}</ADD>\n```", "ADD"),
        ("prose around", f"Let me think.\n<ADD>{body}</ADD>\nDone.", "ADD"),
        # 真·尾随逗号：把数组内最后一个 '}' 后面的 ']' 前插一个逗号
        ("trailing comma", f"<ADD>{body[:-1]},]</ADD>", "ADD"),
        ("wrapped in atoms", f'<ADD>{{"atoms":{body}}}</ADD>', "ADD"),
        ("single object not list", f'<ADD>{json.dumps(item("m1"))}</ADD>', "ADD"),
        ("missing close tag", f"<ADD>{body}", "ADD"),
        ("no tag, bare array",
         json.dumps([{"action": "ADD", **item("m1")}], ensure_ascii=False), "ADD"),
        ("extra non-dict entries", f'<ADD>[1, 2, {json.dumps(item("m1"))}]</ADD>', "ADD"),
        # 模型漏掉外层 '[' 但结尾多一个 ']'
        ("dropped open bracket", f'<ADD>{json.dumps(item("m1"))}]</ADD>', "ADD"),
    ]
    for name, text, want in cases:
        action = E.parse_action(text)
        check(f"{name} -> {want}", action.kind == want, action.kind)
        check(f"{name} payload kept", len(action.payload) == 1, str(len(action.payload)))

    # 全部降级路径都不应崩溃，且必须给出 NOOP 或带错误说明的动作
    for name, text in [
        ("empty output", ""),
        ("prose only", "Nothing needs to change here."),
        ("broken json", '<ADD>[{"memory": "m1",</ADD>'),
        ("empty add", "<ADD>[]</ADD>"),
        ("missing memory", '<ADD>[{"metadata": {"type": "outer"}}]</ADD>'),
    ]:
        action = E.parse_action(text)
        check(f"{name} does not raise", True)
        check(f"{name} safe kind", action.kind in E.ACTIONS, action.kind)

    # 关键：坏输出绝不能变成会修改记忆库的动作
    action = E.parse_action('<ADD>[{"memory": "m1",</ADD>')
    check("broken json yields no executable payload", action.payload == [], str(action.payload))

    # README 里说明标签用法时不该被误判成真实动作（捕获组要求紧跟 '['）
    action = E.parse_action("I will output <ADD> when needed and <DELETE> otherwise. <NOOP></NOOP>")
    check("quoted tag names are not parsed as actions", action.kind == "NOOP", action.kind)

    # 围栏 + 说明文字 + 缺闭合标签组合
    text = "Here you go:\n```\n<UPDATE>" + payload(item("m2", id="x_1")) + "\n```"
    action = E.parse_action(text)
    check("fenced+bare-payload UPDATE", action.kind == "UPDATE", action.kind)
    check("fenced+bare-payload payload", len(action.payload) == 1, str(len(action.payload)))
    check("fenced+bare-payload id", action.payload[0]["metadata"]["id"] == "x_1")


def test_normalize_item() -> None:
    print("normalize: field coercion")
    record = E.normalize_action_item(
        {
            "memory": "  spaced  ",
            "metadata": {
                "id": "a_1",
                "type": "bogus",
                "time": None,
                "tag": ["speaker:A", {"key": "topic", "value": "work"}, "", 42],
                "source": "session_1_1",
                "changelog": [{"time": "t", "content": "c"}, "plain"],
            },
        }
    )
    assert record is not None
    meta = record["metadata"]
    check("memory trimmed", record["memory"] == "spaced")
    check("bad type -> outer", meta["type"] == "outer", meta["type"])
    check("None time -> ''", meta["time"] == "", repr(meta["time"]))
    check("dict tag -> k:v", "topic:work" in meta["tag"], str(meta["tag"]))
    check("empty tag dropped", all(t for t in meta["tag"]), str(meta["tag"]))
    check("id kept", meta["id"] == "a_1", meta["id"])
    check("source str -> list", meta["source"] == ["session_1_1"], str(meta["source"]))
    check("changelog normalized", all(isinstance(c, dict) for c in meta["changelog"]))
    check("no memory -> None", E.normalize_action_item({"metadata": {}}) is None)
    check("blank memory -> None", E.normalize_action_item({"memory": "   "}) is None)


def test_apply_add() -> None:
    print("apply: ADD")
    memories: list[dict] = []
    active: set[str] = set()
    action = E.parse_action(f"<ADD>{payload(item('new fact A'), item('new fact B'))}</ADD>")
    result = E.apply_action(action, memories, active, "session_9_9")

    check("two added", len(result.added_ids) == 2, str(result.added_ids))
    check("ids anchored", result.added_ids == ["session_9_9_1", "session_9_9_2"], str(result.added_ids))
    check("active updated", active == set(result.added_ids), str(active))
    check("changelog stamped", result.memories[0]["metadata"]["changelog"][0]["content"] == "created")
    check("original untouched", memories == [])

    # 连跑：计数器不得与已有 id 冲突
    result2 = E.apply_action(
        E.parse_action(f"<ADD>{payload(item('new fact C'))}</ADD>"), result.memories, active, "session_9_9"
    )
    check("counter continues", result2.added_ids == ["session_9_9_3"], str(result2.added_ids))

    # anchor 自身的 id 不能与新增的 _1 冲突（真实 atommem 的 id 形如 session_1_1_1）
    # 注意 dict 合并顺序：id 必须在 **meta 之后，否则会被 item() 里的空 id 覆盖。
    seeded = [item("anchor", id="session_9_9_1")]
    result3 = E.apply_action(
        E.parse_action(f"<ADD>{payload(item('another'))}</ADD>"), seeded, {"session_9_9_1"}, "session_9_9"
    )
    check("no id collision with existing", result3.added_ids == ["session_9_9_2"], str(result3.added_ids))

    # 重复文本被跳过：库里已有 "new fact C"，再 ADD 两条同文本应当全被拒绝
    dup = E.parse_action(f"<ADD>{payload(item('new fact C'), item('new fact C'))}</ADD>")
    result4 = E.apply_action(dup, result2.memories, active, "session_9_9")
    check("duplicates all skipped", result4.added_ids == [], str(result4.added_ids))
    check("duplicate reported", sum("duplicate" in e for e in result4.errors) == 2, str(result4.errors))

    # 同一批里重复、但库里没有的文本：只保留第一条
    fresh_dup = E.parse_action(f"<ADD>{payload(item('brand new fact'), item('brand new fact'))}</ADD>")
    result5 = E.apply_action(fresh_dup, result2.memories, active, "session_9_9")
    check("same-batch duplicate keeps first", len(result5.added_ids) == 1, str(result5.added_ids))
    check("same-batch duplicate reported", any("duplicate" in e for e in result5.errors),
          str(result5.errors))


def test_apply_update() -> None:
    print("apply: UPDATE")
    memories = [
        {"memory": "old text one", "metadata": {"id": "s_1", "type": "outer", "time": "2023-01-01",
                                                "tag": ["speaker:A"], "source": ["session_1_1"],
                                                "changelog": []}},
        {"memory": "keep me", "metadata": {"id": "s_2", "type": "outer", "time": "",
                                           "tag": ["speaker:A"], "source": [], "changelog": []}},
    ]
    active = {"s_1", "s_2"}
    action = E.parse_action(f"<UPDATE>{payload(item('new text one', id='s_1', time='2023-05-08'))}</UPDATE>")
    result = E.apply_action(action, memories, active, "s")

    check("updated id", result.updated_ids == ["s_1"], str(result.updated_ids))
    updated = result.memories[0]
    check("position preserved", updated["metadata"]["id"] == "s_1")
    check("memory replaced", updated["memory"] == "new text one", updated["memory"])
    check("changelog archives old", updated["metadata"]["changelog"][0]["content"] == "old text one",
          str(updated["metadata"]["changelog"]))
    check("changelog archives old time",
          any(c["content"] == "time: 2023-01-01" for c in updated["metadata"]["changelog"]),
          str(updated["metadata"]["changelog"]))
    check("source union", updated["metadata"]["source"] == ["session_1_1"], str(updated["metadata"]["source"]))
    check("other entry untouched", result.memories[1] == memories[1])
    check("original untouched", memories[0]["memory"] == "old text one")

    # 未知 id / 已删除 / 无变化
    unknown = E.apply_action(E.parse_action(f"<UPDATE>{payload(item('x', id='nope'))}</UPDATE>"),
                             memories, active, "s")
    check("unknown id rejected", unknown.updated_ids == [], str(unknown.updated_ids))
    check("unknown id error", any("unknown id" in e for e in unknown.errors), str(unknown.errors))

    deleted_active = {"s_2"}
    dead = E.apply_action(E.parse_action(f"<UPDATE>{payload(item('x', id='s_1'))}</UPDATE>"),
                          memories, deleted_active, "s")
    check("deleted id rejected", dead.updated_ids == [], str(dead.updated_ids))

    noop_like = E.apply_action(E.parse_action(f"<UPDATE>{payload(item('old text one', id='s_1'))}</UPDATE>"),
                               memories, active, "s")
    check("unchanged update skipped", noop_like.updated_ids == [], str(noop_like.updated_ids))
    check("unchanged reported", any("unchanged" in e for e in noop_like.errors), str(noop_like.errors))

    missing_id = E.apply_action(E.parse_action(f"<UPDATE>{payload(item('x'))}</UPDATE>"), memories, active, "s")
    check("missing id rejected", missing_id.updated_ids == [], str(missing_id.updated_ids))


def test_apply_delete() -> None:
    print("apply: DELETE")
    memories = [
        {"memory": "a", "metadata": {"id": "s_1", "type": "outer", "time": "", "tag": [], "source": [], "changelog": []}},
        {"memory": "b", "metadata": {"id": "s_2", "type": "outer", "time": "", "tag": [], "source": [], "changelog": []}},
    ]
    active = {"s_1", "s_2"}
    action = E.parse_action(f"<DELETE>{payload(item('a', id='s_1'))}</DELETE>")
    result = E.apply_action(action, memories, active, "s")

    check("deleted id", result.deleted_ids == ["s_1"], str(result.deleted_ids))
    check("active shrunk", active == {"s_2"}, str(active))
    check("record retained (soft delete)", len(result.memories) == 2, str(len(result.memories)))
    check("original untouched", "s_1" in {"s_1", "s_2"})

    again = E.apply_action(action, result.memories, active, "s")
    check("double delete rejected", again.deleted_ids == [], str(again.deleted_ids))

    unknown = E.apply_action(E.parse_action(f"<DELETE>{payload(item('z', id='nope'))}</DELETE>"),
                             memories, {"s_1", "s_2"}, "s")
    check("unknown delete rejected", unknown.deleted_ids == [], str(unknown.deleted_ids))


def test_verify_and_active() -> None:
    print("verify: payload content vs stored")
    memories = [{"memory": "stored text", "metadata": {"id": "s_1", "type": "outer", "time": "",
                                                       "tag": [], "source": [], "changelog": []}}]
    action = E.parse_action(f"<DELETE>{payload(item('stored text', id='s_1'))}</DELETE>")
    check("matching content -> no warning",
          E.verify_payload_matches(action.payload, memories, "DELETE") == [])

    action = E.parse_action(f"<DELETE>{payload(item('WRONG text', id='s_1'))}</DELETE>")
    warnings = E.verify_payload_matches(action.payload, memories, "DELETE")
    check("mismatched content -> warning", len(warnings) == 1, str(warnings))

    check("active filter", len(E.current_active(memories, {"s_1"})) == 1)
    check("active filter excludes", len(E.current_active(memories, set())) == 0)
    check("active None keeps all", len(E.current_active(memories)) == 1)


def test_apply_noop() -> None:
    print("apply: NOOP")
    memories = [{"memory": "a", "metadata": {"id": "s_1", "type": "outer", "time": "",
                                             "tag": [], "source": [], "changelog": []}}]
    action = E.parse_action("<NOOP></NOOP>")
    result = E.apply_action(action, memories, {"s_1"}, "s")
    check("no ids", result.added_ids == result.updated_ids == result.deleted_ids == [])
    check("memories equal", result.memories == memories)
    check("no errors", result.errors == [], str(result.errors))


def test_end_to_end_turn() -> None:
    """模拟完整的一轮：ADD 后 UPDATE 该条，再 DELETE 它。"""
    print("integration: turn sequence")
    memories: list[dict] = []
    active: set[str] = set()

    r1 = E.apply_action(E.parse_action(f"<ADD>{payload(item('Melanie has two kids'))}</ADD>"),
                        memories, active, "session_1_1")
    memories = r1.memories
    new_id = r1.added_ids[0]
    check("added", new_id == "session_1_1_1", new_id)

    r2 = E.apply_action(
        E.parse_action(f"<UPDATE>{payload(item('Melanie has two kids, Alan and Beth', id=new_id, time='2023-05-08'))}</UPDATE>"),
        memories, active, "session_1_1")
    memories = r2.memories
    check("updated", r2.updated_ids == [new_id], str(r2.updated_ids))
    check("changelog grew", len(memories[0]["metadata"]["changelog"]) == 2,
          str(memories[0]["metadata"]["changelog"]))

    r3 = E.apply_action(
        E.parse_action(f"<DELETE>{payload(item('Melanie has two kids, Alan and Beth', id=new_id))}</DELETE>"),
        memories, active, "session_1_1")
    memories = r3.memories
    check("deleted", r3.deleted_ids == [new_id], str(r3.deleted_ids))
    check("active empty", active == set(), str(active))
    check("delete content matched", E.verify_payload_matches(r3.memories and E.parse_action(
        f"<DELETE>{payload(item('Melanie has two kids, Alan and Beth', id=new_id))}</DELETE>").payload,
        memories, "DELETE") == [], "content should still match stored value")


def main() -> None:
    for test in (
        test_parse_happy_paths,
        test_parse_robustness,
        test_normalize_item,
        test_apply_add,
        test_apply_update,
        test_apply_delete,
        test_verify_and_active,
        test_apply_noop,
        test_end_to_end_turn,
    ):
        test()
        print()

    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
