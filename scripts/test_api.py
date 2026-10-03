#!/usr/bin/env python
"""封装 API 的纯 CPU 自测：假模型 + 真工具 / 真存储，零模型成本、零网络。

覆盖挑战赛协议里最容易出错、也最该被钉住的几件事：

1. **Add 幂等**：同一个 ``request_id`` 送两次，语料只写一次（协议硬要求）。
2. **多 user 隔离**：不同 ``user_id`` 的语料互不可见。
3. **msg_id / 时间**：分配 ``session_{n}_{i}``；``timestamp``（毫秒）转 ISO-8601 UTC。
4. **Search 产物映射**：evidence 记录 → 协议 ``data[]``（字段齐全、按 score 降序、top_k 截断、
   无证据 → ``[]``）。
5. **多模态归一化**：有序 ``ContentPart[]`` → 可 grep 的文本（图片降级成 ``[image: ...]``）。

做法：把 ``codemem.llm.chat_completion`` 换成一个**脚本化**的假实现 —— 它按 system prompt 判断
这次调用是 agent / verifier / index，返回预先写好的回复。工具、存储、agent loop、verifier
编排、evidence 校验全部是**真的**，只有"模型"是假的。

    python scripts/test_api.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import codemem.llm as llm_mod  # noqa: E402
from codemem.api import retriever, store as store_mod  # noqa: E402
from codemem.api.models import normalize_content  # noqa: E402
from codemem.api.server import create_app  # noqa: E402

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {label}")
    else:
        FAIL += 1
        print(f"  ✗ {label} {detail}")


# ---------------------------------------------------------------------------
# 假模型：按 system prompt 分发
# ---------------------------------------------------------------------------

_EVIDENCE_RECORD = {
    "content": "Caroline went to an LGBTQ support group on 7 May 2023 "
               "(she said 'yesterday' in the session of 8 May 2023).",
    "metadata": {"source": ["session_1_1"], "score": 0.9},
}


async def fake_chat_completion(client, config, messages):  # noqa: ANN001
    """脚本化的模型：按 system prompt 找回应的那一类。

    必须是 **async**（真实现是 async；这里替换它，签名要一致）。

    - verifier（"evidence auditor"）→ 判足够
    - resolve（"You rewrite ONE message"）→ 消解后的 content/time
    - agent（默认）→ 第一轮写 evidence.jsonl，之后回纯文本表示完成
    """
    system = str(messages[0].get("content", "")) if messages else ""

    if "evidence auditor" in system:
        return json.dumps({
            "reasoning": "The record states the date.",
            "answer": "7 May 2023",
            "missing": [],
            "sufficient": True,
        })

    if "You rewrite ONE message" in system:
        return json.dumps({
            "content": "Caroline went to a LGBTQ support group on 7 May 2023.",
            "time": "7 May 2023", "time_kind": "point", "time_raw": "yesterday",
        })

    # agent loop: 第一轮 append 一条证据，之后回纯文本表示完成
    has_assistant = any(m.get("role") == "assistant" for m in messages)
    if not has_assistant:
        body = json.dumps(_EVIDENCE_RECORD, ensure_ascii=False) + "\n"
        return json.dumps({"tool": "edit", "args": {
            "path": "evidence.jsonl", "append": body}})
    return "The evidence already answers the question."   # 纯文本 = 完成信号


def _install_fake_model() -> None:
    llm_mod.chat_completion = fake_chat_completion  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 各段测试
# ---------------------------------------------------------------------------

def test_normalize_content() -> None:
    print("\n[1] 多模态 content 归一化")
    check("字符串原样", normalize_content("hello") == "hello")
    parts = [
        {"type": "text", "text": "first"},
        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
        {"type": "text", "text": "last"},
    ]
    got = normalize_content(parts)
    check("有序拼接且图片降级", got == "first\n[image: http://x/y.png]\nlast", repr(got))
    check("空数组 -> 空串", normalize_content([]) == "")
    check("None -> 空串", normalize_content(None) == "")


def test_iso_from_ms() -> None:
    print("\n[2] timestamp 毫秒 -> ISO")
    got = store_mod._iso_from_ms(1683561600000)   # 2023-05-08T16:00:00Z
    check("转成 ISO-8601 UTC", got == "2023-05-08T16:00:00Z", got)
    check("缺失 -> 空串", store_mod._iso_from_ms(None) == "")
    check("非法 -> 空串", store_mod._iso_from_ms("nope") == "")


def test_slug() -> None:
    print("\n[3] user_id -> 目录名")
    a = store_mod.slug("eval:run1:locomo:conv-0")
    b = store_mod.slug("eval/run1/locomo/conv-0")
    check("路径不安全字符被替换", ":" not in a and "/" not in a)
    check("不同 user_id 不撞目录", a != b)


def test_records_to_data() -> None:
    print("\n[4] evidence 记录 -> 协议 data[]")
    records = [
        {"content": "low", "metadata": {"source": ["s1"], "score": 0.2}},
        {"content": "high", "metadata": {"source": ["s2"], "score": 0.9,
                                          "changelog": [{"time": "2026-01-01T00:00:00Z",
                                                         "content": "created"}]}},
        {"content": "mid", "metadata": {"source": ["s3"], "score": 0.5}},
    ]
    data = retriever.records_to_data(records)
    check("按 score 降序", [d["content"] for d in data] == ["high", "mid", "low"])
    check("字段齐全", all({"id", "content", "score", "created_at"} <= set(d) for d in data))
    check("id 稳定且带前缀", data[0]["id"].startswith("mem_") and len(data[0]["id"]) == 16)
    check("created_at 取自 changelog", data[0]["created_at"] == "2026-01-01T00:00:00Z")
    check("相同内容 -> 相同 id", retriever.records_to_data(records)[0]["id"] == data[0]["id"])
    check("top_k 截断", len(retriever.records_to_data(records, top_k=2)) == 2)
    check("空记录 -> []", retriever.records_to_data([]) == [])


def _app_with_temp_store(tmp: Path):
    config = {
        "store_dir": str(tmp / "store"),
        "tool_config": str(PROJECT_ROOT / "configs" / "tool.json"),
        "generator": {"base_url": "http://127.0.0.1:1/v1", "model": "fake"},
        "index": {"base_url": "http://127.0.0.1:1/v1", "model": "fake"},
        "top_k": 100,
        "concurrency": 4,
        "add_concurrency": 4,
        "max_steps": 6,
        "max_verify_rounds": 2,
        "log": {"level": "error", "output": ""},
    }
    return create_app(config)


def test_endpoints(tmp: Path) -> None:
    from fastapi.testclient import TestClient

    print("\n[5] HTTP 端点（Add / Search）")
    app = _app_with_temp_store(tmp)
    client = TestClient(app)

    check("health", client.get("/health").json() == {"status": "ok"})

    add_body = {
        "request_id": "eval:run1:locomo:conv-0:chunk-0",
        "user_id": "eval:run1:locomo:conv-0",
        "session_id": "eval:run1:sample:0",
        "messages": [
            {"role": "user", "content": "I went to a LGBTQ support group yesterday",
             "timestamp": 1683561600000},
            {"role": "assistant", "content": "That sounds meaningful."},
        ],
    }
    resp = client.post("/add", json=add_body)
    check("Add 返回 200", resp.status_code == 200, resp.text)
    body = resp.json()
    check("success=true", body["success"] is True)
    check("原样回显三个 id",
          body["request_id"] == add_body["request_id"]
          and body["user_id"] == add_body["user_id"]
          and body["session_id"] == add_body["session_id"])

    store = store_mod.UserStore(add_body["user_id"], Path(app.state.api.store_root))
    sessions_after_first = store.sessions()
    check("写入了 2 条消息", len(sessions_after_first) == 2)
    check("msg_id 形如 session_1_1", sessions_after_first[0]["msg_id"] == "session_1_1")
    check("原始会话时间锚点保留在 source_time",
          sessions_after_first[0]["source_time"] == "2023-05-08T16:00:00Z")
    check("消息已被消解（content 是消解结果）",
          "7 May 2023" in sessions_after_first[0]["content"]
          and sessions_after_first[0]["time"] == "7 May 2023"
          and sessions_after_first[0]["time_kind"] == "point",
          str(sessions_after_first[0]))
    check("原话保留在 source_content",
          sessions_after_first[0]["source_content"] == "I went to a LGBTQ support group yesterday")
    check("消息落在 sessions/session_1.jsonl",
          (store.root / "sessions" / "session_1.jsonl").exists())

    # 幂等：同一个 request_id 再送一次
    resp2 = client.post("/add", json=add_body)
    check("重放仍返回 200", resp2.status_code == 200)
    store2 = store_mod.UserStore(add_body["user_id"], Path(app.state.api.store_root))
    check("幂等：语料未被重复写入", len(store2.sessions()) == 2, str(len(store2.sessions())))
    check("幂等：消解语料也未变", len(store2.sessions()) == 2)

    # 同一 session 再追加（新 request_id）→ 序号接续
    follow = dict(add_body, request_id="eval:run1:locomo:conv-0:chunk-1")
    follow["messages"] = [{"role": "user", "content": "Another message"}]
    client.post("/add", json=follow)
    store3 = store_mod.UserStore(add_body["user_id"], Path(app.state.api.store_root))
    ids = [r["msg_id"] for r in store3.sessions()]
    check("追加消息序号接续", ids == ["session_1_1", "session_1_2", "session_1_3"], str(ids))

    # 多 user 隔离
    other = dict(add_body, request_id="other-1", user_id="other-user", session_id="s-other")
    client.post("/add", json=other)
    other_store = store_mod.UserStore("other-user", Path(app.state.api.store_root))
    check("多 user 隔离：各写各的",
          len(other_store.sessions()) == 2 and len(store3.sessions()) == 3)

    # Search：跑真 agent loop + 假模型
    resp = client.post("/search", json={
        "query": "When did Caroline go to the LGBTQ support group?",
        "user_id": add_body["user_id"],
        "top_k": 100,
    })
    check("Search 返回 200", resp.status_code == 200, resp.text)
    data = resp.json()["data"]
    check("data 非空且字段齐全",
          len(data) >= 1 and {"id", "content", "score", "created_at"} <= set(data[0]))
    check("content 非空字符串", isinstance(data[0]["content"], str) and data[0]["content"].strip())
    check("按 score 降序",
          all((data[i]["score"] or 0) >= (data[i + 1]["score"] or 0) for i in range(len(data) - 1)))

    # Search：不存在的 user → 空 data（且不报错）
    resp = client.post("/search", json={"query": "anything", "user_id": "nobody", "top_k": 100})
    check("无记忆 user -> data=[]", resp.status_code == 200 and resp.json()["data"] == [])

    # 协议：data 字段永不被省略
    check("响应含 data 键", "data" in resp.json())


def test_validation(tmp: Path) -> None:
    from fastapi.testclient import TestClient

    print("\n[6] 请求校验")
    app = _app_with_temp_store(tmp)
    client = TestClient(app)
    # 缺 user_id
    resp = client.post("/search", json={"query": "x"})
    check("缺必填字段 -> 422", resp.status_code == 422, str(resp.status_code))
    # 缺 messages
    resp = client.post("/add", json={"request_id": "r", "user_id": "u", "session_id": "s"})
    check("Add 缺 messages -> 422", resp.status_code == 422, str(resp.status_code))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    _install_fake_model()
    test_normalize_content()
    test_iso_from_ms()
    test_slug()
    test_records_to_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        test_endpoints(tmp)
        test_validation(tmp)

    print(f"\n{'=' * 60}\n通过 {PASS} / 失败 {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
