#!/usr/bin/env python
"""封装 API 的纯 CPU 自测：假模型 + 真工具 / 真存储，零模型成本、零网络。

覆盖挑战赛协议里最容易出错、也最该被钉住的几件事：

1. **Add 幂等**：同一个 ``request_id`` 送两次，语料只写一次（协议硬要求）。
2. **多 user 隔离**：不同 ``user_id`` 的语料互不可见。
3. **msg_id / 时间**：分配 ``session_{n}_{i}``；``timestamp``（毫秒）转 ISO-8601 UTC。
4. **Search 产物映射**：evidence 记录 → 协议 ``data[]``（字段齐全、按 score 降序、top_k 截断、
   无证据 → ``[]``）。
5. **多模态归一化**：有序 ``ContentPart[]`` → 可 grep 的文本（图片降级成 ``[image: ...]``）。
6. **中断续跑**：写前日志（WAL）保证"被 kill 后不丢不重" —— 刷新失败/硬中断的 Add 留在
   ``meta.pending``，重发同一 ``request_id`` 不重复写，重启（新 ``UserStore`` 实例）或
   ``recover_all`` 能独立补完。

做法：把 ``codemem.llm.chat_completion`` 换成一个**脚本化**的假实现 —— 它按 system prompt 判断
这次调用是 agent / verifier / index，返回预先写好的回复。工具、存储、agent loop、verifier
编排、evidence 校验全部是**真的**，只有"模型"是假的。

    python scripts/test_api.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import codemem.llm as llm_mod  # noqa: E402
from codemem import adapters  # noqa: E402
from codemem.io import read_jsonl  # noqa: E402
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
    - summem（"You summarize ONE session"）→ 一段会话总结
    - atommem **pipeline**（"extracting atomic facts"）→ 一个 JSON 数组（time_expr 交给代码解析）
    - search agent（"You build the evidence"）→ 写 evidence.jsonl
    """
    system = str(messages[0].get("content", "")) if messages else ""

    if "evidence auditor" in system:
        return json.dumps({
            "reasoning": "The record states the date.",
            "answer": "7 May 2023",
            "missing": [],
            "sufficient": True,
        })

    if "You summarize ONE session" in system:
        return json.dumps({
            "memory": "Caroline went to a LGBTQ support group on 7 May 2023.",
        })

    # atommem pipeline：单次抽取，返回 JSON 数组（memory 不含时间词，时间走 time_expr）
    if "extracting atomic facts" in system:
        return json.dumps([
            {"memory": "Caroline went to an LGBTQ support group",
             "time_expr": "yesterday", "role": "Caroline", "tag": ["activity:group"]},
        ])

    # search 的 agent loop（system 说 "You build the evidence"）：写 evidence.jsonl
    if "You build the evidence" in system or "evidence.jsonl" in system:
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        if not has_assistant:
            body = json.dumps(_EVIDENCE_RECORD, ensure_ascii=False) + "\n"
            return json.dumps({"tool": "edit", "args": {
                "path": "evidence.jsonl", "append": body}})
        return "The evidence already answers the question."

    # 其它（add agent 模式等）：第一轮 append 一条 atom，之后回纯文本表示完成
    has_assistant = any(m.get("role") == "assistant" for m in messages)
    if not has_assistant:
        body = json.dumps({
            "memory": "Caroline went to an LGBTQ support group on 7 May 2023.",
            "metadata": {"time": "7 May 2023", "role": "Caroline", "tag": ["activity:group"]},
        }, ensure_ascii=False) + "\n"
        return json.dumps({"tool": "edit", "args": {"append": body}})
    return "The atoms are written."   # 纯文本 = 完成信号


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
        "add_max_steps": 6,
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
    msgs = store.sessions()
    check("msgmem 写入 2 条消息", len(msgs) == 2)
    check("id 形如 session_1_1", msgs[0]["metadata"]["id"] == "session_1_1")
    check("type=msg", msgs[0]["metadata"]["type"] == "msg")
    check("memory 是原始消息", msgs[0]["memory"] == "I went to a LGBTQ support group yesterday")
    check("原始会话时间锚点保留在 time",
          msgs[0]["metadata"]["time"] == "2023-05-08T16:00:00Z")
    check("role 是原始说话人", msgs[0]["metadata"]["role"] == "user")
    check("msgmem.jsonl 已落盘", store.msgmem_path.exists())
    # 返回前已生成 summem 与 atommem（"Add 返回即立即可检索"）
    check("summem.jsonl 已生成", store.summem_path.exists())
    check("summem 有一条会话总结",
          any((r.get("metadata") or {}).get("type") == "summary"
              for r in read_jsonl(store.summem_path)),
          store.summem_path.read_text()[:200])
    check("atommem.jsonl 已生成", store.atommem_path.exists())
    atoms = read_jsonl(store.atommem_path)
    check("atommem 有原子且补齐系统字段",
          atoms and atoms[0]["metadata"]["type"] == "atom"
          and "session_1_1" in atoms[0]["metadata"]["source"],
          str(atoms[:1]))
    # pipeline 把时间从 memory 里剥离、解析进 metadata.time（"yesterday" → 锚点前一日）
    check("atom 的时间解析进 metadata.time",
          atoms and atoms[0]["metadata"]["time"] == "7 May 2023", str(atoms[:1]))
    check("atom 的 memory 不含时间词",
          atoms and "yesterday" not in atoms[0]["memory"].lower(), str(atoms[:1]))

    # 幂等：同一个 request_id 再送一次
    resp2 = client.post("/add", json=add_body)
    check("重放仍返回 200", resp2.status_code == 200)
    store2 = store_mod.UserStore(add_body["user_id"], Path(app.state.api.store_root))
    check("幂等：语料未被重复写入", len(store2.sessions()) == 2, str(len(store2.sessions())))

    # 同一 session 再追加（新 request_id）→ 序号接续
    follow = dict(add_body, request_id="eval:run1:locomo:conv-0:chunk-1")
    follow["messages"] = [{"role": "user", "content": "Another message"}]
    client.post("/add", json=follow)
    store3 = store_mod.UserStore(add_body["user_id"], Path(app.state.api.store_root))
    ids = [r["metadata"]["id"] for r in store3.sessions()]
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

    # 原始请求留痕：requests.jsonl 记下原始 add 报文与 search query（排错/分析用）
    requests = read_jsonl(store3.requests_path)
    kinds = [r.get("endpoint") for r in requests]
    check("requests.jsonl 记录 add + search",
          "add" in kinds and "search" in kinds, str(kinds))
    add_req = next((r for r in requests if r.get("endpoint") == "add"), {})
    check("add 请求留痕含原始 messages 与 request_id",
          add_req.get("request_id") == add_body["request_id"]
          and any("LGBTQ support group" in str(m.get("content", ""))
                  for m in add_req.get("messages", [])),
          str(add_req)[:200])
    search_req = next((r for r in requests if r.get("endpoint") == "search"), {})
    check("search 请求留痕含 query",
          "LGBTQ support group" in str(search_req.get("query") or ""),
          str(search_req.get("query"))[:120])
    check("search 留痕带 run_dir",
          bool(search_req.get("result", {}).get("run_dir")), str(search_req.get("result")))


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


def test_resume(tmp: Path) -> None:
    """中断续跑：写前日志（WAL）+ 恢复。证明「被 kill 后不丢不重、能补完」。

    做法：把 ``_refresh_memories`` 换成"失败/抛异常"，模拟"消息已落 msgmem、但记忆刷新没跑完
    就被中断"。随后（a）重试同一 ``request_id`` 不重复写；（b）用一个**新的** ``UserStore`` 实例
    （= 模拟进程重启）``recover`` 独立补完，无需客户端重发。
    """
    print("\n[7] 中断续跑（WAL + 恢复）")

    app = _app_with_temp_store(tmp)
    config = app.state.api.config
    client = app.state.api.add_client
    root = Path(app.state.api.store_root)

    user_id, session_id = "eval:run2:locomo:conv-0", "eval:run2:sample:0"
    req_id = "eval:run2:locomo:conv-0:chunk-0"
    messages = [
        {"role": "user", "content": "I went to a LGBTQ support group yesterday",
         "timestamp": 1683561600000},
        {"role": "assistant", "content": "That sounds meaningful."},
    ]
    store = store_mod.UserStore(user_id, root)

    original = store_mod.UserStore._refresh_memories

    async def fail_refresh(self, **kwargs):  # noqa: ANN001
        return False   # 刷新失败：append 应保留 pending 并仍返回成功

    # --- (a) 刷新失败 → 消息已落盘，请求留在 pending，不记为 applied ---
    store_mod.UserStore._refresh_memories = fail_refresh
    try:
        res = asyncio.run(store.append(
            request_id=req_id, session_id=session_id, messages=messages,
            config=config, client=client,
        ))
        check("刷新失败仍返回成功（不抛）",
              res["duplicate"] is False and res.get("refreshed") is False, str(res))
        check("消息已落 msgmem（2 条）", len(store.sessions()) == 2, str(len(store.sessions())))
        meta = store.meta()
        check("meta.pending 记下该 request_id", req_id in meta["pending"])
        check("尚未记为 applied", req_id not in meta["applied_request_ids"])
        check("summem 尚未生成（刷新没跑完）", not store.summem_path.exists())

        # --- 重试同一 request_id（仍失败）→ 不重复写 ---
        res2 = asyncio.run(store.append(
            request_id=req_id, session_id=session_id, messages=messages,
            config=config, client=client,
        ))
        check("重试幂等：msgmem 仍 2 条", len(store.sessions()) == 2, str(len(store.sessions())))
        check("重试返回 duplicate+pending",
              res2.get("duplicate") is True and res2.get("pending") is True, str(res2))
    finally:
        store_mod.UserStore._refresh_memories = original

    # --- (b) 模拟进程重启：新 UserStore 实例 recover，独立补完 ---
    fresh = store_mod.UserStore(user_id, root)
    finalized = asyncio.run(fresh.recover(config=config, client=client))
    check("恢复落定 1 个 pending", finalized == 1, str(finalized))
    meta2 = fresh.meta()
    check("pending 已清空", meta2["pending"] == {}, str(meta2["pending"]))
    check("恢复后记为 applied", req_id in meta2["applied_request_ids"])
    check("恢复未重复写消息", len(fresh.sessions()) == 2, str(len(fresh.sessions())))
    check("恢复后 summem 已生成", fresh.summem_path.exists())
    check("恢复后 summem 是该会话总结",
          any((r.get("metadata") or {}).get("type") == "summary"
              for r in read_jsonl(fresh.summem_path)))
    atoms = read_jsonl(fresh.atommem_path)
    check("恢复后 atommem 已生成且时间已解析",
          bool(atoms) and atoms[0]["metadata"]["time"] == "7 May 2023", str(atoms[:1]))

    # --- (c) 恢复后序号接续（预留 count 未错位）---
    asyncio.run(fresh.append(
        request_id="eval:run2:locomo:conv-0:chunk-1", session_id=session_id,
        messages=[{"role": "user", "content": "Another message"}],
        config=config, client=client,
    ))
    ids = [r["metadata"]["id"] for r in fresh.sessions()]
    check("恢复后追加序号接续", ids == ["session_1_1", "session_1_2", "session_1_3"], str(ids))

    # --- (d) 硬中断（刷新抛异常，append 不返回）→ recover_all 无需重发即补完 ---
    user2, sess2 = "eval:run3:locomo:conv-0", "eval:run3:sample:0"
    store2 = store_mod.UserStore(user2, root)

    async def boom(self, **kwargs):  # noqa: ANN001
        raise RuntimeError("simulated kill during refresh")

    store_mod.UserStore._refresh_memories = boom
    try:
        asyncio.run(store2.append(
            request_id="r-crash", session_id=sess2,
            messages=[{"role": "user", "content": "hi", "timestamp": 1683561600000}],
            config=config, client=client,
        ))
    except RuntimeError:
        pass
    finally:
        store_mod.UserStore._refresh_memories = original

    check("硬中断后消息已落 msgmem", len(store2.sessions()) == 1, str(len(store2.sessions())))
    check("硬中断后 meta.pending 留存", "r-crash" in store2.meta()["pending"])
    total = asyncio.run(store_mod.recover_all(root, config=config, client=client))
    check("recover_all 无需重发即补完", total >= 1, str(total))
    check("硬中断 Add 恢复后 applied", "r-crash" in store2.meta()["applied_request_ids"])
    check("硬中断 Add 恢复后未重复", len(store2.sessions()) == 1, str(len(store2.sessions())))


# ---------------------------------------------------------------------------
# 形态适配层（codemem/adapters.py）
# ---------------------------------------------------------------------------

def test_adapters() -> None:
    """每个平台形态各来一条，钉住"谁说的 / 什么时候 / 正文是什么"这三件事。"""
    print("\n[8] 形态适配层：把平台编码的信息搬回正确的字段")

    # --- ① 带记账标签的对话：真名在 content 前缀里 ---
    batch = adapters.adapt_messages([{
        "role": "user", "timestamp": 1704153600000,
        "content": "[beam_100k][session_1][D1:21][text] Christina Baker: I'm trying to "
                   "implement city autocomplete using OpenWeather's API.",
    }])
    check("形态=dialogue", batch.family == "dialogue", batch.family)
    check("说话人从 content 前缀取到", batch.speakers == ["Christina Baker"], str(batch.speakers))
    check("role 是真名而不是 user", batch.messages[0].role == "Christina Baker")
    check("记账标签不进正文",
          batch.messages[0].content.startswith("I'm trying to implement"), batch.messages[0].content)
    check("timestamp 原样保留", batch.messages[0].timestamp == 1704153600000)

    # --- ② 正文里本来就有的方括号不能被当成记账标签剥掉 ---
    batch = adapters.adapt_messages([{
        "role": "user", "content": "[clbench_16_32k][session_1][D1:2][text] user: [part 1/2] Background\nOTC is stable.",
    }])
    check("只剥平台记账标签，正文方括号留下",
          batch.messages[0].content == "[part 1/2] Background\nOTC is stable.",
          batch.messages[0].content)
    check("user:/assistant: 收敛成通用角色", batch.messages[0].role == "user")

    # --- ③ 说话人前缀不许被短词前缀匹配吃掉（Andrew 曾被 and 吃掉）---
    speaker, rest = adapters.split_speaker("Andrew: Thanks! That sounds cute.")
    check("Andrew: 认成说话人（不被 and 前缀匹配吞掉）", speaker == "Andrew", str(speaker))
    check("也认编号角色", adapters.split_speaker("#8: I think you're wrong.")[0] == "#8")
    check("正文类型前缀不是说话人", adapters.split_speaker("Corpus: {...}")[0] is None)
    check("语气词不是说话人", adapters.split_speaker("Wait: I think so")[0] is None)

    # --- ④ 旁白 / 舞台指示 ---
    batch = adapters.adapt_messages([{
        "role": "user",
        "content": "[angry][session_1][D1:1][narration] NARRATION: (SCENE.—A sitting-room.)",
    }])
    check("NARRATION: → role=narration", batch.messages[0].role == "narration")
    check("NARRATION 标记不进正文", batch.messages[0].content.startswith("(SCENE."))
    batch = adapters.adapt_messages([
        {"role": "user", "content": "[angry][session_1][D1:2][narration] NARRATION: FOREMAN: 35 years old."},
        {"role": "user", "content": "[angry][session_1][D1:3][dialogue] JUDGE: Pardon me, gentlemen."},
    ])
    check("旁白里连着的角色名被认出来",
          batch.messages[0].role == "FOREMAN" and batch.messages[0].content == "35 years old.",
          f"{batch.messages[0].role} / {batch.messages[0].content!r}")
    check("剧本角色进 speakers", batch.speakers == ["FOREMAN", "JUDGE"], str(batch.speakers))

    # --- ⑤ 文档 ---
    batch = adapters.adapt_messages([{
        "role": "user",
        "content": "[docpp][session_1][D1:1][text] document: Document: a.pdf\nNative clip page: 1 of 5\n\n```text\nBody",
    }])
    check("形态=document", batch.family == "document", batch.family)
    check("document: 标记被剥、文档名保留",
          batch.messages[0].role == "document"
          and batch.messages[0].content.startswith("Document: a.pdf"), batch.messages[0].content)

    # --- ⑥ 知识图谱三元组 ---
    batch = adapters.adapt_messages([{
        "role": "user",
        "content": 'Corpus: {"subject": "Latvia", "relation": "official language", '
                   '"object": "Latvian", "subject_id": "Q211", "relation_id": "P37"}',
    }])
    check("形态=triples", batch.family == "triples", batch.family)
    check("三元组渲染成 subject | relation | object",
          batch.messages[0].content == "Latvia | official language | Latvian", batch.messages[0].content)
    check("Q-id 不进正文", "Q211" not in batch.messages[0].content)
    check("三元组没有说话人", batch.messages[0].role == "corpus" and batch.speakers == [])
    batch = adapters.adapt_messages([{
        "role": "user",
        "content": 'Corpus: UPDATE: replace the prior value of this subject and relation with '
                   'the following fact.\n{"subject": "India", "relation": "continent", "object": "Africa"}',
    }])
    check("UPDATE 语义被保留", batch.messages[0].content.startswith("UPDATE: India | continent | Africa"),
          batch.messages[0].content)

    # --- ⑦ [Time: ...] 标签里的时间要搬到 timestamp ---
    batch = adapters.adapt_messages([{
        "role": "user", "timestamp": None,
        "content": "[Time: Sep 04, 2025, 09:05:32] Hello! I'd like to share my info.",
    }])
    check("[Time: ...] 解析成毫秒时间戳", batch.messages[0].timestamp == 1756976732000,
          str(batch.messages[0].timestamp))
    check("时间标签不进正文",
          batch.messages[0].content.startswith("Hello!"), batch.messages[0].content)
    batch = adapters.adapt_messages([{
        "role": "user", "timestamp": None, "content": "[Time: Unknown] FOREMAN: 35 years old.",
    }])
    check("[Time: Unknown] 不编造时间且仍拆出说话人",
          batch.messages[0].timestamp is None and batch.messages[0].role == "FOREMAN")

    # --- ⑧ 流式 QA 反馈：转义 JSON → 可读多行 ---
    batch = adapters.adapt_messages([{
        "role": "user", "content": 'Streaming online QA feedback:\n{"qa_id": "q1", '
                                   '"question": "When did Jon lose his job?\\n\\nA. 2022\\nB. 2023", '
                                   '"predicted_answer": "(B)", "score": 0.0}',
    }])
    check("形态=qa-feedback", batch.family == "qa-feedback", batch.family)
    check("不再是转义的一行（真换行）", "\nB. 2023" in batch.messages[0].content,
          batch.messages[0].content[:80])
    check("旧答案被标明不是权威", "not authoritative" in batch.messages[0].content)

    # --- ⑨ agent 轨迹：展开成逐条消息，且任务号可 grep ---
    trajectory = {
        "dataset": "ALF", "task_index": 86,
        "trajectory": [
            {"role": "system", "content": "You are in a household."},
            {"role": "user", "content": "Your task is to: examine the cd with the desklamp."},
        ],
    }
    batch = adapters.adapt_messages([{
        "role": "user",
        "content": "Streaming intra-transfer train trajectory:\n" + json.dumps(trajectory),
    }])
    check("形态=trajectory", batch.family == "trajectory", batch.family)
    check("展开成逐条消息（2 条）", len(batch.messages) == 2, str(len(batch.messages)))
    check("每条的 role 来自轨迹内部", [m.role for m in batch.messages] == ["system", "user"])
    check('查询字面串 "ALF task index 86" 可 grep',
          all("ALF task index 86" in m.content for m in batch.messages))

    # --- ⑩ 没有前缀的人机对话：role 用协议给的 ---
    batch = adapters.adapt_messages([
        {"role": "user", "content": "医生您好，我想问问我这个膝盖的事。", "timestamp": 1704240000000},
        {"role": "assistant", "content": "听起来这几天的膝盖确实让你有点不安。"},
    ])
    check("形态=plain", batch.family == "plain", batch.family)
    check("role 退回协议值", [m.role for m in batch.messages] == ["user", "assistant"])
    check("正文一字未改", batch.messages[0].content.startswith("医生您好"))

    # --- ⑪ 说话人名单渲染 ---
    check("两人名单", adapters.speakers_phrase(["Caroline", "Melanie"]) == "Caroline and Melanie")
    check("空名单有兜底", "user" in adapters.speakers_phrase([]))
    check("人数超上限时写清总数", "7 distinct speakers" in
          adapters.speakers_phrase([f"s{i}" for i in range(7)], limit=3))
    check("speaker_pair 单人时的第二位不是自己",
          adapters.speaker_pair(["Solo"]) == ("Solo", "the other participant"))


def test_adapters_end_to_end(tmp: Path) -> None:
    """走完整的 store 写入：适配层 → msgmem → summem/atommem 的 prompt 里有没有真名。"""
    print("\n[9] 形态适配贯通到三层记忆（说话人一路传下去）")

    seen_prompts: list[str] = []
    inner = llm_mod.chat_completion

    async def spy(client, config, messages):  # noqa: ANN001
        seen_prompts.append(str(messages[0].get("content", "")) if messages else "")
        return await inner(client, config, messages)

    llm_mod.chat_completion = spy  # type: ignore[assignment]
    try:
        app = _app_with_temp_store(tmp)
        config = app.state.api.config
        client = app.state.api.add_client
        root = Path(app.state.api.store_root)
        user_id, session_id = "eval:fmt:beam:conv-0", "eval:fmt:beam:sess-0"
        store = store_mod.UserStore(user_id, root)

        asyncio.run(store.append(
            request_id="fmt-1", session_id=session_id,
            messages=[
                {"role": "user", "timestamp": 1704153600000,
                 "content": "[beam_100k][session_1][D1:21][text] Christina Baker: I'm trying to "
                            "implement city autocomplete using OpenWeather's API."},
                {"role": "user", "timestamp": 1704153660000,
                 "content": "[beam_100k][session_1][D1:22][text] Christina Baker: hmm, what about "
                            "when the API response takes longer?"},
            ],
            config=config, client=client,
        ))

        records = store.sessions()
        check("msgmem 落的是真名", {r["metadata"]["role"] for r in records} == {"Christina Baker"},
              str({r["metadata"]["role"] for r in records}))
        check("msgmem 正文没有记账标签",
              all("[beam_100k]" not in r["memory"] for r in records))
        check("msgmem 时间锚点还在",
              records[0]["metadata"]["time"] == "2024-01-02T00:00:00Z",
              records[0]["metadata"]["time"])

        atom_prompts = [p for p in seen_prompts if "extracting atomic facts" in p]
        check("atom 抽取的 prompt 收到了真名",
              bool(atom_prompts) and all("Christina Baker" in p for p in atom_prompts),
              str(len(atom_prompts)))
        check("atom prompt 不再出现 speaker_a 字面量",
              all("speaker_a" not in p and "speaker_b" not in p for p in atom_prompts))

        summ_prompts = [p for p in seen_prompts if "You summarize ONE session" in p]
        check("summem 的 prompt 收到了真名",
              bool(summ_prompts) and all("Christina Baker" in p for p in summ_prompts),
              str(len(summ_prompts)))
        check("summem 不再把参与者写死成两个人",
              all("between" not in p for p in summ_prompts))
    finally:
        llm_mod.chat_completion = inner  # type: ignore[assignment]


def test_participants_block(tmp: Path) -> None:
    """search 的 user_begin 里那段"这份语料有谁"：数据驱动，且不该说的话不说。"""
    print("\n[10] search 的参与者说明（user_begin 的 {{PARTICIPANTS}} 槽）")
    from codemem.add.memory import write_memories
    from codemem.io import make_memory
    from codemem.search.runner import render_participants

    inputs = tmp / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    def write(*roles: str) -> None:
        write_memories(inputs / "msgmem.jsonl", [
            make_memory(memory=f"m{i}", memory_type="msg", mem_id=f"session_1_{i + 1}", role=role)
            for i, role in enumerate(roles)
        ])

    write("Caroline", "Melanie")
    text = render_participants(inputs)
    check("列出语料里真实的人名", "Caroline" in text and "Melanie" in text, text)
    check("不提醒'别编造说话人'（对话语料没有这个问题）", "never invent a speaker" not in text)
    check("不提三元组记法（这份语料里没有）", "|" not in text)

    write("user", "assistant")
    check("纯 user/assistant 语料整段不渲染", render_participants(inputs) == "",
          render_participants(inputs))

    write("corpus", "corpus", "corpus")
    text = render_participants(inputs)
    check("三元组语料说明 `A | relation | B` 怎么读", "subject's relation is the" in text, text)
    check("三元组语料提醒别编造说话人", "never invent a speaker" in text)

    write("Caroline", "corpus")
    text = render_participants(inputs)
    check("混合语料两段都说", "Caroline" in text and "never invent a speaker" in text)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    _install_fake_model()
    test_normalize_content()
    test_iso_from_ms()
    test_slug()
    test_records_to_data()
    test_adapters()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        test_endpoints(tmp)
        test_validation(tmp)
        test_resume(tmp)
        test_adapters_end_to_end(tmp / "e2e")
        test_participants_block(tmp / "participants")

    print(f"\n{'=' * 60}\n通过 {PASS} / 失败 {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
