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
import random
import re
import sys
import tempfile
import time
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

    # --- (e) 卡住的 Add 再重试一次：这次该把它推完（幂等路径也不能把遗留 pending 晾着）---
    user3, sess3 = "eval:run4:locomo:conv-0", "eval:run4:sample:0"
    store3 = store_mod.UserStore(user3, root)
    stuck = [{"role": "user", "content": "hello", "timestamp": 1683561600000}]

    store_mod.UserStore._refresh_memories = fail_refresh
    try:
        asyncio.run(store3.append(request_id="r-stuck", session_id=sess3,
                                  messages=stuck, config=config, client=client))
    finally:
        store_mod.UserStore._refresh_memories = original
    check("卡住的 Add 留在 pending", "r-stuck" in store3.meta()["pending"])

    res = asyncio.run(store3.append(request_id="r-stuck", session_id=sess3,
                                     messages=stuck, config=config, client=client))
    meta3 = store3.meta()
    check("重试同一条：这一次把它推完（离开 pending、记为 applied）",
          res.get("duplicate") is True and "r-stuck" not in meta3["pending"]
          and "r-stuck" in meta3["applied_request_ids"], f"{res} / {meta3['pending']}")
    check("推完后 summem 已生成", store3.summem_path.exists())


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


def test_atommem_budget() -> None:
    """add · atommem 的**输入预算**：超出模型上下文时到底会发生什么（纯 CPU，零模型成本）。

    这是"Add 一条超长消息 / 一整个文档"时的保命逻辑：预算内的原样发；超出就把上下文按行
    从最新往回截、目标消息切块；块内还放不下（或输出被 max_tokens 截断）就砍半再切细。
    要钉住的是三件事：**每份 prompt 都在预算内**（否则就是 400 报错）、**内容不丢**、
    **截断有标记**。
    """
    print("\n[11] atommem 的输入预算（超出模型上下文时会发生什么）")
    from codemem.add import atommem_pipeline as ap

    check("CJK 一字算一个 token", ap.estimate_tokens("中文测试") == 4, str(ap.estimate_tokens("中文测试")))
    check("ASCII 四字符算一个", ap.estimate_tokens("abcdefgh") == 2, str(ap.estimate_tokens("abcdefgh")))
    check("空串 0", ap.estimate_tokens("") == 0)

    # --- 超预算的目标消息被切开，且一块都不超预算、内容一个词不丢 ---
    text = "\n".join(f"[session_1_{i}] (t) user: " + "word " * 30 for i in range(1, 41))
    chunks = ap.split_message(text, 200, overlap_tokens=20)
    check("超预算的消息被切开", len(chunks) > 1, str(len(chunks)))
    check("每块都在预算内", all(ap.estimate_tokens(c) <= 200 for c in chunks),
          str([ap.estimate_tokens(c) for c in chunks]))
    check("切开不丢内容（词集合相同）", set(text.split()) == set("".join(chunks).split()))

    # --- 单行就超预算（长文档常见）：硬切也得每块都在预算内 ---
    hard = ap.split_message("x" * 4000, 100)
    check("单行超预算也能硬切", len(hard) > 1 and all(ap.estimate_tokens(c) <= 100 for c in hard),
          str([ap.estimate_tokens(c) for c in hard][:5]))
    check("硬切无重叠时不留空隙", "".join(hard) == "x" * 4000)

    # --- 混合语料：按开头估步长会切超，必须逐块量（CJK 的字符/token 比是 ASCII 的 4 倍）---
    mixed = ("a" * 400 + "中" * 400) * 3
    pieces = ap.split_message(mixed, 120)
    check("中英混排也每块都在预算内", all(ap.estimate_tokens(p) <= 120 for p in pieces),
          str([ap.estimate_tokens(p) for p in pieces][:6]))

    # --- 上下文按行从最新往回保留（消解指代靠最近几轮），并留标记 ---
    context = "\n".join(f"[session_1_{i}] user: line {i}" for i in range(1, 21))
    clipped = ap.clip_context(context, 40)
    check("上下文保留最近的、丢掉最早的",
          "line 20" in clipped and "line 1]" not in clipped, clipped)
    check("上下文截断有标记", "omitted" in clipped)
    check("上下文截断后仍不超预算", ap.estimate_tokens(clipped) <= 40,
          str(ap.estimate_tokens(clipped)))

    # --- plan_units：每一份输入（上下文 + 内容）都必须落在预算内 ---
    units = ap.plan_units(context, text, 400)
    check("按预算切成多份输入", len(units) > 1, str(len(units)))
    check("每份（上下文 + 内容）都在预算内",
          all(ap.estimate_tokens(u.context) + ap.estimate_tokens(u.content) <= 400 for u in units),
          str([ap.estimate_tokens(u.context) + ap.estimate_tokens(u.content) for u in units]))
    check("只有第一份带上下文，其余是接续标记",
          "session_1_" in units[0].context and "SAME message" in units[1].context,
          f"{units[0].context[:40]!r} / {units[1].context[:40]!r}")

    # --- 预算关掉（0）时退回旧行为：一次调用发全文 ---
    off = ap.plan_units(context, text, 0)
    check("预算=0 表示不控制（整段一次发）", len(off) == 1 and off[0].content == text)

    # --- 输出被截断：按对象边界抢救（末条写到一半，前面的必须全保住）---
    from codemem.log import Logger as _Logger

    quiet = _Logger(level="error")
    rescued = ap.parse_atom_array(
        '[{"memory": "a"}, {"memory": "b"}, {"memory": "unfin', quiet, "session_1_1"
    )
    check("截断的数组能抢救出完整记录", len(rescued) == 2, str(rescued))

    # --- 端到端：端点报"超上下文" → 砍半重切，再试成功 ---
    from codemem import llm as llm_mod
    from codemem.add.atommem_pipeline import run_message_pipeline
    from codemem.io import make_memory

    class _TooLong(RuntimeError):
        status_code = 400

    prompts: list[int] = []

    async def shrink_model(client, config, messages):  # noqa: ANN001
        prompts.append(len(str(messages[1]["content"])))
        if len(prompts) == 1:               # 第一次：假装 prompt 放不下
            raise _TooLong("This model's maximum context length is 4096 tokens")
        return json.dumps([{"memory": "Caroline archived a very long document",
                            "time_expr": "", "role": "Caroline", "tag": ["topic:doc"]}])

    original = llm_mod.chat_completion
    llm_mod.chat_completion = shrink_model          # type: ignore[assignment]
    try:
        target = make_memory(memory=" ".join(["word"] * 4000), memory_type="msg",
                             mem_id="session_1_1", time="2023-05-08T16:00:00Z", role="Caroline")
        outcome = asyncio.run(run_message_pipeline(
            target=target, msgmem=[target], existing_atoms=[], client=None,
            config={"max_input_tokens": 2000, "context_window": 12}, log=quiet,
            speakers=["Caroline"],
        ))
    finally:
        llm_mod.chat_completion = original          # type: ignore[assignment]
    check("超上下文后缩预算重切、最终抽到了 atom",
          outcome.ok and len(outcome.atoms) >= 1, f"ok={outcome.ok} atoms={len(outcome.atoms)}")
    check("重切后的 prompt 确实更短",
          len(prompts) >= 2 and prompts[1] < prompts[0], str(prompts[:3]))


def test_add_concurrency(tmp: Path) -> None:
    """Add 的**并发与增量**：不同 session 的 Add 同时跑，同一条 message 不重复抽。

    这两件事都是"锁的作用域"的直接后果：
    - 旧实现把整段记忆刷新放在语料写锁里 → 同一 user 的并发 Add 全被串起来（峰值并发 1）；
    - 旧实现每次都抽**整个会话**的 message → 追加一条新消息要重抽前面全部（O(n²) 次调用）。
    """
    print("\n[12] Add 的并发与增量（锁不再串起整个 user）")
    app = _app_with_temp_store(tmp)
    config = app.state.api.config
    root = Path(app.state.api.store_root)
    user_id = "eval:conc:locomo:conv-0"
    store = store_mod.UserStore(user_id, root)

    inner = llm_mod.chat_completion
    live = {"now": 0, "peak": 0, "atom_calls": 0}

    async def spy(client, config, messages):  # noqa: ANN001
        system = str(messages[0].get("content", "")) if messages else ""
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])
        try:
            await asyncio.sleep(0.05)          # 模拟"模型要想一会儿"
            if "extracting atomic facts" in system:
                live["atom_calls"] += 1
                body = str(messages[1].get("content", ""))
                # 取**最后一个** id：prompt 里先给上下文（也带 id），目标消息在其后。
                found = re.findall(r"\[(session_\d+_\d+)\]", body)
                msg_id = found[-1] if found else "session_1_0"
                # 每条 message 产出**内容互不重叠**的 atom —— 措辞相近的会被 ``_dedup_atoms``
                # 按设计合并成一条（那是它的职责），这里要数的是"抽了几次"，不是"留了几条"。
                unique = live["atom_calls"] * 17
                return json.dumps([{
                    "memory": f"{msg_id} noted alpha{unique} omega{unique}",
                    "time_expr": "", "role": "Caroline", "tag": ["topic:test"],
                }])
            return await inner(client, config, messages)
        finally:
            live["now"] -= 1

    llm_mod.chat_completion = spy  # type: ignore[assignment]
    try:
        # 整段跑在**同一个事件循环**里：并发争用的 ``asyncio.Lock`` 会绑定到首次争用的那个
        # loop，跨 ``asyncio.run`` 复用会 RuntimeError（服务本身是单循环长驻，不受影响）。
        async def scenario() -> None:
            gate = asyncio.Semaphore(8)

            async def add_one(index: int) -> dict:
                return await store.append(
                    request_id=f"conc-{index}", session_id=f"eval:conc:sess-{index}",
                    messages=[{"role": "user", "timestamp": 1683561600000,
                               "content": f"I went to the support group {index} yesterday"}],
                    config=config, client=app.state.api.add_client, gate=gate,
                )

            results = list(await asyncio.gather(*(add_one(i) for i in range(8))))
            check("8 个并发 Add 全部成功",
                  all(r["written"] == 1 and r["refreshed"] for r in results), str(results[:2]))
            check("不同 session 的 Add 真正并行（旧实现峰值只有 1）", live["peak"] > 1,
                  f"peak={live['peak']}")
            check("并发下的模型调用数 = 8 session × (1 总结 + 1 抽取)",
                  live["atom_calls"] == 8, str(live["atom_calls"]))

            # --- 增量：同一 session 再追加一条，只抽**新的**那一条 ---
            before = live["atom_calls"]
            await store.append(
                request_id="conc-0-again", session_id="eval:conc:sess-0",
                messages=[{"role": "user", "timestamp": 1683561660000,
                           "content": "Another message about the group"}],
                config=config, client=app.state.api.add_client, gate=gate,
            )
            check("追加消息只抽新增的那条（不重抽整会话）",
                  live["atom_calls"] - before == 1, f"新增调用 {live['atom_calls'] - before}")
            atoms = read_jsonl(store.atommem_path)
            check("旧会话的 atom 没被冲掉",
                  any("session_1_1" in a["memory"] for a in atoms), str(atoms))
            check("新消息的 atom 也进来了",
                  any("session_1_2" in a["memory"] for a in atoms), str(atoms))
            check("抽取进度有记账（下次才敢跳过）", store.atom_state_path.exists())

            # --- 同一 session 的并发 Add：一条 message 也只抽一次（会话锁 + 进度记账）---
            before_same = live["atom_calls"]

            async def add_same(index: int) -> dict:
                return await store.append(
                    request_id=f"same-{index}", session_id="eval:conc:shared",
                    messages=[{"role": "user", "timestamp": 1683561900000,
                               "content": f"Concurrent message {index}"}],
                    config=config, client=app.state.api.add_client, gate=gate,
                )

            await asyncio.gather(*(add_same(i) for i in range(4)))
            check("同一 session 的 4 条并发消息各抽一次（不重抽）",
                  live["atom_calls"] - before_same == 4,
                  f"新增调用 {live['atom_calls'] - before_same}")

            # --- 抽取失败不记进度：下次还会重试，而不是把这条消息永久钉成 0 条 ---
            async def fail_atoms(client, config, messages):  # noqa: ANN001
                system = str(messages[0].get("content", "")) if messages else ""
                if "extracting atomic facts" in system:
                    raise RuntimeError("simulated endpoint failure")
                return await inner(client, config, messages)

            llm_mod.chat_completion = fail_atoms  # type: ignore[assignment]
            await store.append(
                request_id="conc-0-failed", session_id="eval:conc:sess-0",
                messages=[{"role": "user", "timestamp": 1683561720000,
                           "content": "A third message that will fail to extract"}],
                config=config, client=app.state.api.add_client, gate=gate,
            )
            state = json.loads(store.atom_state_path.read_text(encoding="utf-8"))
            check("失败的那条没被记成已处理",
                  "session_1_3" not in (state.get("1", {}).get("atoms") or []), str(state))

        asyncio.run(scenario())
    finally:
        llm_mod.chat_completion = inner  # type: ignore[assignment]


def test_corpus_cache(tmp: Path) -> None:
    """语料派生结果（tag 索引 / 说话人 / 审计索引）**跨运行目录**只建一次。

    在线形态下每条 query 都是一个新的运行目录（``runs/{ts}/inputs``），而 ``inputs/*`` 是
    **硬链**（与原文件同 inode）—— 所以按**语料指纹**缓存，32 路并发才只解析一遍语料；
    按路径缓存等于不缓存（旧实现就是这样，32 条 query 各读一遍 1.5MB 语料，还都在同一个
    事件循环里）。
    """
    print("\n[13] 语料派生结果的跨运行目录缓存（按 inode 指纹，不按路径）")
    import os

    from codemem import audit
    from codemem.add.memory import write_memories
    from codemem.io import make_memory
    from codemem.search import runner as runner_mod

    src = tmp / "corpus"
    src.mkdir(parents=True, exist_ok=True)
    write_memories(src / "msgmem.jsonl", [
        make_memory(memory="m1", memory_type="msg", mem_id="session_1_1", role="Caroline"),
    ])
    write_memories(src / "atommem.jsonl", [
        make_memory(memory="Caroline likes hiking", memory_type="atom",
                    mem_id="session_1_1_1", role="Caroline", tag=["activity:hiking"]),
    ])

    runs = [tmp / "run_a" / "inputs", tmp / "run_b" / "inputs"]
    for run in runs:
        run.mkdir(parents=True, exist_ok=True)
        for name in ("msgmem.jsonl", "atommem.jsonl"):
            os.link(src / name, run / name)     # 硬链：与原文件同一个 inode

    runner_mod._TAG_INDEX_CACHE.clear()
    runner_mod._PARTICIPANTS_CACHE.clear()
    audit._CORPUS_INDEX_CACHE.clear()

    counts = {"tag": 0, "people": 0, "audit": 0}
    real_tag, real_people = runner_mod.render_tag_index, runner_mod.render_participants
    real_index = audit.build_corpus_index

    def counting_tag(*args, **kwargs):  # noqa: ANN001
        counts["tag"] += 1
        return real_tag(*args, **kwargs)

    def counting_people(*args, **kwargs):  # noqa: ANN001
        counts["people"] += 1
        return real_people(*args, **kwargs)

    def counting_index(*args, **kwargs):  # noqa: ANN001
        counts["audit"] += 1
        return real_index(*args, **kwargs)

    runner_mod.render_tag_index = counting_tag
    runner_mod.render_participants = counting_people
    audit.build_corpus_index = counting_index
    try:
        first = runner_mod.tag_index_for(runs[0])
        check("第一次渲染出 tag 索引", "activity:hiking" in first, first)
        same = runner_mod.tag_index_for(runs[1])
        check("另一个运行目录命中同一份缓存（硬链同 inode）", counts["tag"] == 1, str(counts))
        check("两份结果一模一样", same == first)
        check("说话人名单同理",
              runner_mod.participants_for(runs[0]) == runner_mod.participants_for(runs[1])
              and counts["people"] == 1, str(counts))
        audit.corpus_index_for(runs[0])
        audit.corpus_index_for(runs[1])
        check("审计语料索引同理（它要读全文，最贵）", counts["audit"] == 1, str(counts))

        # --- 语料变了 → 指纹变 → 重建（缓存不能给出过期结果）---
        write_memories(runs[0] / "atommem.jsonl", [
            make_memory(memory="Caroline likes painting", memory_type="atom",
                        mem_id="session_1_1_1", role="Caroline", tag=["activity:painting"]),
        ])
        changed = runner_mod.tag_index_for(runs[0])
        check("语料一变就重建", counts["tag"] == 2, str(counts))
        check("重建后拿到的是新语料", "activity:painting" in changed, changed)
        check("没变的那个目录仍走缓存",
              runner_mod.tag_index_for(runs[1]) == first and counts["tag"] == 2, str(counts))
    finally:
        runner_mod.render_tag_index = real_tag
        runner_mod.render_participants = real_people
        audit.build_corpus_index = real_index


def test_config_precedence(tmp: Path) -> None:
    """配置优先级：``api.yaml`` **显式写下的键**不被离线配置覆盖。

    规则本身很短，但它决定了"把 API 并发调到 32"这句话能不能说出口 —— `search.yaml` 的
    `concurrency` 是"跑一次评测几条 QA 并发"，与"服务同时接多少请求"只是恰好同名；
    没有这条规则，写什么都会被离线那份盖回去。
    """
    print("\n[14] 配置优先级（显式 vs 离线）")
    from codemem.add import atommem as atommem_mod
    from codemem.api.server import load_config
    from codemem.search import runner as search_runner_mod

    explicit = tmp / "explicit.yaml"
    explicit.write_text("concurrency: 7\nadd_concurrency: 9\n", encoding="utf-8")
    cfg = load_config(explicit)
    check("显式写的并发不被离线覆盖",
          cfg["concurrency"] == 7 and cfg["add_concurrency"] == 9,
          f"{cfg['concurrency']}/{cfg['add_concurrency']}")

    bare = tmp / "bare.yaml"
    bare.write_text("store_dir: data/api_store\n", encoding="utf-8")
    cfg2 = load_config(bare)
    search_path = PROJECT_ROOT / "configs" / "search.yaml"
    add_path = PROJECT_ROOT / "configs" / "add.yaml"
    if search_path.exists():
        offline = search_runner_mod.load_config(search_path)
        check("没写就跟随离线 search.yaml",
              cfg2["concurrency"] == offline.get("concurrency"),
              f"{cfg2['concurrency']} vs {offline.get('concurrency')}")
    if add_path.exists():
        offline_add = atommem_mod.load_config(add_path)
        check("没写就跟随离线 add.yaml",
              cfg2["add_concurrency"] == offline_add.get("concurrency"),
              f"{cfg2['add_concurrency']} vs {offline_add.get('concurrency')}")
        check("离线的抽取预算被带进 index 段",
              (cfg2.get("index") or {}).get("max_input_tokens")
              == offline_add.get("max_input_tokens"),
              str((cfg2.get("index") or {}).get("max_input_tokens")))


def test_atom_retry_and_repair(tmp: Path) -> None:
    """抽取失败的**两层重试**：Add 内就地重试，以及重启/手动触发的补漏。

    为什么必须有这两层：``llm.py`` 的退避重试对付的是"单次调用"的抖动，而真实失败长这样 ——
    `APITimeoutError ... (cause: ConnectTimeout)`，一次批量抽取里十几条一起撞上（远程端点在
    并发新建连接时丢 SYN），退避 2s/4s 不够。更关键的是：**抽取失败不影响 Add 成功** ——
    消息已可检索、只是少了几条 atom，所以它不会留在 pending 里，重启的"续跑"也捞不到它。
    """
    print("\n[15] 抽取失败的就地重试与补漏")
    app = _app_with_temp_store(tmp)
    config = dict(app.state.api.config)
    config["atom_retry_rounds"] = 2
    config["atom_retry_backoff"] = 0.0        # 测试不等退避
    root = Path(app.state.api.store_root)
    user_id = "eval:retry:locomo:conv-0"
    store = store_mod.UserStore(user_id, root)

    inner = llm_mod.chat_completion
    live = {"mode": "flaky", "seen": set(), "atom_calls": 0}

    async def model(client, config, messages):  # noqa: ANN001
        system = str(messages[0].get("content", "")) if messages else ""
        if "extracting atomic facts" not in system:
            return await inner(client, config, messages)
        live["atom_calls"] += 1
        found = re.findall(r"\[(session_\d+_\d+)\]", str(messages[1].get("content", "")))
        target = found[-1] if found else "session_1_0"
        if live["mode"] == "dead":
            raise RuntimeError("simulated connect timeout")
        if live["mode"] == "flaky" and target not in live["seen"]:
            live["seen"].add(target)          # 每条的**第一次**必失败
            raise RuntimeError("simulated connect timeout")
        return json.dumps([{
            "memory": f"{target} noted the alpine route",
            "time_expr": "", "role": "Caroline", "tag": ["topic:test"],
        }])

    llm_mod.chat_completion = model  # type: ignore[assignment]
    try:
        async def scenario() -> None:
            gate = asyncio.Semaphore(4)
            add = dict(config=config, client=app.state.api.add_client, gate=gate)

            # --- ① 抖动型失败：Add 内重试后应当补全，不留欠账 ---
            await store.append(
                request_id="retry-1", session_id="eval:retry:sess-0",
                messages=[{"role": "user", "timestamp": 1683561600000,
                           "content": f"message {i}"} for i in range(3)],
                **add,
            )
            state = json.loads(store.atom_state_path.read_text(encoding="utf-8"))
            check("抖动失败被就地重试并补全",
                  not state["1"].get("failed") and len(state["1"]["atoms"]) == 3, str(state))
            check("确实重试过（调用数 > message 数）", live["atom_calls"] > 3,
                  str(live["atom_calls"]))
            atoms = read_jsonl(store.atommem_path)
            sources = {s for atom in atoms for s in (atom["metadata"].get("source") or [])}
            check("三条 message 的 atom 都在盘上（同事实合并、source 取并集）",
                  {"session_1_1", "session_1_2", "session_1_3"} <= sources, str(sources))

            # --- ② 端点一直不通：失败要**记账**，而不是悄悄少几条 ---
            live["mode"] = "dead"
            before = live["atom_calls"]
            await store.append(
                request_id="retry-2", session_id="eval:retry:sess-1",
                messages=[{"role": "user", "timestamp": 1683561600000,
                           "content": "a message that will never extract"}],
                **add,
            )
            state = json.loads(store.atom_state_path.read_text(encoding="utf-8"))
            check("一直失败的那条被记进 failed",
                  state["2"].get("failed") == ["session_2_1"], str(state.get("2")))
            check("重试轮数生效（1 条 × (1+2) 轮）", live["atom_calls"] - before == 3,
                  str(live["atom_calls"] - before))
            check("欠账的会话能被找出来", store.sessions_needing_repair() == [2],
                  str(store.sessions_needing_repair()))

            # --- ③ 端点恢复 → 补漏把它补上（不依赖"同会话的下一次 Add"）---
            live["mode"] = "ok"
            fixed = await store.repair(config=config, client=app.state.api.add_client,
                                       gate=gate)
            state = json.loads(store.atom_state_path.read_text(encoding="utf-8"))
            check("补漏修好了欠账的会话", fixed == 1, str(fixed))
            check("补漏后不再欠账", not state["2"].get("failed") and state["2"]["atoms"],
                  str(state.get("2")))
            check("补漏后没有欠账的会话", store.sessions_needing_repair() == [],
                  str(store.sessions_needing_repair()))

            # --- ④ 模块级补漏（启动时跑的那个）：没欠账时一个模型调用都不发 ---
            calls_before = live["atom_calls"]
            stats = await store_mod.repair_all(root, config=config,
                                               client=app.state.api.add_client, gate=gate)
            check("没欠账时补漏是空转（不烧模型调用）",
                  stats["sessions"] == 0 and live["atom_calls"] == calls_before, str(stats))

            # --- ⑤ 有欠账时 repair_all 会修 ---
            live["mode"] = "dead"
            await store.append(
                request_id="retry-3", session_id="eval:retry:sess-2",
                messages=[{"role": "user", "timestamp": 1683561600000, "content": "again"}],
                **add,
            )
            live["mode"] = "ok"
            stats = await store_mod.repair_all(root, config=config,
                                               client=app.state.api.add_client, gate=gate)
            check("repair_all 修好欠账会话并记账",
                  stats["sessions"] == 1 and stats["fixed"] == 1 and stats["remaining"] == 0,
                  str(stats))

            # --- ⑥ 慢失败不磨请求：超出重试预算就不再试，交给补漏 ---
            # （真机上"建连超时 10s × 3 次"就是这种慢失败：请求延迟必须有个上界。）
            live["mode"] = "dead"
            before = live["atom_calls"]
            await store.append(
                request_id="retry-4", session_id="eval:retry:sess-3",
                messages=[{"role": "user", "timestamp": 1683561600000, "content": "slow fail"}],
                config=dict(config, atom_retry_budget=1e-6),
                client=app.state.api.add_client, gate=gate,
            )
            check("超出重试预算就只试一次（不磨请求）", live["atom_calls"] - before == 1,
                  str(live["atom_calls"] - before))
            state = json.loads(store.atom_state_path.read_text(encoding="utf-8"))
            check("没重试的那条照样记账（补漏会接手）",
                  state["4"].get("failed") == ["session_4_1"], str(state.get("4")))

            # --- ⑦ 旧记账格式（没有 failed 键）：靠"覆盖名单 vs 语料"的差集把欠账找出来 ---
            # 这正是**升级场景**：服务在旧版本下跑出过失败（那时还不记 failed），重启后只有
            # 差集能发现它们；而"从没抽过"的会话（状态里没条目）不会被算进来，不会整库重抽。
            # （先把 ⑥ 留下的欠账清掉 —— 走的是同一套补漏。）
            live["mode"] = "ok"
            await store.repair(config=config, client=app.state.api.add_client, gate=gate)
            check("慢失败留下的欠账也能补上", store.sessions_needing_repair() == [],
                  str(store.sessions_needing_repair()))

            await store.append(
                request_id="retry-5", session_id="eval:retry:sess-4",
                messages=[{"role": "user", "timestamp": 1683561600000, "content": f"legacy {i}"}
                          for i in range(2)],
                **add,
            )
            state = json.loads(store.atom_state_path.read_text(encoding="utf-8"))
            entry = state["5"]
            entry.pop("failed", None)                 # 抹成旧格式
            entry["atoms"] = entry["atoms"][:1]       # 假装当时有一条没抽成
            store.atom_state_path.write_text(json.dumps(state), encoding="utf-8")
            check("旧格式条目：差集能发现欠账", store.sessions_needing_repair() == [5],
                  str(store.sessions_needing_repair()))
            fixed = await store.repair(config=config, client=app.state.api.add_client,
                                       gate=gate)
            state = json.loads(store.atom_state_path.read_text(encoding="utf-8"))
            check("旧格式的欠账也被补上并转成新格式",
                  fixed == 1 and state["5"].get("failed") == [] and len(state["5"]["atoms"]) == 2,
                  str(state.get("5")))

            # --- ⑧ 差集不能误报：旧格式但覆盖完整 → 不算欠账；记账为空（旧语料）→ 也不欠账 ---
            state = json.loads(store.atom_state_path.read_text(encoding="utf-8"))
            state["5"].pop("failed", None)             # 旧格式，但覆盖是完整的
            store.atom_state_path.write_text(json.dumps(state), encoding="utf-8")
            check("旧格式但覆盖完整 → 不算欠账", store.sessions_needing_repair() == [],
                  str(store.sessions_needing_repair()))
            store.atom_state_path.write_text("{}", encoding="utf-8")
            check("记账为空（旧语料没有记账文件）→ 不认为欠账，不会整库重抽",
                  store.sessions_needing_repair() == [], str(store.sessions_needing_repair()))

        asyncio.run(scenario())
    finally:
        llm_mod.chat_completion = inner  # type: ignore[assignment]


def test_store_checker(tmp: Path) -> None:
    """只读体检脚本（`scripts/check_api_store.py`）：把"缺哪些 atom"算对。

    它是升级/补漏前后的那把尺子：补漏前看欠了什么、补漏后看是不是真补齐了。判据必须跟补漏本身
    （`UserStore.sessions_needing_repair`）一致，否则会出现"体检说干净、补漏还在干活"。
    """
    print("\n[16] 体检脚本：欠了哪些 atom")
    import importlib.util

    from codemem.add.memory import write_memories
    from codemem.io import make_memory

    path = PROJECT_ROOT / "scripts" / "check_api_store.py"
    spec = importlib.util.spec_from_file_location("check_api_store", path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)          # type: ignore[union-attr]

    root = tmp / "store"
    user_id = "eval:check:locomo:conv-0"
    store = store_mod.UserStore(user_id, root)
    store.root.mkdir(parents=True, exist_ok=True)
    write_memories(store.msgmem_path, [
        make_memory(memory=f"message {i}", memory_type="msg", mem_id=f"session_1_{i}",
                    role="Caroline") for i in (1, 2)
    ])
    write_memories(store.summem_path, [
        make_memory(memory="a summary", memory_type="summary", mem_id="session_1_summary"),
    ])
    write_memories(store.atommem_path, [
        make_memory(memory="Caroline likes hiking", memory_type="atom",
                    mem_id="session_1_1_1", role="Caroline", source=["session_1_1"]),
    ])

    # 旧格式记账（没有 failed 键）：靠差集发现 session_1_2 缺 atom
    store.atom_state_path.write_text(
        json.dumps({"1": {"summarized": 2, "atoms": ["session_1_1"]}}), encoding="utf-8"
    )
    info = checker.inspect_user(store)
    check("旧记账：差集算出缺哪一条",
          info["missing"] == [(1, ["session_1_2"], "旧记账（按差集推）")], str(info["missing"]))
    check("体检认得会话与总结", info["sessions"] == 1 and info["summaries"] == 1, str(info))

    # 新格式记账：显式 failed 说了算，理由要写成"记账说失败"
    store.atom_state_path.write_text(
        json.dumps({"1": {"summarized": 2, "atoms": ["session_1_1"],
                          "failed": ["session_1_2"]}}), encoding="utf-8"
    )
    info = checker.inspect_user(store)
    check("新记账：failed 说了算", info["missing"] == [(1, ["session_1_2"], "记账说失败")],
          str(info["missing"]))

    # 旧格式 + 空总结：老版本失败时写的是空总结，记账里没有痕迹 → 只能看 summem 有没有字
    write_memories(store.summem_path, [
        make_memory(memory="", memory_type="summary", mem_id="session_1_summary"),
    ])
    store.atom_state_path.write_text(
        json.dumps({"1": {"summarized": 2, "atoms": ["session_1_1", "session_1_2"]}}),
        encoding="utf-8",
    )
    info = checker.inspect_user(store)
    check("旧记账 + 空总结也算欠账",
          info["missing"] == [(1, [], "旧记账：总结为空")], str(info["missing"]))
    write_memories(store.summem_path, [
        make_memory(memory="a summary", memory_type="summary", mem_id="session_1_summary"),
    ])

    # 干净：覆盖完整 + 记账说没失败
    store.atom_state_path.write_text(
        json.dumps({"1": {"summarized": 2, "atoms": ["session_1_1", "session_1_2"],
                          "failed": []}}), encoding="utf-8"
    )
    check("补齐后体检干净", checker.inspect_user(store)["missing"] == [],
          str(checker.inspect_user(store)["missing"]))

    # 总结为空也算欠账（补漏会重跑它）
    store.atom_state_path.write_text(
        json.dumps({"1": {"atoms": ["session_1_1", "session_1_2"], "failed": [],
                          "summary_failed": True}}), encoding="utf-8"
    )
    check("总结为空也报出来",
          checker.inspect_user(store)["missing"] == [(1, [], "总结未完成")],
          str(checker.inspect_user(store)["missing"]))

    # 被端点拒绝（内容策略）：报出来，但**不是**会被自动重试的那种
    store.atom_state_path.write_text(
        json.dumps({"1": {"atoms": ["session_1_1"], "failed": [], "summary_failed": True,
                          "blocked": {"session_1_2": "BadRequestError: content_filter"}}}),
        encoding="utf-8",
    )
    info = checker.inspect_user(store)
    check("被端点拒绝的单独报，且说清不自动重试",
          info["missing"] == [(1, ["session_1_2"], "被端点拒绝（不自动重试）")],
          str(info["missing"]))


def test_moderation_and_summary_budget(tmp: Path) -> None:
    """两类"重发也没用"的失败 + summem 的输入预算。

    实测遇到的两条：

    1. ``400 content_filter``：prompt 被端点的内容策略拦下 —— 缩输入救不回来（拦的是内容不是长度），
       所以**不能**像抖动那样一直重试（否则每轮补漏都会把同一份 prompt 再烧一遍）。但触发词常常在
       **前文上下文**里，所以先去掉上下文试一次是值得的；试过还不行就记 ``blocked``（留痕、不自动重试）。
    2. ``400 context_length_exceeded``（summem）：把一个 15 万 token 的会话整段塞进总结 prompt。
       现在按预算**分段总结再拼起来**。
    """
    print("\n[17] 内容策略拦截与 summem 的输入预算")
    app = _app_with_temp_store(tmp)
    config = app.state.api.config
    root = Path(app.state.api.store_root)
    store = store_mod.UserStore("eval:mod:locomo:conv-0", root)
    inner = llm_mod.chat_completion

    class Filtered(RuntimeError):
        status_code = 400

    live = {"atom_calls": 0, "summem_calls": 0}

    def _split(body: str) -> tuple[str, str]:
        """把 prompt 拆成（前文上下文, 目标消息）—— 触发词在**哪一边**决定还能不能救。"""
        marker = "**Target message to extract**"
        if marker not in body:
            return "", body
        head, tail = body.split(marker, 1)
        return head, tail

    async def model(client, config, messages):  # noqa: ANN001
        system = str(messages[0].get("content", "")) if messages else ""
        body = str(messages[1].get("content", "")) if len(messages) > 1 else ""
        context, target = _split(body)
        if "extracting atomic facts" in system:
            live["atom_calls"] += 1
            # 触发词在**前文**里 → 端点拦；把上下文去掉就该放行（这正是那一层降级的意义）
            if "SENSITIVE" in context:
                raise Filtered("The response was filtered due to the prompt triggering "
                               "content management policy")
            # 触发词在**目标消息本身**里 → 去掉上下文也躲不掉
            if "banned-word" in target:
                raise Filtered("The response was filtered due to the prompt triggering "
                               "content management policy")
            found = re.findall(r"\[(session_\d+_\d+)\]", body)
            target_id = found[-1] if found else "session_1_0"
            return json.dumps([{"memory": f"{target_id} noted the harbor",
                                "time_expr": "", "role": "Caroline", "tag": ["topic:test"]}])
        if "You summarize ONE session" in system:
            live["summem_calls"] += 1
            if "banned-word" in body:
                raise Filtered("The response was filtered due to the prompt triggering "
                               "content management policy")
            return json.dumps({"memory": f"summary of a chunk ({len(body)} chars)"})
        return await inner(client, config, messages)

    llm_mod.chat_completion = model  # type: ignore[assignment]
    try:
        async def scenario() -> None:
            gate = asyncio.Semaphore(4)
            add = dict(config=config, client=app.state.api.add_client, gate=gate)

            # --- ① 触发词在**前文**里：去掉上下文重试一次就成功了 ---
            await store.append(
                request_id="mod-1", session_id="eval:mod:sess-0",
                messages=[
                    {"role": "user", "timestamp": 1683561600000,
                     "content": "something SENSITIVE happened"},
                    {"role": "user", "timestamp": 1683561660000, "content": "anyway, I moved"},
                ],
                **add,
            )
            state = json.loads(store.atom_state_path.read_text(encoding="utf-8"))
            check("先去掉上下文重试，救回来了（没记成 blocked）",
                  not state["1"].get("blocked") and not state["1"].get("failed")
                  and len(state["1"]["atoms"]) == 2, str(state.get("1")))
            check("确实是带上下文失败、去掉上下文成功（2 条消息 3 次调用）",
                  live["atom_calls"] == 3, str(live["atom_calls"]))

            # --- ② 触发词在**目标消息本身**里：重发没用 → 记 blocked，且**不再自动重试** ---
            before = live["atom_calls"]
            await store.append(
                request_id="mod-2", session_id="eval:mod:sess-1",
                messages=[{"role": "user", "timestamp": 1683561600000,
                           "content": "this one has a banned-word in it"}],
                **add,
            )
            state = json.loads(store.atom_state_path.read_text(encoding="utf-8"))
            entry = state["2"]
            check("端点不接受的那条记进 blocked", list(entry.get("blocked") or {}) == ["session_2_1"],
                  str(entry.get("blocked")))
            check("没有记成 failed（不该被自动重试）", entry.get("failed") == [], str(entry))
            check("只试了两次（带上下文 + 不带上下文），没进重试轮次",
                  live["atom_calls"] - before == 2, str(live["atom_calls"] - before))
            check("blocked 不算欠账", store.sessions_needing_repair() == [],
                  str(store.sessions_needing_repair()))
            check("--retry-blocked 时才把它算进来",
                  store.sessions_needing_repair(include_blocked=True) == [2],
                  str(store.sessions_needing_repair(include_blocked=True)))

            # --- ③ 换端点（这里用"不再拦内容"的模型）后重来一遍 ---
            live["summem_calls"] = 0
            llm_mod.chat_completion = _allow_all(inner)  # type: ignore[assignment]
            fixed = await store.repair(config=config, client=app.state.api.add_client, gate=gate,
                                       include_blocked=True)
            state = json.loads(store.atom_state_path.read_text(encoding="utf-8"))
            check("重来一遍后 blocked 清空、atom 抽出",
                  fixed >= 1 and not state["2"].get("blocked")
                  and "session_2_1" in state["2"]["atoms"], str(state.get("2")))

            # --- ④ summem 的输入预算：长会话分段总结，不再整段塞 ---
            live["summem_calls"] = 0
            lorem = " ".join(["word"] * 1200)          # 每条 ~1500 token
            await store.append(
                request_id="mod-3", session_id="eval:mod:sess-long",
                messages=[{"role": "user", "timestamp": 1683561600000, "content": lorem}
                          for _ in range(3)],
                config=dict(config, index={**(config.get("index") or {}),
                                           "max_input_tokens": 1600}),
                client=app.state.api.add_client, gate=gate,
            )
            check("长会话被分段总结（多次 summem 调用）", live["summem_calls"] >= 2,
                  str(live["summem_calls"]))
            summaries = read_jsonl(store.summem_path)
            check("分段总结带 part 标记",
                  any("(part " in str(s.get("memory") or "") for s in summaries),
                  str([s.get("memory") for s in summaries])[:200])

            # --- ⑤ 总结被端点拦下：记 summary_blocked，不自动重试 ---
            llm_mod.chat_completion = model  # type: ignore[assignment]  ← 换回"会拦内容"的端点
            await store.append(
                request_id="mod-4", session_id="eval:mod:sess-bad",
                messages=[{"role": "user", "timestamp": 1683561600000,
                           "content": "a session with a banned-word inside"}],
                **add,
            )
            before = live["summem_calls"]                  # 这一轮之后它就不该再被调用了
            state = json.loads(store.atom_state_path.read_text(encoding="utf-8"))
            check("总结被拦下记 summary_blocked", state["4"].get("summary_blocked") is True,
                  str(state.get("4")))
            check("被拦的总结不自动重试", store.sessions_needing_repair() == [],
                  str(store.sessions_needing_repair()))
            await store.repair(config=config, client=app.state.api.add_client, gate=gate)
            check("普通补漏不会再去碰它", live["summem_calls"] == before,
                  f"{live['summem_calls']} vs {before}")

        def _allow_all(fallback):  # noqa: ANN001, ANN202
            async def allowed(client, config, messages):  # noqa: ANN001
                system = str(messages[0].get("content", "")) if messages else ""
                body = str(messages[1].get("content", "")) if len(messages) > 1 else ""
                if "extracting atomic facts" in system:
                    found = re.findall(r"\[(session_\d+_\d+)\]", body)
                    target = found[-1] if found else "session_1_0"
                    return json.dumps([{"memory": f"{target} noted the harbor",
                                        "time_expr": "", "role": "Caroline",
                                        "tag": ["topic:test"]}])
                if "You summarize ONE session" in system:
                    live["summem_calls"] += 1
                    return json.dumps({"memory": "a clean summary"})
                return await fallback(client, config, messages)
            return allowed

        asyncio.run(scenario())
    finally:
        llm_mod.chat_completion = inner  # type: ignore[assignment]


def test_cli_surface() -> None:
    """CLI 表面：``python -m codemem.api`` 的参数能解析（恢复 / 补漏 / 重来历次被拒的）。

    `--repair` / `--recover` 是"服务没在跑时"手动补的那两条路，参数名写错了就等于线上少一条
    补救手段 —— 而它们平时不会被任何用例走到。
    """
    print("\n[18] CLI 入口")
    import importlib

    module = importlib.import_module("codemem.api.__main__")
    check("默认既不起服务也不补漏",
          module.parse_args([]).recover is False and module.parse_args([]).repair is False)
    check("--recover 可解析", module.parse_args(["--recover"]).recover is True)
    check("--repair 可解析", module.parse_args(["--repair"]).repair is True)
    check("--repair --retry-blocked 可解析",
          module.parse_args(["--repair", "--retry-blocked"]).retry_blocked is True)
    check("--port / --host 仍在", module.parse_args(["--port", "9000"]).port == 9000)


def test_merge_atoms_equivalence() -> None:
    """索引版合并（``merge_atoms``）与暴力版（``_dedup_atoms``）**结果一致**。

    这两条路必须等价 —— 前者是"把新抽的并进库里"的加速版（避免每次 Add 重跑 O(n²) 的两两比较），
    后者是它退化成"拿一整份列表去重"的特例。判据本身（Jaccard / 去情绪动词 / 专名不同不并）
    一条都不能因为加速而变。
    """
    print("\n[19] 去重：索引版与暴力版等价")
    from codemem.add.atommem import _dedup_atoms, merge_atoms
    from codemem.io import make_memory

    def atom(text: str, source: str) -> dict:
        return make_memory(memory=text, memory_type="atom", mem_id=f"{source}_1",
                           role="Caroline", source=[source])

    existing = [
        atom("Caroline likes hiking", "session_1_1"),
        atom("Caroline values self-expression", "session_2_1"),
        atom("Caroline has a pet named Luna", "session_3_1"),
        atom("Caroline went to the lake last summer", "session_4_1"),
    ]
    incoming = [
        atom("caroline likes hiking.", "session_5_1"),          # 精确重复（大小写/句号）
        atom("Caroline appreciates self-expression", "session_5_2"),  # 换情绪动词
        atom("Caroline has a pet named Bailey", "session_5_3"),  # 专名不同 → 不该并
        atom("Caroline likes painting", "session_5_4"),          # 全新
        atom("Caroline went to the lake last summer", "session_5_5"),  # 与已有完全相同
    ]
    merged = merge_atoms([dict(a) for a in existing], [dict(a) for a in incoming])
    brute = _dedup_atoms([dict(a) for a in existing] + [dict(a) for a in incoming])
    check("合并条数一致", len(merged) == len(brute), f"{len(merged)} vs {len(brute)}")
    check("留下的正文逐条一致",
          [a["memory"] for a in merged] == [a["memory"] for a in brute],
          f"{[a['memory'] for a in merged]} vs {[a['memory'] for a in brute]}")
    check("source 并集逐条一致",
          [sorted(a["metadata"]["source"]) for a in merged]
          == [sorted(a["metadata"]["source"]) for a in brute],
          str([a["metadata"]["source"] for a in merged]))
    check("专名不同的两条各自留着",
          any("Luna" in a["memory"] for a in merged)
          and any("Bailey" in a["memory"] for a in merged), str([a["memory"] for a in merged]))


def test_dedup_scale() -> None:
    """去重的**量级**：这是"API 看起来卡住"那一步的实测数字（默认只跑最小的一档）。

    为什么单列一条：它只是给人看的数字，不是断言 —— 但换掉实现时得能一眼看出差多少。
    ``python scripts/test_api.py --bench`` 会多跑几档（几千条那一档用旧实现要几十秒）。
    """
    sizes = (1000, 2000) if "--bench" in sys.argv else (300,)
    print("\n[20] 去重量级（库里已有 N 条 + 本次新增 20 条 = 一次 Add 的合并）")
    from codemem.add.atommem import _absorb, merge_atoms
    from codemem.io import make_memory
    from codemem.similarity import dedup_key, is_same_fact

    def brute(atoms: list[dict]) -> list[dict]:
        """**旧实现**（留在这里当基准与参照）：整份列表两两比 —— O(n²) 的那一版。"""
        exact: dict[str, dict] = {}
        order: list[str] = []
        for atom in atoms:
            key = dedup_key(str(atom.get("memory") or ""))
            if not key:
                continue
            kept = exact.get(key)
            if kept is None:
                exact[key] = atom
                order.append(key)
            else:
                _absorb(kept, atom)
        deduped: list[dict] = []
        for key in order:
            atom = exact[key]
            text = str(atom.get("memory") or "")
            target = None
            for kept in deduped:
                if is_same_fact(text, str(kept.get("memory") or "")):
                    target = kept
                    break
            if target is None:
                deduped.append(atom)
            else:
                _absorb(target, atom)
        return deduped

    words = ("hiking painting adoption agency interviews pottery camp lake concert guitar "
             "garden therapy school daughter husband summer winter charity race window "
             "market client project meeting design review code test deploy server").split()

    def atom(index: int) -> dict:
        rng = random.Random(index)
        return make_memory(memory=f"Caroline {' '.join(rng.sample(words, 6))} number{index}",
                           memory_type="atom", mem_id=f"session_1_{index}_1",
                           role="Caroline", source=[f"session_1_{index}"])

    def build(size: int) -> list[dict]:
        """按**真实路径**造库：一条条 merge 进来（所以库里本来就没有近重复）。

        直接造一个近重复密集的列表去比，比的就不是"两条路等价与否"了 —— 那个列表本来就
        只有暴力版会去重（它连"库里已有的两两之间"也比一遍）。
        """
        corpus: list[dict] = []
        for index in range(size):
            corpus = merge_atoms(corpus, [atom(index)])
        return corpus

    for size in sizes:
        kept = build(size)
        fresh = [atom(size + i) for i in range(20)]
        started = time.monotonic()
        old = brute([dict(a) for a in kept] + [dict(a) for a in fresh])
        old_time = time.monotonic() - started
        started = time.monotonic()
        new = merge_atoms([dict(a) for a in kept], [dict(a) for a in fresh])
        new_time = time.monotonic() - started
        same = [a["memory"] for a in old] == [a["memory"] for a in new]
        check(f"库里 {size} 条：索引版与暴力版结果一致", same,
              f"{len(old)} vs {len(new)}")
        print(f"    暴力(O(n²)) {old_time:7.2f}s / 索引 {new_time:6.3f}s"
              f"（快 {old_time / max(new_time, 1e-6):5.0f}×）")


def test_server_knobs(tmp: Path) -> None:
    """服务端的几把闸门：Add / Search / 模型调用 / 补漏，各自的额度从哪来。

    补漏必须**有自己的小额度**：它跑得久又是后台活，占了 add 那 32 个名额就会让前台请求排队
    等好几分钟 —— 那正是"服务刚起来像是卡住了"的一种形态。
    """
    print("\n[21] 服务端闸门")
    from codemem.api.server import ApiState

    app = _app_with_temp_store(tmp)
    base = dict(app.state.api.config)
    state = ApiState(dict(base, concurrency=32, add_concurrency=32))
    check("Search / Add / 模型 三把闸门按配置",
          state.search_sem._value == 32 and state.add_sem._value == 32
          and state.add_gate._value == 32, "")
    check("补漏闸门独立、且远小于 Add 闸门",
          state.repair_gate is not state.add_gate and 0 < state.repair_gate._value <= 8,
          str(state.repair_gate._value))
    tuned = ApiState(dict(base, add_concurrency=32, model_concurrency=4, repair_concurrency=2))
    check("model_concurrency 只压模型调用、不压接请求数",
          tuned.add_sem._value == 32 and tuned.add_gate._value == 4, "")
    check("repair_concurrency 可单独配", tuned.repair_gate._value == 2, "")


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
    test_atommem_budget()
    test_cli_surface()
    test_merge_atoms_equivalence()
    test_dedup_scale()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        test_config_precedence(tmp)
        test_endpoints(tmp)
        test_validation(tmp)
        test_resume(tmp)
        test_adapters_end_to_end(tmp / "e2e")
        test_participants_block(tmp / "participants")
        test_corpus_cache(tmp / "cache")
        test_add_concurrency(tmp / "conc")
        test_atom_retry_and_repair(tmp / "retry")
        test_store_checker(tmp / "check")
        test_moderation_and_summary_budget(tmp / "mod")
        test_server_knobs(tmp / "knobs")

    print(f"\n{'=' * 60}\n通过 {PASS} / 失败 {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
