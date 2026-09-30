"""Search 后端：复用本仓库的 code-agent 检索管线，把 ``query`` 变成协议要求的 ``data[]``。

流程（与 ``search.runner.run_qa`` 同一套编排，只是单条 query、且产物要映射成协议）：

    query ─► 建运行目录（硬链两层语料进 inputs/）
          ─► agent loop（grep 检索 → 写 evidence.jsonl，read/write/edit/bash）
          ─► verifier 判"够不够" → 不够就把 missing[] 回灌再跑一轮
          ─► 校验 evidence.jsonl → 排序列成 data[]

**为什么复用而不是另写一个轻量检索**：本仓库的核心主张就是"证据是**推导**出来的" ——
agent 会把 ``yesterday`` 折算成绝对日期、消解指代、把散落事实合并成一条能直接回答的记录。
一个朴素的关键词检索做不到这些，而协议要的恰恰是"能直接回答问题的记忆列表"。

代价：每条 query 一次（或多次）LLM agent 调用 —— 慢、有成本、非确定性。这是刻意的取舍。

**不改动 search 现有代码**：只 import 它的组件（``ModelRunner`` / ``run_agent_round`` / ``verify`` /
``validate_file`` / ``load_catalog`` / ``build_tools``）。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..log import Logger
from ..search import agent as agent_mod
from ..search import evidence as evidence_mod
from ..search import runner as runner_mod
from ..search import toolcfg
from ..search import tools as tools_mod
from .store import UserStore


# ---------------------------------------------------------------------------
# 运行目录
# ---------------------------------------------------------------------------

def _new_run_dir(store: UserStore) -> Path:
    """给这次 Search 建一个带时间戳的运行目录（可审计：能看到 inputs/ 与 evidence.jsonl）。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = store.root / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _link_inputs(store: UserStore, run_dir: Path) -> Path:
    """把该 user 的两层语料硬链进 ``run_dir/inputs/``（只读输入，不改动源文件）。"""
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for name in runner_mod.CORPUS_FILES:
        source = store.root / name
        if source.exists():
            runner_mod.link_or_copy(source, inputs_dir / name)
    # 空文件占位：agent 的 grep 面对不存在的文件会报错，给个空文件更友好（"确实没有记忆"）
    for name in runner_mod.CORPUS_FILES:
        target = inputs_dir / name
        if not target.exists():
            target.touch()
    return inputs_dir


# ---------------------------------------------------------------------------
# 产物 → 协议 data[]
# ---------------------------------------------------------------------------

def _memory_id(record: dict[str, Any]) -> str:
    """稳定 id：``mem_{sha1(content + sorted(source))[:12]}``。

    同一份证据在多次 Search 里得到同一个 id（内容与来源都相同 → 同一条记忆），
    便于外部按 id 去重 / 追踪。
    """
    content = str(record.get("content") or "")
    source = (record.get("metadata") or {}).get("source") or []
    material = content + "\x00" + "\x00".join(sorted(str(s) for s in source))
    return "mem_" + hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]


def _created_at(record: dict[str, Any]) -> str | None:
    """取该记忆的时间：优先 ``metadata.changelog[0].time``（持久化时间），否则空。"""
    meta = record.get("metadata") or {}
    changelog = meta.get("changelog")
    if isinstance(changelog, list) and changelog and isinstance(changelog[0], dict):
        value = changelog[0].get("time")
        if isinstance(value, str) and value:
            return value
    for key in ("time", "created_at"):
        value = meta.get(key) or record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def records_to_data(
    records: list[dict[str, Any]], top_k: int | None = None
) -> list[dict[str, Any]]:
    """把 evidence 记录映射成协议 ``data[]``。

    - ``content`` 非空（``validate_file`` 已保证）
    - ``score`` 取 ``metadata.score``（0-1，已在 ``evidence.normalize`` 里夹取）
    - 按 ``score`` 降序 → 保证"数值越高越相关"，与协议一致
    - ``top_k`` 截断（None 或 <=0 表示不截）
    """
    items: list[dict[str, Any]] = []
    for record in records:
        content = str(record.get("content") or "").strip()
        if not content:
            continue
        meta = record.get("metadata") or {}
        score = meta.get("score")
        score = float(score) if isinstance(score, (int, float)) else None
        items.append({
            "id": _memory_id(record),
            "content": content,
            "score": score,
            "created_at": _created_at(record),
        })

    items.sort(key=lambda item: (item["score"] is not None, item["score"] or 0.0), reverse=True)
    if top_k is not None and top_k > 0:
        items = items[:top_k]
    return items


# ---------------------------------------------------------------------------
# agent 检索
# ---------------------------------------------------------------------------

def _agent_config(config: dict[str, Any]) -> dict[str, Any]:
    """search agent loop 的参数：``generator`` 段与顶层混成一份。

    直接复用 ``search.runner`` 的默认值，只覆盖 api 配置显式给了的那几个键。
    """
    merged = json.loads(json.dumps(runner_mod.DEFAULT_CONFIG))
    generator = config.get("generator")
    if isinstance(generator, dict):
        merged["generator"].update(generator)
    for key in (
        "max_steps", "no_progress_patience", "max_verify_rounds", "observation_max_chars",
        "write_nudge_after", "evidence_soft_limit", "context_max_chars", "context_keep_recent",
        "bash_timeout", "timeout", "max_retries", "backoff_base", "rate_limit_backoff",
        "max_backoff", "backoff_jitter", "repeat_limit", "max_parse_retries",
    ):
        if key in config and config[key] is not None:
            merged[key] = config[key]
    return merged


async def search(
    *,
    store: UserStore,
    query: str,
    config: dict[str, Any],
    client: Any,
    tool_config: str | Path | None = None,
    log: Logger | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """对一条 query 跑检索，返回协议 ``data[]``（无记忆/无证据 → 空列表）。

    ``query`` 已由调用方归一化成字符串（多模态 → 文本）。
    """
    log = log or Logger(level="warn")
    if not store.exists():
        log.info("该 user 没有任何记忆，返回空结果")
        return []

    run_dir = _new_run_dir(store)
    inputs_dir = _link_inputs(store, run_dir)
    qa_dir = run_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = qa_dir / "evidence.jsonl"
    # agent 的 cwd 是 qa/，而它的 prompt 与 system 里每个检索示例都写 `inputs/...`
    # （相对 cwd）。所以 qa/ 下必须有 `inputs` 这一项 —— 与 `runner.run_qa` 完全一致。
    runner_mod.link_dir_or_copy(inputs_dir, qa_dir / "inputs")

    catalog = toolcfg.load_catalog(tool_config)
    catalog.validate(["read", "write", "edit", "bash"])

    agent_cfg = _agent_config(config)
    shell_prefix = runner_mod.build_shell_prefix(inputs_dir=inputs_dir, config=agent_cfg)
    tools = tools_mod.build_tools(
        cwd=qa_dir,
        shell_command_prefix=shell_prefix,
        bash_timeout=float(agent_cfg.get("bash_timeout", 60)),
    )
    catalog.apply(tools)

    model = runner_mod.ModelRunner(
        client, runner_mod.gen_call_config(agent_cfg), log,
        run_dir / "model_calls.jsonl",
        full_messages=bool(agent_cfg.get("log_full_messages", False)),
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": catalog.render_prompt(query)},
        {"role": "user", "content": catalog.prompt_text("user_begin")},
    ]

    max_steps = int(agent_cfg.get("max_steps", 30))
    max_verify_rounds = max(1, int(agent_cfg.get("max_verify_rounds", 2)))
    step_offset = 0

    try:
        for round_index in range(1, max_verify_rounds + 1):
            outcome, messages = await agent_mod.run_agent_round(
                messages=messages,
                tools=tools,
                model_call=model,
                evidence_path=evidence_path,
                cwd=qa_dir,
                max_steps=max_steps,
                no_progress_patience=int(agent_cfg.get("no_progress_patience", 6)),
                step_offset=step_offset,
                round_index=round_index,
                log=log,
                observation_max_chars=int(agent_cfg.get("observation_max_chars", 8000)),
                write_nudge_after=int(agent_cfg.get("write_nudge_after", 5)),
                evidence_soft_limit=int(agent_cfg.get("evidence_soft_limit", 12)),
                max_parse_retries=int(agent_cfg.get("max_parse_retries", 2)),
                repeat_limit=int(agent_cfg.get("repeat_limit", 1)),
                duplicate_warn_at=int(agent_cfg.get("duplicate_warn_at", 10)),
                context_max_chars=int(agent_cfg.get("context_max_chars", 0)),
                context_keep_recent=int(agent_cfg.get("context_keep_recent", 8)),
            )
            step_offset += len(outcome.steps)

            report = evidence_mod.validate_file(evidence_path)
            verdict = await _verify(
                query=query, report=report, model=model, log=log,
                round_index=round_index, evidence_path=evidence_path,
            )
            log.info(f"round {round_index}：{outcome.summary()} | {verdict.describe()}")
            if verdict.sufficient or round_index >= max_verify_rounds:
                break

            missing = verdict.missing or [
                "(the auditor gave no specific gap; re-read the question and your evidence)"
            ]
            messages.append({
                "role": "user",
                "content": catalog.prompt_text(
                    "user_gaps",
                    missing="\n".join(f"- {item}" for item in missing),
                    attempted_answer=verdict.answer or "(no answer could be produced)",
                ),
            })
    finally:
        model.close()

    # 收尾：规范化产物（形态偏离"一行一条"时写回），再取交付列表
    report = evidence_mod.validate_file(evidence_path)
    if report.rejected or report.parse_errors or report.empty_lines or report.multi_record_lines:
        evidence_mod.write_normalized(evidence_path, report)
        report = evidence_mod.validate_file(evidence_path)

    data = records_to_data(report.valid, top_k=top_k)
    log.info(f"检索完成：evidence {len(report.valid)} 条 -> data {len(data)} 条（{run_dir}）")
    return data


async def _verify(
    *,
    query: str,
    report: Any,
    model: Any,
    log: Logger,
    round_index: int,
    evidence_path: Path,
) -> Any:
    """把校验后的证据交给 verifier 判"够不够"（空证据时 verifier 自己会跳过模型调用）。"""
    from ..search import verifier as verifier_mod

    return await verifier_mod.verify(
        question=query,
        memories=report.valid,
        model_call=model,
        log=log,
        round_index=round_index,
        evidence_present=(evidence_path.exists() and evidence_path.stat().st_size > 0),
    )
