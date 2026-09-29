"""answer 步骤：question + evidence.jsonl → 答案。

**输入是 search 的产物 ``evidence.jsonl``**，不是整个记忆库 —— 这正是整条链路的意义：
search 已经把"回答这个问题所需的记忆"提炼成一份小列表（含推理结论、时间已解析），
answer 只需要基于它作答，不需要再做检索。

设计要点：

- **只在证据内作答**。prompt 明确要求"仅用给定记忆，不得编造日期/数字/人名"；
  不足以回答时输出 ``{"answer": null, "unsupported": true}``，而不是猜。
  这个 ``unsupported`` 是 eval 阶段区分"答错"和"证据不足"的依据。
- **已解析过的时间不再重算**。search 阶段的 agent 已经把相对表述锚定成可独立理解的
  时间（含"9 June 之前的那一周"这类**本来就该保留锚点**的表述），所以 prompt 里强调
  直接使用，避免二次推理引入新错误。
- **输入格式统一**：映 evidence 记录时带上 ``time`` 与 ``speaker`` —— 时间推理依赖它。

用法::

    python -m codemem.answer --sample Caroline_Melanie
    python -m codemem.answer --evidence data/search_runs/x/qa_0/evidence.jsonl
    python -m codemem.answer --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import dataset, llm
from ..io import DATA_DIR, PROJECT_ROOT, msg_content, read_jsonl
from ..log import Logger
from ..prompts import ANSWER_PROMPT

CONFIG_FILE = PROJECT_ROOT / "configs" / "answer.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "model": "local",
    "base_url": "http://127.0.0.1:30000/v1",
    "api_key": "sglang",
    "temperature": 0.2,
    "max_tokens": 2048,
    "concurrency": 8,
    "timeout": 300,
    "max_retries": 3,
    "enable_thinking": False,
    "log": {"level": "info", "output": "data/answer_runs"},
}


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def load_config(path: Path | None = None) -> dict[str, Any]:
    """读 ``configs/answer.yaml``，缺省值兜底。"""
    import yaml

    config = json.loads(json.dumps(DEFAULT_CONFIG))
    target = path or CONFIG_FILE
    if target.exists():
        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                    config[key].update(value)
                else:
                    config[key] = value
    return config


# ---------------------------------------------------------------------------
# 证据 → 上下文
# ---------------------------------------------------------------------------

def render_evidence(memories: list[dict[str, Any]]) -> str:
    """把 evidence 记录渲染成给 answer 模型的上下文。

    **必须走 ``io.msg_content``**。这里曾经读 v2 的 ``memory`` 字段，而新的 evidence 是
    ``content`` —— 于是每条渲染成 ``[unknown time] unknown:``，**正文是空的**。
    实测后果：全量 1422/1487 条回答报"证据不足"，因为模型确实什么都没看到。

    带上 ``source``：answer 阶段的推理需要知道这条证据出自哪些消息（时间锚点、
    指代关系常常依赖它）。
    """
    if not memories:
        return "(no memories provided)"
    lines: list[str] = []
    for memory in memories:
        meta = memory.get("metadata") or {}
        text = msg_content(memory) or json.dumps(memory, ensure_ascii=False)
        when = str(memory.get("time") or meta.get("time") or "").strip()
        source = meta.get("source")
        source_text = ", ".join(str(s) for s in source) if isinstance(source, list) else ""
        head = f"[{when}] " if when else ""
        tail = f"  (from: {source_text})" if source_text else ""
        lines.append(f"{head}{text}{tail}")
    return "\n".join(lines)


def current_date_hint(memories: list[dict[str, Any]]) -> str:
    """从证据里取最晚的日期作为"当前日期"。

    提示模型时间轴的方向（"a month ago" 是相对哪一天）。**扫正文而不只扫 time 字段**：
    evidence 记录通常没有 ``time``（模板里只有 content 与 source），日期是写在正文里的
    （search 阶段已把相对时间折算成绝对日期）。取不到就返回今天。
    """
    latest: datetime | None = None
    for memory in memories:
        candidates = [str(memory.get("time") or "")]
        candidates.append(msg_content(memory))
        for text in candidates:
            for match in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", text or ""):
                try:
                    moment = datetime(*(int(match.group(i)) for i in (1, 2, 3)))
                except ValueError:
                    continue
                if latest is None or moment > latest:
                    latest = moment
    return (latest or datetime.now()).strftime("%Y-%m-%d")


def build_messages(question: str, memories: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": ANSWER_PROMPT.format(
                question=question,
                context=render_evidence(memories),
                current_date=current_date_hint(memories),
            ),
        },
        {"role": "user", "content": "Return the answer now, as the JSON object described above."},
    ]


# ---------------------------------------------------------------------------
# 输出解析
# ---------------------------------------------------------------------------

@dataclass
class AnswerResult:
    """一条回答。``unsupported=True`` 表示模型认为证据不足（与"答错"不同）。"""

    answer: str | None = None
    unsupported: bool = False
    reasoning: str = ""
    raw: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "unsupported": self.unsupported,
            "reasoning": self.reasoning,
            "error": self.error,
        }


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def parse_answer(text: str) -> AnswerResult:
    """稳健解析模型输出为 ``AnswerResult``。任何失败都退化为 ``answer=None``。

    容错顺序：整段 JSON → 剥代码围栏 → 抓最外层 ``{...}`` → ``<answer>`` 标签 → 纯文本。
    最后两级不是"猜"：模型偶尔会只回答答案本身，那也是有信息的，记下来并交给 judge 判。
    """
    raw = text or ""
    if not raw.strip():
        return AnswerResult(unsupported=True, raw=raw, error="empty reply")

    parsed = _extract_json(raw)
    if parsed is not None:
        return _from_dict(parsed, raw)

    tag = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.DOTALL | re.IGNORECASE)
    if tag:
        return AnswerResult(answer=tag.group(1).strip() or None, raw=raw)

    cleaned = _FENCE_RE.sub("", raw.strip()).strip().strip("`").strip()
    return AnswerResult(answer=cleaned or None, raw=raw)


def _extract_json(text: str) -> dict[str, Any] | None:
    candidates = [text.strip(), _FENCE_RE.sub("", text.strip()).strip()]
    for candidate in candidates:
        if not candidate.startswith("{"):
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
        if isinstance(value, dict):
            return value
    return None


def _from_dict(parsed: dict[str, Any], raw: str) -> AnswerResult:
    """从解析出的 JSON 建结果。

    **``unsupported`` 尊重模型的显式判断，只在缺失时才由"答案是否为空"推断。**
    这一点是刻意的：``unsupported`` 是 eval 区分"证据不足"和"答错了"的唯一依据，而这两者
    的修法完全不同（前者改 search 的检索/推理，后者改 answer 的 prompt）。模型说
    "证据够但我答不出"（``unsupported=false`` + 空答案）是**answer 步骤**的问题，
    不该被归成证据不足 —— 所以那个矛盾的组合按模型的显式声明走，由 judge 判错。
    """
    answer = parsed.get("answer", parsed.get("prediction", parsed.get("final_answer")))
    if isinstance(answer, str):
        answer = answer.strip()
    if answer == "":
        answer = None
    unsupported = parsed.get("unsupported")
    if not isinstance(unsupported, bool):
        unsupported = answer is None
    reasoning = parsed.get("reasoning", parsed.get("reason", parsed.get("explanation", "")))
    if not isinstance(reasoning, str):
        reasoning = json.dumps(reasoning, ensure_ascii=False)
    return AnswerResult(
        answer=answer, unsupported=bool(unsupported), reasoning=reasoning.strip(), raw=raw
    )


# ---------------------------------------------------------------------------
# 单条 / 批量
# ---------------------------------------------------------------------------

async def answer_question(
    question: str,
    memories: list[dict[str, Any]],
    config: dict[str, Any],
    client: Any,
    log: Logger | None = None,
) -> AnswerResult:
    """回答一条问题。模型调用失败退化为 ``error``（不抛异常，让调用方决定怎么记账）。"""
    messages = build_messages(question, memories)
    try:
        raw = await llm.chat_completion(client, config, messages)
    except Exception as exc:  # noqa: BLE001 - 单条失败不该拖垮整批
        if log is not None:
            log.warn(f"answer 调用失败：{type(exc).__name__}: {exc}")
        return AnswerResult(unsupported=True, error=f"{type(exc).__name__}: {exc}")
    if log is not None:
        log.debug(f"evidence {len(memories)} 条 -> answer={parse_answer(raw).answer!r}")
    return parse_answer(raw)


@dataclass
class AnswerJob:
    """一条待回答的任务：question + 它的证据文件。"""

    qa_index: int | None
    question: str
    evidence_path: Path | None = None
    memories: list[dict[str, Any]] = field(default_factory=list)
    reference: str = ""
    category: Any = None
    # QA 的标注证据标记（如 "D1:3"），透传给 eval 算 evidence_recall —— 少了它 recall 恒为 0
    evidence: list[str] = field(default_factory=list)


def resolve_evidence_path(candidate: Path) -> Path | None:
    """``candidate`` 既可以是 evidence.jsonl 文件，也可以是 qa_{idx} 目录。"""
    if candidate.is_file() and candidate.suffix == ".jsonl":
        return candidate
    nested = candidate / "evidence.jsonl"
    if nested.is_file():
        return nested
    return None


def collect_jobs(
    evidence_dir: Path | None,
    label: str = "",
    limit: int = 0,
    skip_category5: bool = True,
) -> list[AnswerJob]:
    """收集待回答任务。

    两种模式：

    - ``evidence_dir`` 指向 search 的一次运行目录（含 ``{dir}/qa_{idx}/evidence.jsonl``）：
      问题从数据集取，答案写到该目录下。
    - ``evidence_dir`` 指向单个 ``evidence.jsonl``：只能拿到证据，问题需要 ``--question``。
    """
    jobs: list[AnswerJob] = []
    if evidence_dir is None:
        return jobs

    single = resolve_evidence_path(evidence_dir)
    if single is not None:
        jobs.append(AnswerJob(qa_index=None, question="", evidence_path=single))
        return jobs[:limit] if limit else jobs

    labels = [label] if label else sorted(
        path.name for path in evidence_dir.iterdir() if path.is_dir()
    )
    for dir_name in labels:
        directory = evidence_dir / dir_name
        if not directory.is_dir():
            continue
        try:
            questions = dataset.questions_for_dir(dir_name, skip_category5=skip_category5)
        except (OSError, ValueError, FileNotFoundError):
            questions = []
        by_index = {index: qa for index, qa in questions}
        for qa_dir in sorted(
            directory.glob("qa_*"), key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0
        ):
            path = resolve_evidence_path(qa_dir)
            if path is None:
                continue
            index = int(qa_dir.name.split("_")[1])
            qa = by_index.get(index, {})
            jobs.append(
                AnswerJob(
                    qa_index=index,
                    question=str(qa.get("question") or ""),
                    evidence_path=path,
                    reference=str(dataset.reference_answer(qa)) if qa else "",
                    category=qa.get("category"),
                    evidence=list(qa.get("evidence") or []),
                )
            )
    return jobs[:limit] if limit else jobs


async def run_jobs(
    jobs: list[AnswerJob],
    config: dict[str, Any],
    log: Logger,
    concurrency: int | None = None,
) -> list[dict[str, Any]]:
    """并发回答全部任务，返回逐条结果（含 question/reference，便于 eval 直接用）。"""
    client = llm.make_client(config)
    semaphore = asyncio.Semaphore(max(1, concurrency or int(config.get("concurrency", 8))))
    results: list[dict[str, Any]] = []

    async def worker(job: AnswerJob) -> dict[str, Any]:
        async with semaphore:
            memories = read_jsonl(job.evidence_path) if job.evidence_path else job.memories
            if not job.question:
                # 单文件模式：没有问题文本，用 QA 数据源按目录猜是做不到的 —— 交给调用方
                log.warn(f"{job.evidence_path}: 没有对应的问题文本，跳过")
                return {}
            result = await answer_question(job.question, memories, config, client, log)
            log.info(
                f"qa{job.qa_index} evidence={len(memories)} "
                f"answer={result.answer!r}"
                + (" (unsupported)" if result.unsupported else "")
            )
            payload = {
                "qa_index": job.qa_index,
                "question": job.question,
                "reference": job.reference,
                "category": job.category,
                "evidence": job.evidence,
                "evidence_count": len(memories),
                "evidence_path": str(job.evidence_path) if job.evidence_path else "",
                **result.to_dict(),
            }
            return payload

    try:
        for coroutine in asyncio.as_completed([worker(job) for job in jobs]):
            payload = await coroutine
            if payload:
                results.append(payload)
    finally:
        await client.close()
    results.sort(key=lambda item: (item.get("qa_index") is None, item.get("qa_index") or 0))
    return results


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m codemem.answer",
        description="answer：根据 question + evidence.jsonl 回答（只用证据，不检索）",
    )
    parser.add_argument("--config", default=str(CONFIG_FILE), help="配置（默认 configs/answer.yaml）")
    parser.add_argument("--evidence-dir", default="", help="search 的运行目录；缺省取最新一次")
    parser.add_argument("--sample", default="", help="只处理指定 speaker 目录")
    parser.add_argument("--limit", type=int, default=0, help="最多回答 N 条（调试用）")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖并发数")
    parser.add_argument("--output", default="", help="结果输出路径（默认写到运行目录下）")
    parser.add_argument("--include-category5", action="store_true", help="包含 adversarial 类")
    parser.add_argument("--log-level", default="", help="debug/info/warn/error/silent")
    parser.add_argument("--log-output", default=None, help="日志目录；空串=只打终端")
    return parser.parse_args(argv)


def latest_search_run(root: Path | None = None) -> Path | None:
    """找最近一次 search 运行目录（``data/search_runs/{experiment}_{ts}``）。"""
    base = root or (DATA_DIR / "search_runs")
    if not base.exists():
        return None
    candidates = sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name)
    return candidates[-1] if candidates else None


def build_logger(config: dict[str, Any], args: argparse.Namespace) -> Logger:
    section = dict(config.get("log") or {})
    if args.log_level:
        section["level"] = args.log_level
    if args.log_output is not None:
        section["output"] = args.log_output
    return Logger.from_config({"log": section}, prefix="answer")


async def async_main(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    log = build_logger(config, args)

    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else latest_search_run()
    if evidence_dir is None or not evidence_dir.exists():
        log.error(
            "找不到 search 的运行目录。先用 `python -m codemem.search` 生成 evidence.jsonl，"
            "或用 --evidence-dir 指定。"
        )
        return
    log.info(f"evidence 目录：{evidence_dir}")

    jobs = collect_jobs(
        evidence_dir,
        label=args.sample,
        limit=args.limit,
        skip_category5=not args.include_category5,
    )
    if not jobs:
        log.error(f"{evidence_dir} 下没有找到 evidence.jsonl（期待 {{dir}}/qa_{{idx}}/evidence.jsonl）")
        return
    log.info(f"待回答 {len(jobs)} 条 / 并发 {args.concurrency or config.get('concurrency')}")

    results = await run_jobs(jobs, config, log, args.concurrency)

    output = Path(args.output) if args.output else evidence_dir / "answers.jsonl"
    from ..io import write_jsonl

    write_jsonl(output, results)
    unsupported = sum(1 for item in results if item.get("unsupported"))
    log.info(f"完成 {len(results)} 条（证据不足 {unsupported} 条）-> {output}")
    log.info(log.summary())
    log.close()


def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(parse_args(argv)))


if __name__ == "__main__":
    main()
