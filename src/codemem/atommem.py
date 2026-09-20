"""
基于从原始对话数据初始化的memory.jsonl去继续尽可能地提取生成原子记忆，包括显示隐式的原子记忆，尽可能覆盖每个message memory的每个有实际含义的单词，原子记忆要求简要且信息完整，核心在于提取的准确性，
时间需要准确，只有在确定的时候才写入确定性地时间，当比较模糊不确定的时候写入相对时间描述等。每次处理一个message memory，但是需要给一定上下文窗口的信息，可以通过设置上下文窗口的大小来控制，默认是前后各3条message memory。上下文窗口内的message memory可以作为当前message memory的上下文信息，帮助提取原子记忆。

流程：
    1. 读取 data/{speaker_a}_{speaker_b}/msgmem.jsonl 中的原始 message（按文件顺序，即会话顺序）。
    2. 对每条 message memory，截取前后各 N 条（N = context_window，默认 3）作为上下文窗口。
    3. 并行调用大模型（OpenAI 兼容 /v1/chat/completions），一次处理一个 message memory，
       要求其参考 memory_template.json 的定义，把该 message 拆成若干原子记忆。
    4. 原始 msgmem.jsonl 保持不变，原子记忆单独写入 atommem.jsonl。

用法：
    python -m codemem.atommem                      # 处理全部 speaker 目录
    python -m codemem.atommem Caroline_Melanie     # 只处理指定目录
    python -m codemem.atommem --limit 50           # 每个目录只处理前 50 条 raw（调试用）
    python -m codemem.atommem --concurrency 4 --context-window 3
    python -m codemem.atommem --resume Caroline_Melanie  # 跳过已完成，只重试失败项
"""

import argparse
import asyncio
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai
import yaml

from .prompts import ATOM_EXTRACTION_SYSTEM_PROMPT, ATOM_EXTRACTION_USER_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_FILE = PROJECT_ROOT / "configs" / "model.yaml"
TEMPLATE_FILE = PROJECT_ROOT / "memory_template_init_extract.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "base_url": "http://127.0.0.1:11434/v1",
    "api_key": "ollama",
    "model": "qwen3:14b",
    "temperature": 0.2,
    "max_tokens": 4096,
    "concurrency": 4,
    "timeout": 300,
    "max_retries": 6,
    "backoff_base": 2.0,
    "rate_limit_backoff": 10.0,
    "max_backoff": 120.0,
    "backoff_jitter": 0.5,
    "context_window": 3,
    "enable_thinking": False,
}

# 原子记忆 id 前缀：以源 raw memory id 为根，追加序号
ATOM_ID_RE = re.compile(r"^session_\d+_\d+_\d+$")


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------

def load_config() -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        config.update(loaded)
    return config


def load_template() -> dict:
    if TEMPLATE_FILE.exists():
        return json.loads(TEMPLATE_FILE.read_text(encoding="utf-8"))
    return {}


def _is_you_field(spec: Any) -> bool:
    """模板里含 [YOU] 表示由模型填写（如 type 为 '[SYSTEM for raw] ... [YOU for inner/outer]'）。"""
    if isinstance(spec, dict):
        return "[YOU" in str(spec.get("description", ""))
    return False


def _field_line(name: str, spec: dict) -> str:
    """把模板字段渲染成一行紧凑说明：name (type[, values]) - description"""
    ftype = spec.get("type", "string")
    if spec.get("values"):
        ftype = " | ".join(str(v) for v in spec["values"])
    # 去掉 [YOU]/[SYSTEM] 标记，保留可读描述
    desc = re.sub(r"\[(?:SYSTEM[^\]]*|YOU[^\]]*)\]\s*", "", str(spec.get("description", "")))
    desc = " ".join(desc.split())
    examples = spec.get("examples")
    if examples:
        desc += " e.g. " + "; ".join(json.dumps(e, ensure_ascii=False) for e in examples)
    return f"- {name} ({ftype}): {desc}"


def build_schema_prompt(template: dict) -> str:
    """从 memory_template.json 中提取 [YOU] 字段，生成给模型看的字段定义。"""
    if not template:
        return "- memory (string): core content\n- metadata: {type: inner|outer, time, tag, source}"

    lines: list[str] = []
    for name, spec in template.items():
        if name == "metadata" and isinstance(spec, dict):
            continue
        if _is_you_field(spec):
            lines.append(_field_line(name, spec))

    meta = template.get("metadata", {})
    meta_lines = [
        _field_line(name, spec)
        for name, spec in meta.items()
        if _is_you_field(spec)
    ]
    if meta_lines:
        lines.append("metadata:")
        lines.extend("  " + line for line in meta_lines)
    return "\n".join(lines)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            position = 0
            while position < len(line):
                while position < len(line) and line[position].isspace():
                    position += 1
                if position >= len(line):
                    break
                try:
                    item, end = decoder.raw_decode(line, position)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}:{exc.colno}: invalid JSON"
                    ) from exc
                if not isinstance(item, dict):
                    raise ValueError(
                        f"{path}:{line_number}:{position + 1}: expected a JSON object"
                    )
                items.append(item)
                position = end
    return items


def write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, items: list[dict]) -> None:
    """追加 JSONL 记录，兼容已有文件末尾缺少换行的情况。"""
    if not items:
        return
    with path.open("a+", encoding="utf-8") as f:
        f.seek(0, 2)
        if f.tell() > 0:
            f.seek(f.tell() - 1)
            if f.read(1) != "\n":
                f.write("\n")
        f.seek(0, 2)
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def sort_memory_records(records: list[dict], raw_order: list[str]) -> list[dict]:
    """按 raw 会话顺序排列记录，atom 在其源 raw 后按 id 排列。"""
    raw_positions = {raw_id: position for position, raw_id in enumerate(raw_order)}
    fallback_position = len(raw_order)

    def sort_key(item: dict) -> tuple[int, int, str]:
        metadata = item.get("metadata") or {}
        record_type = metadata.get("type")
        record_id = metadata.get("id")
        record_id = record_id if isinstance(record_id, str) else ""
        if record_type == "raw":
            return raw_positions.get(record_id, fallback_position), 0, ""

        sources = metadata.get("source") or []
        source_id = sources[0] if sources and isinstance(sources[0], str) else ""
        source_position = raw_positions.get(source_id, fallback_position)
        return source_position, 1, record_id

    return sorted(records, key=sort_key)


def _source_ids(record: dict) -> set[str]:
    sources = (record.get("metadata") or {}).get("source") or []
    return {source for source in sources if isinstance(source, str)}


def split_memory_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """将混合 memory 文件拆成 raw 消息和 atom 记录。"""
    raws: list[dict] = []
    atoms: list[dict] = []
    for record in records:
        if (record.get("metadata") or {}).get("type") == "raw":
            raws.append(record)
        else:
            atoms.append(record)
    return raws, atoms


# ---------------------------------------------------------------------------
# 上下文窗口与 prompt 构造
# ---------------------------------------------------------------------------

def format_time(time_value: Any) -> str:
    if isinstance(time_value, dict):
        return json.dumps(time_value, ensure_ascii=False)
    return str(time_value)


def build_context_window(raws: list[dict], index: int, window: int) -> str:
    """截取 index 前后各 window 条 raw memory 作为上下文字符串。

    目标消息本身不放入上下文窗口（它在 user prompt 里单独给出），避免同一文本出现两次
    导致模型重复抽取。
    """
    start = max(0, index - window)
    end = min(len(raws), index + window + 1)
    lines = []
    for i in range(start, end):
        if i == index:
            continue
        meta = raws[i]["metadata"]
        speaker = next(
            (t.split(":", 1)[1] for t in meta.get("tag", []) if t.startswith("speaker:")),
            "unknown",
        )
        lines.append(
            f"[{meta['id']}] ({format_time(meta.get('time'))}) {speaker}: {raws[i]['memory']}"
        )
    return "\n".join(lines) if lines else "(none)"


def build_messages(
    system_prompt: str, raws: list[dict], index: int, window: int
) -> list[dict]:
    target = raws[index]
    tmeta = target["metadata"]
    target_speaker = next(
        (t.split(":", 1)[1] for t in tmeta.get("tag", []) if t.startswith("speaker:")),
        "unknown",
    )
    before = min(index, window)
    after = min(len(raws) - index - 1, window)
    user = ATOM_EXTRACTION_USER_PROMPT.format(
        window_before=before,
        window_after=after,
        context=build_context_window(raws, index, window),
        target_time=format_time(tmeta.get("time")),
        target_speaker=target_speaker,
        target_text=target["memory"],
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# 大模型调用
# ---------------------------------------------------------------------------

JSON_RE = re.compile(r"[\[{].*[\]}]", re.DOTALL)


def _repair_json(text: str) -> str:
    """修常见的小模型 JSON 语法错误：缺失逗号、多余逗号、尾随逗号、非法控制字符。"""
    # 去掉对象/数组之间的多余逗号（会自动补回）
    text = re.sub(r"\}\s*\{", "},{", text)
    text = re.sub(r"\]\s*\[", "],[", text)
    # 小模型偶尔漏掉最后一个 atom 的对象闭合，并多写一个数组闭合：... ]}]]}
    text = re.sub(r"}\s*\]\s*\]}\s*$", "}}]}" , text)
    # 另一种变体只是多写一个数组闭合：... ] ]}
    text = re.sub(r"\]\s*\]\s*}", "]}", text)
    # 去掉尾随逗号： {"a":1,}  /  [1,]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # 去掉字符串内的裸控制字符
    text = "".join(ch if ch >= " " or ch in "\n\t" else " " for ch in text)
    return text


def _parse_any(text: str) -> Any:
    """尝试把文本解析成 JSON（dict 或 list），失败则修复后再试。"""
    for candidate in (text, _repair_json(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None


def extract_json(text: str) -> dict:
    """从模型输出里取出 JSON，统一成 dict。

    兼容两种形态：{"atoms": [...]} 与裸数组 [...]（小模型常省略外层包装）。
    强模型有时会在 JSON 前后加说明文字，需要更激进的提取策略。
    """
    text = text.strip()

    # 1. 去除 markdown 代码块标记
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # 2. 先尝试直接解析整个文本
    parsed = _parse_any(text)
    if parsed is not None:
        if isinstance(parsed, list):
            return {"atoms": parsed}
        if isinstance(parsed, dict):
            return parsed

    # 3. 尝试提取最大的 JSON 对象（从第一个 { 到最后一个 }）
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        parsed = _parse_any(candidate)
        if parsed is not None and isinstance(parsed, dict):
            return parsed

    # 4. 尝试提取最大的 JSON 数组（从第一个 [ 到最后一个 ]）
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidate = text[first_bracket:last_bracket + 1]
        parsed = _parse_any(candidate)
        if parsed is not None and isinstance(parsed, list):
            return {"atoms": parsed}

    # 5. 使用正则提取任意 JSON 结构（最后手段）
    match = JSON_RE.search(text)
    if match:
        parsed = _parse_any(match.group(0))
        if parsed is not None:
            if isinstance(parsed, list):
                return {"atoms": parsed}
            if isinstance(parsed, dict):
                return parsed

    # 6. 全部失败，抛出详细错误
    raise ValueError(f"no valid JSON in model output: {text[:500]!r}")


def _status_code(exc: Exception) -> int | None:
    """取出异常对应的 HTTP 状态码（openai.APIStatusError 及其子类）。"""
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _retry_after_seconds(exc: Exception) -> float | None:
    """解析响应头里的 Retry-After（秒数或 HTTP-date）。"""
    response = getattr(exc, "response", None)
    raw = response.headers.get("Retry-After") if response is not None else None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    # HTTP-date 形式
    try:
        from email.utils import parsedate_to_datetime

        target = parsedate_to_datetime(raw)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


def _backoff_seconds(exc: Exception, attempt: int, config: dict) -> float:
    """计算本次失败后的等待时间。

    - 429 / 503：优先用 Retry-After；否则按 rate_limit_backoff 指数退避（更长）。
    - 其它错误：普通指数退避。
    - 统一叠加抖动，避免并发请求同时重试再次打爆限流。
    """
    base = float(config.get("backoff_base", 2.0))
    cap = float(config.get("max_backoff", 120.0))
    jitter = float(config.get("backoff_jitter", 0.5))

    status = _status_code(exc)

    if status == 429:
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            delay = min(retry_after, cap)
        else:
            rl_base = float(config.get("rate_limit_backoff", 10.0))
            delay = min(rl_base * (2 ** (attempt - 1)), cap)
    elif status == 503:
        rl_base = float(config.get("rate_limit_backoff", 10.0))
        delay = min(rl_base * (2 ** (attempt - 1)), cap)
    else:
        delay = min(base * (2 ** (attempt - 1)), cap)

    return delay + random.uniform(0, jitter * delay)


class TruncatedCompletion(RuntimeError):
    """模型输出因达到 max_tokens 而被截断（finish_reason == 'length'）。"""


async def chat_completion(
    client: openai.AsyncOpenAI, config: dict, messages: list[dict]
) -> str:
    # 配置里可能指定了与 client 不同的 base_url（例如 eval 脚本共用 DEFAULT_LLM_CONFIG
    # 构造 client，再按 generator 段覆盖地址）。OpenAI SDK 的默认 client 会忽略 base_url，
    # 导致请求被静默发到旧地址，这里显式按请求切换。
    base_url = config.get("base_url")
    if base_url and str(client.base_url).rstrip("/") != str(base_url).rstrip("/"):
        client = client.with_options(base_url=base_url)

    kwargs: dict[str, Any] = {
        "model": config["model"],
        "messages": messages,
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "stream": False,
    }
    # # ollama / qwen3 专用：chat_template_kwargs 用来关闭思考过程。
    # # 注意：OpenAI 及其它托管服务会因未知参数直接返回 400，所以只在 ollama 端点才发送。
    # base_url = str(config.get("base_url", ""))
    # is_ollama = "11434" in base_url or "ollama" in base_url.lower()
    # if is_ollama and config.get("enable_thinking") is False:
    #     kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    # 模型名称里有 qwen3 就关闭思考过程
    model_name = config.get("model", "")
    is_qwen3 = "qwen3" in model_name.lower()

    if is_qwen3 and config.get("enable_thinking") is False:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    # 不可重试的客户端错误（除 429 外），直接抛出
    non_retryable = {400, 401, 403, 404, 422}

    last_error: Exception | None = None
    for attempt in range(1, int(config["max_retries"]) + 1):
        try:
            response = await client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            if not content.strip():
                # 小模型偶发空回复（尤其关思考后）：补上 reasoning 兜底，仍为空则当瞬时错误重试
                content = (getattr(message, "reasoning", None) or "").strip()
            if not content:
                raise RuntimeError("empty completion content")
            # 被 max_tokens 截断：内容必然是不完整的 JSON，重试（思考长度会波动，重试常能成功）
            if getattr(choice, "finish_reason", None) == "length":
                raise TruncatedCompletion(
                    "completion truncated by max_tokens"
                    f" (alignment: {content[-80:]!r})"
                )
            return content
        except openai.APIStatusError as exc:
            last_error = exc
            if exc.status_code in non_retryable:
                raise
            if attempt < int(config["max_retries"]):
                await asyncio.sleep(_backoff_seconds(exc, attempt, config))
        except Exception as exc:  # noqa: BLE001 - 连接/超时/解析等瞬时错误
            last_error = exc
            if attempt < int(config["max_retries"]):
                await asyncio.sleep(_backoff_seconds(exc, attempt, config))

    # 保留截断这一具体原因，便于定位（max_tokens 不足）
    if isinstance(last_error, TruncatedCompletion):
        raise TruncatedCompletion(f"chat completion failed after retries: {last_error}")
    raise RuntimeError(f"chat completion failed after retries: {last_error}")


# ---------------------------------------------------------------------------
# 原子记忆后处理
# ---------------------------------------------------------------------------

def normalize_atom(atom: dict, source_raw: dict) -> dict | None:
    """把模型输出的原子记忆补全为完整的 memory item。

    模型只负责 memory / type / time / tag；
    系统负责 id / source（固定为源 message 的 id）/ target（初始为空）/ changelog。
    """
    memory = (atom.get("memory") or "").strip()
    if not memory:
        return None
    meta = atom.get("metadata") or {}
    mem_type = meta.get("type")
    if mem_type not in ("inner", "outer"):
        mem_type = "outer"

    tags: list[str] = []
    for tag in meta.get("tag") or []:
        if isinstance(tag, str) and tag.strip():
            tags.append(tag)
        elif isinstance(tag, dict):
            key, value = tag.get("key"), tag.get("value")
            if isinstance(key, str) and isinstance(value, str) and key and value:
                tags.append(f"{key}:{value}")
    if not any(t.startswith("speaker:") for t in tags):
        # 兜底：继承源 raw memory 的 speaker 标签
        tags = [t for t in source_raw["metadata"].get("tag", []) if t.startswith("speaker:")] + tags

    # source 由系统固定为源 message 的 id，不采用模型输出
    raw_id = source_raw["metadata"]["id"]
    source = [raw_id]

    time_value = meta.get("time")
    if time_value is None or (isinstance(time_value, str) and not time_value.strip()):
        time_value = source_raw["metadata"].get("time")

    return {
        "memory": memory,
        "metadata": {
            "type": mem_type,
            "time": time_value,
            "tag": tags,
            "source": source,
        },
    }


def assign_ids(atoms: list[dict], raw_id: str, counter: int) -> int:
    """按 <raw_id>_<n> 分配唯一 id，返回更新后的计数器。"""
    for atom in atoms:
        counter += 1
        atom["metadata"]["id"] = f"{raw_id}_{counter}"
    return counter


def deduplicate_atoms(atoms: list[dict]) -> list[dict]:
    """移除同一 raw 输出中的重复原子记忆，保留首次出现的记录。"""
    unique: list[dict] = []
    seen: set[str] = set()
    for atom in atoms:
        metadata = atom["metadata"]
        key = json.dumps(
            {
                "memory": atom["memory"],
                "type": metadata.get("type"),
                "time": metadata.get("time"),
                "tag": metadata.get("tag", []),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if key not in seen:
            seen.add(key)
            unique.append(atom)
    return unique


def finalize_atom(atom: dict) -> dict:
    """补齐 target / changelog（changelog 时间为真实创建时间）。"""
    meta = atom["metadata"]
    meta["target"] = []
    meta["changelog"] = [{"time": _now(), "content": "created"}]
    # 保持字段顺序与模板一致
    ordered = {
        "memory": atom["memory"],
        "metadata": {
            "id": meta["id"],
            "type": meta["type"],
            "time": meta["time"],
            "tag": meta["tag"],
            "source": meta["source"],
            "target": meta["target"],
            "changelog": meta["changelog"],
        },
    }
    return ordered


# ---------------------------------------------------------------------------
# 单目录处理
# ---------------------------------------------------------------------------

async def process_index(
    client: openai.AsyncOpenAI,
    config: dict,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    raws: list[dict],
    index: int,
    window: int,
) -> tuple[int, list[dict], str | None]:
    """处理第 index 条 raw memory，返回 (index, 原子记忆原始输出, 错误信息)。"""
    messages = build_messages(system_prompt, raws, index, window)
    async with semaphore:
        try:
            content = await chat_completion(client, config, messages)
            parsed = extract_json(content)
            atoms = parsed.get("atoms") or []
            if not isinstance(atoms, list):
                atoms = [atoms]
            # 只保留 dict 形态的原子记忆，避免单个畸形项拖垮整条消息
            atoms = [a for a in atoms if isinstance(a, dict)]
            return index, atoms, None
        except Exception as exc:  # noqa: BLE001
            return index, [], f"{type(exc).__name__}: {exc}"


def resolve_dir(target: str) -> Path:
    """把参数解析为 speaker 目录，兼容裸目录名、相对路径与绝对路径。"""
    path = Path(target)
    if path.is_dir() and ((path / "msgmem.jsonl").exists() or (path / "memory.jsonl").exists()):
        return path
    candidate = DATA_DIR / target
    if candidate.is_dir():
        return candidate
    return path


async def process_dir(
    client: openai.AsyncOpenAI,
    config: dict,
    dir_name: str,
    limit: int | None,
    resume: bool,
) -> None:
    directory = resolve_dir(dir_name)
    label = directory.name
    msgmem_file = directory / "msgmem.jsonl"
    atom_file = directory / "atommem.jsonl"
    done_file = directory / "atommem.done.json"
    source_file = msgmem_file if msgmem_file.exists() else directory / "memory.jsonl"
    if not source_file.exists():
        print(f"[skip] {label}: no msgmem.jsonl")
        return

    source_records = read_jsonl(source_file)
    all_raws, embedded_atoms = split_memory_records(source_records)
    if not all_raws:
        print(f"[skip] {label}: no raw memory in msgmem.jsonl")
        return

    # 兼容之前写入 memory.jsonl 的 atom：读取但不再修改原始 memory.jsonl。
    existing_atoms: list[dict] = list(embedded_atoms)
    if atom_file.exists():
        _, stored_atoms = split_memory_records(read_jsonl(atom_file))
        existing_atoms.extend(stored_atoms)

    raw_order = [raw["metadata"]["id"] for raw in all_raws]
    raws = all_raws
    if limit is not None:
        raws = raws[:limit]
    if not raws:
        print(f"[skip] {label}: no raw memory")
        return

    completed_ids: set[str] = set()
    completed_ids.update(
        source_id
        for atom in existing_atoms
        for source_id in _source_ids(atom)
    )
    if resume and done_file.exists():
        try:
            saved_ids = json.loads(done_file.read_text(encoding="utf-8"))
            if isinstance(saved_ids, list):
                completed_ids.update(item for item in saved_ids if isinstance(item, str))
        except (OSError, json.JSONDecodeError):
            print(f"[warn] {label}: invalid {done_file.name}, rebuilding resume state")

    # 非 resume 模式重跑指定 raw 时先移除旧 atom，避免重复累积。
    selected_ids = {raw["metadata"]["id"] for raw in raws}
    if not resume:
        existing_atoms = [atom for atom in existing_atoms if not _source_ids(atom) & selected_ids]
        completed_ids.difference_update(selected_ids)
    write_jsonl(atom_file, existing_atoms)

    pending_indices = [
        index for index, raw in enumerate(raws)
        if raw["metadata"]["id"] not in completed_ids
    ]
    skipped = len(raws) - len(pending_indices)
    if resume:
        print(f"[resume] {skipped} 条已完成，{len(pending_indices)} 条待处理")
    if not pending_indices:
        print(f"[skip] {label}: 全部 raw 已完成")
        return

    window = int(config["context_window"])
    system_prompt = ATOM_EXTRACTION_SYSTEM_PROMPT.replace("{{SCHEMA}}", build_schema_prompt(load_template()))
    # print(system_prompt)
    semaphore = asyncio.Semaphore(int(config["concurrency"]))

    print(f"[run ] {label}: {len(raws)} raw messages, concurrency={config['concurrency']}, window={window}")
    print(f"[info] 实时追加到 {atom_file}")

    tasks = [
        asyncio.create_task(
            process_index(client, config, semaphore, system_prompt, raws, i, window)
        )
        for i in pending_indices
    ]
    print(f"[info] submitted {len(tasks)} requests to {config['model']}")

    results: dict[int, list[dict]] = {}
    errors: list[str] = []
    atom_count = 0
    done = 0
    for coro in asyncio.as_completed(tasks):
        index, atoms, error = await coro
        done += 1
        if error:
            errors.append(f"{raws[index]['metadata']['id']}: {error}")
            print(f"[error] {raws[index]['metadata']['id']}: {error}")
        else:
            results[index] = atoms
            # 实时处理并追加这条 raw 的原子记忆到 atommem.jsonl
            raw = raws[index]
            raw_id = raw["metadata"]["id"]
            normalized = [normalize_atom(a, raw) for a in atoms]
            normalized = [a for a in normalized if a]
            normalized = deduplicate_atoms(normalized)
            assign_ids(normalized, raw_id, 0)
            finalized = [finalize_atom(a) for a in normalized]
            # 实时追加到 atommem.jsonl
            append_jsonl(atom_file, finalized)
            atom_count += len(finalized)
            completed_ids.add(raw_id)
            done_file.write_text(
                json.dumps(sorted(completed_ids), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if not finalized:
                print(f"[empty] {raws[index]['metadata']['id']}: no atom extracted")
        if done % 20 == 0 or done == len(raws):
            print(f"        {done}/{len(raws)} messages processed, {atom_count} atoms written")

    # 并发请求按完成顺序追加，全部完成后恢复会话顺序。
    sorted_records = sort_memory_records(read_jsonl(atom_file), raw_order)
    write_jsonl(atom_file, sorted_records)
    print(f"[sort] {label}: {len(sorted_records)} atom records ordered by raw conversation")

    print(
        f"[done] {label}: {atom_count} atoms from {len(raws)} messages; "
        f"atom output -> {atom_file.name}"
    )
    if errors:
        error_file = directory / "atom_errors.log"
        error_file.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"[warn] {label}: {len(errors)} failed message(s) -> {error_file.name}")


async def run(config: dict, targets: list[str], limit: int | None, resume: bool) -> None:
    if targets:
        dir_names = targets
    else:
        dir_names = sorted(
            p.name
            for p in DATA_DIR.iterdir()
            if (p / "msgmem.jsonl").exists() or (p / "memory.jsonl").exists()
        )

    client = openai.AsyncOpenAI(
        base_url=config["base_url"],
        api_key=config.get("api_key") or "not-needed",
        timeout=float(config["timeout"]),
        max_retries=0,  # 重试由本模块的 _backoff_seconds 统一处理
    )
    try:
        for dir_name in dir_names:
            await process_dir(client, config, dir_name, limit, resume)
    finally:
        await client.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 raw message memory 抽取原子记忆")
    parser.add_argument("dirs", nargs="*", help="speaker 目录名，如 Caroline_Melanie；缺省处理全部")
    parser.add_argument("--limit", type=int, default=None, help="每个目录只处理前 N 条 raw（调试用）")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖配置里的并发数")
    parser.add_argument("--context-window", type=int, default=None, help="覆盖配置里的上下文窗口大小")
    parser.add_argument("--resume", action="store_true", help="跳过已完成的 raw，只重试失败或未完成项")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    config = load_config()
    if args.concurrency is not None:
        config["concurrency"] = args.concurrency
    if args.context_window is not None:
        config["context_window"] = args.context_window
    asyncio.run(run(config, args.dirs, args.limit, args.resume))


if __name__ == "__main__":
    main()
