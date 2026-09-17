"""
先格式化将data/correct_locomo10.json里面的session转换为memory item, memory_item参考memory_template.json
在data/{speaker1}_{speaker2}/memory.jsonl里面存储，memory为原始内容，id为session_{session_index}_{message_index}, type为raw，tag为[speaker:xxx]，无source，初始target也为空，changelog的time为创建时间，content为created
为后续原子记忆抽取做准备

用法：
    python -m codemem.init_mem
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "correct_locomo10.json"
DATA_DIR = PROJECT_ROOT / "data"

SESSION_INDEX_RE = re.compile(r"^session_(\d+)$")


def _created_at() -> str:
    """创建时间，统一的 UTC ISO-8601 格式。"""
    return datetime.now(timezone.utc).isoformat()


def parse_session_datetime(date_time: str) -> str:
    """将 locomo 的 '1:56 pm on 8 May, 2023' 转为 ISO-8601 字符串（无时区、不加 Z），解析失败则原样返回。"""
    try:
        dt = datetime.strptime(date_time, "%I:%M %p on %d %B, %Y")
    except (ValueError, TypeError):
        return date_time
    return dt.isoformat()


def build_memory_item(
    text: str, session_index: int, message_index: int, speaker: str, session_time: str
) -> dict:
    """把一条原始对话消息包装成 memory item。

    - 会话首条消息的 time 即该会话发生时间（ISO-8601，无时区）。
    - 后续消息的 time 为 'few minutes in {session_time}'，表示发生在同一会话开始后的几分钟。
    - changelog.time 始终为真实的创建时间。
    """
    created = _created_at()
    if message_index == 1:
        time = session_time
    else:
        time = f"few minutes in {session_time}"
    return {
        "memory": text,
        "metadata": {
            "id": f"session_{session_index}_{message_index}",
            "type": "raw",
            "time": time,
            "tag": [f"speaker:{speaker}"],
            "source": [],
            "target": [],
            "changelog": [{"time": created, "content": "created"}],
        },
    }


def iter_sessions(conversation: dict):
    """按 session_index 从小到大依次产出 (session_index, session_date_time, messages)。"""
    sessions = []
    for key, value in conversation.items():
        match = SESSION_INDEX_RE.match(key)
        if match:
            sessions.append((int(match.group(1)), value))
    sessions.sort(key=lambda item: item[0])

    for session_index, messages in sessions:
        raw_date_time = conversation.get(f"session_{session_index}_date_time", "")
        yield session_index, parse_session_datetime(raw_date_time), messages


def convert_sample(sample: dict) -> tuple[str, list[dict]]:
    """将一个 sample（一对 speaker）转换为 (目录名, memory item 列表)。"""
    conversation = sample["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    items = []
    for session_index, session_time, messages in iter_sessions(conversation):
        for message_index, message in enumerate(messages, start=1):
            items.append(
                build_memory_item(
                    text=message["text"],
                    session_index=session_index,
                    message_index=message_index,
                    speaker=message["speaker"],
                    session_time=session_time,
                )
            )

    return f"{speaker_a}_{speaker_b}", items


def write_memory_jsonl(dir_name: str, items: list[dict]) -> Path:
    out_dir = DATA_DIR / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "msgmem.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return out_path


def main() -> None:
    with DATA_FILE.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    for sample in samples:
        dir_name, items = convert_sample(sample)
        out_path = write_memory_jsonl(dir_name, items)
        print(f"{sample['sample_id']} -> {out_path} ({len(items)} items)")


if __name__ == "__main__":
    main()
