"""code agent 的四个通用工具：``read`` / ``write`` / ``edit`` / ``bash``。

语义移植自 ``reference/tools.py``（原文件依赖未安装的 ``tau_agent`` 包），保留其关键行为：

- ``read``：UTF-8 读文本，截断到 2000 行 / 50KB，附续读提示（offset/limit）。
- ``write``：覆盖写，自动建父目录。
- ``edit``：``edits[].oldText`` 必须**唯一且互不重叠**，**全部校验通过才落盘**；返回 diff。
- ``bash``：stdout+stderr 合并，**tail** 截断 2000 行 / 50KB，超时杀整个进程组；
  支持 ``shell_command_prefix``（本方案用它注入 PATH / PYTHONPATH / 索引路径）。

相对参考实现的三处必要改动（都是为了"agent 只允许写自己的工作目录"这条约束）：

1. **去掉图像支持**。这个 agent 只读 JSONL，图像分支纯属死代码，且牵出一堆依赖。
2. **``write`` / ``edit`` 加路径约束**（``confine_to``）：目标路径 ``resolve()`` 后必须落在
   工作目录内，否则报错。软链指向外部时也会被拦下 —— 所以 ``qa_{idx}/inputs`` 这个软链
   只能读、不能写。这是**第一道**防线；绕过它的写入由 harness 的 sha256 校验兜底
   （见 ``evomem_v3.run_qa``）——检测而非防御，因为 bash 原则上能写任何地方。
3. **返回 ``ToolResult``**（自带给模型看的 ``text`` + 给日志/轨迹用的 ``details`` + ``ok``），
   而不是 ``tau_agent`` 的 ``AgentToolResult``。

本模块是**纯 CPU、纯 stdlib** 的：不打模型、不碰网络。所有输入错误都抛 ``ToolError``，
由调用方渲染成一次"失败的观测"返回给模型（模型自己修），不中断循环。

自测：``python scripts/test_evomem_v3.py``。
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import signal
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any

# 输出截断上限（与 reference/tools.py 一致）
DEFAULT_MAX_OUTPUT_BYTES = 50 * 1024
DEFAULT_MAX_OUTPUT_LINES = 2_000
UTF8_BOM = "﻿"


class ToolError(ValueError):
    """工具入参非法 / 前置条件不满足。会被渲染成失败的观测返回给模型。"""


@dataclass
class ToolResult:
    """一次工具调用的结果。

    ``text``     —— 给模型看的观测（会被写进对话历史）
    ``details``  —— 结构化明细（落 ``steps.jsonl``、供日志与排错，不进 prompt）
    ``ok``       —— 是否成功。失败只用于记账与"无进展检测"，不中断循环。
    """

    text: str
    details: dict[str, Any] = field(default_factory=dict)
    ok: bool = True


@dataclass
class ToolDefinition:
    """一个工具：给模型看的元信息 + JSON Schema + 异步执行器。

    ``description`` / ``guidelines`` / ``input_schema`` 由 ``tool.json`` 提供
    （统一管理、统一注入上下文）；执行器在本文件里按名字注册。
    """

    name: str
    description: str
    guidelines: tuple[str, ...]
    input_schema: Mapping[str, Any]
    executor: Callable[[Mapping[str, Any]], Awaitable[ToolResult]]
    # 一次调用是否**可能**改变工作目录的内容。无进展检测用它判断"这一步有没有可能推进"。
    mutating: bool = False

    async def run(self, args: Mapping[str, Any]) -> ToolResult:
        return await self.executor(args)


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

def format_size(bytes_count: int) -> str:
    if bytes_count < 1024:
        return f"{bytes_count}B"
    if bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f}KB"
    return f"{bytes_count / (1024 * 1024):.1f}MB"


def atomic_write_text(path: Path, content: str) -> None:
    """**原子**写文本：先写同目录临时文件 → fsync → ``os.replace``。

    为什么必须这样：agent 会在多轮之间用 ``write`` 重写 ``evidence.jsonl``。若直接
    ``write_text``，模型输出被 ``max_tokens`` 截断时就会**原地留下一个残缺文件** ——
    而截断发生在写到一半时，文件长度甚至可能被截短，之前写好的证据同时丢失。
    原子替换保证任何时刻磁盘上的文件要么是旧版本、要么是完整的新版本。

    临时文件必须放**同一目录**（跨设备 ``os.replace`` 不是原子的）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        # 任何失败都不要留下临时文件（它会被 agent 的 `ls`/`find` 看到，造成困惑）
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def shrink_notice(path: Path, before: int, after: int) -> str:
    """文件被大幅缩短时的提示语（**不拒绝**，只提醒）。

    这是**截断检测**：模型输出被 ``max_tokens`` 切断时，典型表现就是重写后的文件比原来
    短很多。此时既不能静默接受（会丢证据），也不能直接拒绝（agent 可能真的是在精简）——
    所以把事实摆到观测里，让模型自己判断。空内容则更明确，直接拒绝（见 ``write``）。
    """
    if before >= 200 and after < before * 0.5:
        return (
            f"\n\n[WARNING: {path} shrank from {format_size(before)} to "
            f"{format_size(after)}. If your output was truncated, the content is now "
            f"incomplete -- read the file back and rewrite it in full.]"
        )
    return ""


@dataclass(frozen=True, slots=True)
class Truncation:
    """截断结果 + 元信息（``to_details`` 进 JSONL，``note`` 进观测文本）。"""

    content: str
    truncated: bool
    truncated_by: str | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool = False
    first_line_exceeds_limit: bool = False

    def to_details(self) -> dict[str, Any]:
        return {
            "truncated": self.truncated,
            "truncated_by": self.truncated_by,
            "total_lines": self.total_lines,
            "total_bytes": self.total_bytes,
            "output_lines": self.output_lines,
            "output_bytes": self.output_bytes,
        }


def _split_lines_for_counting(content: str) -> list[str]:
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def truncate_head(content: str, *, max_lines: int, max_bytes: int) -> Truncation:
    """保留开头（``read`` 用）。"""
    lines = _split_lines_for_counting(content)
    total_lines, total_bytes = len(lines), len(content.encode())
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return Truncation(content, False, None, total_lines, total_bytes, total_lines, total_bytes)

    first_line_bytes = len(lines[0].encode()) if lines else 0
    if first_line_bytes > max_bytes:
        return Truncation("", True, "bytes", total_lines, total_bytes, 0, 0,
                          first_line_exceeds_limit=True)

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by = "lines"
    for index, line in enumerate(lines[:max_lines]):
        line_bytes = len(line.encode()) + (1 if index > 0 else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines.append(line)
        output_bytes += line_bytes
    output = "\n".join(output_lines)
    return Truncation(output, True, truncated_by, total_lines, total_bytes,
                      len(output_lines), len(output.encode()))


def truncate_tail(content: str, *, max_lines: int, max_bytes: int) -> Truncation:
    """保留结尾（``bash`` 用：报错和最终结果都在尾部）。"""
    lines = _split_lines_for_counting(content)
    total_lines, total_bytes = len(lines), len(content.encode())
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return Truncation(content, False, None, total_lines, total_bytes, total_lines, total_bytes)

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by = "lines"
    last_line_partial = False
    for line in reversed(lines):
        line_bytes = len(line.encode()) + (1 if output_lines else 0)
        if len(output_lines) >= max_lines:
            truncated_by = "lines"
            break
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not output_lines:
                clipped = line.encode()[-max_bytes:].decode(errors="ignore")
                output_lines.insert(0, clipped)
                output_bytes = len(clipped.encode())
                last_line_partial = True
            break
        output_lines.insert(0, line)
        output_bytes += line_bytes
    output = "\n".join(output_lines)
    return Truncation(output, True, truncated_by, total_lines, total_bytes,
                      len(output_lines), len(output.encode()), last_line_partial=last_line_partial)


# ---------------------------------------------------------------------------
# 入参解析（一律抛 ToolError，让模型知道错在哪）
# ---------------------------------------------------------------------------

def _str_arg(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ToolError(f"{name} must be a string, got {type(value).__name__}")
    return value


def _optional_int_arg(arguments: Mapping[str, Any], name: str) -> int | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"{name} must be an integer")
    return value


def _optional_float_arg(arguments: Mapping[str, Any], name: str) -> float | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{name} must be a number")
    return float(value)


def _path_arg(arguments: Mapping[str, Any], name: str, *, cwd: Path) -> Path:
    value = _str_arg(arguments, name).strip()
    if not value:
        raise ToolError(f"{name} must not be empty")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path


def _ensure_inside(path: Path, root: Path) -> None:
    """路径约束：``path`` 解析后必须落在 ``root`` 内。

    用 ``resolve()``（follow symlink）而不是字面比较 —— 否则 ``qa_0/inputs`` 这个指向
    只读输入目录的软链就能被用来写到外面去。
    """
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise ToolError(
            f"refusing to write outside the workspace: {path} "
            f"(resolves to {resolved}, workspace is {root.resolve()})"
        ) from None


# ---------------------------------------------------------------------------
# 每个路径一把写锁（同进程内串行化，避免并发写同一文件交错）
# ---------------------------------------------------------------------------

_file_locks: dict[Path, asyncio.Lock] = {}


class _FileLock:
    def __init__(self, path: Path) -> None:
        self._key = path.resolve()
        self._lock: asyncio.Lock | None = None

    async def __aenter__(self) -> None:
        lock = _file_locks.setdefault(self._key, asyncio.Lock())
        self._lock = lock
        await lock.acquire()

    async def __aexit__(self, *_exc: object) -> None:
        if self._lock is not None:
            self._lock.release()


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------

def create_read_tool(
    *,
    cwd: Path,
    max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> ToolDefinition:
    async def execute(arguments: Mapping[str, Any]) -> ToolResult:
        raw_path = _str_arg(arguments, "path")
        path = _path_arg(arguments, "path", cwd=cwd)
        offset = _optional_int_arg(arguments, "offset")
        limit = _optional_int_arg(arguments, "limit")
        if offset is not None and offset < 0:
            raise ToolError("offset must be at least 0")
        if limit is not None and limit < 1:
            raise ToolError("limit must be at least 1")
        if not path.exists():
            raise ToolError(f"File not found: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {path}")

        text = path.read_bytes().decode("utf-8", errors="replace")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        all_lines = text.split("\n")
        start_line = 0 if offset is None or offset == 0 else offset - 1
        if start_line >= len(all_lines):
            raise ToolError(
                f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)"
            )

        user_limited: int | None = None
        if limit is not None:
            end_line = min(start_line + limit, len(all_lines))
            selected = "\n".join(all_lines[start_line:end_line])
            user_limited = end_line - start_line
        else:
            selected = "\n".join(all_lines[start_line:])

        truncation = truncate_head(selected, max_lines=max_lines, max_bytes=max_bytes)
        start_display = start_line + 1
        details: dict[str, Any] = {
            "path": str(path),
            "lines": len(all_lines),
            "bytes": len(text.encode()),
            "truncation": truncation.to_details(),
        }

        if truncation.first_line_exceeds_limit:
            output = (
                f"[Line {start_display} is "
                f"{format_size(len(all_lines[start_line].encode()))}, exceeds the "
                f"{format_size(max_bytes)} limit. Read a slice of it with bash instead, e.g. "
                f"`sed -n '{start_display}p' {raw_path} | head -c {max_bytes}`]"
            )
        else:
            output = truncation.content
            end_display = start_display + max(truncation.output_lines, 1) - 1
            next_offset = end_display + 1
            if truncation.truncated:
                why = "lines" if truncation.truncated_by == "lines" else format_size(max_bytes)
                output += (
                    f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)} "
                    f"(limit: {why}). Use offset={next_offset} to continue.]"
                )
            elif user_limited is not None and start_line + user_limited < len(all_lines):
                remaining = len(all_lines) - (start_line + user_limited)
                output += (
                    f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
                )
        return ToolResult(text=output or "(empty file)", details=details)

    return ToolDefinition(
        name="read",
        description="",              # 由 tool.json 提供
        guidelines=(),
        input_schema={},
        executor=execute,
    )


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

def create_write_tool(*, cwd: Path, confine: bool = True,
                      protect_filenames: tuple[str, ...] = ("evidence.jsonl",)) -> ToolDefinition:
    protected = tuple(protect_filenames)

    async def execute(arguments: Mapping[str, Any]) -> ToolResult:
        path = _path_arg(arguments, "path", cwd=cwd)
        content = _str_arg(arguments, "content")
        if confine:
            _ensure_inside(path, cwd)

        previous = path.stat().st_size if path.exists() else 0
        existed = path.exists()

        # 空内容**拒绝**：多轮之间最常见的截断表现，会静默销毁之前写好的证据。
        # 这里拒绝比接受更安全，而且模型看到错误后会重写。
        if existed and not content.strip() and path.name in protected:
            raise ToolError(
                f"refusing to overwrite {path.name} with empty content -- that would erase "
                f"{previous} bytes of existing evidence. Write the full list of records, or "
                f"delete the file first if you really mean to start over."
            )

        async with _FileLock(path):
            atomic_write_text(path, content)

        line_count = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        return ToolResult(
            text=(
                f"Successfully wrote {len(content)} characters ({line_count} lines) to {path}."
                + shrink_notice(path, previous, len(content.encode()))
            ),
            details={
                "path": str(path),
                "characters": len(content),
                "lines": line_count,
                "existed": existed,
                "previous_bytes": previous,
                "shrank": existed and len(content.encode()) < previous * 0.5,
                "atomic": True,
            },
        )

    return ToolDefinition(
        name="write", description="", guidelines=(), input_schema={},
        executor=execute, mutating=True,
    )


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------

def _normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _count_occurrences(content: str, text: str) -> int:
    count, start = 0, 0
    while True:
        index = content.find(text, start)
        if index == -1:
            return count
        count += 1
        start = index + len(text)


def _edits_arg(arguments: Mapping[str, Any]) -> list[dict[str, str]]:
    value = arguments.get("edits")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            value = parsed
    if not isinstance(value, list) or not value:
        raise ToolError(
            "edits must be a non-empty array of {oldText, newText} objects"
        )
    edits: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ToolError(f"edits[{index}] must be an object")
        old_text, new_text = item.get("oldText"), item.get("newText")
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ToolError(f"edits[{index}].oldText and edits[{index}].newText must be strings")
        edits.append({"oldText": old_text, "newText": new_text})
    return edits


def apply_edits(normalized: str, edits: list[dict[str, str]], path: str) -> tuple[str, str]:
    """把若干条精确替换应用到 ``normalized``，返回 ``(原文, 新文)``。

    **全部校验通过才返回** —— 任一条 ``oldText`` 找不到或出现多次，就抛错，文件不动。
    """
    normalized_edits = [
        {"oldText": _normalize_to_lf(e["oldText"]), "newText": _normalize_to_lf(e["newText"])}
        for e in edits
    ]
    total = len(normalized_edits)
    for index, edit in enumerate(normalized_edits):
        if not edit["oldText"]:
            raise ToolError(f"edits[{index}].oldText must not be empty in {path}")

    spans: list[tuple[int, int, str]] = []
    for index, edit in enumerate(normalized_edits):
        occurrences = _count_occurrences(normalized, edit["oldText"])
        label = "the text" if total == 1 else f"edits[{index}]"
        if occurrences == 0:
            raise ToolError(
                f"Could not find {label} in {path}. oldText must match exactly, "
                f"including all whitespace and newlines."
            )
        if occurrences > 1:
            raise ToolError(
                f"Found {occurrences} occurrences of {label} in {path}. Each oldText must be "
                f"unique -- add more surrounding context to disambiguate."
            )
        start = normalized.index(edit["oldText"])
        spans.append((start, start + len(edit["oldText"]), edit["newText"]))

    previous_end = -1
    for start, end, _ in sorted(spans):
        if start < previous_end:
            raise ToolError("Edits must not overlap")
        previous_end = end

    new_content = normalized
    for start, end, new_text in sorted(spans, reverse=True):
        new_content = f"{new_content[:start]}{new_text}{new_content[end:]}"
    if new_content == normalized:
        raise ToolError(
            f"No changes made to {path}: the replacement produced identical content."
        )
    return normalized, new_content


def create_edit_tool(*, cwd: Path, confine: bool = True) -> ToolDefinition:
    async def execute(arguments: Mapping[str, Any]) -> ToolResult:
        path = _path_arg(arguments, "path", cwd=cwd)
        edits = _edits_arg(arguments)
        if confine:
            _ensure_inside(path, cwd)
        if not path.exists():
            raise ToolError(f"Could not edit file: {path}. File not found.")
        if path.is_dir():
            raise ToolError(f"Could not edit file: {path}. Path is a directory.")

        async with _FileLock(path):
            raw = path.read_text(encoding="utf-8")
            bom, content = (UTF8_BOM, raw[1:]) if raw.startswith(UTF8_BOM) else ("", raw)
            ending = "\r\n" if content.find("\r\n") != -1 and (
                content.find("\n") == -1 or content.find("\r\n") < content.find("\n")
            ) else "\n"
            normalized = _normalize_to_lf(content)
            base_content, new_content = apply_edits(normalized, edits, str(path))
            restored = new_content.replace("\n", "\r\n") if ending == "\r\n" else new_content
            atomic_write_text(path, bom + restored)

        old_lines, new_lines = base_content.splitlines(), new_content.splitlines()
        diff = "\n".join(difflib.ndiff(old_lines, new_lines))
        return ToolResult(
            text=f"Successfully applied {len(edits)} replacement(s) to {path}.\n\n{diff}",
            details={"path": str(path), "edits": len(edits), "diff": diff},
        )

    return ToolDefinition(
        name="edit", description="", guidelines=(), input_schema={},
        executor=execute, mutating=True,
    )


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------

def _shell_executable() -> str | None:
    """本机 bash 的绝对路径；找不到返回 ``None``。

    这是名为 ``bash`` 的工具，命令就该由 bash 解释 —— 尤其是引号规则（``'.content="x"'``
    这类在 POSIX shell 里合法、在 Windows ``cmd.exe`` 里会被吃掉引号）。只有当本机确实
    没有 bash 时才退回 ``create_subprocess_shell``（POSIX 上是 ``/bin/sh``，Windows 上是
    ``cmd.exe``）—— 那里命令的语法就不保证了，但至少不是直接崩。

    路径必须是**绝对路径**：POSIX 上 ``execvp`` 会查 PATH，Windows 的 ``CreateProcess``
    不会（它要求带扩展名），裸 ``bash`` 直接 ``WinError 2``。
    """
    import shutil

    return shutil.which("bash")


def create_bash_tool(
    *,
    cwd: Path,
    shell_command_prefix: str | None = None,
    default_timeout: float | None = None,
    max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> ToolDefinition:
    prefix = shell_command_prefix.strip() if shell_command_prefix else None
    shell_executable = _shell_executable()

    async def execute(arguments: Mapping[str, Any]) -> ToolResult:
        command = _str_arg(arguments, "command")
        timeout = _optional_float_arg(arguments, "timeout")
        if timeout is not None and timeout <= 0:
            raise ToolError("timeout must be greater than 0")
        effective_timeout = timeout if timeout is not None else default_timeout
        start = monotonic()
        # 用 ``bash -c`` 执行，而不是 create_subprocess_shell(executable=...)。
        #
        # 两个理由：① 前缀里是 POSIX 语法（``export PATH="a:b"``），只有 bash 会解释；
        # ② Windows 上 ``create_subprocess_shell`` 会把 executable 拼进 ``cmd.exe /c``
        # 的命令行 —— 实测 ``C:\Program Files\...\bash.EXE`` 被拆成 ``/c/Program: Files\...``
        # 直接 "No such file or directory"。走 bash -c 没有这一层：脚本是**一个参数**，
        # 引号规则由我们自己定。
        if shell_executable:
            argv = [shell_executable, "-c", f"{prefix}\n{command}" if prefix else command]
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
        else:
            process = await asyncio.create_subprocess_shell(
                f"{prefix}\n{command}" if prefix else command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
        communicate = asyncio.create_task(process.communicate())
        timed_out = False
        try:
            output_bytes, _ = await asyncio.wait_for(communicate, timeout=effective_timeout)
        except asyncio.TimeoutError:
            timed_out = True
            _kill_process_tree(process)
            try:
                output_bytes, _ = await communicate
            except asyncio.CancelledError:
                output_bytes = b""
        except asyncio.CancelledError:
            _kill_process_tree(process)
            communicate.cancel()
            raise

        output = output_bytes.decode(errors="replace")
        truncation = truncate_tail(output, max_lines=max_lines, max_bytes=max_bytes)
        # 空输出**保持为空**，不在这里补一个 "(no output)" 占位串。
        #
        # 原因（实测踩的坑）：占位串会让 `agent.render_observation` 里"命令成功但没有任何
        # 输出"的检测**永远不成立**，于是那条专门解释"jq 选不到东西"的提示成了死代码。
        # 实测后果是一条 QA 因此空转 6 步：模型查一个不存在的 id → 得到 "(no output)" →
        # 以为是环境问题 → 换个写法再查 → 又被重复检测拒掉 → 循环到预算耗尽。
        # 观测文本由渲染层统一决定（见 render_observation），工具只负责如实返回空串。
        output_text = truncation.content
        full_output_path: str | None = None
        if truncation.truncated:
            full_output_path = _write_temp_output(output)
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                output_text += (
                    f"\n\n[Showing the last {format_size(truncation.output_bytes)} of line "
                    f"{end_line}. Full output: {full_output_path}]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. "
                    f"Full output: {full_output_path}]"
                )

        exit_code = process.returncode
        status: str | None = None
        if timed_out:
            status = (
                f"Command timed out after {effective_timeout:g} seconds"
                if effective_timeout else "Command timed out"
            )
        elif exit_code not in (0, None):
            status = f"Command exited with code {exit_code}"
        if status:
            output_text = f"{output_text}\n\n{status}" if output_text else status

        return ToolResult(
            text=output_text,
            details={
                "command": command,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "duration_seconds": round(monotonic() - start, 3),
                "truncation": truncation.to_details(),
                "full_output_path": full_output_path,
                "output_bytes": len(output.encode()),
                "output_lines": truncation.total_lines,
            },
            ok=not timed_out and exit_code in (0, None),
        )

    return ToolDefinition(
        name="bash", description="", guidelines=(), input_schema={},
        executor=execute, mutating=True,
    )


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    """杀掉整个进程组 —— 否则管道/复合命令的子进程会继续跑（参考实现同此处理）。"""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return
    else:
        try:
            process.kill()
        except ProcessLookupError:
            return


def _write_temp_output(output: str) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="codemem-bash-", suffix=".log", delete=False
    ) as handle:
        handle.write(output)
        return handle.name


# ---------------------------------------------------------------------------
# 工具集
# ---------------------------------------------------------------------------

def build_tools(
    *,
    cwd: Path,
    shell_command_prefix: str | None = None,
    bash_timeout: float | None = None,
    confine: bool = True,
    max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    protect_filenames: tuple[str, ...] = ("evidence.jsonl",),
) -> dict[str, ToolDefinition]:
    """建出这一套工具。返回 ``{name: ToolDefinition}``。

    描述与 schema 是**空壳**，由 ``toolcfg.apply_catalog`` 从 ``tool.json`` 填 ——
    元信息统一管理，执行器在此注册。
    """
    tools = [
        create_read_tool(cwd=cwd, max_lines=max_lines, max_bytes=max_bytes),
        create_write_tool(cwd=cwd, confine=confine, protect_filenames=protect_filenames),
        create_edit_tool(cwd=cwd, confine=confine),
        create_bash_tool(
            cwd=cwd,
            shell_command_prefix=shell_command_prefix,
            default_timeout=bash_timeout,
            max_lines=max_lines,
            max_bytes=max_bytes,
        ),
    ]
    return {tool.name: tool for tool in tools}
