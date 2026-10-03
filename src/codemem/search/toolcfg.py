"""``configs/tool.json`` 的加载与渲染：agent 上下文的**唯一**来源。

职责边界：

- ``tools.py``  —— 执行语义（怎么读写文件、怎么跑 bash）。
- ``toolcfg.py`` —— 元信息（工具叫什么、怎么描述、schema 长什么样、system prompt 怎么写）。

这样"改 agent 行为"= 改 ``tool.json``，不动代码。同时保留 JSON Schema，所以日后要换成
模型原生 ``tools=`` 参数，只需要在这里加一个适配器，执行层完全不动。

对外接口：

    load_catalog(path)            读 + 校验 tool.json
    catalog.apply(tools)          把 description/guidelines/input_schema 填进 ToolDefinition
    catalog.render_tools()        渲染 <tools> 段
    catalog.render_commands()     渲染 grep/date 的用法与示例
    catalog.render_workspace()    渲染目录布局与只读规则
    catalog.render_schema()       渲染 evidence 记录字段说明
    catalog.render_time_policy()  渲染"时间精度"规则段
    catalog.render_prompt(question)  渲染完整 system prompt（替换 {{TOOLS}} 等槽位）
    catalog.prompt_text(key)      取 user_begin / user_gaps / ... 文案
    parse_tool_call(text)         解析模型输出为一次工具调用
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 从 io 导入**唯一**的 PROJECT_ROOT，不要自己 parents[N] 推 ——
# 本文件在子包里深度不同，重算会落到 src/ 而不是项目根。
from ..io import PROJECT_ROOT
DEFAULT_TOOL_CONFIG = PROJECT_ROOT / "configs" / "tool.json"

# system prompt 里的槽位名 -> catalog 属性。渲染时逐一替换。
# 新增槽位时记得同时加进 ``validate`` 的检查，否则拼错槽位会静默留一个 {{XXX}} 在 prompt 里。
PROMPT_SLOTS = ("TOOLS", "COMMANDS", "WORKSPACE", "SCHEMA", "PROTOCOL", "WRITING_POLICY",
                "TIME_POLICY", "EVIDENCE_STANDARD", "STRATEGY")


class ToolConfigError(ValueError):
    """tool.json 缺失或结构非法。启动时就该炸，而不是跑到一半才发现。"""


# ---------------------------------------------------------------------------
# 响应解析
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """一次工具调用。``args`` 已保证是 dict（否则解析阶段就丢掉）。"""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return f"{self.tool}({_short_args(self.args)})"


def _short_args(args: dict[str, Any], limit: int = 120) -> str:
    parts = []
    for key, value in args.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        text = re.sub(r"\s+", " ", str(text)).strip()
        if len(text) > limit:
            text = text[:limit] + "…"
        parts.append(f"{key}={text!r}")
    return ", ".join(parts)


_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*|\s*```\s*$")
_JSON_OBJECT_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE_RE.sub("", stripped, count=1)
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def _first_json_object(text: str) -> dict[str, Any] | None:
    """从文本里找出**第一个能解析成 dict** 的对象。

    这里刻意不做括号修复：模型偶尔会在命令字符串里用未转义的引号，那种情况下"修"出来的
    对象往往语义已经错了。宁可把这一步判为"没有工具调用"，也不要执行一条猜出来的命令。
    """
    # 先试整段（严格 JSON / 代码围栏包裹的 JSON）
    for candidate in (text, _strip_fences(text)):
        candidate = candidate.strip()
        if candidate.startswith("{"):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(parsed, dict):
                    return parsed
    # 再试"文字里夹了一个对象"
    for match in _JSON_OBJECT_RE.finditer(text):
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_tool_call(text: str) -> ToolCall | None:
    """把模型输出解析成 ``ToolCall``；返回 ``None`` 表示**模型说自己做完了**。

    判定顺序（每一步失败都退化为"没有工具调用"= 当作完成信号，交 verifier 判断）：

    1. 空输出 → 完成信号。
    2. 能解析出 ``{"tool": ..., "args": {...}}`` → 工具调用。
    3. 能解析出对象但没有 ``tool`` 键（例如模型直接吐了一条 base_mem 记录）→ 记为
       "格式错"，仍按完成信号处理（由 verifier 兜底），并把原文留在轨迹里。
    4. 完全解析不出对象 → 完成信号。
    """
    if not text or not text.strip():
        return None
    parsed = _first_json_object(text)
    if parsed is None:
        return None
    tool = parsed.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        return None
    args = parsed.get("args")
    if not isinstance(args, dict):
        # 允许把参数平铺在顶层（{"tool":"bash","command":"..."}）
        args = {
            key: value
            for key, value in parsed.items()
            if key not in ("tool", "args")
        }
    return ToolCall(tool=tool.strip(), args=args)


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------

@dataclass
class ToolCatalog:
    """``tool.json`` 的运行时形式。"""

    protocol: dict[str, Any]
    tools: dict[str, dict[str, Any]]
    commands: dict[str, dict[str, Any]]
    workspace: dict[str, Any]
    schema: dict[str, Any]
    prompt: dict[str, str]
    writing_policy: dict[str, Any] = field(default_factory=dict)
    time_policy: dict[str, Any] = field(default_factory=dict)
    evidence_standard: str = ""
    strategy: str = ""
    source_path: Path | None = None

    # -- 校验 ---------------------------------------------------------------

    def validate(self, available: list[str]) -> None:
        """与执行层注册的工具名对齐。缺描述 / 多描述都在启动时就报出来。

        ``available`` 是执行层注册的工具（read / edit / bash）。tool.json 的描述必须与它
        完全一致 —— 没有"控制工具"例外了（``next`` 已删除，流程回到朴素 loop）。
        """
        described = set(self.tools)
        missing = [name for name in available if name not in described]
        if missing:
            raise ToolConfigError(
                f"tool.json 缺少这些工具的描述：{missing}（执行层已注册 {available}）"
            )
        extra = [name for name in described if name not in available]
        if extra:
            raise ToolConfigError(
                f"tool.json 描述了执行层没有的工具：{extra}（可用 {available}）"
            )
        for slot in PROMPT_SLOTS:
            if "{{" + slot + "}}" not in self.prompt.get("system", ""):
                raise ToolConfigError(f"tool.json prompt.system 缺少槽位 {{{{{slot}}}}}")

    # -- 注入 ---------------------------------------------------------------

    def apply(self, tools: dict[str, Any]) -> None:
        """把元信息填进 ToolDefinition（就地修改）。"""
        for name, definition in tools.items():
            meta = self.tools.get(name)
            if not meta:
                continue
            definition.description = str(meta.get("description", ""))
            definition.guidelines = tuple(str(g) for g in meta.get("guidelines", []))
            definition.input_schema = meta.get("input_schema") or {}

    # -- 渲染 ---------------------------------------------------------------

    def render_tools(self) -> str:
        blocks: list[str] = []
        for name, meta in self.tools.items():
            lines = [f"### {name}", str(meta.get("description", "")).strip()]
            for guideline in meta.get("guidelines", []):
                lines.append(f"  - {guideline}")
            schema = json.dumps(
                {"type": "object",
                 "properties": (meta.get("input_schema") or {}).get("properties", {}),
                 "required": (meta.get("input_schema") or {}).get("required", [])},
                ensure_ascii=False,
            )
            lines.append(f"  args schema: {schema}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def render_commands(self) -> str:
        blocks: list[str] = []
        for name, meta in self.commands.items():
            lines = [f"### {name}", str(meta.get("summary", "")).strip()]
            if meta.get("usage"):
                lines.append(f"  usage: {meta['usage']}")
            fields = meta.get("output_fields")
            if isinstance(fields, dict):
                lines.append("  output fields:")
                for key, note in fields.items():
                    lines.append(f"    - {key}: {note}")
            recipes = meta.get("recipes") or []
            if recipes:
                lines.append("  examples:")
                for recipe in recipes:
                    lines.append(f"    # {recipe.get('why', '')}".rstrip())
                    lines.append(f"    {recipe.get('cmd', '')}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def render_workspace(self) -> str:
        lines = ["Files:", *(f"  {item}" for item in self.workspace.get("layout", [])),
                 "", "Rules:", *(f"  - {r}" for r in self.workspace.get("rules", []))]
        return "\n".join(lines)

    def render_schema(self) -> str:
        lines: list[str] = []
        for entry in self.schema.get("fields", []):
            flag = "required" if entry.get("required") else "optional"
            lines.append(f"  - {entry.get('name')}  ({flag})  {entry.get('note', '')}")
        return "\n".join(lines)

    def render_writing_policy(self) -> str:
        """渲染"证据是推导出来的、不是抄来的"这一段（本方案的核心写作策略）。"""
        lines: list[str] = []
        headline = str(self.writing_policy.get("headline", "")).strip()
        if headline:
            lines.append(headline)
        lines.extend(f"- {rule}" for rule in self.writing_policy.get("rules", []))
        return "\n".join(lines)

    def render_time_policy(self) -> str:
        """渲染"时间精度"这一段。

        它约束的是**答案的粒度**：源语料说"去年"，答案就该是年份，不能编一个具体日期；
        源语料依附于某个锚点（"9 June 之前的那个星期"），那这个锚点表述本身就是正确答案。
        实测 LoCoMo 的 "when" 问题里，绝大多数参考答就是这种相对/区间形态，而不是单点日期。
        """
        lines: list[str] = []
        headline = str(self.time_policy.get("headline", "")).strip()
        if headline:
            lines.append(headline)
        lines.extend(f"- {rule}" for rule in self.time_policy.get("rules", []))
        return "\n".join(lines)

    def render_evidence_standard(self) -> str:
        """渲染"证据标准"这一段。

        它约束的是**一条证据的写法**（而不是检索策略）：自足、指代消解、时间带锚点、
        不编造精度、是断言而非抄录、source 真实、该合并就合并。与检索能力无关 ——
        实测最常见的失败是"检索到了却写成一条下游读不懂的记录"。

        正文来自 ``codemem.prompts.EVIDENCE_STANDARD``（单独维护的一份可核对清单），
        由 ``load_catalog`` 填进 ``self.evidence_standard``。
        """
        return self.evidence_standard.strip()

    def render_strategy(self) -> str:
        """渲染"检索策略引导"这一段：什么题该关键词检索、什么题该通篇读。

        刻意只给**判据 + 一句做法**，不给固定流程 —— 策略由 agent 自己选，这是设计的一部分。
        """
        return self.strategy.strip()

    def render_protocol(self) -> str:
        lines = [
            str(self.protocol.get("response_format", "")).strip(),
            f"  shape: {json.dumps(self.protocol.get('call_schema', {}), ensure_ascii=False)}",
            f"  done:  {self.protocol.get('done_signal', '')}",
            f"  note:  {self.protocol.get('quote_arg_safety', '')}",
        ]
        return "\n".join(line for line in lines if line.strip())

    def render_prompt(self, question: str) -> str:
        """渲染完整 system prompt。``{{TOOLS}}`` 等槽位由 catalog 自己填，``{{QUESTION}}`` 用入参。"""
        text = self.prompt.get("system", "")
        replacements = {
            "{{TOOLS}}": self.render_tools(),
            "{{COMMANDS}}": self.render_commands(),
            "{{WORKSPACE}}": self.render_workspace(),
            "{{SCHEMA}}": self.render_schema(),
            "{{PROTOCOL}}": self.render_protocol(),
            "{{WRITING_POLICY}}": self.render_writing_policy(),
            "{{TIME_POLICY}}": self.render_time_policy(),
            "{{EVIDENCE_STANDARD}}": self.render_evidence_standard(),
            "{{STRATEGY}}": self.render_strategy(),
            "{{QUESTION}}": question.strip(),
        }
        for slot, value in replacements.items():
            text = text.replace(slot, value)
        return text

    def prompt_text(self, key: str, **kwargs: Any) -> str:
        """取 user_* 文案并按 ``{{KEY}}`` 做替换（键名大写）。"""
        text = str(self.prompt.get(key, ""))
        for name, value in kwargs.items():
            text = text.replace("{{" + name.upper() + "}}", str(value))
        return text


def load_catalog(path: str | Path | None = None) -> ToolCatalog:
    """读 ``tool.json``。缺字段就报错 —— 配置是设计的一部分，不该静默降级。"""
    target = Path(path) if path else DEFAULT_TOOL_CONFIG
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    if not target.exists():
        raise ToolConfigError(f"找不到工具配置：{target}")
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ToolConfigError(f"{target} 不是合法 JSON：{exc}") from exc
    if not isinstance(raw, dict):
        raise ToolConfigError(f"{target} 顶层必须是对象")

    for key in ("protocol", "tools", "commands", "workspace", "schema", "prompt"):
        if key not in raw:
            raise ToolConfigError(f"{target} 缺少顶层字段 '{key}'")
    if not raw["tools"]:
        raise ToolConfigError(f"{target} 的 tools 为空")

    return ToolCatalog(
        protocol=raw["protocol"],
        tools=raw["tools"],
        commands=raw["commands"],
        workspace=raw["workspace"],
        schema=raw["schema"],
        prompt=raw["prompt"],
        writing_policy=raw.get("writing_policy") or {},
        time_policy=raw.get("time_policy") or {},
        # 证据标准：优先取 tool.json 里的覆盖，缺省用 prompts.py 的常量（单一事实源）
        evidence_standard=str(raw.get("evidence_standard") or _default_evidence_standard()),
        strategy=str(raw.get("strategy") or ""),
        source_path=target,
    )


def _default_evidence_standard() -> str:
    """证据标准的缺省正文 —— ``codemem.prompts.EVIDENCE_STANDARD``。

    放在 prompts.py 而不是 tool.json，是因为它是一份**与检索策略无关的清单**，
    与 add 侧其它 prompt 同居更便于统一维护；tool.json 仍可覆盖它（``evidence_standard``
    顶层键），以便不改代码就调措辞。
    """
    from ..prompts import EVIDENCE_STANDARD

    return EVIDENCE_STANDARD
