# ---------------------------------------------------------------------------
# 原子记忆抽取
# ---------------------------------------------------------------------------

ATOM_EXTRACTION_SYSTEM_PROMPT = """You are an atomic memory extraction engine. Decompose ONE target conversation message (with a small context window) into atomic memories an agent can later recall without the conversation.

## Information to Remember
- Personal preferences: likes, dislikes, favorites, opinions.
- Important personal details: names, relationships, family structure, durations, significant life facts.
- Plans and intentions: explicit future goals, plans, or intentions.
- Activities and routines: travel, visited places, recurring habits, hobbies with context.
- Professional details: job titles, career goals, professional interests, work habits.
- Miscellaneous meaningful facts: books, movies, creative work, projects, notable activities.
- Health and wellness (non-diagnostic): wellness experiences or preferences; do NOT infer diagnoses.

## What to Ignore (apply FIRST)
- System-level instructions and any non-dialogue control text.
- The phatic ACT of a greeting or courtesy phrase ("Good to see you", "Thanks", "That sounds great").
- The bare ACT of a question or dialogue act (asks, says hello, thanks) - the act itself is NOT a fact.
- Generic statements, opinions without substance, and common knowledge.
- If, after removing the above, no meaningful fact remains, output an empty atoms list. Many turns are legitimately empty.
- IMPORTANT: only the ACT is discarded. Any INFORMATION carried inside a greeting-like turn is still extracted - see Core Extraction Rules.

## Core Extraction Rules
- ONE core fact per atom. Split compound/multi-fact sentences into separate atoms.
- Brief and self-contained: <= 20 words, understandable without the conversation.
- Resolve pronouns and deixis to concrete entities using the context window.
- Extract facts for BOTH speakers; extract explicit content AND clear implications.
- **Extract implied durable facts**: a turn that mentions a lasting attribute implies that attribute as its own fact. E.g. "swamped with the kids & work" implies BOTH "Melanie has kids" AND "Melanie has a job"; "my husband and I" implies "X has a spouse". Extract these even when the turn is otherwise casual.
- **Extract what greetings/messages carry**: nicknames, address forms, and terms of endearment ARE facts (e.g. "Hey Mel!" -> "Mel is a nickname for Melanie"). Extract the information, never the greeting act.
- **Presupposition extraction with duality**: When a speaker states a fact involving another entity, extract the underlying relationship or involvement as a fact too (e.g. "Melanie is Caroline's friend").
- **Symmetric extraction**: Compound elements joined by conjunctions (and/or/&) must each yield a separate atom. Do not selectively extract only some elements.
- **Lexical fidelity**: Use exact words from source text. Only rephrase for pronoun resolution, article insertion, or minimal tense normalization.

## Self-Contained Fact Rules (CRITICAL)
- Every fact must be understandable when retrieved alone.
- Every fact MUST explicitly name the subject speaker (e.g., "John ...", "Tim ...").
- Avoid unresolved pronouns (`he`, `she`, `they`, `them`, `it`, `this`, `that`) unless the noun is in the same fact.
- Rewrite vague references to explicit entities. If the entity cannot be resolved, do NOT store the fact.
- Style: third person, starting with the subject name. Good: "John wants to keep reaching for new goals". Bad: "Wants to keep reaching for new goals".

## Intent vs Event Rule
- Past events (what happened) and intentions or goals (what the speaker wants or plans) MUST be extracted as SEPARATE atoms.

## Inner vs Outer Guide
- **inner**: Fundamental, long-term identity or life facts. Includes presupposed life roles (parent, employee, student, spouse, etc.), core relationships, occupation, residence, values, major life events.
- **outer**: Everything else. Transient facts, actions, preferences, plans, intentions, emotions, and temporary states.

## Scope: extract from the TARGET only
- Extract facts ONLY from the TARGET message. The context window is provided solely to resolve
  pronouns, references, and implicit entities in the TARGET - never extract a fact that is stated
  entirely in a context message.
- Every atom's fact must be supported by the TARGET message itself. If a fact is already fully
  stated in the context (another message), do NOT output it again here; it belongs to that message.
- Do not duplicate an atom within this turn either: output each distinct fact exactly once.
- If the TARGET carries no new fact (e.g. it only asks or greets), output an empty atoms list even
  when the surrounding context is full of facts.

## Prohibited Behaviors
- NEVER merge multiple distinct facts (multiple events, timeframes, motivations, or reflections) into a single atom.
- NEVER fabricate facts not stated or reasonably implied.
- NEVER selectively extract only part of a compound list while dropping other elements.
- NEVER store a fact that depends on another fact for context.

## Output
Return ONLY valid JSON, no prose, no fences. Wrap atoms as {"atoms": [ ... ]}.
- Keep it compact: field order memory, then metadata.{type,time,tag}; no pretty-printing, no trailing text.
- Output at most 12 atoms per message. If the message seems to yield more, keep the 12 most
  important and distinct facts - quality over exhaustiveness. A truncated response is useless.

Field definitions:

{{SCHEMA}}
"""


ATOM_EXTRACTION_USER_PROMPT = """## Context window ({window_before} before, {window_after} after)
Reference ONLY - use these to resolve who/what the TARGET refers to. Do NOT extract facts from them.

{context}

## TARGET message (extract ONLY from this one)

[id] ({target_time}) {target_speaker}: {target_text}

Extract the atomic facts stated in the TARGET message. Do not copy facts that belong to other
messages in the context window, and do not repeat the same fact twice.
Output ONLY the JSON object."""


# ---------------------------------------------------------------------------
# 弱模型（DPO 的 rejected 侧）使用的简化 prompt
# ---------------------------------------------------------------------------
# 刻意比强模型版本精简：只保留最基本的约束与样例，让能力较弱的小模型暴露其真实水平，
# 从而产出有区分度的 rejected 样本。生成 chosen 的强模型仍使用完整的
# ATOM_EXTRACTION_SYSTEM_PROMPT。

WEAK_ATOM_EXTRACTION_SYSTEM_PROMPT = """Extract atomic facts from the conversation message.

Rules:
- One fact per atom, short and self-contained.
- Start each fact with the speaker's name (third person). No pronouns like he/she/they.
- Ignore greetings and questions.
- type: "inner" for lasting facts (identity, job, family, relationships), else "outer".
- Set time only when it is known, else use "".
- Always include a "speaker:<name>" tag.

Example:
Input: Melanie: "I'm swamped with the kids & work."
Output: {"atoms": [{"memory": "Melanie has kids", "metadata": {"type": "inner", "time": "", "tag": ["speaker:Melanie"]}}]}

Return ONLY JSON: {"atoms": [ ... ]}. No prose, no fences.

Field definitions:

{{SCHEMA}}
"""
