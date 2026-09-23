# ---------------------------------------------------------------------------
# 原子记忆抽取
# ---------------------------------------------------------------------------

ATOM_EXTRACTION_SYSTEM_PROMPT = """CRITICAL: Your response must be ONLY a JSON object. No explanation, no reasoning, no markdown fences, no text before or after the JSON.

You are an atomic memory extraction engine. Decompose ONE target conversation message (with a small context window) into atomic memories an agent can later recall without the conversation.

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
CRITICAL OUTPUT FORMAT:
- Your entire response must be ONLY the JSON object starting with { and ending with }
- No markdown code fences (```json or ```)
- No explanatory text before or after the JSON
- No reasoning or commentary
- Just the raw JSON object

Format: {"atoms": [ ... ]}
- Keep it compact: field order memory, then metadata.{type,time,tag}; no pretty-printing, no trailing text.
- Output at most 12 atoms per message. If the message seems to yield more, keep the 12 most
  important and distinct facts - quality over exhaustiveness. A truncated response is useless.

Field definitions:

{{SCHEMA}}

REMINDER: Output ONLY the JSON object. Start with { and end with }. Nothing else."""


ATOM_EXTRACTION_USER_PROMPT = """## Context window ({window_before} before, {window_after} after)
Reference ONLY - use these to resolve who/what the TARGET refers to. Do NOT extract facts from them.

{context}

## TARGET message (extract ONLY from this one)

[id] ({target_time}) {target_speaker}: {target_text}

Extract the atomic facts stated in the TARGET message. Do not copy facts that belong to other
messages in the context window, and do not repeat the same fact twice.

CRITICAL: Output ONLY the JSON object. No markdown fences, no explanation. Start with {{ and end with }}."""


# ---------------------------------------------------------------------------
# 弱模型（DPO 的 rejected 侧）使用的简化 prompt
# ---------------------------------------------------------------------------
# 刻意保留几类常见但可学习的抽取错误，让能力较弱的小模型产出有区分度的 rejected
# 样本。仍要求输出可解析 JSON，避免所有 rejected 都退化成请求失败文本。

WEAK_ATOM_EXTRACTION_SYSTEM_PROMPT = """Extract facts from the conversation.

Rules:
- Return a few facts that seem relevant; do not try to be exhaustive.
- It is acceptable to copy a useful fact from the context when resolving the target.
- It is acceptable to combine closely related facts into one atom.
- You may keep simple pronouns or vague references when the meaning is obvious.
- Do not always split conjunctions into separate atoms.
- It is acceptable to miss implied roles, relationships, or secondary facts.
- It is acceptable to repeat an important fact if it appears more than once.
- Treat questions, greetings, and opinions as facts when they contain personal-sounding content.
- type: "inner" for lasting facts (identity, job, family, relationships), else "outer".
- time may be omitted or left as "" even when the message contains relative time.
- Include a speaker tag when convenient, but do not spend effort resolving every speaker.

Example:
Input: Melanie: "I'm swamped with the kids & work."
Output: {"atoms": [{"memory": "Melanie is swamped with the kids and work", "metadata": {"type": "outer", "time": "", "tag": ["speaker:Melanie"]}}]}

CRITICAL: Output ONLY the JSON object. No markdown fences, no explanation. Start with { and end with }.

Field definitions:

{{SCHEMA}}

IMPORTANT: Sometimes you will receive input that is confusing, contradictory, or has missing information. In those cases, do your best to still extract atoms. If you cannot extract any reasonable atoms, output a single atom with memory set to "No clear facts extracted." and metadata type "outer". Never refuse to output JSON. Never ask clarifying questions. Always produce a valid JSON object.
"""

ANSWER_PROMPT = """
Prompt template: memory-grounded question answering

You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. NEVER leave a relative time expression in the final answer. Replace every relative time with the concrete date, month, or year it resolves to, computed from the timestamp of the memory that states it. For example, a memory dated 2023-05-08 saying "last year" resolves to "2022"; a memory dated 2023-05-08 saying "two months ago" resolves to "March 2023".
7. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
8. If memories are insufficient and the question is about a general world fact, you may use reliable general world knowledge.
9. Keep the final answer concise, typically no more than 10-12 words; do not omit essential entities or dates.

# GROUNDING RULES (CRITICAL):
- Use ONLY facts stated in the memories, or clear logical consequences of them. Never invent a date, number, name, place, or detail.
- If the memories partially support an answer, give the part that is supported and mark the missing part as unknown rather than guessing.
- If the memories do not support an answer at all, set "answer" to null and "unsupported": true.

# SPECIFICITY RULES (CRITICAL):
- Resolve every relative time indicator into an absolute date, month, or year.
- Resolve every pronoun and deictic reference (he, she, they, it, this, that, here, there, the other day, that time) into the concrete entity, place, or date it refers to.
- Name every person, place, organization, and event explicitly instead of referring to them indirectly.
- Prefer the most specific form the memories actually support, e.g. "2022" over "a couple of years ago", "Melanie" over "she", "Paris" over "there".
- Make the answer understandable on its own, without the question or the memories.
- If a detail cannot be resolved from the memories, do NOT invent it; either omit it or mark it unknown.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based on the evidence in the memories, using general world knowledge only if memories are insufficient
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Memories:
{context}

Question: {question}

# OUTPUT
CRITICAL: Your entire response must be ONLY a JSON object. No markdown fences, no explanation, no text before or after the JSON.

Format:
{{"answer": "<final answer, or null if the memories do not support one>", "unsupported": <true|false>, "reasoning": "<brief: evidence used, and how relative times/pronouns were resolved>"}}

- "answer": concise and specific (normally <= 10-12 words); resolve all relative times and pronouns as described above.
- "unsupported": true only when the memories do not support any answer; then "answer" must be null.
- "reasoning": one or two short sentences, used for auditing only.

Example (memory dated 2023-05-08: "Melanie went to India last year"):
{{"answer": "2022", "unsupported": false, "reasoning": "Memory dated 2023-05-08 says Melanie went to India last year, which resolves to 2022."}}

REMINDER: Output ONLY the JSON object. Start with {{ and end with }}. Nothing else.
"""

BASELINE_ANSWER_PROMPT = """
You are an intelligent memory assistant answering a question from the complete conversation memory of two speakers.

Current Date: {current_date}

Instructions:
- Analyze all supplied memories and use direct evidence whenever available.
- Each memory includes a timestamp (YYYY-MM-DD HH:MM:SS) and speaker information - use these carefully.
- If the question asks "when" (about time), you MUST provide a specific date, year, or time period (e.g., "2022", "May 2022", "2021-03-15"). NEVER answer with relative terms like "yesterday", "last year", "recently", "two months ago".
- If the question asks "who" (about people), you MUST provide the specific person's name from the speaker or memory content.
- Convert any relative time references in the memories to absolute dates using the memory timestamp and current date.
- Do not invent facts, infer beyond what is stated, or add extra information not present in the memories.
- Answer concisely, normally in no more than 10-12 words, without omitting essential names, dates, or entities.
- If the memories do not contain enough information, answer briefly that it is unknown. Use general knowledge only for genuinely general facts.

# GROUNDING RULES (CRITICAL):
- Use ONLY facts stated in the memories, or clear logical consequences of them. Never invent a date, number, name, place, or detail.
- If the memories partially support an answer, give the supported part and mark the missing part as unknown.
- If the memories do not support an answer at all, set "answer" to null and "unsupported": true.

# SPECIFICITY RULES (CRITICAL):
- Replace every relative time expression (yesterday, last year, recently, two months ago, the other day) with the absolute date, month, or year it resolves to from the memory timestamp and current date.
- Replace every pronoun and deictic reference (he, she, they, it, this, that, here, there) with the concrete entity, place, or date it refers to.
- Name every person, place, organization, and event explicitly.
- Make the answer understandable on its own, without the question or the memories.
- Prefer the most specific form the memories actually support; if a detail cannot be resolved, do NOT invent it.

Complete conversation memories (with timestamps and speakers):
{context}

Question: {question}

# OUTPUT
CRITICAL: Your entire response must be ONLY a JSON object. No markdown fences, no explanation, no text before or after the JSON.

Format:
{{"answer": "<final answer, or null if the memories do not support one>", "unsupported": <true|false>, "reasoning": "<brief: evidence used, and how relative times/pronouns were resolved>"}}

- "answer": concise and specific (normally <= 10-12 words); resolve all relative times and pronouns as described above.
- "unsupported": true only when the memories do not contain enough information; then "answer" must be null.
- "reasoning": one or two short sentences, used for auditing only.

REMINDER: Output ONLY the JSON object. Start with {{ and end with }}. Nothing else.
"""

RAG_ANSWER_PROMPT = """
You are an intelligent memory assistant answering a question from retrieved conversation memories.

Current Date: {current_date}

Instructions:
- Treat the retrieved memories below as the only evidence for the answer.
- Each memory includes a timestamp (YYYY-MM-DD HH:MM:SS) and speaker information - use these carefully.
- If the question asks "when" (about time), you MUST provide a specific date, year, or time period (e.g., "2022", "May 2022", "2021-03-15"). NEVER answer with relative terms like "yesterday", "last year", "recently", "two months ago".
- If the question asks "who" (about people), you MUST provide the specific person's name from the speaker or memory content.
- Convert any relative time references in the memories to absolute dates using the memory timestamp and current date.
- Do not invent facts, infer beyond what is stated, or add extra information not present in the memories.
- Answer concisely, normally in no more than 10-12 words, without omitting essential names, dates, or entities.
- If the retrieved memories do not support an answer, answer briefly that it is unknown.

# GROUNDING RULES (CRITICAL):
- Use ONLY facts stated in the retrieved memories, or clear logical consequences of them. Never invent a date, number, name, place, or detail.
- If the retrieved memories partially support an answer, give the supported part and mark the missing part as unknown.
- If the retrieved memories do not support an answer at all, set "answer" to null and "unsupported": true.

# SPECIFICITY RULES (CRITICAL):
- Replace every relative time expression (yesterday, last year, recently, two months ago, the other day) with the absolute date, month, or year it resolves to from the memory timestamp and current date.
- Replace every pronoun and deictic reference (he, she, they, it, this, that, here, there) with the concrete entity, place, or date it refers to.
- Name every person, place, organization, and event explicitly.
- Make the answer understandable on its own, without the question or the retrieved memories.
- Prefer the most specific form the memories actually support; if a detail cannot be resolved, do NOT invent it.

Retrieved memories (with timestamps and speakers):
{context}

Question: {question}

# OUTPUT
CRITICAL: Your entire response must be ONLY a JSON object. No markdown fences, no explanation, no text before or after the JSON.

Format:
{{"answer": "<final answer, or null if the retrieved memories do not support one>", "unsupported": <true|false>, "reasoning": "<brief: evidence used, and how relative times/pronouns were resolved>"}}

- "answer": concise and specific (normally <= 10-12 words); resolve all relative times and pronouns as described above.
- "unsupported": true only when the retrieved memories do not support an answer; then "answer" must be null.
- "reasoning": one or two short sentences, used for auditing only.

REMINDER: Output ONLY the JSON object. Start with {{ and end with }}. Nothing else.
"""

JUDGE_PROMPT = """You are an evaluator judging whether a predicted answer correctly answers the question based on the reference answer.

Question: {question}
Reference Answer: {reference}
Predicted Answer: {prediction}

Evaluation Criteria:

1. TEMPORAL ACCURACY:
   - If the question asks "when", check if the prediction conveys the SAME point in time as the reference
   - Absolute dates (e.g., "2022", "7 May 2023") are preferred, but relative time expressions CAN be accepted IF they can be reasonably inferred from context
   - Example: In a conversation dated June 2023, "last year" = "2022" → ACCEPTABLE
   - However, if NO conversation date context is available, relative times should be marked INCORRECT
   - Key test: Would someone reading both answers agree they refer to the same time?

2. CORE FACTUAL ACCURACY:
   - The main factual claim of the prediction must match the reference
   - Minor additions (e.g., "for vacation" added to "went to India") are ACCEPTABLE if they don't change the core fact
   - Major additions that introduce new claims NOT in reference → INCORRECT
   - Omission of key information from reference → INCORRECT

3. ENTITY HANDLING:
   - Pronouns (he/she/they) are ACCEPTABLE if the entity is clear from context or the question itself
   - Naming the wrong person → INCORRECT
   - Adding extra people not mentioned in reference → INCORRECT if it changes the answer

4. SEMANTIC EQUIVALENCE:
   - Accept reasonable paraphrasing and rephrasing
   - Different word order, synonyms, or grammatical variations are OK
   - The meaning must be substantially the same
   - "studies psychology" ≈ "is studying psychology" ≈ "pursues a degree in psychology" → ACCEPTABLE

5. COMPLETENESS:
   - The prediction must cover all ESSENTIAL parts of the reference answer
   - Non-essential elaborations or stylistic differences are acceptable
   - A prediction that is a SUBSET of the reference (missing key facts) → INCORRECT

Decision Rules:
- When in doubt about temporal alignment, mark INCORRECT
- For all other criteria, lean towards ACCEPTANCE if the core answer is clearly present
- A prediction that adds reasonable, non-contradictory context around the correct answer should generally be accepted

Examples:
- Q: "When did Melanie paint a sunrise?" | Ref: "2022" | Pred: "last year"
  → Depends on context. With conversation date=2023: CORRECT. Without context: INCORRECT

- Q: "When did Melanie paint a sunrise?" | Ref: "2022" | Pred: "Melanie painted the sunrise last year."
  → Same as above. Context-dependent.

- Q: "When did Melanie paint a sunrise?" | Ref: "2022" | Pred: "In 2022" → CORRECT

- Q: "What did Caroline research?" | Ref: "Adoption agencies" | Pred: "Adoption agencies and career options"
  → INCORRECT if "career options" is not supported by evidence. But if evidence supports both, could be CORRECT.

- Q: "Where did they go?" | Ref: "Paris" | Pred: "They went to Paris for vacation"
  → CORRECT. Core fact preserved, minor addition doesn't contradict.

- Q: "Who went to Paris?" | Ref: "John" | Pred: "He went there"
  → CORRECT if question context makes clear "he" refers to John. INCORRECT if ambiguous.

- Q: "Where has Melanie camped?" | Ref: "beach, mountains, forest" | Pred: "beach and mountains"
  → INCORRECT. Missing "forest" is a key omission.

# OUTPUT
CRITICAL: Your entire response must be ONLY a JSON object. No markdown fences, no explanation, no text before or after the JSON.

Format:
{{"label": "CORRECT" | "INCORRECT", "reason": "<brief explanation, referencing the specific criteria above>"}}

- "label" must be exactly "CORRECT" or "INCORRECT" (uppercase).
- When the prediction is a JSON object, judge its "answer" field; a null answer with "unsupported": true counts as "unknown" and is INCORRECT when the reference answer is known.
- "reason": one or two short sentences naming the criteria that decided the verdict.

REMINDER: Output ONLY the JSON object. Start with {{ and end with }}. Nothing else."""

# ---------------------------------------------------------------------------
# EvoMem：逐步演化原子记忆
# ---------------------------------------------------------------------------
# 一次调用 = 一条原子的一轮演化：给出**这一条**目标记忆 + 召回的相关记忆（附源消息
# 上下文）+ 这条原子之前各轮已经做过的动作，要求模型输出**一个**动作块。同一条原子
# 反复调用直到 NOOP（或到 max_evomem_turn），然后游标移到下一条原子。
#
# 四个占位符，都由 evomem.build_messages 填充：
#   {{CURRENT}}  当前要精修的那一条（单条，不是整个库）
#   {{RECALLED}} 召回的证据（源消息 + 前后上下文）
#   {{HISTORY}}  这条原子之前几轮的动作与执行结果
#   {{SCHEMA}}   字段定义，由 memory_template_evomem.json 生成
#
# 结构刻意保持"模块化"：ROLE / TARGET / EVIDENCE / HISTORY / WHEN TO ACT / ACTIONS /
# FIELDS / OUTPUT。字段细节**只**出现在 {{SCHEMA}} 里（由模板生成），正文不再重复 ——
# 旧版本把字段规则手写了一遍又一遍，改模板时两处会对不上。

EVOMEM_PROMPT = """Output exactly one tag: <ADD>...</ADD>, <UPDATE>...</UPDATE>, <DELETE>...</DELETE>, or <NOOP></NOOP>. No other text.

# STOP — check {{HISTORY}} first
{{HISTORY}}
If history shows REJECTED, IDENTICAL, or 2+ consecutive failures → <NOOP></NOOP>

# TASK
Refine one memory atom. Make it accurate, self-contained, connected.
Do NOT change meaning unless evidence proves it wrong.

## TARGET
{{CURRENT}}

## EVIDENCE (sole source of truth)
{{RECALLED}}

## DECIDE — first match wins
1. **WRONG/OUTDATED**: Evidence contradicts target → UPDATE.
2. **NOT SELF-CONTAINED**: Uses "yesterday"/"he"/"there" etc → UPDATE with absolute values.
3. **SPLIT/IMPLICIT**: Target + evidence only make sense together → UPDATE one or ADD bridge. Clear implication missing → ADD it.
4. **REDUNDANT**: Evidence states exact same fact → DELETE target. Same info ≠ same topic.
5. **ELSE** → <NOOP></NOOP>

## FORMATS
<ADD>[{"memory":"...","metadata":{"id":"","type":"outer","time":"","tag":["speaker:X"],"source":[],"changelog":[]}}]</ADD>
<UPDATE>[{"memory":"...","metadata":{"id":"EXACT_ID","type":"...","time":"...","tag":[...],"source":[...],"changelog":[]}}]</UPDATE>
<DELETE>[{"memory":"exact text","metadata":{"id":"EXACT_ID","type":"...","time":"...","tag":[...],"source":[...],"changelog":[]}}]</DELETE>
<NOOP></NOOP>

## RULES
- UPDATE: token-F1≥0.6 vs original. id must match. No paraphrase-only rewrites.
- ADD: id="", source non-empty. Max 3. Must add new info, not copy target.
- DELETE: id + memory must match stored entry. Only when evidence already says same thing.
- NOOP is default. Don't force changes.
- Never invent names/dates/places. Source must reference real IDs.
- Fields: {{SCHEMA}}"""


# 「当前要精修的那一条原子」的标题。evomem 按 id 顺序逐条处理，prompt 里只有这一条是
# 目标，其余（recalled）只是证据。所以不要叫 "Library" —— 模型会以为它看见了整个库，
# 进而对着看不见的全集去删（实测过的乱删就是这么来的）。
#
# 这个标题属于 **prompt 模板**（写在 EVOMEM_PROMPT 的 `# TARGET` 段里），不属于被替换
# 进去的值：值里再带一个二级标题会让结构看起来错乱。这里保留常量供 format_current 在
# 空值分支复用。
EVOMEM_CURRENT_PLACEHOLDER = "{{CURRENT}}"
EVOMEM_CURRENT_HEADER = "## Target entry"

EVOMEM_HISTORY_EMPTY = "(nothing yet - this is the first attempt on this entry)"

EVOMEM_USER_SUFFIX = (
    "Produce the single action block now, following the format exactly."
)
