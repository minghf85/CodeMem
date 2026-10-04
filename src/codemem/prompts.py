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
- Tag keys are FIXED (speaker/topic/activity/action/event/state/sentiment/relation/entity/time/other). A hobby or pastime is `activity:<name>` -- never `action:` or `hobby:`. Something merely talked about is `topic:<name>`. Never invent a new key.
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
5. **The memory timestamp is WHEN THE CONVERSATION HAPPENED -- it is almost never when the event happened.** A memory dated 2023-06-09 saying "I gave a speech last week" means the speech was in the week before 9 June 2023, NOT on 9 June 2023. Never answer with the memory's own timestamp as if it were the event date.
6. **A relative expression that carries its anchor is a COMPLETE answer.** "the week before 9 June 2023", "the Friday before 15 July 2023", "the weekend before 20 October 2023" are exact and need no further context -- answer with that form, or with the single date it resolves to, whichever the memory supports. Do NOT flatten an anchored expression into the anchor itself.
7. **Only answer with a bare relative word if the anchor is resolvable and you state it.** "next month" alone is WRONG; "June 2023" (i.e. the month after the 2023-05-25 conversation) is right. Same for "last year" -> give the year, "recently" -> give the week or month. Never leave the reader to guess what "next" or "recent" is relative to.
8. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
9. If memories are insufficient and the question is about a general world fact, you may use reliable general world knowledge.
10. Keep the final answer concise, typically no more than 10-12 words; do not omit essential entities or dates.

# GROUNDING RULES (CRITICAL):
- Use ONLY facts stated in the memories, or clear logical consequences of them. Never invent a date, number, name, place, or detail.
- If the memories partially support an answer, give the part that is supported and mark the missing part as unknown rather than guessing.
- If the memories do not support an answer at all, set "answer" to null and "unsupported": true.

# SPECIFICITY RULES (CRITICAL):
- **Resolve a relative time against its own anchor, not against the conversation date.** "last Saturday" in a memory dated 2023-05-25 is "the Saturday before 25 May 2023"; "ten years ago" in a memory dated 2023-06-27 is "10 years ago" (the year, since that is all the speaker gave).
- An anchored phrase ("the week before <date>", "the Friday before <date>") is already absolute -- keep it as-is, or give the date it resolves to. Both are accepted; the anchor alone is NOT.
- Do NOT add precision the memory does not have. If the speaker said "last year", a year is the correct granularity; do not invent a specific day.
- Resolve every pronoun and deictic reference (he, she, they, it, this, that, here, there, that time) into the concrete entity, place, or date it refers to.
- Name every person, place, organization, and event explicitly instead of referring to them indirectly.
- Prefer the most specific form the memories actually support, e.g. "2022" over "a couple of years ago", "Melanie" over "she", "Paris" over "there".
- Make the answer understandable on its own, without the question or the memories.
- If a detail cannot be resolved from the memories, do NOT invent it; either omit it or mark it unknown.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation, DO the calculation and show it in reasoning: name the anchor date and the offset ("memory dated 2023-06-09 + 'last week' -> the week before 9 June 2023"). Never write the anchor date as the answer to a "when did X happen" question
5. Formulate a precise, concise answer based on the evidence in the memories, using general world knowledge only if memories are insufficient
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Memories:
{context}

Question: {question}

# OUTPUT
CRITICAL: Your entire response must be ONLY a JSON object. No markdown fences, no explanation, no text before or after the JSON.

Format:
{{"reasoning": "<first: name the memory you used, its date, and how you resolved the time>", "answer": "<final answer, or null if the memories do not support one>", "unsupported": <true|false>}}

- "reasoning": FIRST. One or two sentences: which memory you used, that memory's date, and how you turned its wording into the answer. Write this before the answer.
- "answer": concise and specific (normally <= 10-12 words); resolve all relative times and pronouns as described above.
- "unsupported": true only when the memories do not support any answer; then "answer" must be null.

Example (memory dated 2023-05-08: "Melanie went to India last year"):
{{"reasoning": "Memory dated 2023-05-08 says Melanie went to India last year; one year before 2023 is 2022.", "answer": "2022", "unsupported": false}}

Example (memory dated 2023-06-09: "I gave a speech at a school last week"):
{{"reasoning": "Memory dated 2023-06-09; the speech was 'last week', i.e. the week before 9 June 2023 -- not on 9 June itself.", "answer": "the week before 9 June 2023", "unsupported": false}}

REMINDER: Output ONLY the JSON object. Start with {{ and end with }}. Nothing else.
"""

BASELINE_ANSWER_PROMPT = """
You are an intelligent memory assistant answering a question from the complete conversation memory of two speakers.

Current Date: {current_date}

Instructions:
- Analyze all supplied memories and use direct evidence whenever available.
- Each memory includes a timestamp (YYYY-MM-DD HH:MM:SS) and speaker information - use these carefully.
- If the question asks "when" (about time), give a date, year, or time period (e.g., "2022", "May 2022", "2021-03-15"). An anchored phrase such as "the week before 9 June 2023" is acceptable and exact -- it needs no outside context.
- **The memory timestamp is WHEN THE CONVERSATION HAPPENED, not when the event happened.** A memory dated 2023-06-09 saying "last week" refers to the week before 9 June 2023, NOT 9 June 2023. Never return the memory's own timestamp as the event date.
- Anchor every relative expression before using it: "next month" is meaningless on its own -- resolve it to "June 2023" from the memory's date. Never answer with an unanchored relative word.
- If the question asks "who" (about people), you MUST provide the specific person's name from the speaker or memory content.
- Do not invent facts, infer beyond what is stated, or add extra information not present in the memories.
- Answer concisely, normally in no more than 10-12 words, without omitting essential names, dates, or entities.
- If the memories do not contain enough information, answer briefly that it is unknown. Use general knowledge only for genuinely general facts.

# GROUNDING RULES (CRITICAL):
- Use ONLY facts stated in the memories, or clear logical consequences of them. Never invent a date, number, name, place, or detail.
- If the memories partially support an answer, give the supported part and mark the missing part as unknown.
- If the memories do not support an answer at all, set "answer" to null and "unsupported": true.

# SPECIFICITY RULES (CRITICAL):
- Resolve a relative time against its own anchor (the memory it appears in), not against the conversation date. Keep an anchored phrase as-is if that is what the memory supports ("the week before 9 June 2023").
- Do NOT add precision the memory lacks: "last year" -> a year; "last Saturday" -> that weekday.
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
{{"reasoning": "<first: name the memory you used, its date, and how you resolved the time>", "answer": "<final answer, or null if the memories do not support one>", "unsupported": <true|false>}}

- "reasoning": FIRST. One or two sentences: which memory you used, that memory's date, and how you turned its wording into the answer. Write this before the answer.
- "answer": concise and specific (normally <= 10-12 words); resolve all relative times and pronouns as described above.
- "unsupported": true only when the memories do not contain enough information; then "answer" must be null.

REMINDER: Output ONLY the JSON object. Start with {{ and end with }}. Nothing else.
"""

RAG_ANSWER_PROMPT = """
You are an intelligent memory assistant answering a question from retrieved conversation memories.

Current Date: {current_date}

Instructions:
- Treat the retrieved memories below as the only evidence for the answer.
- Each memory includes a timestamp (YYYY-MM-DD HH:MM:SS) and speaker information - use these carefully.
- If the question asks "when" (about time), give a date, year, or time period (e.g., "2022", "May 2022", "2021-03-15"). An anchored phrase such as "the week before 9 June 2023" is acceptable and exact -- it needs no outside context.
- **The memory timestamp is WHEN THE CONVERSATION HAPPENED, not when the event happened.** A memory dated 2023-06-09 saying "last week" refers to the week before 9 June 2023, NOT 9 June 2023. Never return the memory's own timestamp as the event date.
- Anchor every relative expression before using it: "next month" is meaningless on its own -- resolve it to "June 2023" from the memory's date. Never answer with an unanchored relative word.
- If the question asks "who" (about people), you MUST provide the specific person's name from the speaker or memory content.
- Do not invent facts, infer beyond what is stated, or add extra information not present in the memories.
- Answer concisely, normally in no more than 10-12 words, without omitting essential names, dates, or entities.
- If the retrieved memories do not support an answer, answer briefly that it is unknown.

# GROUNDING RULES (CRITICAL):
- Use ONLY facts stated in the retrieved memories, or clear logical consequences of them. Never invent a date, number, name, place, or detail.
- If the retrieved memories partially support an answer, give the supported part and mark the missing part as unknown.
- If the retrieved memories do not support an answer at all, set "answer" to null and "unsupported": true.

# SPECIFICITY RULES (CRITICAL):
- Resolve a relative time against its own anchor (the memory it appears in), not against the conversation date. Keep an anchored phrase as-is if that is what the memory supports ("the week before 9 June 2023").
- Do NOT add precision the memory lacks: "last year" -> a year; "last Saturday" -> that weekday.
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
{{"reasoning": "<first: name the memory you used, its date, and how you resolved the time>", "answer": "<final answer, or null if the retrieved memories do not support one>", "unsupported": <true|false>}}

- "reasoning": FIRST. One or two sentences: which memory you used, that memory's date, and how you turned its wording into the answer. Write this before the answer.
- "answer": concise and specific (normally <= 10-12 words); resolve all relative times and pronouns as described above.
- "unsupported": true only when the retrieved memories do not support an answer; then "answer" must be null.

REMINDER: Output ONLY the JSON object. Start with {{ and end with }}. Nothing else.
"""

JUDGE_PROMPT = """You are an evaluator judging whether a predicted answer correctly answers the question based on the reference answer.

Question: {question}
Reference Answer: {reference}
Predicted Answer: {prediction}

Evaluation Criteria:

1. TEMPORAL ACCURACY:
   - If the question asks "when", check whether the prediction and the reference denote the SAME point in time.
   - **A reference phrased relatively and a prediction phrased absolutely are EQUIVALENT when the arithmetic works out.**
     This is the single most common way judges get it wrong -- do the arithmetic before deciding.
   - A [PRECOMPUTED] line may be appended below. When it says the two are the SAME date, treat
     the temporal criterion as SATISFIED and do not re-derive the arithmetic yourself.
   - "the week before X" / "a week before X" / "N weeks before X" = X minus 7 (or 7N) days
   - "the weekend before X" = the Saturday/Sunday immediately preceding X (strictly before).
     "N weekends before X" = the Nth such weekend back.
   - "the <weekday> before X" = the nearest that weekday strictly before X
   - "last week"/"last year" relative to a conversation dated Y = the corresponding period before Y
   - WORKED EXAMPLES -- verify the weekday yourself before trusting these:
       "The week before 6 July 2023"            -> 2023-06-29      (6 Jul is a Thursday; -7d)
       "The Friday before 15 July 2023"         -> 2023-07-14      (15 Jul is a SATURDAY; prior Friday = 14 Jul)
       "The weekend before 17 July 2023"        -> 2023-07-15/16   (17 Jul is a MONDAY; prior weekend = 15-16 Jul)
       "The Tuesday before 20 July 2023"        -> 2023-07-18      (20 Jul is a THURSDAY; prior Tuesday = 18 Jul)
       "two weekends before 17 July 2023"       -> 2023-07-08/09   (one weekend back = 15-16 Jul; two = 8-9 Jul)
   - IMPORTANT: a prediction that equals the CONVERSATION date rather than the event date is
     INCORRECT. If the memory is dated 9 June 2023 and says "last week", then "9 June 2023" is
     wrong and "the week before 9 June 2023" (or 2 June 2023) is right.
   - Allow +/- 1 day on a computed weekday/weekend, and accept a month-only answer when the
     reference is month-only (or vice versa) as long as the month matches.
   - Only mark INCORRECT when the two answers denote DIFFERENT points in time after the
     arithmetic, or when the prediction leaves a time un-resolved that the reference resolves.
   - Key test: convert BOTH to a concrete date if you can, then compare; do not compare the
     surface wording.

2. CORE FACTUAL ACCURACY:
   - The main factual claim of the prediction must match the reference
   - **An empty prediction is ALWAYS INCORRECT** when the reference states an answer. A prediction of
     null / "unsupported" / "unable to determine" asserts nothing -- it cannot "match" a reference that
     does state the fact. This is the single most common way a judge inflates the score, so check it
     FIRST: if the prediction is empty or says the memories are insufficient, mark INCORRECT.
   - **Extra content is NOT a defect when the question asks for a SET or a LIST.** For "what activities
     / books / places / events / people", the reference is usually an incomplete list. A prediction that
     contains every reference item PLUS additional correct items has a SUPERSET of the answer and is
     CORRECT -- the extra items are more of the answer, not a contradiction. Only mark INCORRECT if an
     added item is wrong, or if the question asked for a specific single fact.
   - Minor additions (e.g., "for vacation" added to "went to India") are ACCEPTABLE if they don't change the core fact
   - Additions that CONTRADICT the reference, or that misstate the asked-about fact → INCORRECT
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
{{"reason": "<the comparison you made, in one or two sentences>", "label": "CORRECT" | "INCORRECT"}}

- "reason": FIRST. One or two sentences comparing the two answers. Write this before the label.
  Do NOT name or recite the criteria headings -- just make the comparison.
- **Decide by DEFAULT to CORRECT.** The prediction does not have to be worded like the reference. Ask only: does it state the same fact (allowing paraphrase, extra correct detail, and a superset for set questions)? Mark INCORRECT only for a clear defect: a different fact, a wrong entity, an omitted key item, or an empty answer.
- "label": exactly "CORRECT" or "INCORRECT" (uppercase).
- When the prediction is a JSON object, judge its "answer" field; a null answer with "unsupported": true counts as "unknown" and is INCORRECT when the reference answer is known.

REMINDER: Output ONLY the JSON object. Start with {{ and end with }}. Nothing else."""


# ---------------------------------------------------------------------------
# summem：一个会话 → 一段总结（逐 session 一次 LLM 调用）
# ---------------------------------------------------------------------------
#
# summem 是三条记忆链里"会话粒度"的那一层：把一次会话里的对话总结成一段自足的叙述，
# 供检索时定位"哪次会话谈过这件事"。它**不是**给 agent 当作答案的（实测顶层印象会诱导
# 模型不检索就作答），所以 search 的会话清单只给"session_N (时间)"，summary 正文要 agent
# 自己 grep。
#
# 三条硬约束与 resolve 一致：**自足**（时间绝对或带锚点、指代消解）、**精度只减不增**、
# 不编造事实。总结里保留可检索的具体实体与时间。

SUMMEM_SYSTEM_PROMPT = """CRITICAL: Your response must be ONLY a JSON object. No explanation, no reasoning, no markdown fences, no text before or after the JSON.

You summarize ONE session of a conversation between {speaker_a} and {speaker_b}. A reader who has never seen the conversation must understand the summary, and be able to find the session again by grepping a name, place, or fact in it.

## What the summary must contain
- The main events, facts, and topics of the session, in the order they came up.
- Concrete entities: every person, place, organization, book, object, and event by NAME.
- Time, resolved: the session's own date is the anchor. A fact stated as "last week" becomes "the week before {session_date}" (or the exact date when the arithmetic is certain). Never leave a bare relative word.

## Rules
- Resolve every pronoun and pointing word (he / she / they / it / there / then) to the concrete person, place, or time.
- Do NOT invent facts, dates, or names that the session does not support.
- Do NOT add precision the source lacks: "last year" stays a year, not a date.
- 3-8 sentences. Dense and specific, not vague ("they talked about life" is useless).
- Write it as a standalone paragraph a downstream reader can use directly.

## Output
Your entire response must be ONLY this JSON object, starting with { and ending with }:

{"memory": "<the session summary, self-contained>"}
"""


SUMMEM_USER_PROMPT = """## Session {session_index} of the conversation
Session time (the anchor for all relative times): {session_time}

## Messages of this session

{messages}

Summarize this ONE session into a self-contained paragraph (resolve time + references, keep the concrete entities).
"""


# ---------------------------------------------------------------------------
# summem/msgmem 的软解析：从模型输出里取 JSON 对象（容忍围栏/前后说明/截断尾巴）
# ---------------------------------------------------------------------------

def salvage_json_object(raw: str) -> dict | None:
    """从模型输出里取一个 JSON 对象；容忍围栏、前后说明、被截断的尾巴。

    add 侧（summem 总结、atommem agent 的字段）共用这一份，避免多个模块各写一份导致不一致。
    """
    import json
    import re as _re

    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = _re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = _re.sub(r"\s*```$", "", text).strip()
    candidates = [text]
    if "{" in text:
        candidates.append(text[text.find("{"): text.rfind("}") + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    try:
        from ..io import salvage_json_records  # type: ignore[import-not-found]

        salvaged, _ = salvage_json_records(text)
        return salvaged[0] if salvaged else None
    except Exception:  # noqa: BLE001 - 解析失败一律返回 None，调用方保留原状
        return None


# ---------------------------------------------------------------------------
# atommem：add agent 的原子事实标准（agent 只写 memory + metadata.{time, role, tag}）
# ---------------------------------------------------------------------------
#
# atommem 是三条记忆链里最细的那一层：**逐个 message** 抽原子事实。核心原理是"后面的
# message 可能受前面内容影响、可能跨 session 影响"，所以 agent 处理一条 message 时要能
# 检索 msgmem / summem / 已经生成的 atommem 里**相关的**前面内容，才能把指代消解对、
# 把隐含事实补全，最终得到高质量的 atommem。
#
# **agent 只填四个字段**：``memory`` 与 ``metadata.{time, role, tag}``。``id`` / ``source`` /
# ``type`` / ``changelog`` 一律由系统生成 —— 所以 prompt 里明确只输出这四样。
#
# 标准与旧的 ATOM_EXTRACTION_* 一脉相承（一事实一 atom / 自足 / 显式+隐含 / 拆复合），
# 但多了两件事：① 时间要**解决**（自足、精度只减不增，锚点是该 message 的会话时间）；
# ② 指代要**解决**（可检索前文确认）。见 `add/atommem.py` 的 agent loop。

ATOM_AGENT_STANDARD = """An atom is ONE self-contained fact that a reader can understand with no conversation and no other atom. Every atom must pass all of these:

1. SELF-CONTAINED. It names its subject explicitly (a real name), so it makes sense alone. Never "he"/"she"/"it"/"there" as the only reference.
2. ONE FACT. One core fact per atom. Split compound sentences (and / or / &) into separate atoms -- extract EVERY element, never selectively.
3. TIME RESOLVED. If the message carries time, the atom's `time` must be self-contained: an absolute time ("7 May 2023") or a relative phrase that carries its own anchor ("the week before 9 June 2023"). A bare "yesterday"/"last week" is NOT allowed -- the message's own session time is the anchor. Match the source's granularity exactly: "last year" -> "2022", never a day; "yesterday" -> the exact date. Empty string when the fact has no time.
4. REFERENCES RESOLVED. Replace every pronoun / pointing word with what it really is (real name / real place / real thing / resolved time). If you cannot tell what a pronoun means, search the earlier messages and summaries for it -- do not guess.
5. A FACT, NOT AN ACT. Do not store greetings, thanks, bare questions, or the act of speaking. A turn that is only a greeting yields no atom.
6. IMPLIED DURABLE FACTS COUNT. A turn mentioning a lasting attribute implies that attribute as its own atom: "swamped with the kids & work" implies BOTH "Melanie has kids" AND "Melanie has a job"; "my husband and I" implies "X has a spouse". Extract these even from casual turns.
7. WHAT GREETINGS CARRY COUNTS. Nicknames, address forms, terms of endearment ARE facts: "Hey Mel!" -> "Mel is a nickname for Melanie". Extract the INFORMATION, never the greeting act.
8. NO FABRICATION. Never invent a fact, a name, a date, or a detail the message does not state or clearly imply.
9. KEEP THE SOURCE'S WORDS where you can. Only rephrase for pronoun resolution, article insertion, or minimal tense normalization.

Style: third person, starting with the subject's name. "Caroline went to an LGBTQ support group on 7 May 2023." (good) / "Went to a support group" (bad -- no subject)."""


ATOM_AGENT_SYSTEM_PROMPT = """You extract ATOMIC MEMORIES from ONE message of a conversation between {speaker_a} and {speaker_b}.

You are a code agent with `read`, `edit`, and `bash` tools. You work in a directory that holds the conversation's memory files.

# RESPONSE FORMAT

{{PROTOCOL}}

# THE MEMORY FILES

{{WORKSPACE}}

# HOW TO SEARCH

{{STRATEGY}}

# TOOLS

{{TOOLS}}

# COMMANDS (use through bash)

{{COMMANDS}}

# WHAT AN ATOM IS

{{ATOM_STANDARD}}

# RULES

- You fill ONLY `memory` and `metadata.{time, role, tag}`. Do NOT emit `id`, `source`, `type`, or `changelog` -- the system adds them.
- Write TARGET facts only. Use the other messages ONLY to resolve references; never write an atom for a fact stated entirely in another message.
- If the TARGET is a bare greeting or a question with no fact, write nothing and finish immediately.
- Do NOT edit msgmem.jsonl or summem.jsonl -- only atommem.jsonl is yours to append.
- When you have written the target's atoms, reply with plain text (no JSON) to finish.
"""


ATOM_AGENT_USER_PROMPT = """## The conversation is split into these sessions
{sessions}

## Earlier messages in THIS session (message {msg_id} is the target -- do NOT extract from these)

{window}

## TARGET message (extract atoms ONLY from this one)

{target}

Extract this ONE message's atomic facts. Search the memory files first if you need to resolve a pronoun or a reference, then append each atom to atommem.jsonl (only `memory` + `metadata.{{time, role, tag}}`). Reply with plain text when done.
"""


# ---------------------------------------------------------------------------
# 证据标准（search 的核心：一条证据要写法上合格，找到只是第一步）
# ---------------------------------------------------------------------------
#
# 为什么单独抽成一个变量：**"找到了证据"和"写出来的证据能用"是两件事**。实测里最
# 常见的失败不是没检索到，而是检索到了却写成了一条下游读不懂的记录 —— 指代没消解
# （"she went there"）、时间只有个无锚点的相对词（"next month"）、或者把两条只有合看
# 才有意义的记录原样抄成两条。这一类问题与"检索策略"无关，只与"记录的写法"有关。
#
# 所以它被提成一份**可核对的清单**（每条都能用眼睛验），由 tool.json 的
# {{EVIDENCE_STANDARD}} 槽位注入 agent 的 system prompt，与 writing_policy / time_policy
# 并列。改了它就等于改了"什么算一条合格的证据"。

EVIDENCE_STANDARD = """A record is ACCEPTABLE only if it passes every check below. Run these checks on each record before you consider it written.

1. SELF-CONTAINED. A reader who has never seen the conversation, the memory files, or the question must understand it. If it needs anything from outside the record itself to make sense, rewrite it or drop it.
2. REFERENTS RESOLVED. No bare pronoun or pointing word survives: she / he / they / it / this / that / there / then / the other one. Replace each with the concrete person, place, or thing. (A name used as the subject of the sentence is the target; "Caroline" is resolved, "she" is not.)
3. TIME ANCHORED. Any time expression either (a) is already an absolute date or period ("7 May 2023", "June 2023", "the week of 23 August 2023"), or (b) is a relative phrase that carries its own anchor in the record ("the week before 9 June 2023"). A relative word with no anchor ("next month", "recently", "the other day") is NOT acceptable -- write the anchor next to it.
4. NO INVENTED PRECISION. State the time at the granularity the source gave. If the source said "last year", the record says a year, not a day. Do not compute a date the source did not support.
5. A CLAIM, NOT A QUOTE. The record states a fact, not a transcript turn. Paraphrase into a standalone statement; do not paste the raw message.
6. SOURCED. metadata.source lists the real record ids you read to support this record -- never an id you only remember or guessed. (A message id is `session_N_M`; an atom id is `session_N_M_K`.) If you cannot point to an id, you cannot write the record.
7. ONE FACT, JOINED IF NEEDED. If answering needs two records joined, write the joined conclusion as one record (and list both sources). If a question wants a SET, list the members, ideally in one record -- not one record per member.

If a candidate record fails a check, fix it (usually by adding the name or the anchor) rather than dropping the evidence. Drop it only when you genuinely cannot resolve it -- a partially-resolved record is worse than a resolved one, but a dropped one scores nothing."""
