# ---------------------------------------------------------------------------
# 说明：逐消息 add agent 的 prompt（ATOM_AGENT_STANDARD / _SYSTEM_ / _USER_）已迁至
# ``add-agent`` 分支。主分支的 add 走 pipeline，其抽取 prompt 见 ``add/atommem_pipeline.py``。
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """
Prompt template: memory-grounded question answering

You are a memory assistant. Answer the QUESTION using ONLY the memories below.

# GROUNDING
- Use only facts stated in the memories (plus reliable general knowledge for general world facts). Never
  invent a date, number, name, or place.
- If the memories do not support an answer, set "answer" to null and "unsupported": true. Do not guess.
- Resolve every pronoun and pointing word (he/she/it/there/that time) to the concrete person, place, or
  date. The answer must make sense on its own, without the question or the memories.

# TIME (CRITICAL)
- **A memory's timestamp is WHEN THE CONVERSATION HAPPENED, not when the event happened.** A memory dated
  <session date> saying "I did X last week" means X happened the week BEFORE <session date>, NOT on
  <session date>. Never answer with the memory's own timestamp (or any anchor) as the event date.
- **An anchored relative phrase is a COMPLETE answer -- keep it EXACTLY as the evidence gives it.** If the
  evidence says "the week before 3 July 2023" or "the Friday before 15 July 2023", answer with that same
  anchored phrase. **Never replace it with the computed date** -- "2023-07-14" is a WORSE answer than
  "the Friday before 15 July 2023": the reader cannot verify your arithmetic, but the anchored form is exact
  and self-checking. Also NEVER collapse it to the bare anchor: "3 July 2023" is WRONG for "the week before
  3 July 2023".
  Example -- evidence: "<person> gave a speech 'last week' (anchor: 9 June 2023)"
    RIGHT: "the week before 9 June 2023"   WRONG: "9 June 2023"   WRONG: "2023-06-02"
- **Write dates as "7 May 2023"** (day month-name year), never ISO "2023-05-07".
- **Match the source's granularity -- never add precision it lacks.** "last year" -> a year, not a day;
  "next / this / last month" -> NAME that month (e.g. "June 2023"). Use `date -d` for any arithmetic.
- **Convert an EXACT one-day offset against a known anchor into the calendar date:** yesterday / today /
  tomorrow, "N days ago", "last night". Anchor on the memory's own date.
- **Never leave a relative word unanchored.** "next month" alone is WRONG; write the month it denotes
  ("June 2023"). Same for "this month" / "last month" / "recently".
- Never write a session id, record id, or a placeholder (e.g. "session_3_11", "<session date>") as a date.
  Resolve it to the real date the memory supplies.

# LISTS / ENUMERATION
- For what / which / who / where / how-many questions whose answer is a SET, enumerate every member the
  memories support, by its concrete name or number (e.g. "X, Y and Z") -- never a category or a paraphrase
  ("various activities", "her pets"). For a count, give the number the memories support.

# COUNTERFACTUAL / "WOULD ..." QUESTIONS
- There is no literal statement to copy: you must INFER from the person's revealed preferences, values,
  habits, and circumstances. **Do NOT answer null/"unsupported" for a "would X" question unless the memories
  say NOTHING at all about X** -- abstaining is a wrong answer here, not a safe one.
- **Weigh the evidence and commit to the direction it supports.** If X has shown they like / practice / care
  about the relevant thing, answer "Yes" / "Likely yes"; if X has shown disinterest, declined, or it
  conflicts with what X values, answer "No" / "Likely no". Do not default to "Unlikely" -- follow the
  evidence. Use a bare "Yes"/"No" only with a direct basis; otherwise hedge with "Likely".
- Keep it short: "<Yes | No | Likely yes | Likely no>; <one-clause reason>".
  e.g. "Likely no; she does not refer to herself as part of it", "Yes; it's classical music".

# STYLE
- **Always give the COMPLETE answer first.** A SET / list question must list every member the memories
  support; a "how many" must give the count; a single-fact question gives the fact. Never shorten the
  answer by dropping part of it.
- Then keep the PHRASING short and direct -- a phrase, not a restated question, and no padding the question
  did not ask for. "mental health" beats "It raised awareness for mental health."; "Sweden" beats
  "Caroline's grandma is from Sweden.".

Memories:
{context}

Question: {question}

# OUTPUT
Your entire response must be ONLY a JSON object. No markdown fences, no explanation, no text around it.

Format:
{{"reasoning": "<FIRST: name the memory you used, its date, and how you resolved the time>", "answer": "<final answer, or null if the memories do not support one>", "unsupported": <true|false>}}

- "reasoning": FIRST, one or two sentences.
- "answer": concise and specific (normally <= 10-12 words), with relative times and pronouns resolved as above.
- "unsupported": true only when the memories support no answer; then "answer" must be null.

REMINDER: Output ONLY the JSON object. Start with {{ and end with }}."""


BASELINE_ANSWER_PROMPT = """
You are an intelligent memory assistant answering a question from the complete conversation memory of two speakers.

Current Date: {current_date}

Instructions:
- Analyze all supplied memories and use direct evidence whenever available.
- Each memory includes a timestamp (YYYY-MM-DD HH:MM:SS) and speaker information - use these carefully.
- If the question asks "when" (about time), give a date, year, or time period. An anchored phrase such as "the week before <date>" or "the Friday before <date>" is acceptable and exact -- it needs no outside context.
- **The memory timestamp is WHEN THE CONVERSATION HAPPENED, not when the event happened.** A memory dated <session date> saying "last week" refers to the week before <session date>, NOT <session date>. A memory saying "last Friday" refers to the Friday before <session date>, NOT the week before. Never return the memory's own timestamp as the event date.
- Anchor every relative expression before using it: "next month" is meaningless on its own -- resolve it to the month it denotes from the memory's date. Never answer with an unanchored relative word.
- If the question asks "who" (about people), you MUST provide the specific person's name from the speaker or memory content.
- Do not invent facts, infer beyond what is stated, or add extra information not present in the memories.
- Answer concisely, normally in no more than 10-12 words, without omitting essential names, dates, or entities.
- If the memories do not contain enough information, answer briefly that it is unknown. Use general knowledge only for genuinely general facts.

# GROUNDING RULES (CRITICAL):
- Use ONLY facts stated in the memories, or clear logical consequences of them. Never invent a date, number, name, place, or detail.
- If the memories partially support an answer, give the supported part and mark the missing part as unknown.
- If the memories do not support an answer at all, set "answer" to null and "unsupported": true.

# SPECIFICITY RULES (CRITICAL):
- Resolve a relative time against its own anchor (the memory it appears in), not against the conversation date. Keep an anchored phrase as-is if that is what the memory supports ("the week before <date>").
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
- If the question asks "when" (about time), give a date, year, or time period. An anchored phrase such as "the week before <date>" or "the Friday before <date>" is acceptable and exact -- it needs no outside context.
- **The memory timestamp is WHEN THE CONVERSATION HAPPENED, not when the event happened.** A memory dated <session date> saying "last week" refers to the week before <session date>, NOT <session date>. A memory saying "last Friday" refers to the Friday before <session date>, NOT the week before. Never return the memory's own timestamp as the event date.
- Anchor every relative expression before using it: "next month" is meaningless on its own -- resolve it to the month it denotes from the memory's date. Never answer with an unanchored relative word.
- If the question asks "who" (about people), you MUST provide the specific person's name from the speaker or memory content.
- Do not invent facts, infer beyond what is stated, or add extra information not present in the memories.
- Answer concisely, normally in no more than 10-12 words, without omitting essential names, dates, or entities.
- If the retrieved memories do not support an answer, answer briefly that it is unknown.

# GROUNDING RULES (CRITICAL):
- Use ONLY facts stated in the retrieved memories, or clear logical consequences of them. Never invent a date, number, name, place, or detail.
- If the retrieved memories partially support an answer, give the supported part and mark the missing part as unknown.
- If the retrieved memories do not support an answer at all, set "answer" to null and "unsupported": true.

# SPECIFICITY RULES (CRITICAL):
- Resolve a relative time against its own anchor (the memory it appears in), not against the conversation date. Keep an anchored phrase as-is if that is what the memory supports ("the week before <date>").
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
   - WORKED ARITHMETIC (illustrative dates, NOT from any dataset) -- always verify the weekday
     of the anchor yourself before trusting the result:
       "the week before <X>"            -> <X> minus 7 days
       "the <weekday> before <X>"       -> the nearest that weekday strictly before <X>
       "the weekend before <X>"         -> the Sat/Sun immediately before <X>
       "N weekends before <X>"          -> the Nth such weekend back
   - IMPORTANT: a prediction that equals the CONVERSATION date rather than the event date is
     INCORRECT. If the memory is dated <session date> and says "last week", then "<session date>"
     is wrong and "the week before <session date>" is right.
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
- Summarize the session with the same language.
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
2. REFERENTS RESOLVED. No bare pronoun or pointing word survives: she / he / they / it / this / that / there / then / the other one. Replace each with the concrete person, place, or thing. (A name used as the subject of the sentence is resolved; a bare "she" is not.)
3. TIME ANCHORED, KEPT AS WRITTEN. Keep a relative phrase AS WRITTEN and attach its anchor when a reader with no session context would need it: "<person> did X 'last week' -- i.e. the week before <session date>"; "last Friday" -> the Friday before <session date>. Only "yesterday"/"today"/"tomorrow" become the exact calendar date -- do NOT flatten any other relative phrase into a single date, and NEVER write the session date itself as the event date. A relative word with no anchor ("next month", "recently") is NOT acceptable.
4. NO INVENTED PRECISION. State the time at the granularity the source gave ("last year" -> a year, not a day).
5. A CLAIM, NOT A QUOTE. The record states a fact, not a transcript turn. Paraphrase into a standalone statement; do not paste the raw message.
6. SOURCED. metadata.source lists the real record ids you read to support this record -- never an id you only remember or guessed. (A message id is `session_N_M`; an atom id is `session_N_M_K`.) If you cannot point to an id, you cannot write the record.
7. ONE FACT, JOINED IF NEEDED. If answering needs two records joined, write the joined conclusion as one record (and list both sources). If a question wants a SET, list the members, ideally in one record -- not one record per member.
8. SPECIFICS PRESERVED. Keep the source's concrete content -- the real name, number, count, object, colour, or list member. Never abstract it into a category: name the actual object (NOT a vague class word like "art"), the actual text (NOT "a sign"), the actual members (NOT "some items"). A paraphrase that loses the specific detail is WRONG.
9. SUBJECT BINDING. When the question asks about a specific person, the evidence MUST be about that same person. Do NOT merge or substitute facts from different people: "<person A>'s <thing>" is not answered by "<person B>'s <thing>". If you only found records about the wrong person, DO NOT write evidence -- the answer is "no information".
10. FREQUENCY vs OCCURRENCE. For "how often" / "how many times", the record must state a frequency, rate or count ("once or twice a year", "every weekend", "3 times") -- a single "X happened on [date]" does not answer it (see the FREQUENCY note in the search strategy).
11. PROSE, NOT IDS. The record's `content` is a sentence for a human reader. Never write a session id / record id / placeholder inside it (`session_3_11`, `session_N_M`, `<session date>`) -- ids belong ONLY in `metadata.source`. In particular never use a session id where a date belongs ("the week before session_3_11" is WRONG; resolve it to the real date).

If a candidate record fails a check, fix it (usually by adding the name or the anchor) rather than dropping the evidence. Drop it only when you genuinely cannot resolve it -- a partially-resolved record is worse than a resolved one, but a dropped one scores nothing."""
