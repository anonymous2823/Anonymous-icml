import json
from typing import Any, Dict, Tuple

from litellm import completion

from config import AppConfig
from utils.claim_utils import parse_json_block


SYSTEM_PROMPT = """You are a careful scientific reader.

DEFINITION
A “scientific claim” is a concise, declarative set of statements of a finding, result, or
methodological fact asserted by the authors. A valid scientific claim:
- can stand alone as a set of factual statements,
- is specific and empirically or theoretically verifiable,
- reflects the scope and conditions of the work when needed,
- does NOT refer to the paper, authors, or presentation (e.g., no “this paper”, “we”, “the authors”).

TASK:
Read the provided paper text and GENERATE (do not quote) the paper’s core scientific claims.

OUTPUT RULES
- Write claims as factual statements, NOT as descriptions of what the paper does.
- Do NOT use phrases such as:
  “the paper proposes”, “this work shows”, “the authors introduce”, “we demonstrate”.
- Don't mention the numerical results of the paper as claim statement, instead focus on the conceptual findings, methods, or insights.
- Each claim must be EXACTLY ONE 'crisp' consolidated sentence covering all aspects of the claim.
- Return ONLY valid JSON in the following format:
  {
    "claims": [
      { "text": "<one claim>" },
      { "text": "<one claim>" }
    ]
  }
- Try to keep only one consolidated claim.
- If more claims are there in the paper: return maximum 5 claims.
- If no clear scientific claim is present, return:
  { "claims": [] }

END OF INSTRUCTIONS. Produce only the lines of claims as specified.
"""


# QA_SYSTEM_PROMPT = """
# You are an expert exam-item writer.

# DEFINITION
# A “scientific claim” is a concise, declarative statement that stands as a fact-like proposition asserted by the paper, is specific and verifiable (empirically or theoretically), and includes necessary scope/conditions (task, dataset, domain, population) without marketing language or citations.

# TASK
# You will receive ONE scientific claim (a single sentence). Create assessment items that test the ENTIRE content of that claim (not just a fragment).

# OUTPUT FORMAT (JSON ONLY)
# Return ONLY valid JSON in exactly this schema (no extra commentary, no code fences):
# {
#   "mcq": [
#     { "question": "Choose the correct option (A–D): <stem covering the full claim> [A] <option A> [B] <option B> [C] <option C> [D] <option D>", "answer": "A" },
#     { "question": "Choose the correct option (A–D): <stem covering the full claim> [A] <option A> [B] <option B> [C] <option C> [D] <option D>", "answer": "C" }
#   ],
#   "true_false": [
#     { "question": "<single True/False statement that covers the full claim>", "answer": "True" },
#     { "question": "<single True/False statement that covers the full claim>", "answer": "False" }
#   ],
#   "fill_blank": [
#     { "question": "<rephrase the full claim with exactly one blank: … ______ …>", "answer": "<one word or one short line that correctly fills the blank>" },
#     { "question": "<rephrase the full claim with exactly one blank: … ______ …>", "answer": "<one word or one short line that correctly fills the blank>" }
#   ],
#   "assertion_reason": [
#     { "question": "Assertion (A): <statement covering the full claim>. Reason (R): <single reason statement>", "answer": "A is True, R is True, and R explains A" },
#     { "question": "Assertion (A): <statement covering the full claim>. Reason (R): <single reason statement>", "answer": "A is True, R is False" }
#   ]
# }

# STRICT RULES
# - Coverage: Every item MUST assess the entire claim; do not focus on a partial detail.
# - Quantity: Exactly 2 items for each category (mcq, true_false, fill_blank, assertion_reason).
# - Keys: Each item must contain ONLY two keys — "question" and "answer".
# - MCQ: Embed exactly four options inside the question text labeled [A], [B], [C], [D]; the "answer" must be a single letter "A"/"B"/"C"/"D".
# - True/False: "answer" must be exactly "True" or "False".
# - Fill-in-the-blank: The question must contain exactly one blank represented by "______"; the "answer" must be one word or a single short line.
# - Assertion–Reason: Provide exactly one assertion and one reason in the "question" string; the "answer" must be one of:
#   "A is True, R is True, and R explains A" | "A is True, R is True, but R does not explain A" | "A is True, R is False" | "A is False, R is True" | "A is False, R is False".
# - No double-barrel items: Do not ask two things in one item; no multiple blanks; no multi-sentence questions.
# - Tone: Use precise, neutral academic language. Do not introduce information beyond the claim.
# - Output: Return ONLY the JSON object exactly as specified, with no surrounding text.

# CRITICAL CONSTRAINTS

# - Treat the claim as an atomic unit; do NOT refer to any method, approach, model, author, or paper.
# - Do NOT introduce phrases such as “the paper states,” “the authors propose,” “this method,” or similar constructs.
# - Each generated question MUST be fully self-contained and independently understandable without seeing the original claim.
# - Rewrite or restate the claim within each question so that it can be answered in isolation.
# - Do NOT assume the test-taker has access to the original claim.
# - Each question must directly assess the full claim itself, not its origin or context.

# """


QA_SYSTEM_PROMPT = """
You are an expert exam-item writer.

DEFINITION
A “scientific claim” is a concise, declarative statement that stands as a fact-like proposition asserted by the paper, is specific and verifiable (empirically or theoretically), and includes necessary scope/conditions (task, dataset, domain, population) without marketing language or citations.

TASK
You will receive ONE scientific claim (a single sentence). Create assessment items that test the ENTIRE content of that claim (not just a fragment).

OUTPUT FORMAT (JSON ONLY)
Return ONLY valid JSON in exactly this schema (no extra commentary, no code fences):
{
  "mcq": [
    { "question": "Choose the correct option (A–D): <stem covering the full claim> [A] <option A> [B] <option B> [C] <option C> [D] <option D>", "answer": "A" },
    { "question": "Choose the correct option (A–D): <stem covering the full claim> [A] <option A> [B] <option B> [C] <option C> [D] <option D>", "answer": "C" }
  ],
  "true_false": [
    { "question": "<single True/False statement that covers the full claim>", "answer": "True" },
    { "question": "<single True/False statement that covers the full claim>", "answer": "False" }
  ],
  "fill_blank": [
    { "question": "<rephrase the full claim with exactly one blank and exactly two inline options in this format: _________(correct_option/wrong_option)>", "answer": "<the correct option exactly as written inside the parentheses>" },
    { "question": "<rephrase the full claim with exactly one blank and exactly two inline options in this format: _________(correct_option/wrong_option)>", "answer": "<the correct option exactly as written inside the parentheses>" }
  ],
  "assertion_reason": [
    { "question": "Assertion (A): <statement covering the full claim>. Reason (R): <single reason statement>", "answer": "A is True, R is True, and R explains A" },
    { "question": "Assertion (A): <statement covering the full claim>. Reason (R): <single reason statement>", "answer": "A is True, R is False" }
  ]
}

STRICT RULES
- Coverage: Every item MUST assess the entire claim; do not focus on a partial detail.
- Quantity: Exactly 2 items for each category (mcq, true_false, fill_blank, assertion_reason).
- Keys: Each item must contain ONLY two keys — "question" and "answer".
- MCQ: Embed exactly four options inside the question text labeled [A], [B], [C], [D]; the "answer" must be a single letter "A"/"B"/"C"/"D".
- True/False: "answer" must be exactly "True" or "False".
- Fill-in-the-blank:
  - The question must contain exactly one blank represented by "_________". Immediately after the blank, include exactly two answer choices inline in parentheses using this exact format:
    _________(option1/option2)
  - One option must be correct and the other must be a plausible but incorrect distractor.
  - The "answer" must be exactly the correct option as written inside the parentheses.
  - Do NOT use more than two options.
  - Do NOT use multiple blanks.
  - Do not miss the inline options or use a different format for them.
- Assertion–Reason: Provide exactly one assertion and one reason in the "question" string; the "answer" must be one of:
  "A is True, R is True, and R explains A" | "A is True, R is True, but R does not explain A" | "A is True, R is False" | "A is False, R is True" | "A is False, R is False".
- No double-barrel items: Do not ask two things in one item; no multiple blanks; no multi-sentence questions.
- Tone: Use precise, neutral academic language. Do not introduce information beyond the claim.
- Output: Return ONLY the JSON object exactly as specified, with no surrounding text.

CRITICAL CONSTRAINTS
- Treat the claim as an atomic unit; do NOT refer to any method, approach, model, author, or paper.
- Do NOT introduce phrases such as “the paper states,” “the authors propose,” “this method,” or similar constructs.
- Each generated question MUST be fully self-contained and independently understandable without seeing the original claim.
- Rewrite or restate the claim within each question so that it can be answered in isolation.
- Do NOT assume the test-taker has access to the original claim.
- Each question must directly assess the full claim itself, not its origin or context.

EXAMPLE FOR FILL_BLANK FORMAT
- "Traditional RETE-based pattern matching is _________(insufficient/abstract) for frequently updated working memories."
- "The framework translates OWL 2 RL ontologies into _________(Horn-clause/graphical) rules."
"""

VERBATIM_CLAIM_SYSTEM_PROMPT = """
You are a careful scientific reader.

TASK
You will be given:
- paper title
- full extracted paper text

Your task is to identify verbatim scientific claims that appear specifically in:
- the abstract
- the introduction
- the conclusion

A “scientific claim” is a concise, declarative statement of a finding, result, or
methodological fact asserted by the authors. A valid scientific claim:
- can stand alone as a factual statement,
- is specific and empirically or theoretically checkable,
- reflects the scope and conditions of the work when needed
- Does not cover the numerical results of the paper, instead focus on the conceptual findings, methods, or insights.

IMPORTANT
- "Verbatim" means the returned claim text must be copied exactly as it appears in the provided paper text.
- Do NOT paraphrase.
- Do NOT rewrite.
- Do NOT merge multiple sentences.
- Only extract claims that truly come from abstract, introduction, or conclusion.
- You must infer the source section from the paper text itself.
- If you are uncertain about the section, do not include the claim.
- Only extract strong scientific claims that describe:
  - a proposed method, model, or system
  - a key capability or function
  - a demonstrated outcome or performance insight
- Do NOT extract:
  - background information
  - scenario descriptions
  - assumptions (e.g., "it is assumed that...")
  - evaluation setup descriptions
- Prefer claims that reflect the main contribution or novelty of the paper.
- Avoid extracting minor or peripheral statements.
- The claim must be meaningful on its own without requiring surrounding sentences.
- Avoid sentences that depend heavily on prior context.
- Only include claims that describe what the system/method achieves, not just what it was tested on or analyzed.
- Exclude statements that only report observational or post-hoc analysis, unless tied to a key capability or guarantee.

- Do NOT extract:
  - definitions of terms, models, or components (e.g., "X is defined as...")
  - descriptions of internal structure (e.g., "consists of three components")

- Only extract claims that describe at least one of the following:
  - a novel system, method, or model
  - a key capability (what the system can do)
  - a performance advantage or outcome
- Prefer claims that directly support the main contribution of the paper.

OUTPUT FORMAT
Return ONLY valid JSON in exactly this schema:
{
  "verbatim_claims": [
    {
      "text": "<exact copied text>",
      "source_section": "abstract"
    },
    {
      "text": "<exact copied text>",
      "source_section": "introduction"
    },
    {
      "text": "<exact copied text>",
      "source_section": "conclusion"
    }
  ]
}

RULES
- source_section must be one of: "abstract", "introduction", "conclusion"
- The text must be copied exactly from the provided paper text.
- Prefer concise sentences or clauses that clearly express a finding, contribution, capability, or methodological statement.
- You may return multiple items from the same section.
- If no suitable claim is found, return:
  { "verbatim_claims": [] }
- Return JSON only.
""".strip()

INTERNAL_RETAIN_SYSTEM_PROMPT = """
You are a careful scientific reader and exam-item writer.

TASK
Read the provided paper text and produce:

1) a compact "paper_base" that captures the core foundational identity of the paper
2) paper-level assessment items in the SAME structured categories:
   - mcq
   - true_false
   - fill_blank
   - assertion_reason

VERY IMPORTANT
You will also be given a list of scientific claims already extracted from this paper.

Those claims are PROVIDED ONLY AS CONTEXT so that you can AVOID generating questions that directly restate, verify, or paraphrase those claims.

Your job is to generate BASIC / FOUNDATIONAL / CONCEPT-LEVEL paper questions that are APART FROM the given claims.

GOOD QUESTION TYPES
These questions should test:
- what the main concept / model / system is
- what the method fundamentally is
- what task/problem it addresses
- what kind of method family it belongs to
- what broad input-output behavior it models
- what kind of setup / framework the paper is built around
- what broad role the central concept plays in the paper
- But all the questions must be strictly atomic and standalone, does not mention the paper or authors (when asking the question don't consider that there is a paper context, the question must be standalone without referring to any paper or author) and does not require understanding of specific claims.

EXAMPLES
If a paper is about InstructPix2Pix, acceptable questions include:
- What is InstructPix2Pix?
- InstructPix2Pix is mainly used for what type of task?
- InstructPix2Pix belongs to what broad model/method family?
- InstructPix2Pix takes what kind of input and produces what kind of output?

These are GOOD because they test foundational understanding of the paper,
not the narrow scientific claims/results.

DO NOT GENERATE
- questions that directly test the provided claims
- result-verification questions
- highly specific numerical findings
- claim paraphrases
- author/presentation questions
- citation questions
- trivial metadata-only questions
- ask what the paper says about a concept (instead, ask what the concept itself is)

OUTPUT FORMAT
Return ONLY valid JSON in exactly this schema:
{
  "paper_base": {
    "topic": "<broad topic>",
    "paper_type": "<paper type>",
    "task_or_problem": "<main problem or task>",
    "method_family": "<broad method family>",
    "data_or_domain": "<domain / data / setting>",
    "core_concept": "<the main concept / method / system in the paper>"
  },
  "qa_by_base": {
    "mcq": [
      { "question": "Choose the correct option (A–D): ... [A] ... [B] ... [C] ... [D] ...", "answer": "A" },
      { "question": "Choose the correct option (A–D): ... [A] ... [B] ... [C] ... [D] ...", "answer": "C" }
    ],
    "true_false": [
      { "question": "...", "answer": "True" },
      { "question": "...", "answer": "False" }
    ],
    "fill_blank": [
      { "question": "... ______ ...", "answer": "..." },
      { "question": "... ______ ...", "answer": "..." }
    ],
    "assertion_reason": [
      { "question": "Assertion (A): ... Reason (R): ...", "answer": "A is True, R is True, and R explains A" },
      { "question": "Assertion (A): ... Reason (R): ...", "answer": "A is True, R is False" }
    ]
  }
}

STRICT RULES
- Questions must be paper-level and concept-level.
- Questions must be foundational/basic, not claim-level.
- Use the provided claims only to AVOID overlap with claim questions.
- Generate exactly 2 questions for each category.
- MCQ questions must embed exactly four options labeled [A], [B], [C], [D].
- True/False answers must be exactly "True" or "False".
- Fill-blank questions must contain exactly one blank represented by "______".
- Assertion–Reason answers must use the standard allowed labels.
- Return only JSON.


CRITICAL CONSTRAINTS

- Treat the questions as an atomic unit; do NOT refer to any method, approach, model, author, or paper.
- Do NOT introduce phrases such as “the paper states,” “the authors propose,” “this method,” or similar constructs.
- Each generated question MUST be fully self-contained and independently understandable without seeing the original claim.

""".strip()


DERIVED_QUESTION_SYSTEM_PROMPT = """
You are an expert question transformation engine.

TASK
You will be given:
1) a pool of atomic source questions
2) their answers
3) optionally the source claims from which the questions came

Your job is to generate DERIVED QUESTIONS.

DEFINITION OF A DERIVED QUESTION
A derived question is:
- grounded in the same atomic knowledge as the source question pool,
- but asked from a more conceptually distant angle.
- and NOT a paraphrase of any original question.

IMPORTANT
The derived question must remain ATOMIC.
Do NOT broaden it into paper-level or document-level discussion.

STRICT RULES
- Do NOT use words such as:
  "paper", "work", "authors", "study", "this paper", "the paper", "the authors", "this work".
- Do NOT paraphrase any source question.
- Do NOT restate the same answer in slightly different wording.
- Do NOT ask the exact same fact in another surface form.
- Do NOT introduce unrelated information.

WHAT TO DO INSTEAD
Generate questions that target a conceptually more distant but still valid aspect of the same atomic meaning, such as:
- a property
- a category
- a role
- a constraint
- a mechanism
- an implication
- an abstraction level above the original wording
- a representation shift

OUTPUT TASK
Generate exactly 12 derived questions total:
- 3 mcq
- 3 true_false
- 3 fill_blank
- 3 assertion_reason

OUTPUT FORMAT
Return ONLY valid JSON in exactly this schema:
{
  "derived_qa": {
    "mcq": [
      { "question": "Choose the correct option (A–D): ... [A] ... [B] ... [C] ... [D] ...", "answer": "A" }
    ],
    "true_false": [
      { "question": "...", "answer": "True" }
    ],
    "fill_blank": [
      { "question": "... ______ ...", "answer": "..." }
    ],
    "assertion_reason": [
      { "question": "Assertion (A): ... Reason (R): ...", "answer": "A is True, R is True, and R explains A" }
    ]
  }
}

FORM RULES
- MCQ must have exactly 4 options labeled [A], [B], [C], [D].
- True/False answer must be exactly "True" or "False".
- Fill blank must contain exactly one blank: "______".
- Assertion–Reason answer must be one of:
  - "A is True, R is True, and R explains A"
  - "A is True, R is True, but R does not explain A"
  - "A is True, R is False"
  - "A is False, R is True"
  - "A is False, R is False"

CRITICAL CONSTRAINTS

- Treat the questions as an atomic unit; do NOT refer to any method, approach, model, author, or paper.
- Do NOT introduce phrases such as “the paper states,” “the authors propose,” “this method,” or similar constructs.
- Each generated question MUST be fully self-contained and independently understandable without seeing the original claim.

DISTANCE RULE
Each derived question must be at least one conceptual step away from the source questions.
It must target the same atomic knowledge, but through a more abstract, property-based, role-based, or implication-based angle.

Return JSON only.
""".strip()



CS_PAPER_FILTER_SYSTEM_PROMPT = """
You are a precise academic-domain classifier.

TASK
You will be given a paper title and abstract.
Your job is to determine whether the paper should be classified as a Computer Science paper.

IMPORTANT
Use only the semantic content of the title and abstract.
Do NOT rely on venue names, citation style, or external assumptions.

Return ONLY valid JSON in exactly this schema:
{
  "is_computer_science": true,
  "confidence": "high",
  "reason": "short explanation"
}

RULES
- "is_computer_science" must be either true or false.
- "confidence" must be one of: "high", "medium", "low".
- "reason" must be short and concise.
- If the paper is primarily in another domain (e.g. biology, chemistry, physics, medicine, economics) but uses computational methods, return true.
- If the paper is clearly about algorithms, machine learning, AI, image processing, computer vision, computer graphics,
  NLP, robotics, systems, databases, programming languages, software engineering,
  security, HCI, graphics, theory, networks, or other core CS areas, return true.

Return JSON only.
""".strip()



PAPER_TYPE_FILTER_SYSTEM_PROMPT = """
You are a precise academic paper classifier.

TASK
You will be given a paper title and abstract.
Choose EXACTLY ONE paper type from the allowed set.

ALLOWED PAPER TYPES
- experimental_original_research
- review_survey
- benchmark_dataset
- theoretical_methodological
- position_opinion_editorial
- case_report_clinical
- other

DEFINITIONS
- experimental_original_research:
  Original research that includes empirical evaluation, experiments, measurements,
  comparative results, or tested hypotheses/methods on data/systems/tasks.
- review_survey:
  A survey, review, or overview paper summarizing prior work.
- benchmark_dataset:
  A paper mainly introducing a dataset, benchmark, shared task, or evaluation resource.
- theoretical_methodological:
  A mainly theoretical, formal, or methodological paper with limited or no empirical experimentation.
- position_opinion_editorial:
  Commentary, perspective, editorial, manifesto, or non-research opinion piece.
- case_report_clinical:
  Clinical case reports or case-study style medical papers centered on individual cases.
- other:
  Anything that does not clearly fit the above.

IMPORTANT
- Use only the semantic content of title and abstract.
- Choose exactly one label.
- If uncertain, choose the most conservative label.
- Do not output multiple labels.

OUTPUT FORMAT
Return ONLY valid JSON in exactly this schema:
{
  "paper_type": "experimental_original_research",
  "confidence": "high",
  "reason": "short explanation"
}

RULES
- "paper_type" must be exactly one of the allowed labels.
- "confidence" must be one of: "high", "medium", "low".
- "reason" must be short and concise.
- Return JSON only.
""".strip()



def ensure_cost_log_dir(config: AppConfig) -> None:
    config.cost_log_dir.mkdir(parents=True, exist_ok=True)


def cost_from_usage(usage: dict, config: AppConfig) -> Tuple[float, float, float]:
    prompt_toks = usage.get("prompt_tokens", 0) or 0
    completion_toks = usage.get("completion_tokens", 0) or 0

    prompt_cost = (prompt_toks / 1000.0) * config.price_per_1k_prompt
    completion_cost = (completion_toks / 1000.0) * config.price_per_1k_completion
    total_cost = prompt_cost + completion_cost
    return prompt_cost, completion_cost, total_cost


def append_cost_log(entry: Dict[str, Any], config: AppConfig) -> None:
    ensure_cost_log_dir(config)
    with open(config.cost_log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def call_llm_json(
    messages: list[dict],
    config: AppConfig,
    task_type: str = "generic_json",
    model_name: str | None = None,
) -> str:
    """
    Generic Azure/LiteLLM JSON call helper.
    Reuses configured Azure model and logs usage cost.
    Returns raw text content from the model.
    """
    resp = completion(
        model=model_name or config.llm_model_name,
        messages=messages,
        temperature=config.llm_temperature,
    )

    usage = getattr(resp, "usage", {}) or {}
    prompt_cost, completion_cost, total_cost = cost_from_usage(usage, config)

    append_cost_log(
        {
            "type": task_type,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
        },
        config,
    )

    return resp.choices[0].message.content.strip()


def call_llm_single_pass(text: str, corpus_id: int, config: AppConfig) -> str:
    if len(text) > config.max_chars_for_model:
        text = text[: config.max_chars_for_model]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Paper text:\n\n{text}"},
    ]

    resp = completion(
        model=config.llm_model_name,
        messages=messages,
        temperature=config.llm_temperature,
    )

    usage = getattr(resp, "usage", {}) or {}
    prompt_cost, completion_cost, total_cost = cost_from_usage(usage, config)

    append_cost_log(
        {
            "type": "claim_extraction",
            "corpus_id": corpus_id,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
        },
        config,
    )

    out = resp.choices[0].message.content.strip()
    return " ".join(out.split()).strip()


def call_llm_questions(claim: str, config: AppConfig) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": f"Claim:\n{claim}"},
    ]

    resp = completion(
        model=config.llm_model_name,
        messages=messages,
        temperature=config.llm_temperature,
    )

    usage = getattr(resp, "usage", {}) or {}
    prompt_cost, completion_cost, total_cost = cost_from_usage(usage, config)

    append_cost_log(
        {
            "type": "qa_generation",
            "claim": claim,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
        },
        config,
    )

    content = resp.choices[0].message.content
    return parse_json_block(content)


def call_llm_internal_retain_questions(
    text: str,
    paper_title: str,
    paper_claims: list[str],
    config: AppConfig,
) -> Dict[str, Any]:
    """
    Generate:
    - paper_base
    - concept-level / foundational structured QA
      (mcq, true_false, fill_blank, assertion_reason)

    The forget-side claims are included only to prevent overlap with claim-based questions.
    """
    if len(text) > config.max_chars_for_model:
        text = text[: config.max_chars_for_model]

    claims_block = "\n".join(f"- {c}" for c in paper_claims if isinstance(c, str) and c.strip())
    if not claims_block:
        claims_block = "(no prior claims available)"

    messages = [
        {"role": "system", "content": INTERNAL_RETAIN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Paper title:\n{paper_title}\n\n"
                f"Previously extracted claims from this paper:\n{claims_block}\n\n"
                f"Paper text:\n\n{text}"
            ),
        },
    ]

    resp = completion(
        model=config.llm_model_name,
        messages=messages,
        temperature=config.llm_temperature,
    )

    usage = getattr(resp, "usage", {}) or {}
    prompt_cost, completion_cost, total_cost = cost_from_usage(usage, config)

    append_cost_log(
        {
            "type": "retain_internal_generation",
            "paper_title": paper_title,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
        },
        config,
    )

    content = resp.choices[0].message.content
    return parse_json_block(content)


def call_llm_verbatim_claims(
    paper_title: str,
    full_text: str,
    config: AppConfig,
) -> Dict[str, Any]:
    """
    Extract verbatim claims from abstract/introduction/conclusion
    directly from the full paper text (no regex section extraction).
    """
    if len(full_text) > config.max_chars_for_model:
        full_text = full_text[: config.max_chars_for_model]

    user_content = (
        f"Paper title:\n{paper_title}\n\n"
        f"Full extracted paper text:\n\n{full_text}"
    )

    messages = [
        {"role": "system", "content": VERBATIM_CLAIM_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    resp = completion(
        model=config.llm_model_name,
        messages=messages,
        temperature=1.0,  # important for verbatim-style extraction
    )

    usage = getattr(resp, "usage", {}) or {}
    prompt_cost, completion_cost, total_cost = cost_from_usage(usage, config)

    append_cost_log(
        {
            "type": "verbatim_claim_extraction",
            "paper_title": paper_title,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
        },
        config,
    )

    content = resp.choices[0].message.content
    return parse_json_block(content)


def call_llm_derived_questions(
    source_questions: list[dict],
    source_claims: list[str],
    config: AppConfig,
) -> Dict[str, Any]:
    """
    Generate exactly 4 derived questions total from a pool of up to 8 source atomic questions:
    - 1 mcq
    - 1 true_false
    - 1 fill_blank
    - 1 assertion_reason
    """
    claims_block = "\n".join(
        f"- {c}" for c in source_claims if isinstance(c, str) and c.strip()
    )
    if not claims_block:
        claims_block = "(no source claims provided)"

    src_lines = []
    for idx, item in enumerate(source_questions, start=1):
        src_lines.append(
            f"[{idx}] type={item.get('source_type')} | question={item.get('question')} | answer={item.get('answer')}"
        )
    source_block = "\n".join(src_lines)

    messages = [
        {"role": "system", "content": DERIVED_QUESTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Source claims:\n{claims_block}\n\n"
                f"Source atomic questions:\n{source_block}\n\n"
                f"Generate 12 derived questions according to the rules."
            ),
        },
    ]

    resp = completion(
        model=config.llm_model_name,
        messages=messages,
        temperature=config.llm_temperature,
    )


    usage = getattr(resp, "usage", {}) or {}
    prompt_cost, completion_cost, total_cost = cost_from_usage(usage, config)

    append_cost_log(
        {
            "type": "derived_question_generation",
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
        },
        config,
    )

    content = resp.choices[0].message.content
    return parse_json_block(content)



def call_llm_is_computer_science_paper(
    title: str,
    abstract: str,
    config: AppConfig,
) -> Dict[str, Any]:
    """
    Classify whether a paper is Computer Science using title + abstract.
    """
    title = (title or "").strip()
    abstract = (abstract or "").strip()

    user_content = (
        f"Title:\n{title or '(missing)'}\n\n"
        f"Abstract:\n{abstract or '(missing)'}"
    )

    messages = [
        {"role": "system", "content": CS_PAPER_FILTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    resp = completion(
        model=config.llm_model_name,
        messages=messages,
        temperature=1.0,
    )

    usage = getattr(resp, "usage", {}) or {}
    prompt_cost, completion_cost, total_cost = cost_from_usage(usage, config)

    append_cost_log(
        {
            "type": "cs_paper_filter",
            "title": title,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
        },
        config,
    )

    content = resp.choices[0].message.content
    return parse_json_block(content)


def call_llm_paper_type(
    title: str,
    abstract: str,
    config: AppConfig,
) -> Dict[str, Any]:
    """
    Classify paper type from title + abstract.
    """
    title = (title or "").strip()
    abstract = (abstract or "").strip()

    user_content = (
        f"Title:\n{title or '(missing)'}\n\n"
        f"Abstract:\n{abstract or '(missing)'}"
    )

    messages = [
        {"role": "system", "content": PAPER_TYPE_FILTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    resp = completion(
        model=config.llm_model_name,
        messages=messages,
        temperature=1.0,
    )

    usage = getattr(resp, "usage", {}) or {}
    prompt_cost, completion_cost, total_cost = cost_from_usage(usage, config)

    append_cost_log(
        {
            "type": "paper_type_filter",
            "title": title,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
        },
        config,
    )

    content = resp.choices[0].message.content
    return parse_json_block(content)