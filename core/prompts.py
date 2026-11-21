"""All LLM prompts used by the lecture summarization workflow."""

from __future__ import annotations


SUMMARY_SYSTEM_PROMPT = """You are a precise lecture summarization assistant.
Use ONLY the supplied lecture context. Never add outside facts. When the context
does not support a claim, write 'not in lecture'. Every point must include the
source reference supplied in the context. Return JSON only, with no Markdown
fences or commentary."""


def chunk_summary_prompt(context: str) -> str:
    return f"""Summarize these lecture excerpts into JSON with this exact shape:
{{"summary": "one concise paragraph in plain words", "points": [{{"text": "claim in plain words", "source_reference": {{"timestamp": "00:00", "slide": null}}}}]}}
Use 2-5 points. Replace 00:00 with an actual timestamp copied from the lecture context. Never output the literal placeholder 00:00, MM:SS, or any made-up timestamp. Write the summary as readable sentences, not JSON text inside the claim.

LECTURE CONTEXT:
{context}"""


def section_summary_prompt(context: str) -> str:
    return f"""Combine these chunk summaries into one coherent section JSON:
{{"title": "short section title", "summary": "concise section summary in plain words", "points": [{{"text": "claim in plain words", "source_reference": {{"timestamp": "00:00", "slide": null}}}}]}}
Use only the supplied summaries and retain their actual source references. Never output the literal placeholder 00:00 or MM:SS.

CHUNK SUMMARIES:
{context}"""


def final_summary_prompt(context: str) -> str:
    return f"""Create the final lecture summary as JSON with exactly this shape:
{{"tldr": [{{"text": "line in plain words", "source_reference": {{"timestamp": "00:00", "slide": null}}}}], "key_points": [{{"text": "point in plain words", "source_reference": {{"timestamp": "00:00", "slide": null}}}}], "detailed_notes": [{{"text": "note in plain words", "source_reference": {{"timestamp": "00:00", "slide": null}}}}]}}
The tldr must contain exactly 3 lines. Make key_points and detailed_notes complete but concise, using readable sentences.
The detailed_notes must be a comprehensive list of points only (bullet points).
Do not put timestamps or bracketed numbers in the text field; provide pure summary text in plain words.
Every item must cite an actual supplied timestamp and optional slide. Never output 00:00, MM:SS, or a made-up citation. Use only the section summaries.

SECTION SUMMARIES:
{context}"""


def retry_json_prompt(original_prompt: str, invalid_response: str, error: str) -> str:
    short_invalid = invalid_response[:250] if len(invalid_response) > 250 else invalid_response
    return f"""{original_prompt}

The previous response failed validation:
{short_invalid}
Error: {error}
You MUST return valid JSON starting with {{"questions": [...]}}. Do NOT return chunk metadata or other keys."""


QUIZ_SYSTEM_PROMPT = """You create lecture-grounded assessment questions.
Use ONLY the supplied lecture context. Do not use outside knowledge. Distractors
must be plausible but contradicted by or absent from the lecture. Every question
must cite a supplied source reference. Return JSON only, without Markdown fences."""


def quiz_generation_prompt(
    context: str,
    count: int,
    difficulty: str,
    bloom_level: str,
    weak_concepts: list[str],
) -> str:
    weak_focus = ", ".join(weak_concepts) if weak_concepts else "none recorded"
    return f"""LECTURE CONTEXT:
{context}

---
TASK:
Based on the LECTURE CONTEXT above, generate exactly {count} lecture-grounded questions as JSON.
You MUST output a valid JSON object whose ONLY top-level key is "questions".

Target difficulty: {difficulty}
Target bloom_level: {bloom_level}
Previously missed concepts to emphasize if present in context: {weak_focus}

JSON Schema:
{{
  "questions": [
    {{
      "question": "question text",
      "type": "mcq",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option A",
      "bloom_level": "{bloom_level}",
      "difficulty": "{difficulty}",
      "explanation": "brief explanation based on the lecture",
      "source_reference": {{"timestamp": "00:00", "slide": null}},
      "concept": "concept name"
    }}
  ]
}}

CRITICAL RULES:
1. Root object must have key "questions". Do NOT output any other keys like "version" or "chunks".
2. Question types: "mcq" (4 options), "true_false" (options: ["True", "False"]), "fill_blank" (options: []), "open_ended" (options: []).
3. Return JSON only, starting with {{"questions":"""


def quiz_verification_prompt(questions_json: str) -> str:
    return f"""Verify each MCQ and true/false question below against the lecture context already embedded in its source reference. Return JSON only:
{{"checks": [{{"question_index": 0, "is_valid": true, "correct_option": "exact option text", "reason": "brief reason"}}]}}
There must be exactly one correct option for every checked question. Mark is_valid false if there are zero or multiple correct options. Check all questions, using zero-based indexes.

QUESTIONS:
{questions_json}"""


QA_SYSTEM_PROMPT = """You answer questions about a lecture using only the supplied retrieved context.
If the answer is not supported, answer exactly 'not in lecture', set grounded to false,
and return an empty citations list. Otherwise, answer concisely and cite one or more
source references copied exactly from the context. Return JSON only."""


def qa_prompt(question: str, context: str) -> str:
    return f"""RETRIEVED LECTURE CONTEXT:
{context}

---
QUESTION:
{question}

TASK:
Answer the question using ONLY the retrieved lecture context above.
If the answer is not supported, answer exactly 'not in lecture', set grounded to false, and set citations to [].
Otherwise, provide a concise grounded answer, set grounded to true, and cite the matching timestamp(s) from the context.

JSON Format:
{{"answer": "concise answer in plain words", "citations": [{{"timestamp": "00:00", "slide": null}}], "grounded": true}}
Return JSON only:"""


CONCEPT_SYSTEM_PROMPT = """You extract concepts from lecture context only.
Do not add outside facts. Definitions must be grounded in the supplied text.
Return JSON only, with source references copied from the context."""


def concept_extraction_prompt(context: str) -> str:
    return f"""Extract the important concepts from this lecture context as JSON:
{{"concepts": [{{"name": "short concept name", "definition": "grounded definition in plain words", "first_mention_timestamp": "00:00", "slide_no": null, "related_concepts": ["other extracted concept"]}}]}}
Include 5-30 concepts. Use only concepts supported by the context and preserve actual source timestamps/slides. Replace 00:00 with the timestamp in the SOURCE line where the concept is first mentioned. Never output the literal 00:00 or MM:SS. Definitions must be readable sentences, not template text.

LECTURE CONTEXT:
{context}"""
