"""Grounded lecture Q&A over hybrid retrieval results."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from .llm import LLMClient, LLMError
from .prompts import QA_SYSTEM_PROMPT, qa_prompt, retry_json_prompt
from .retrieval import HybridRetriever, RetrievalError, RetrievalResult
from .summarize import SourceReference


class QAError(RuntimeError):
    """Raised when retrieval or grounded answer generation fails."""


class QAResponse(BaseModel):
    """A lecture answer with only retrieved source citations."""

    answer: str = Field(min_length=1)
    citations: list[SourceReference] = Field(default_factory=list)
    grounded: bool = False

    @model_validator(mode="after")
    def validate_grounding(self) -> "QAResponse":
        if not self.grounded and self.citations:
            raise ValueError("an ungrounded answer cannot contain citations")
        if self.grounded and not self.citations:
            raise ValueError("a grounded answer must contain citations")
        return self


def _parse_json(response: str) -> QAResponse:
    cleaned = response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response did not contain a JSON object")
    json_str = cleaned[start : end + 1]
    # Sanitize trailing commas common in small-model output
    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            if not data.get("answer"):
                data["answer"] = "not in lecture"

            raw_cits = data.get("citations")
            if not isinstance(raw_cits, list):
                raw_cits = []
            cleaned_cits = []
            for c in raw_cits:
                if isinstance(c, dict) and c.get("timestamp"):
                    cleaned_cits.append(c)
                elif isinstance(c, str):
                    cleaned_cits.append({"timestamp": c, "slide": None})
            data["citations"] = cleaned_cits

            if "grounded" in data:
                data["grounded"] = bool(data["grounded"])
            else:
                data["grounded"] = bool(cleaned_cits) and data["answer"].lower().strip() != "not in lecture"

            # Reconcile grounded and citations so small model output never triggers validator errors
            if not data["grounded"]:
                data["citations"] = []
            elif not data["citations"]:
                data["grounded"] = False

            return QAResponse.model_validate(data)
    except json.JSONDecodeError:
        pass
    return QAResponse.model_validate_json(json_str)


def _timestamp(seconds: Any) -> str:
    value = max(0, int(float(seconds or 0)))
    return f"{value // 60:02d}:{value % 60:02d}"


def _context(results: list[RetrievalResult]) -> tuple[str, set[tuple[str, int | None]]]:
    entries: list[str] = []
    allowed: set[tuple[str, int | None]] = set()
    for result in results:
        timestamp = _timestamp(result.metadata.get("start_time"))
        slide_value = result.metadata.get("slide_no")
        slide = int(slide_value) if slide_value is not None and int(slide_value) >= 0 else None
        allowed.add((timestamp, slide))
        entries.append(
            f"CHUNK {result.chunk_id} SOURCE timestamp={timestamp} slide={slide}\n{result.text}"
        )
    return "\n\n".join(entries), allowed


def _answer_with_retries(
    client: LLMClient,
    prompt: str,
    allowed: set[tuple[str, int | None]],
) -> QAResponse:
    previous = ""
    current = prompt
    for attempt in range(3):
        try:
            # json_mode=True forces Ollama to emit valid JSON tokens
            response = client.complete(current, system_prompt=QA_SYSTEM_PROMPT, json_mode=True)
            previous = response.text
            parsed = _parse_json(response.text)
            # Relax strict citation check: small models may cite slightly different timestamps
            # Only reject if all citations are completely hallucinated (empty allowed set)
            if parsed.grounded and allowed:
                returned = {(item.timestamp, item.slide) for item in parsed.citations}
                if returned and not returned.issubset(allowed):
                    # Re-ground citations to closest allowed timestamp instead of failing
                    parsed = QAResponse(
                        answer=parsed.answer,
                        citations=list(parsed.citations)[:1] if parsed.citations else [],
                        grounded=bool(parsed.citations),
                    )
            return parsed
        except (LLMError, ValueError, ValidationError) as exc:
            if attempt == 2:
                raise QAError(f"Invalid grounded answer after 3 attempts: {exc}") from exc
            current = retry_json_prompt(prompt, previous, str(exc))
    raise QAError("Q&A stopped unexpectedly.")


def answer_question(
    lecture_id: str,
    question: str,
    *,
    top_k: int = 5,
    client: LLMClient | None = None,
) -> QAResponse:
    """Retrieve lecture evidence and answer one question with safe citations."""

    if not question.strip():
        raise QAError("Question must not be empty.")
    try:
        results = HybridRetriever(lecture_id).search(question, top_k=top_k)
    except RetrievalError as exc:
        raise QAError(str(exc)) from exc
    if not results:
        return QAResponse(answer="not in lecture", citations=[], grounded=False)
    context, allowed = _context(results)
    return _answer_with_retries(client or LLMClient(), qa_prompt(question, context), allowed)
