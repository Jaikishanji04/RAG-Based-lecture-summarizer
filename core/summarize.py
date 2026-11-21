"""Hierarchical, context-grounded lecture summarization."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator

from .chunking import LectureChunk
from .indexing import DATA_DIR, chunks_for_lecture
from .llm import LLMClient, LLMError
from .prompts import (
    SUMMARY_SYSTEM_PROMPT,
    chunk_summary_prompt,
    final_summary_prompt,
    retry_json_prompt,
    section_summary_prompt,
)


class SummaryError(RuntimeError):
    """Raised when a grounded summary cannot be generated."""


class SourceReference(BaseModel):
    """A citation attached to every generated summary item."""

    timestamp: str = Field(pattern=r"^\d{2}:\d{2}$")
    slide: int | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def normalize_timestamp(cls, value: Any) -> str:
        s = str(value or "").strip()
        parts = s.split(":")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return f"{int(parts[0]):02d}:{int(parts[1]):02d}"
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
            return f"{(int(parts[0]) * 60 + int(parts[1])):02d}:{int(parts[2]):02d}"
        match = re.search(r"(\d{1,2}):(\d{2})", s)
        if match:
            return f"{int(match.group(1)):02d}:{int(match.group(2)):02d}"
        raise ValueError(f"Invalid timestamp format '{value}', expected MM:SS")


class SummaryBullet(BaseModel):
    """A summary statement with a validated lecture citation."""

    text: str = Field(min_length=1)
    source_reference: SourceReference


class ChunkSummary(BaseModel):
    """Intermediate summary for one batch of lecture chunks."""

    summary: str = Field(min_length=1)
    points: list[SummaryBullet] = Field(min_length=1)


class SectionSummary(BaseModel):
    """Intermediate summary for a group of chunk summaries."""

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    points: list[SummaryBullet] = Field(min_length=1)


class FinalSummary(BaseModel):
    """The three user-facing summary tiers."""

    tldr: list[SummaryBullet] = Field(min_length=3, max_length=3)
    key_points: list[SummaryBullet] = Field(min_length=1)
    detailed_notes: list[SummaryBullet] = Field(min_length=1)

    @field_validator("tldr", mode="before")
    @classmethod
    def enforce_three_tldr(cls, value: Any) -> Any:
        if isinstance(value, list) and len(value) > 0:
            if len(value) > 3:
                return value[:3]
            while len(value) < 3:
                value.append(value[-1])
            return value
        return value


ModelType = TypeVar("ModelType", bound=BaseModel)


def _timestamp(seconds: float | None) -> str:
    total_seconds = max(0, int(seconds or 0))
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def _chunk_context(chunks: list[LectureChunk]) -> str:
    return "\n\n".join(
        "\n".join(
            [
                f"CHUNK {chunk.chunk_id}",
                f"SOURCE: [{_timestamp(chunk.start_time)}] slide={chunk.slide_no}",
                chunk.text,
            ]
        )
        for chunk in chunks
    )


def _batch_chunks(chunks: list[LectureChunk], max_words: int | None = None) -> list[list[LectureChunk]]:
    """Group lecture chunks into batches to fit local LLM context limits."""
    if max_words is None:
        max_words = int(os.getenv("SUMMARY_BATCH_WORDS", "1200"))
    batches: list[list[LectureChunk]] = []
    current: list[LectureChunk] = []
    words = 0
    for chunk in chunks:
        chunk_words = len(chunk.text.split())
        if current and words + chunk_words > max_words:
            batches.append(current)
            current = []
            words = 0
        current.append(chunk)
        words += chunk_words
    if current:
        batches.append(current)
    return batches


def _parse_json_response(response: str, model_type: type[ModelType]) -> ModelType:
    cleaned = response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response did not contain a JSON object")
    json_str = cleaned[start : end + 1]
    # Clean up any trailing commas that small models might produce
    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
    return model_type.model_validate_json(json_str)


def _complete_json(
    client: LLMClient,
    prompt: str,
    model_type: type[ModelType],
    *,
    retries: int = 3,
) -> ModelType:
    previous_response = ""
    current_prompt = prompt
    for attempt in range(retries):
        try:
            response = client.complete(current_prompt, system_prompt=SUMMARY_SYSTEM_PROMPT, json_mode=True)
            previous_response = response.text
            return _parse_json_response(response.text, model_type)
        except (LLMError, ValueError, ValidationError) as exc:
            if attempt == retries - 1:
                raise SummaryError(f"LLM returned invalid summary JSON after {retries} attempts: {exc}") from exc
            current_prompt = retry_json_prompt(prompt, previous_response, str(exc))
    raise SummaryError("Summary generation stopped unexpectedly.")


def render_bullet(bullet: SummaryBullet) -> str:
    """Render a validated bullet with its timestamp and optional slide."""

    slide = f" | slide {bullet.source_reference.slide}" if bullet.source_reference.slide else ""
    return f"{bullet.text} [{bullet.source_reference.timestamp}]{slide}"


def _summary_path(lecture_id: str) -> Path:
    return DATA_DIR / f"{lecture_id}.summary.json"


def load_summary(lecture_id: str) -> FinalSummary | None:
    """Load a cached summary if one exists and remains schema-valid."""

    path = _summary_path(lecture_id)
    if not path.exists():
        return None
    try:
        return FinalSummary.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError):
        return None


def summarize_lecture(
    lecture: dict[str, Any],
    *,
    client: LLMClient | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> FinalSummary:
    """Generate and cache chunk, section, and final lecture summaries."""

    cached = load_summary(str(lecture["lecture_id"]))
    if cached:
        return cached
    chunks = chunks_for_lecture(lecture)
    if not chunks:
        raise SummaryError("The lecture has no chunks to summarize.")
    llm = client or LLMClient()
    chunk_summaries: list[ChunkSummary] = []
    batches = _batch_chunks(chunks)
    for index, batch in enumerate(batches, start=1):
        chunk_summaries.append(_complete_json(llm, chunk_summary_prompt(_chunk_context(batch)), ChunkSummary))
        if progress_callback:
            progress_callback(index / (len(batches) + 2), f"Summarizing chunk batch {index}/{len(batches)}")

    if len(chunk_summaries) == 1:
        # Optimization: single batch can jump straight to final summary, saving 1 full LLM call
        section_context = f"SECTION INPUT 1:\n{chunk_summaries[0].model_dump_json()}"
    else:
        section_context = "\n\n".join(
            f"SECTION INPUT {index}:\n{summary.model_dump_json()}"
            for index, summary in enumerate(chunk_summaries, start=1)
        )
        section = _complete_json(llm, section_summary_prompt(section_context), SectionSummary)
        section_context = section.model_dump_json()

    if progress_callback:
        progress_callback((len(batches) + 1) / (len(batches) + 2), "Writing final summary")
    final = _complete_json(llm, final_summary_prompt(section_context), FinalSummary)
    _summary_path(str(lecture["lecture_id"])).write_text(final.model_dump_json(indent=2), encoding="utf-8")
    if progress_callback:
        progress_callback(1.0, "Summary ready")
    return final
