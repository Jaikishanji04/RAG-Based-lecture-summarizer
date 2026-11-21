"""Create timestamped, retrieval-friendly lecture chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass
class LectureChunk:
    """A bounded text unit with the source location needed for citations."""

    chunk_id: str
    lecture_id: str
    text: str
    start_time: float | None
    end_time: float | None
    slide_no: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable chunk."""

        return asdict(self)


def _word_count(text: str) -> int:
    return len(text.split())


def _overlap_segments(segments: list[dict[str, Any]], overlap_words: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    words = 0
    for segment in reversed(segments):
        selected.insert(0, segment)
        words += _word_count(str(segment.get("text", "")))
        if words >= overlap_words:
            break
    return selected


def chunk_transcript(
    segments: Iterable[dict[str, Any]],
    lecture_id: str,
    *,
    target_words: int = 320,
    overlap_words: int = 45,
) -> list[LectureChunk]:
    """Group timestamped transcript segments into roughly 250-400 word chunks."""

    normalized = [segment for segment in segments if str(segment.get("text", "")).strip()]
    chunks: list[LectureChunk] = []
    current: list[dict[str, Any]] = []
    current_words = 0

    for segment in normalized:
        current.append(segment)
        current_words += _word_count(str(segment["text"]))
        gap = 0.0
        if len(current) > 1:
            gap = float(segment.get("start", 0)) - float(current[-2].get("end", 0))
        at_natural_boundary = str(segment["text"]).rstrip().endswith(('.', '?', '!', ':'))
        should_flush = current_words >= target_words and (at_natural_boundary or gap >= 1.5)
        if should_flush:
            chunks.append(_make_transcript_chunk(current, lecture_id, len(chunks)))
            current = _overlap_segments(current, overlap_words)
            current_words = sum(_word_count(str(item["text"])) for item in current)

    if current:
        chunks.append(_make_transcript_chunk(current, lecture_id, len(chunks)))
    return chunks


def _make_transcript_chunk(segments: list[dict[str, Any]], lecture_id: str, index: int) -> LectureChunk:
    return LectureChunk(
        chunk_id=f"{lecture_id}-chunk-{index:05d}",
        lecture_id=lecture_id,
        text=" ".join(str(segment["text"]).strip() for segment in segments),
        start_time=float(segments[0].get("start", 0.0)),
        end_time=float(segments[-1].get("end", segments[-1].get("start", 0.0))),
        slide_no=None,
    )


def chunk_pages(
    pages: Iterable[dict[str, Any]],
    lecture_id: str,
    *,
    target_words: int = 320,
    overlap_words: int = 45,
) -> list[LectureChunk]:
    """Group PDF/PPTX pages while retaining the first page number as metadata."""

    chunks: list[LectureChunk] = []
    current: list[dict[str, Any]] = []
    current_words = 0
    for page in pages:
        text = str(page.get("text", "")).strip()
        if not text:
            continue
        current.append(page)
        current_words += _word_count(text)
        if current_words >= target_words:
            chunks.append(_make_page_chunk(current, lecture_id, len(chunks)))
            overlap: list[dict[str, Any]] = []
            words = 0
            for item in reversed(current):
                overlap.insert(0, item)
                words += _word_count(str(item["text"]))
                if words >= overlap_words:
                    break
            current = overlap
            current_words = sum(_word_count(str(item["text"])) for item in current)
    if current:
        chunks.append(_make_page_chunk(current, lecture_id, len(chunks)))
    return chunks


def _make_page_chunk(pages: list[dict[str, Any]], lecture_id: str, index: int) -> LectureChunk:
    return LectureChunk(
        chunk_id=f"{lecture_id}-chunk-{index:05d}",
        lecture_id=lecture_id,
        text=" ".join(str(page["text"]).strip() for page in pages),
        start_time=None,
        end_time=None,
        slide_no=int(pages[0]["slide_no"]),
    )
