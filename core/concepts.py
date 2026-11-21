"""Concept extraction and learning-graph construction."""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .indexing import DATA_DIR, chunks_for_lecture, get_embedding_model
from .llm import LLMClient, LLMError
from .prompts import CONCEPT_SYSTEM_PROMPT, concept_extraction_prompt, retry_json_prompt
from .summarize import SourceReference


class ConceptError(RuntimeError):
    """Raised when concepts cannot be extracted or graphed."""


class Concept(BaseModel):
    """A lecture concept and its first grounded source location."""

    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    first_mention_timestamp: str = Field(pattern=r"^\d{2}:\d{2}$")
    slide_no: int | None = None
    related_concepts: list[str] = Field(default_factory=list)


class ConceptSet(BaseModel):
    """Validated concepts extracted for one lecture."""

    concepts: list[Concept] = Field(min_length=1)


def _parse_json(response: str) -> ConceptSet:
    cleaned = response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response did not contain a JSON object")
    json_str = cleaned[start : end + 1]
    # Sanitize trailing commas that small models frequently output
    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
    # Normalize timestamps like "1:23" or "01:02:03" inside the JSON string
    def _fix_ts(m: re.Match) -> str:
        raw = m.group(1)
        parts = raw.split(":")
        try:
            if len(parts) == 2:
                return f'"{int(parts[0]):02d}:{int(parts[1]):02d}"'
            if len(parts) == 3:
                total_min = int(parts[0]) * 60 + int(parts[1])
                return f'"{total_min:02d}:{int(parts[2]):02d}"'
        except ValueError:
            pass
        return m.group(0)
    json_str = re.sub(r'"(\d{1,3}:\d{2}(?::\d{2})?)"', _fix_ts, json_str)
    return ConceptSet.model_validate_json(json_str)


def _context(lecture: dict[str, Any], max_words: int = 1200) -> str:
    """Build context capped at max_words to fit the local model's context window."""
    chunks = chunks_for_lecture(lecture)
    entries: list[str] = []
    words = 0
    for chunk in chunks:
        chunk_words = chunk.text.split()
        if entries and words + len(chunk_words) > max_words:
            break
        timestamp = max(0, int(chunk.start_time or 0))
        reference = f"[{timestamp // 60:02d}:{timestamp % 60:02d}] slide={chunk.slide_no}"
        entries.append(f"SOURCE {reference}\n{chunk.text}")
        words += len(chunk_words)
    return "\n\n".join(entries)


def _concept_path(lecture_id: str) -> Path:
    return DATA_DIR / f"{lecture_id}.concepts.json"


def load_concepts(lecture_id: str) -> ConceptSet | None:
    """Load cached, schema-validated concepts."""

    path = _concept_path(lecture_id)
    if not path.exists():
        return None
    try:
        return ConceptSet.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError):
        return None


def extract_concepts(lecture: dict[str, Any], *, client: LLMClient | None = None) -> ConceptSet:
    """Extract and cache concepts for a lecture using the configured LLM."""

    lecture_id = str(lecture["lecture_id"])
    cached = load_concepts(lecture_id)
    if cached:
        return cached
    prompt = concept_extraction_prompt(_context(lecture))
    llm = client or LLMClient()
    previous = ""
    current = prompt
    for attempt in range(3):
        try:
            # json_mode=True forces Ollama to produce valid JSON tokens
            response = llm.complete(current, system_prompt=CONCEPT_SYSTEM_PROMPT, json_mode=True)
            previous = response.text
            concepts = _parse_json(response.text)
            _concept_path(lecture_id).write_text(concepts.model_dump_json(indent=2), encoding="utf-8")
            return concepts
        except (LLMError, ValueError, ValidationError) as exc:
            if attempt == 2:
                raise ConceptError(f"Invalid concept JSON after 3 attempts: {exc}") from exc
            current = retry_json_prompt(prompt, previous, str(exc))
    raise ConceptError("Concept extraction stopped unexpectedly.")


def _concept_mentions(concept: Concept, text: str) -> bool:
    return bool(re.search(rf"\b{re.escape(concept.name)}\b", text, flags=re.IGNORECASE))


def build_concept_graph(concept_set: ConceptSet, chunks: list[Any] | None = None) -> Any:
    """Build a NetworkX graph from co-occurrence and embedding similarity."""

    try:
        import networkx as nx
    except ImportError as exc:
        raise ConceptError("Install networkx before building the concept graph.") from exc

    graph = nx.Graph()
    concepts = concept_set.concepts
    for concept in concepts:
        graph.add_node(
            concept.name,
            definition=concept.definition,
            timestamp=concept.first_mention_timestamp,
            slide=concept.slide_no,
            related=concept.related_concepts,
        )

    if chunks:
        for chunk in chunks:
            mentioned = [concept.name for concept in concepts if _concept_mentions(concept, chunk.text)]
            for source, target in combinations(mentioned, 2):
                if graph.has_edge(source, target):
                    graph[source][target]["weight"] += 1.0
                    graph[source][target]["relation"] = "co-occurrence + similarity"
                else:
                    graph.add_edge(source, target, weight=1.0, relation="co-occurrence")

    try:
        texts = [f"{concept.name}: {concept.definition}" for concept in concepts]
        vectors = get_embedding_model().encode(texts, normalize_embeddings=True).tolist()
        for index, other_index in combinations(range(len(concepts)), 2):
            similarity = sum(
                left * right for left, right in zip(vectors[index], vectors[other_index], strict=False)
            )
            if similarity >= 0.55:
                source, target = concepts[index].name, concepts[other_index].name
                if graph.has_edge(source, target):
                    graph[source][target]["weight"] += float(similarity)
                    graph[source][target]["relation"] = "co-occurrence + similarity"
                else:
                    graph.add_edge(source, target, weight=float(similarity), relation="embedding similarity")
    except Exception:
        # Co-occurrence remains useful if the embedding model is unavailable.
        pass
    return graph
