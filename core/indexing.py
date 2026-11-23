"""Persistent dense and lexical indexes for lecture chunks."""

from __future__ import annotations

import json
import logging
import os
import importlib.machinery
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .chunking import LectureChunk, chunk_pages, chunk_transcript


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
INDEX_DIR = DATA_DIR / "indexes"


class IndexingError(RuntimeError):
    """Raised when a lecture cannot be indexed."""


def _install_transformers_optional_dependency_shim() -> None:
    """Provide optional metrics without loading blocked SciPy DLLs."""

    sklearn_module = types.ModuleType("sklearn")
    sklearn_module.__spec__ = importlib.machinery.ModuleSpec("sklearn", None)
    metrics_module = types.ModuleType("sklearn.metrics")
    metrics_module.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", None)
    metrics_module.roc_curve = lambda *args, **kwargs: None
    sklearn_module.metrics = metrics_module
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.metrics"] = metrics_module

    scipy_module = types.ModuleType("scipy")
    scipy_module.__spec__ = importlib.machinery.ModuleSpec("scipy", None)
    optimize_module = types.ModuleType("scipy.optimize")
    optimize_module.__spec__ = importlib.machinery.ModuleSpec("scipy.optimize", None)
    optimize_module.linear_sum_assignment = lambda matrix: ([], [])
    scipy_module.optimize = optimize_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.optimize"] = optimize_module


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    """Load the configured sentence-transformer once per process."""

    model_name = os.getenv("HF_EMBED_MODEL") or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception as sentence_transformer_error:
        logger.warning("Using direct Transformers embedding fallback: %s", sentence_transformer_error)
        try:
            _install_transformers_optional_dependency_shim()
            import numpy as np
            import torch
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            encoder = AutoModel.from_pretrained(model_name)
            encoder.eval()

            class TransformersEmbeddingModel:
                def encode(
                    self,
                    texts: list[str],
                    *,
                    normalize_embeddings: bool = True,
                    show_progress_bar: bool = False,
                ) -> Any:
                    del show_progress_bar
                    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                    with torch.no_grad():
                        output = encoder(**inputs).last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1).expand(output.size()).float()
                    pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    if normalize_embeddings:
                        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    return np.asarray(pooled.cpu())

            return TransformersEmbeddingModel()
        except Exception as fallback_error:
            raise IndexingError(
                f"Could not load embedding model '{model_name}' with either "
                f"sentence-transformers ({sentence_transformer_error}) or Transformers fallback "
                f"({fallback_error})."
            ) from fallback_error


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexingError(f"Could not read {path.name}: {exc}") from exc


def chunks_for_lecture(lecture: dict[str, Any]) -> list[LectureChunk]:
    """Build chunks from a processed lecture record."""

    lecture_id = str(lecture["lecture_id"])
    if lecture.get("transcript_path"):
        return chunk_transcript(_load_json(Path(lecture["transcript_path"])), lecture_id)
    if lecture.get("pages_path"):
        return chunk_pages(_load_json(Path(lecture["pages_path"])), lecture_id)
    raise IndexingError("Lecture has no transcript or extracted document pages.")


def _get_collection(lecture_id: str) -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise IndexingError("Install chromadb before building an index.") from exc
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=f"lecture_{lecture_id}",
        metadata={"hnsw:space": "cosine"},
    )


def index_lecture(
    lecture: dict[str, Any],
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Persist chunks, dense embeddings, and BM25 source data for one lecture."""

    if progress_callback:
        progress_callback(0.05, "Preparing searchable chunks")
    chunks = chunks_for_lecture(lecture)
    if not chunks:
        raise IndexingError("No text was extracted, so the lecture cannot be indexed.")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    lecture_id = str(lecture["lecture_id"])
    chunk_dicts = [chunk.to_dict() for chunk in chunks]
    (INDEX_DIR / f"{lecture_id}.chunks.json").write_text(json.dumps(chunk_dicts, indent=2), encoding="utf-8")

    model = get_embedding_model()
    try:
        if progress_callback:
            progress_callback(0.2, f"Generating embeddings for {len(chunks)} chunks")
        embeddings = model.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        ).tolist()
        collection = _get_collection(lecture_id)
        metadatas = [
            {
                "lecture_id": chunk.lecture_id,
                "start_time": chunk.start_time if chunk.start_time is not None else -1.0,
                "end_time": chunk.end_time if chunk.end_time is not None else -1.0,
                **({"slide_no": chunk.slide_no} if chunk.slide_no is not None else {}),
            }
            for chunk in chunks
        ]
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
        )
        if progress_callback:
            progress_callback(0.8, "Saving dense search index")
    except Exception as exc:
        raise IndexingError(f"Could not build dense index: {exc}") from exc

    tokenized = [chunk.text.lower().split() for chunk in chunks]
    (INDEX_DIR / f"{lecture_id}.bm25.json").write_text(json.dumps(tokenized), encoding="utf-8")
    if progress_callback:
        progress_callback(0.95, "Saving keyword search index")
    return {"lecture_id": lecture_id, "chunk_count": len(chunks), "collection": f"lecture_{lecture_id}"}
