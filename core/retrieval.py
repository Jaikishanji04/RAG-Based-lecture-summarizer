"""Hybrid BM25/dense retrieval with reciprocal-rank fusion and reranking."""

from __future__ import annotations

import json
import importlib.machinery
import os
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from .indexing import INDEX_DIR, _get_collection, get_embedding_model


class RetrievalError(RuntimeError):
    """Raised when a lecture search cannot be completed."""


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


@dataclass
class RetrievalResult:
    """A ranked chunk returned to downstream summarization and Q&A."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge ranked IDs using standard RRF scoring."""

    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


@lru_cache(maxsize=1)
def get_reranker() -> Any:
    """Load the configured cross-encoder once per process."""

    model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    try:
        from sentence_transformers import CrossEncoder
    except Exception as sentence_transformer_error:
        try:
            _install_transformers_optional_dependency_shim()
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()

            class TransformersReranker:
                def predict(self, pairs: list[list[str]]) -> list[float]:
                    questions = [pair[0] for pair in pairs]
                    documents = [pair[1] for pair in pairs]
                    inputs = tokenizer(
                        questions,
                        documents,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                    with torch.no_grad():
                        logits = model(**inputs).logits.squeeze(-1)
                    return [float(score) for score in logits.cpu().tolist()]

            return TransformersReranker()
        except Exception as fallback_error:
            raise RetrievalError(
                f"Could not load reranker '{model_name}' through sentence-transformers "
                f"({sentence_transformer_error}) or Transformers fallback ({fallback_error})."
            ) from fallback_error
    try:
        return CrossEncoder(model_name)
    except Exception as exc:
        raise RetrievalError(f"Could not load reranker '{model_name}': {exc}") from exc


class HybridRetriever:
    """Search one lecture with BM25, dense vectors, RRF, and a cross-encoder."""

    def __init__(self, lecture_id: str) -> None:
        self.lecture_id = lecture_id
        self._chunks = self._load_chunks()

    def _load_chunks(self) -> dict[str, dict[str, Any]]:
        path = INDEX_DIR / f"{self.lecture_id}.chunks.json"
        if not path.exists():
            raise RetrievalError(f"No chunk index exists for lecture '{self.lecture_id}'.")
        try:
            chunks = json.loads(path.read_text(encoding="utf-8"))
            return {str(chunk["chunk_id"]): chunk for chunk in chunks}
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RetrievalError(f"Could not read chunk index: {exc}") from exc

    def _bm25_ids(self, query: str, limit: int) -> list[str]:
        try:
            from rank_bm25 import BM25Okapi

            token_path = INDEX_DIR / f"{self.lecture_id}.bm25.json"
            tokenized = json.loads(token_path.read_text(encoding="utf-8"))
            scores = BM25Okapi(tokenized).get_scores(query.lower().split())
            indexes = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            chunk_ids = list(self._chunks)
            return [chunk_ids[index] for index in indexes[:limit]]
        except (ImportError, OSError, json.JSONDecodeError) as exc:
            raise RetrievalError(f"Could not run BM25 search: {exc}") from exc

    def _dense_ids(self, query: str, limit: int) -> list[str]:
        try:
            embedding = get_embedding_model().encode([query], normalize_embeddings=True).tolist()
            result = _get_collection(self.lecture_id).query(
                query_embeddings=embedding,
                n_results=min(limit, len(self._chunks)),
                include=["documents", "metadatas"],
            )
            return [str(item_id) for item_id in result.get("ids", [[]])[0]]
        except Exception as exc:
            raise RetrievalError(f"Could not run dense search: {exc}") from exc

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        """Return the most relevant grounded chunks for a query."""

        if not query.strip():
            return []
        candidate_limit = max(top_k * 4, 20)
        rankings = [self._bm25_ids(query, candidate_limit), self._dense_ids(query, candidate_limit)]
        fused = reciprocal_rank_fusion(rankings)
        candidate_ids = [item_id for item_id, _ in fused[:candidate_limit] if item_id in self._chunks]
        reranker = get_reranker()
        pairs = [[query, self._chunks[item_id]["text"]] for item_id in candidate_ids]
        rerank_scores = reranker.predict(pairs)
        ordered = sorted(zip(candidate_ids, rerank_scores), key=lambda item: float(item[1]), reverse=True)
        return [
            RetrievalResult(
                chunk_id=chunk_id,
                text=self._chunks[chunk_id]["text"],
                score=float(score),
                metadata=self._chunks[chunk_id],
            )
            for chunk_id, score in ordered[:top_k]
        ]
