"""RAGAS evaluation helpers for grounded lecture Q&A."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .quiz import QuizSet


class EvaluationError(RuntimeError):
    """Raised when a RAGAS evaluation cannot be completed."""


class EvaluationRecord(BaseModel):
    """One question, generated answer, evidence, and reference answer."""

    user_input: str = Field(min_length=1)
    response: str = Field(min_length=1)
    retrieved_contexts: list[str] = Field(min_length=1)
    reference: str = Field(min_length=1)


class EvaluationResult(BaseModel):
    """Normalized RAGAS metric output."""

    metrics: dict[str, float]
    rows: list[dict[str, Any]]


def records_from_quiz(quiz: QuizSet, chunk_records: list[dict[str, Any]]) -> list[EvaluationRecord]:
    """Create an automatic evaluation set from a validated lecture quiz."""

    contexts = [str(chunk.get("text", "")) for chunk in chunk_records if chunk.get("text")]
    if not contexts:
        raise EvaluationError("No indexed contexts are available for evaluation.")
    return [
        EvaluationRecord(
            user_input=question.question,
            response=question.correct_answer,
            retrieved_contexts=contexts[:5],
            reference=question.correct_answer,
        )
        for question in quiz.questions
    ]


def evaluate_with_ragas(records: list[EvaluationRecord]) -> EvaluationResult:
    """Run RAGAS faithfulness, context precision, and context recall metrics."""

    if not records:
        raise EvaluationError("At least one evaluation record is required.")
    try:
        from datasets import Dataset
        from ragas import evaluate
        try:
            from ragas.metrics import context_precision, context_recall, faithfulness
        except ImportError:
            from ragas.metrics.collections import context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise EvaluationError("Install ragas and datasets to run evaluation.") from exc

    dataset = Dataset.from_list([record.model_dump() for record in records])
    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, context_precision, context_recall],
        )
        if hasattr(result, "to_pandas"):
            frame = result.to_pandas()
            rows = frame.to_dict(orient="records")
            metrics = {
                name: float(frame[name].dropna().mean())
                for name in ("faithfulness", "context_precision", "context_recall")
                if name in frame
            }
        else:
            values = dict(result)
            metrics = {
                name: float(values[name])
                for name in ("faithfulness", "context_precision", "context_recall")
                if name in values and values[name] is not None
            }
            rows = [values]
        return EvaluationResult(metrics=metrics, rows=rows)
    except Exception as exc:
        raise EvaluationError(f"RAGAS evaluation failed: {exc}") from exc
