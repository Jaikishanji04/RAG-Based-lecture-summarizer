import pytest
from pydantic import ValidationError

from core.quiz import QuizQuestion, answer_is_correct
from core.summarize import SourceReference


def _question(question_type: str = "mcq", options: list[str] | None = None) -> QuizQuestion:
    values = options if options is not None else ["A", "B", "C", "D"]
    return QuizQuestion(
        question="Which answer is grounded?",
        type=question_type,
        options=values,
        correct_answer=values[0] if values else "grounded answer",
        bloom_level="Understand",
        difficulty="medium",
        explanation="The lecture supports this answer.",
        source_reference=SourceReference(timestamp="00:12"),
        concept="retrieval",
    )


def test_mcq_requires_four_options_and_grades_exact_answer() -> None:
    question = _question()
    assert answer_is_correct(question, "A")
    assert not answer_is_correct(question, "B")


def test_true_false_requires_fixed_options() -> None:
    question = _question("true_false", ["True", "False"])
    assert answer_is_correct(question, "True")
    with pytest.raises(ValidationError):
        _question("true_false", ["Yes", "No"])


def test_fill_blank_has_no_options() -> None:
    with pytest.raises(ValidationError):
        _question("fill_blank", ["answer"])
