import pytest

from core.evaluation import records_from_quiz
from core.export import quiz_to_csv, summary_to_markdown, summary_to_pdf
from core.quiz import QuizQuestion, QuizSet
from core.summarize import FinalSummary, SourceReference, SummaryBullet


def _bullet(text: str) -> SummaryBullet:
    return SummaryBullet(text=text, source_reference=SourceReference(timestamp="00:12"))


def _quiz() -> QuizSet:
    return QuizSet(
        questions=[
            QuizQuestion(
                question="What is retrieval?",
                type="mcq",
                options=["A", "B", "C", "D"],
                correct_answer="A",
                bloom_level="Understand",
                difficulty="easy",
                explanation="The lecture states this.",
                source_reference=SourceReference(timestamp="00:12"),
                concept="retrieval",
            )
        ]
    )


def test_markdown_and_csv_exports_contain_grounding_fields() -> None:
    summary = FinalSummary(tldr=[_bullet("One"), _bullet("Two"), _bullet("Three")], key_points=[_bullet("Point")], detailed_notes=[_bullet("Detail")])
    assert "[00:12]" in summary_to_markdown(summary)
    csv_data = quiz_to_csv(_quiz()).decode("utf-8-sig")
    assert "Correct Answer" in csv_data
    assert "What is retrieval?" in csv_data


def test_quiz_becomes_evaluation_records() -> None:
    records = records_from_quiz(_quiz(), [{"text": "Retrieval finds relevant context."}])
    assert records[0].reference == "A"
    assert records[0].retrieved_contexts


def test_pdf_export_resets_cursor_between_sections() -> None:
    pytest.importorskip("fpdf")
    summary = FinalSummary(
        tldr=[_bullet("One"), _bullet("Two"), _bullet("Three")],
        key_points=[_bullet("Point")],
        detailed_notes=[_bullet("Detail")],
    )
    assert summary_to_pdf(summary)
