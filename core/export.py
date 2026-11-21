"""Export validated summaries and quizzes to common study formats."""

from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path

from .quiz import QuizSet
from .summarize import FinalSummary, render_bullet


class ExportError(RuntimeError):
    """Raised when an export format cannot be produced."""


def summary_to_markdown(summary: FinalSummary, title: str = "Lecture Summary") -> str:
    """Render all summary tiers as Markdown."""

    lines = [f"# {title}", "", "## TL;DR"]
    lines.extend(f"- {render_bullet(item)}" for item in summary.tldr)
    lines.append("\n## Key Points")
    lines.extend(f"- {render_bullet(item)}" for item in summary.key_points)
    lines.append("\n## Detailed Notes")
    lines.extend(f"- {render_bullet(item)}" for item in summary.detailed_notes)
    return "\n".join(lines) + "\n"


def _pdf_text(value: str) -> str:
    return value.encode("latin-1", errors="replace").decode("latin-1")


def summary_to_pdf(summary: FinalSummary, title: str = "Lecture Summary") -> bytes:
    """Render a summary as a portable PDF using fpdf2."""

    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise ExportError("Install fpdf2 to export PDF files.") from exc
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, _pdf_text(title))
    for heading, bullets in (
        ("TL;DR", summary.tldr),
        ("Key Points", summary.key_points),
        ("Detailed Notes", summary.detailed_notes),
    ):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(0, 8, heading)
        pdf.set_font("Helvetica", size=10)
        for bullet in bullets:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 6, _pdf_text(f"- {render_bullet(bullet)}"))
    output = pdf.output(dest="S")
    return output.encode("latin-1") if isinstance(output, str) else bytes(output)


def quiz_to_csv(quiz: QuizSet) -> bytes:
    """Export quiz questions in a spreadsheet-compatible format."""

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Question", "Type", "Options", "Correct Answer", "Explanation", "Concept", "Timestamp", "Slide"])
    for question in quiz.questions:
        writer.writerow(
            [
                question.question,
                question.type,
                " | ".join(question.options),
                question.correct_answer,
                question.explanation,
                question.concept,
                question.source_reference.timestamp,
                question.source_reference.slide or "",
            ]
        )
    return output.getvalue().encode("utf-8-sig")


def quiz_to_anki(quiz: QuizSet, deck_name: str = "Lecture RAG") -> tuple[bytes, str, str]:
    """Export Anki package bytes, falling back to CSV when Genanki is unavailable."""

    try:
        import genanki
    except ImportError:
        return quiz_to_csv(quiz), "text/csv", "quiz.csv"
    model = genanki.Model(
        1607392319,
        "Lecture RAG Basic",
        fields=[{"name": "Question"}, {"name": "Answer"}],
        templates=[{"name": "Card 1", "qfmt": "{{Question}}", "afmt": "{{FrontSide}}<hr id=answer>{{Answer}}"}],
    )
    deck = genanki.Deck(2059400110, deck_name)
    for question in quiz.questions:
        answer = f"{question.correct_answer}<br>{question.explanation}"
        deck.add_note(genanki.Note(model=model, fields=[question.question, answer]))
    try:
        with tempfile.NamedTemporaryFile(suffix=".apkg", delete=False) as temporary:
            package_path = Path(temporary.name)
        genanki.Package(deck).write_to_file(str(package_path))
        data = package_path.read_bytes()
        package_path.unlink(missing_ok=True)
        return data, "application/octet-stream", "quiz.apkg"
    except Exception as exc:
        raise ExportError(f"Could not create Anki package: {exc}") from exc
