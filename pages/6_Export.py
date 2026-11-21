"""Export cached lecture summaries and quizzes."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from core.export import ExportError, quiz_to_anki, quiz_to_csv, summary_to_markdown, summary_to_pdf
from core.quiz import QuizSet
from core.summarize import load_summary
from core.ui import configure_page, empty_state, lecture_selector, lecture_strip, page_header, require_ready_lectures


configure_page("Export", "↓")
page_header("06 / TAKE AWAY", "Export your study kit", "Download grounded notes for your next study session, or carry quiz cards into Anki.")
lectures = require_ready_lectures()
lecture_id = lecture_selector(lectures)
lecture = lectures[lecture_id]
lecture_strip(lecture, show_player_link=False)
summary = load_summary(lecture_id)
with st.container(border=True):
    st.markdown('<div class="kicker">SUMMARY</div><h3>Study notes</h3>', unsafe_allow_html=True)
    if summary:
        markdown = summary_to_markdown(summary, lecture.get("title", "Lecture Summary"))
        st.download_button("Download Markdown", markdown, file_name=f"{lecture_id}-summary.md", mime="text/markdown")
        try:
            pdf = summary_to_pdf(summary, lecture.get("title", "Lecture Summary"))
            st.download_button("Download PDF", pdf, file_name=f"{lecture_id}-summary.pdf", mime="application/pdf")
        except ExportError as exc:
            st.error(str(exc))
    else:
        empty_state("No summary to export", "Generate a summary in Lecture Player first.", "Open Lecture Player", "pages/2_Lecture_Player.py")

quiz_paths = sorted(Path("data").glob(f"{lecture_id}.quiz-*.json"))
with st.container(border=True):
    st.markdown('<div class="kicker">QUIZ</div><h3>Practice cards</h3>', unsafe_allow_html=True)
    if quiz_paths:
        quiz_path = st.selectbox("Quiz", quiz_paths, format_func=lambda item: item.name)
        try:
            quiz = QuizSet.model_validate(json.loads(quiz_path.read_text(encoding="utf-8")))
            data, mime, name = quiz_to_anki(quiz, lecture.get("title", "Lecture Quiz"))
            st.download_button("Download Anki deck", data, file_name=f"{lecture_id}-{name}", mime=mime)
            st.download_button("Download CSV fallback", quiz_to_csv(quiz), file_name=f"{lecture_id}-quiz.csv", mime="text/csv")
        except (OSError, json.JSONDecodeError, ValueError, ExportError) as exc:
            st.error(f"Could not export quiz: {exc}")
    else:
        empty_state("No quiz to export", "Generate a quiz in Quiz first.", "Open Quiz", "pages/3_Quiz.py")
