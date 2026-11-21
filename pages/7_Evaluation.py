"""RAGAS evaluation dashboard for lecture Q&A quality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from core.evaluation import EvaluationError, EvaluationRecord, EvaluationResult, evaluate_with_ragas, records_from_quiz
from core.indexing import INDEX_DIR
from core.quiz import QuizSet
from core.ui import configure_page, empty_state, lecture_selector, lecture_strip, page_header, require_ready_lectures, stat_card


configure_page("Evaluation", "E")
page_header("07 / QUALITY", "Measure groundedness", "Inspect faithfulness, context precision, and context recall before trusting the study experience.")
lectures = require_ready_lectures()
lecture_id = lecture_selector(lectures)
lecture_strip(lectures[lecture_id], show_player_link=False)

with st.container(border=True):
    uploaded = st.file_uploader("Upload evaluation JSON", type=["json"])
    records: list[EvaluationRecord] | None = None
    if uploaded:
        try:
            payload: Any = json.loads(uploaded.getvalue())
            raw_records = payload.get("records", payload) if isinstance(payload, dict) else payload
            records = [EvaluationRecord.model_validate(item) for item in raw_records]
            st.success(f"✓ Loaded {len(records)} evaluation records.")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            st.error(f"Invalid evaluation JSON: {exc}")
    if st.button("Build evaluation set from cached quiz"):
        quiz_paths = sorted(Path("data").glob(f"{lecture_id}.quiz-*.json"))
        chunk_path = INDEX_DIR / f"{lecture_id}.chunks.json"
        if not quiz_paths or not chunk_path.exists():
            st.error("Generate a quiz and index the lecture before creating an automatic evaluation set.")
        else:
            try:
                quiz = QuizSet.model_validate_json(quiz_paths[0].read_text(encoding="utf-8"))
                records = records_from_quiz(quiz, json.loads(chunk_path.read_text(encoding="utf-8")))
                st.session_state["evaluation_records"] = records
                st.success(f"✓ Built {len(records)} evaluation records.")
            except (OSError, json.JSONDecodeError, ValueError, EvaluationError) as exc:
                st.error(str(exc))

if records is None:
    records = st.session_state.get("evaluation_records")
if records and st.button("Run evaluation", type="primary"):
    try:
        with st.spinner("Running RAGAS metrics..."):
            st.session_state["evaluation_result"] = evaluate_with_ragas(records)
    except EvaluationError as exc:
        st.error(f"Evaluation failed: {exc}")

result: EvaluationResult | None = st.session_state.get("evaluation_result")
if result:
    st.subheader("Metrics")
    cols = st.columns(3)
    for column, (name, value) in zip(cols, result.metrics.items()):
        with column:
            stat_card(name.replace("_", " ").title(), f"{value:.3f}", "Higher is better")
    if result.metrics:
        st.bar_chart(result.metrics)
    st.subheader("Detailed rows")
    st.dataframe(result.rows, use_container_width=True)
elif not records:
    empty_state("No evaluation set yet", "Upload JSON or build one from a cached quiz to inspect quality.")
