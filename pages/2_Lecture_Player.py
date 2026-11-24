"""Unified Lecture Study Workspace bringing Player, Quiz, Q&A, Concept Graph, Export, and Evaluation into one cohesive interface."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

from core.concepts import ConceptError, build_concept_graph, extract_concepts, load_concepts
from core.evaluation import EvaluationError, EvaluationRecord, EvaluationResult, evaluate_with_ragas, records_from_quiz
from core.export import ExportError, quiz_to_anki, quiz_to_csv, summary_to_markdown, summary_to_pdf
from core.indexing import INDEX_DIR, chunks_for_lecture
from core.ingestion import IngestionError
from core.qa import QAError, QAResponse, answer_question
from core.quiz import QuizError, QuizSet, answer_is_correct, generate_quiz, load_quiz
from core.summarize import SummaryError, load_summary, summarize_lecture
from core.ui import (
    configure_page,
    empty_state,
    fmt_ts,
    lecture_selector,
    lecture_strip,
    page_header,
    require_ready_lectures,
    source_chip,
    stat_card,
)


def _clean_summary_text(text: str) -> str:
    cleaned = re.sub(r"\[\d{1,2}:\d{2}(?:\s*\|\s*slide\s*\d+)?\]", "", text)
    cleaned = re.sub(r"\(\d{1,2}:\d{2}\)", "", cleaned)
    cleaned = re.sub(r"^\s*[-*•]\s*", "", cleaned)
    return cleaned.strip()


configure_page("Study Workspace", "🎓")
page_header(
    "01 / WORKBENCH",
    "Lecture Study Workspace",
    "Unified workspace for your processed lectures: playback, grounded summaries, adaptive quizzes, Q&A, concept graph, export, and quality metrics.",
)

lectures = require_ready_lectures()
lecture_id = lecture_selector(lectures)
lecture = lectures[lecture_id]
lecture_strip(lecture, show_player_link=False)

# Six unified study tasks in one view
tab_player, tab_quiz, tab_qa, tab_graph, tab_export, tab_eval = st.tabs(
    [
        "🎧 Player & Notes",
        "📝 Practice Quiz",
        "💬 Ask Q&A",
        "🕸️ Concept Graph",
        "📦 Export Kit",
        "📊 Evaluation",
    ]
)

# ---------------------------------------------------------
# TAB 1: PLAYER & SUMMARY
# ---------------------------------------------------------
with tab_player:
    player_time = int(st.session_state.get("player_time", 0))
    source_path = Path(lecture["path"]) if lecture.get("path") else None

    col_media, col_summary_action = st.columns([1.5, 1], gap="large")
    with col_media:
        if source_path and source_path.suffix.lower() in {".mp4", ".mov"}:
            st.video(str(source_path), start_time=player_time)
        elif source_path and source_path.suffix.lower() in {".mp3", ".wav", ".m4a"}:
            st.audio(str(source_path), start_time=player_time)
        else:
            st.markdown(
                '<div class="card source-card"><h3>Document lecture</h3><p>This source is made of indexed pages or slides. Use the transcript and citations below to study it.</p></div>',
                unsafe_allow_html=True,
            )

    with col_summary_action:
        st.markdown(
            '<div class="card generated-card"><div class="ai-kicker">STUDY GUIDE</div><h3>Lecture Summary</h3><p>Generate a compact, grounded summary with TL;DR, key takeaways, and detailed notes.</p>',
            unsafe_allow_html=True,
        )
        if st.button("Generate Summary", type="primary", use_container_width=True, key="btn_gen_summary"):
            st.session_state["ui_generate_summary"] = True
        st.markdown("</div>", unsafe_allow_html=True)

    transcript: list[dict[str, Any]] = []
    if lecture.get("transcript_path"):
        try:
            transcript = json.loads(Path(lecture["transcript_path"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not load transcript: {exc}")

    summary = load_summary(lecture_id)
    if st.session_state.pop("ui_generate_summary", False) and summary is None:
        progress = st.progress(0.0)
        status = st.status("Generating grounded summary", expanded=True)
        try:
            summary = summarize_lecture(
                lecture,
                progress_callback=lambda value, message: (progress.progress(value, text=message), status.update(label=message)),
            )
            status.update(label="Summary ready", state="complete")
        except (SummaryError, IngestionError) as exc:
            status.update(label="Summary failed", state="error")
            st.error(f"Could not generate the summary: {exc}")
            if st.button("Retry summary", key="btn_retry_summary"):
                st.session_state["ui_generate_summary"] = True
                st.rerun()

    sub_transcript, sub_summary = st.tabs(["Transcript", "Summary"])
    with sub_transcript:
        if transcript:
            st.markdown('<div class="card source-card transcript-scroll">', unsafe_allow_html=True)
            for index, segment in enumerate(transcript):
                start = float(segment.get("start", 0))
                active = abs(start - player_time) <= 2
                row = st.container()
                if active:
                    row.markdown('<div class="active-segment">', unsafe_allow_html=True)
                timestamp_col, text_col = row.columns([0.22, 0.78])
                with timestamp_col:
                    if st.button(fmt_ts(start), key=f"jump-{lecture_id}-{index}", use_container_width=True):
                        st.session_state["player_time"] = int(start)
                        st.rerun()
                with text_col:
                    st.write(str(segment.get("text", "")))
                if active:
                    row.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            empty_state("No audio transcript available", "This document lecture is represented by indexed pages or slides.")

    with sub_summary:
        if not summary:
            empty_state("Your study guide is waiting", "Click 'Generate Summary' above to see a TL;DR, key points, and detailed notes.")
        else:
            st.markdown('<div class="card generated-card"><div class="ai-kicker">AI-GENERATED · TL;DR</div>', unsafe_allow_html=True)
            for bullet in summary.tldr:
                st.markdown(f"• **{_clean_summary_text(bullet.text)}**")
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("Key points")
            for bullet in summary.key_points:
                st.markdown('<div class="card generated-card">', unsafe_allow_html=True)
                st.write(f"• {_clean_summary_text(bullet.text)}")
                st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Detailed summary (points)", expanded=True):
                for bullet in summary.detailed_notes:
                    st.markdown(f"• {_clean_summary_text(bullet.text)}")

# ---------------------------------------------------------
# TAB 2: ADAPTIVE QUIZ
# ---------------------------------------------------------
with tab_quiz:
    with st.container(border=True):
        st.markdown('<div class="kicker">QUIZ SETTINGS</div>', unsafe_allow_html=True)
        q_cols = st.columns(3)
        q_count = q_cols[0].slider("Questions", 1, 20, 5, key="quiz_q_count")
        q_diff = q_cols[1].selectbox("Difficulty", ["easy", "medium", "hard"], index=1, key="quiz_q_diff")
        q_bloom = q_cols[2].selectbox(
            "Bloom level",
            ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"],
            index=1,
            key="quiz_q_bloom",
        )
        weak_concepts = st.session_state.setdefault("quiz_weak_concepts", {}).setdefault(lecture_id, [])
        if weak_concepts:
            st.caption("Focus from your last attempt: " + ", ".join(weak_concepts))

        if st.button("Generate quiz", type="primary", key="btn_generate_quiz"):
            try:
                with st.spinner("Generating and verifying lecture questions..."):
                    st.session_state["active_quiz"] = generate_quiz(
                        lecture_id,
                        count=q_count,
                        difficulty=q_diff,
                        bloom_level=q_bloom,
                        weak_concepts=weak_concepts,
                    )
                    st.session_state["quiz_answers"] = {}
                    st.session_state["quiz_result"] = None
            except QuizError as exc:
                st.error(f"Could not generate quiz: {exc}")

    active_quiz = st.session_state.get("active_quiz") or load_quiz(lecture_id, q_count, q_diff, q_bloom)
    if active_quiz is None:
        empty_state("No quiz loaded", "Choose settings above and generate a grounded quiz to begin practicing.")
    else:
        st.progress(
            len(st.session_state.get("quiz_answers", {})) / len(active_quiz.questions),
            text="Answer progress",
        )
        answers = st.session_state.setdefault("quiz_answers", {})
        for index, question in enumerate(active_quiz.questions):
            with st.container(border=True):
                st.markdown(f'<div class="kicker">QUESTION {index + 1:02d}</div>', unsafe_allow_html=True)
                st.subheader(question.question)
                st.caption(f"{question.difficulty.title()} · {question.bloom_level} · {question.concept}")
                if question.type in {"mcq", "true_false"}:
                    saved_val = answers.get(index)
                    opt_idx = question.options.index(saved_val) if saved_val in question.options else None
                    user_choice = st.radio(
                        "Options",
                        question.options,
                        index=opt_idx,
                        key=f"q_{index}_{lecture_id}",
                        label_visibility="collapsed",
                    )
                    if user_choice:
                        answers[index] = user_choice
                else:
                    user_text = st.text_input(
                        "Your answer",
                        value=answers.get(index, ""),
                        key=f"q_{index}_{lecture_id}",
                    )
                    if user_text:
                        answers[index] = user_text

        if st.button("Submit answers", type="primary", key="btn_submit_quiz"):
            correct_count = sum(
                1 for idx, q in enumerate(active_quiz.questions) if answer_is_correct(q, answers.get(idx, ""))
            )
            st.session_state["quiz_result"] = (correct_count, len(active_quiz.questions))
            st.success(f"Score: {correct_count} / {len(active_quiz.questions)}")

# ---------------------------------------------------------
# TAB 3: ASK LECTURE Q&A
# ---------------------------------------------------------
with tab_qa:
    st.success("✓ Grounded in this lecture")
    all_messages = st.session_state.setdefault("qa_messages", {})
    messages = all_messages.setdefault(lecture_id, [])

    if not messages:
        st.markdown("**Try a question:**")
        examples = [
            "What is the main idea of this lecture?",
            "Explain the most important concepts.",
            "Summarize the key conclusions.",
        ]
        ex_cols = st.columns(3)
        for col, example in zip(ex_cols, examples):
            if col.button(example, key=f"ex_{example}"):
                st.session_state["ui_example_question"] = example

    for message_index, message in enumerate(messages):
        with st.chat_message(str(message["role"])):
            st.write(str(message["content"]))
            resp = message.get("response")
            if isinstance(resp, QAResponse):
                if resp.grounded:
                    for citation_index, citation in enumerate(resp.citations):
                        source_chip(
                            citation.timestamp,
                            citation.slide,
                            key=f"qa-chip-{lecture_id}-{message_index}-{citation_index}",
                        )
                else:
                    st.warning("! This answer is not supported by the indexed lecture context.")

    prompt_q = st.chat_input("Ask something about this lecture...") or st.session_state.pop("ui_example_question", None)
    if prompt_q:
        messages.append({"role": "user", "content": prompt_q})
        with st.chat_message("user"):
            st.write(prompt_q)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the lecture..."):
                    resp = answer_question(lecture_id, prompt_q)
                st.write(resp.answer)
                if resp.grounded:
                    st.caption("Sources:")
                    for c_idx, cit in enumerate(resp.citations):
                        source_chip(cit.timestamp, cit.slide, key=f"qa-live-chip-{c_idx}")
                messages.append({"role": "assistant", "content": resp.answer, "response": resp})
            except QAError as exc:
                st.error(f"Could not answer: {exc}")

# ---------------------------------------------------------
# TAB 4: CONCEPT GRAPH
# ---------------------------------------------------------
with tab_graph:
    concept_set = load_concepts(lecture_id)
    if concept_set is None and st.button("Extract concepts", type="primary", key="btn_extract_concepts"):
        try:
            with st.spinner("Extracting grounded concepts..."):
                concept_set = extract_concepts(lecture)
        except ConceptError as exc:
            st.error(f"Could not extract concepts: {exc}")

    if concept_set is None:
        empty_state("Your graph is waiting", "Click 'Extract concepts' to map concept relationships across this lecture.")
    else:
        try:
            graph = build_concept_graph(concept_set, chunks_for_lecture(lecture))
            graph_col, detail_col = st.columns([1.5, 1], gap="large")
            with graph_col:
                try:
                    from streamlit_agraph import Config, Edge, Node, agraph

                    selected_node = agraph(
                        nodes=[
                            Node(
                                id=name,
                                label=name,
                                size=18 + min(24, graph.degree(name) * 3),
                                color="#1F8A8A",
                            )
                            for name in graph.nodes
                        ],
                        edges=[
                            Edge(
                                source=source,
                                target=target,
                                label=data.get("relation", "related"),
                                width=max(1, min(6, int(float(data.get("weight", 1))))),
                            )
                            for source, target, data in graph.edges(data=True)
                        ],
                        config=Config(
                            width="100%",
                            height=520,
                            directed=False,
                            physics=True,
                            hierarchical=False,
                            nodeHighlightBehavior=True,
                        ),
                    )
                except ImportError:
                    st.info("Interactive visualizer fallback: Search concept nodes below.")
                    filter_txt = st.text_input("Filter concepts", key="graph_filter")
                    for name in graph.nodes:
                        if not filter_txt or filter_txt.lower() in name.lower():
                            st.markdown(f"• **{name}** (Degree: {graph.degree(name)})")

            with detail_col:
                st.markdown(
                    f'<div class="card source-card"><div class="ai-kicker">CONCEPTS IDENTIFIED</div><h3>{len(concept_set.concepts)} Core Ideas</h3><p>Graph nodes reflect key terms and relations extracted directly from lecture chunks.</p></div>',
                    unsafe_allow_html=True,
                )
                for c in concept_set.concepts[:8]:
                    st.markdown(f"**{c.name}** · *{c.summary[:100]}...*")
        except ConceptError as exc:
            st.error(f"Could not build the graph: {exc}")

# ---------------------------------------------------------
# TAB 5: EXPORT KIT
# ---------------------------------------------------------
with tab_export:
    summary_for_export = load_summary(lecture_id)
    with st.container(border=True):
        st.markdown('<div class="kicker">STUDY NOTES</div><h3>Export Notes</h3>', unsafe_allow_html=True)
        if summary_for_export:
            md_text = summary_to_markdown(summary_for_export, lecture.get("title", "Lecture Summary"))
            st.download_button(
                "Download Markdown (.md)",
                md_text,
                file_name=f"{lecture_id}-summary.md",
                mime="text/markdown",
                key="btn_dl_md",
            )
            try:
                pdf_bytes = summary_to_pdf(summary_for_export, lecture.get("title", "Lecture Summary"))
                st.download_button(
                    "Download PDF Document (.pdf)",
                    pdf_bytes,
                    file_name=f"{lecture_id}-summary.pdf",
                    mime="application/pdf",
                    key="btn_dl_pdf",
                )
            except ExportError as exc:
                st.error(str(exc))
        else:
            empty_state("No summary to export", "Generate a summary in the 'Player & Notes' tab first.")

    quiz_paths = sorted(Path("data").glob(f"{lecture_id}.quiz-*.json"))
    with st.container(border=True):
        st.markdown('<div class="kicker">PRACTICE FLASHCARDS</div><h3>Export Cards</h3>', unsafe_allow_html=True)
        if quiz_paths:
            chosen_quiz_path = st.selectbox(
                "Choose Quiz Deck",
                quiz_paths,
                format_func=lambda item: item.name,
                key="export_quiz_select",
            )
            try:
                q_set = QuizSet.model_validate(json.loads(chosen_quiz_path.read_text(encoding="utf-8")))
                data_anki, mime_anki, name_anki = quiz_to_anki(q_set, lecture.get("title", "Lecture Quiz"))
                st.download_button(
                    "Download Anki Deck (.apkg)",
                    data_anki,
                    file_name=f"{lecture_id}-{name_anki}",
                    mime=mime_anki,
                    key="btn_dl_anki",
                )
                st.download_button(
                    "Download CSV Format (.csv)",
                    quiz_to_csv(q_set),
                    file_name=f"{lecture_id}-quiz.csv",
                    mime="text/csv",
                    key="btn_dl_csv",
                )
            except Exception as exc:
                st.error(f"Could not export quiz: {exc}")
        else:
            empty_state("No quiz cards to export", "Generate a quiz in the 'Practice Quiz' tab first.")

# ---------------------------------------------------------
# TAB 6: QUALITY & EVALUATION
# ---------------------------------------------------------
with tab_eval:
    with st.container(border=True):
        st.markdown('<div class="kicker">EVALUATION BENCHMARK</div><h3>Measure Groundedness</h3>', unsafe_allow_html=True)
        uploaded_eval = st.file_uploader("Upload custom evaluation JSON", type=["json"], key="eval_uploader")
        eval_records: list[EvaluationRecord] | None = None
        if uploaded_eval:
            try:
                eval_payload = json.loads(uploaded_eval.getvalue())
                raw_eval = eval_payload.get("records", eval_payload) if isinstance(eval_payload, dict) else eval_payload
                eval_records = [EvaluationRecord.model_validate(item) for item in raw_eval]
                st.success(f"✓ Loaded {len(eval_records)} evaluation records.")
            except Exception as exc:
                st.error(f"Invalid evaluation JSON: {exc}")

        if st.button("Build evaluation set from cached quiz", key="btn_build_eval"):
            q_paths = sorted(Path("data").glob(f"{lecture_id}.quiz-*.json"))
            chunk_file = INDEX_DIR / f"{lecture_id}.chunks.json"
            if not q_paths or not chunk_file.exists():
                st.error("Generate a quiz and index the lecture first to create an automatic evaluation set.")
            else:
                try:
                    q_data = QuizSet.model_validate_json(q_paths[0].read_text(encoding="utf-8"))
                    eval_records = records_from_quiz(q_data, json.loads(chunk_file.read_text(encoding="utf-8")))
                    st.session_state["evaluation_records"] = eval_records
                    st.success(f"✓ Built {len(eval_records)} evaluation records.")
                except Exception as exc:
                    st.error(str(exc))

        if eval_records is None:
            eval_records = st.session_state.get("evaluation_records")

        if eval_records and st.button("Run evaluation", type="primary", key="btn_run_eval"):
            with st.spinner("Running RAGAS evaluation..."):
                try:
                    eval_result = evaluate_with_ragas(eval_records)
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        stat_card("Faithfulness", f"{eval_result.faithfulness:.2f}")
                    with m2:
                        stat_card("Context Precision", f"{eval_result.context_precision:.2f}")
                    with m3:
                        stat_card("Context Recall", f"{eval_result.context_recall:.2f}")
                except Exception as exc:
                    st.warning(f"Evaluation benchmark notice: {exc}")
