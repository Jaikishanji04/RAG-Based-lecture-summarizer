"""Interactive concept graph and concept-focused learning tools."""

from __future__ import annotations

import streamlit as st

from core.concepts import ConceptError, build_concept_graph, extract_concepts, load_concepts
from core.indexing import chunks_for_lecture
from core.qa import QAError, answer_question
from core.quiz import QuizError, generate_quiz
from core.ui import configure_page, empty_state, lecture_selector, lecture_strip, page_header, require_ready_lectures, source_chip


configure_page("Concept Graph", "G")
page_header("05 / CONNECT", "Concept graph", "See how ideas travel together, inspect first mentions, and practice one concept at a time.")
lectures = require_ready_lectures()
lecture_id = lecture_selector(lectures)
lecture = lectures[lecture_id]
lecture_strip(lecture, show_player_link=False)
concept_set = load_concepts(lecture_id)
if concept_set is None and st.button("Extract concepts", type="primary"):
    try:
        with st.spinner("Extracting grounded concepts..."):
            concept_set = extract_concepts(lecture)
    except ConceptError as exc:
        st.error(f"Could not extract concepts: {exc}")
if concept_set is None:
    empty_state("Your graph is waiting", "Extract grounded concepts from this lecture to see relationships.")
    st.stop()
try:
    graph = build_concept_graph(concept_set, chunks_for_lecture(lecture))
except ConceptError as exc:
    st.error(f"Could not build the graph: {exc}")
    st.stop()

selected_name = None
graph_col, detail_col = st.columns([1.5, 1], gap="large")
with graph_col:
    try:
        from streamlit_agraph import Config, Edge, Node, agraph
        selected_name = agraph(
            nodes=[Node(id=name, label=name, size=18 + min(24, graph.degree(name) * 3), color="#1F8A8A") for name in graph.nodes],
            edges=[Edge(source=source, target=target, label=data.get("relation", "related"), width=max(1, min(6, int(float(data.get("weight", 1)))))) for source, target, data in graph.edges(data=True)],
            config=Config(width="100%", height=560, directed=False, physics=True, hierarchical=False, nodeHighlightBehavior=True),
        )
    except ImportError as exc:
        st.warning(f"Interactive graph unavailable ({exc}). Showing the searchable relationship list instead.")
        search = st.text_input("Search concepts")
        for name in graph.nodes:
            if search.lower() in name.lower():
                st.markdown(f"- **{name}** · {graph.degree(name)} relationships")
    names = list(graph.nodes)
    selected_name = st.selectbox("Inspect concept", names, index=names.index(selected_name) if selected_name in names else 0)

selected = next(concept for concept in concept_set.concepts if concept.name == selected_name)
with detail_col:
    st.markdown('<div class="card generated-card"><div class="ai-kicker">SELECTED CONCEPT</div>', unsafe_allow_html=True)
    st.subheader(selected.name)
    st.write(selected.definition)
    st.caption(f"First mention · {selected.first_mention_timestamp}" + (f" · slide {selected.slide_no}" if selected.slide_no else ""))
    source_chip(selected.first_mention_timestamp, selected.slide_no, key=f"concept-watch-{lecture_id}")
    related = sorted(set(graph.neighbors(selected_name)) | set(selected.related_concepts))
    st.markdown("**Related concepts**")
    st.write(" · ".join(related) if related else "No related concepts recorded.")
    st.markdown("</div>", unsafe_allow_html=True)

st.subheader("Three-question mini-quiz")
if st.button("Generate mini-quiz", key=f"mini-{lecture_id}-{selected_name}"):
    try:
        st.session_state["concept_mini_quiz"] = generate_quiz(lecture_id, count=3, difficulty="medium", bloom_level="Understand", weak_concepts=[selected_name])
    except QuizError as exc:
        st.error(f"Could not generate mini-quiz: {exc}")
mini_quiz = st.session_state.get("concept_mini_quiz")
if mini_quiz:
    for index, question in enumerate(mini_quiz.questions, start=1):
        st.markdown(f"**{index}. {question.question}**")
        st.caption(f"Answer: {question.correct_answer} · {question.explanation}")

st.subheader("Ask about this concept")
concept_question = st.text_input("Question", key=f"concept-question-{lecture_id}-{selected_name}")
if st.button("Ask concept Q&A", key=f"concept-ask-{lecture_id}-{selected_name}") and concept_question:
    try:
        with st.spinner("Searching the lecture..."):
            response = answer_question(lecture_id, f"About {selected_name}: {concept_question}")
        st.write(response.answer)
        for index, citation in enumerate(response.citations):
            source_chip(citation.timestamp, citation.slide, key=f"concept-source-{lecture_id}-{index}")
    except QAError as exc:
        st.error(f"Could not answer this question: {exc}")
