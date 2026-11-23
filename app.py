import streamlit as st

from core.ui import configure_page, ready_lectures, stat_card

configure_page("Lecture Workbench", "📚")
lectures = ready_lectures()
st.markdown('<div class="kicker">STUDY DESK / LOCAL RAG</div><h1>Turn long lectures into clear, searchable study material.</h1><p class="page-header-copy">Upload once, then listen, search, practice, connect ideas, and export a grounded study kit.</p>', unsafe_allow_html=True)

steps = [
    ("01", "Bring it in", "Upload audio, video, PDF, PPTX, or a YouTube URL."),
    ("02", "Make it searchable", "Transcribe or extract pages, then index the source."),
    ("03", "Understand", "Generate cited summaries and ask grounded questions."),
    ("04", "Practice", "Use adaptive quizzes and concept relationships."),
    ("05", "Take it with you", "Export notes, quiz cards, and quality metrics."),
]
cols = st.columns(5)
for column, (number, title, copy) in zip(cols, steps):
    with column:
        st.markdown(f'<div class="card"><div class="kicker">{number}</div><strong>{title}</strong><p>{copy}</p></div>', unsafe_allow_html=True)

if lectures:
    lecture = next(iter(lectures.values()))
    st.subheader("Continue studying")
    a, b, c = st.columns(3)
    with a: stat_card("Active lecture", lecture.get("title", "Untitled"))
    with b: stat_card("Source", str(lecture.get("source_type", "—")).upper())
    with c: stat_card("Chunks", lecture.get("index", {}).get("chunk_count", "—"))
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        if st.button("Open Lecture Player", type="primary", use_container_width=True):
            st.switch_page("pages/2_Lecture_Player.py")
    with col_act2:
        if st.button("Open Inbox Agent (Gmail & MCP)", type="secondary", use_container_width=True):
            st.switch_page("pages/8_Inbox_Agent.py")
else:
    st.markdown('<div class="card generated-card"><div class="ai-kicker">START HERE</div><h2>Your desk is clear.</h2><p>Process your first lecture to unlock the player, quiz, Q&A, concept graph, exports, and evaluation.</p></div>', unsafe_allow_html=True)
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        if st.button("Upload a lecture", type="primary", use_container_width=True):
            st.switch_page("pages/1_Upload_and_Process.py")
    with col_act2:
        if st.button("Open Inbox Agent (Gmail & MCP)", type="secondary", use_container_width=True):
            st.switch_page("pages/8_Inbox_Agent.py")
