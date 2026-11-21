"""Upload and process lecture sources."""

from pathlib import Path
from time import monotonic

import streamlit as st

from core.ingestion import IngestionError, SUPPORTED_EXTENSIONS, ingest_source, load_lectures, save_lectures
from core.indexing import IndexingError, index_lecture
from core.ui import configure_page, empty_state, page_header, stage_tracker, stat_card


configure_page("Upload & Process", "↑")
page_header("01 / INTAKE", "Bring a lecture to the desk", "Add a source once. Completed lectures are cached and become searchable across every study tool.")

left, right = st.columns(2)
with left:
    st.markdown('<div class="card source-card"><div class="kicker">SOURCE MATERIAL</div><h3>Upload a file</h3><p>Audio, video, PDF, and PowerPoint are supported.</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a lecture file", type=[suffix[1:] for suffix in sorted(SUPPORTED_EXTENSIONS)], label_visibility="collapsed")
    st.caption("Accepted: " + ", ".join(sorted(suffix.upper() for suffix in SUPPORTED_EXTENSIONS)))
    st.markdown("</div>", unsafe_allow_html=True)
with right:
    st.markdown('<div class="card source-card"><div class="kicker">SOURCE MATERIAL</div><h3>Use a YouTube lecture</h3><p>Paste a public URL. Authentication-required videos may need browser cookies.</p>', unsafe_allow_html=True)
    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

source_label = uploaded_file.name if uploaded_file else youtube_url.strip()
retry_source = st.session_state.pop("ui_retry_source", None)
if retry_source:
    source_label = retry_source
st.markdown("<br>", unsafe_allow_html=True)
st.caption("Choose one source. Processing can take several minutes on the first run.")
process_requested = st.button("Process lecture", type="primary", disabled=not bool(source_label), use_container_width=True) or bool(retry_source)
stages = ["Preparing source", "Transcribing or extracting text", "Creating chunks", "Building search index", "Lecture ready"]
if not source_label:
    stage_tracker(stages)

if process_requested:
    progress = st.progress(0.0)
    status = st.status("Preparing source", expanded=True)
    started_at = monotonic()
    current_progress = [0.0]

    def update_progress(value: float, message: str) -> None:
        bounded = max(0.0, min(value, 1.0))
        current_progress[0] = bounded
        elapsed = monotonic() - started_at
        hint = f" · about {max(0, round(elapsed / bounded - elapsed))}s remaining" if bounded > 0.02 and elapsed > 1 else ""
        progress.progress(bounded, text=f"{message}{hint}")
        status.update(label=message)

    try:
        if retry_source:
            lecture = ingest_source(Path(retry_source), progress_callback=update_progress)
        elif uploaded_file:
            upload_path = Path("data") / "incoming" / Path(uploaded_file.name).name
            upload_path.parent.mkdir(parents=True, exist_ok=True)
            upload_path.write_bytes(uploaded_file.getbuffer())
            lecture = ingest_source(upload_path, progress_callback=update_progress)
        else:
            lecture = ingest_source(youtube_url, progress_callback=update_progress)
        index = index_lecture(lecture, progress_callback=lambda value, message: update_progress(0.65 + value * 0.3, message))
        lecture["index"] = index
        lectures = load_lectures()
        lectures[lecture["lecture_id"]] = lecture
        save_lectures(lectures)
        st.session_state["ui_active_lecture_id"] = lecture["lecture_id"]
        status.update(label="Lecture ready", state="complete")
        progress.progress(1.0, text="Lecture ready")
        st.success(f"✓ {lecture['title']} is ready to study.")
        a, b, c = st.columns(3)
        with a: stat_card("Source", lecture.get("source_type", "—").upper())
        with b: stat_card("Status", "Ready")
        with c: stat_card("Indexed chunks", index.get("chunk_count", "—"))
        if st.button("Open in Lecture Player", type="primary"):
            st.switch_page("pages/2_Lecture_Player.py")
    except (IngestionError, IndexingError) as exc:
        status.update(label="Processing failed", state="error")
        progress.progress(current_progress[0], text="Processing stopped")
        st.error(f"× Could not process the lecture: {exc}")
        st.caption("Nothing is added as ready until every stage succeeds.")
        retry, change = st.columns(2)
        with retry:
            if st.button("Retry", key="retry_processing_button"):
                st.rerun()
        with change:
            st.button("Change source", key="change_source_button")
