"""Lecture playback, transcript navigation, and grounded summary."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

from core.ingestion import IngestionError
from core.summarize import SummaryError, load_summary, summarize_lecture
from core.ui import configure_page, empty_state, fmt_ts, lecture_selector, lecture_strip, page_header, require_ready_lectures, stat_card


def _clean_summary_text(text: str) -> str:
    cleaned = re.sub(r"\[\d{1,2}:\d{2}(?:\s*\|\s*slide\s*\d+)?\]", "", text)
    cleaned = re.sub(r"\(\d{1,2}:\d{2}\)", "", cleaned)
    cleaned = re.sub(r"^\s*[-*•]\s*", "", cleaned)
    return cleaned.strip()


configure_page("Lecture Player", "▶")
page_header("02 / LISTEN + SYNTHESIZE", "Lecture player", "Move through the source, jump from evidence, and turn the lecture into a compact study guide.")
lectures = require_ready_lectures()
lecture_id = lecture_selector(lectures)
lecture = lectures[lecture_id]
lecture_strip(lecture)

player_time = int(st.session_state.get("player_time", 0))
source_path = Path(lecture["path"]) if lecture.get("path") else None
left, right = st.columns([1.5, 1], gap="large")
with left:
    if source_path and source_path.suffix.lower() in {".mp4", ".mov"}:
        st.video(str(source_path), start_time=player_time)
    elif source_path and source_path.suffix.lower() in {".mp3", ".wav", ".m4a"}:
        st.audio(str(source_path), start_time=player_time)
    else:
        st.markdown('<div class="card source-card"><h3>Document lecture</h3><p>This source is made of pages or slides. Use the transcript and citations below to study it.</p></div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="card generated-card"><div class="ai-kicker">STUDY ACTIONS</div><h3>What next?</h3>', unsafe_allow_html=True)
    if st.button("Generate Summary", type="primary", use_container_width=True):
        st.session_state["ui_generate_summary"] = True
    actions = [("Quiz", "pages/3_Quiz.py"), ("Ask the lecture", "pages/4_Ask_Lecture_QA.py"), ("Concept graph", "pages/5_Concept_Graph.py"), ("Export", "pages/6_Export.py")]
    for label, page in actions:
        if st.button(label, use_container_width=True):
            st.switch_page(page)
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
        summary = summarize_lecture(lecture, progress_callback=lambda value, message: (progress.progress(value, text=message), status.update(label=message)))
        status.update(label="Summary ready", state="complete")
    except (SummaryError, IngestionError) as exc:
        status.update(label="Summary failed", state="error")
        st.error(f"Could not generate the summary: {exc}")
        if st.button("Retry summary"):
            st.session_state["ui_generate_summary"] = True
            st.rerun()

transcript_tab, summary_tab = st.tabs(["Transcript", "Summary"])
with transcript_tab:
    if transcript:
        st.markdown('<div class="card source-card transcript-scroll">', unsafe_allow_html=True)
        for index, segment in enumerate(transcript):
            start = float(segment.get("start", 0))
            active = abs(start - player_time) <= 2
            row = st.container()
            if active:
                row.markdown('<div class="active-segment">', unsafe_allow_html=True)
            timestamp, text = row.columns([.22, .78])
            with timestamp:
                if st.button(fmt_ts(start), key=f"jump-{lecture_id}-{index}", use_container_width=True):
                    st.session_state["player_time"] = int(start)
                    st.rerun()
            with text:
                st.write(str(segment.get("text", "")))
            if active:
                row.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        empty_state("No transcript available", "This document lecture is represented by indexed pages or slides.")

with summary_tab:
    if not summary:
        empty_state("Your study guide is waiting", "Generate a grounded summary to see a TL;DR, key points, and detailed notes.", "Generate Summary")
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
