"""Shared Streamlit visual system for the lecture workspace."""

from __future__ import annotations

from html import escape
from typing import Any, Iterable

import streamlit as st

from core.ingestion import load_lectures


NAVIGATION = {
    "WORKBENCH": [
        ("Study Workspace", "pages/2_Lecture_Player.py", "01"),
    ],
    "AGENT": [
        ("Inbox Agent", "pages/8_Inbox_Agent.py", "02"),
    ],
}

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --paper:#F6F0E4; --card:#FFFBF3; --ink:#1E2A2A; --muted:#6B6F66; --border:#E3D9C3; --sidebar:#12343B; --coral:#C94F32; --teal:#1F8A8A; --ok:#2E8B57; --warn:#C98A12; --err:#C0392B; }
html, body, [class*="css"] { font-family:'Space Grotesk',sans-serif; }
.stApp, [data-testid="stAppViewContainer"] { background:var(--paper); color:var(--ink); }
[data-testid="stMainBlockContainer"] { max-width:1320px; padding:2.5rem 3rem 4rem; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:var(--sidebar); }
[data-testid="stSidebar"] * { color:#F6F0E4; }
[data-testid="stSidebarNav"] { display:none; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#C4D2CD; }
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] { border-radius:10px; min-height:44px; padding:.65rem .7rem; }
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover { background:rgba(255,255,255,.1); }
h1,h2,h3 { color:var(--ink); letter-spacing:-.02em; }
h1 { font-size:3.2rem; line-height:1.02; } h2 { margin-top:1.8rem; } h3 { margin-top:1.1rem; }
[data-testid="stCaptionContainer"] { color:var(--muted); }
.stButton > button, .stDownloadButton > button { min-height:44px; border:1px solid var(--coral); border-radius:10px; background:var(--coral); color:#fff; font-weight:600; }
.stButton > button:hover, .stDownloadButton > button:hover { background:#A83F28; border-color:#A83F28; transform:translateY(-1px); }
.stButton > button:focus-visible, input:focus-visible, textarea:focus-visible { outline:3px solid rgba(31,138,138,.45); outline-offset:2px; }
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"], [data-testid="stExpander"], [data-testid="stStatusWidget"] { background:var(--card); border:1px solid var(--border); border-radius:14px; box-shadow:0 8px 24px rgba(30,42,42,.06); }
[data-testid="stMetric"] { padding:1rem; }
[data-testid="stMetricValue"] { color:var(--teal); }
[data-testid="stFileUploaderDropzone"], [data-baseweb="select"] > div, [data-testid="stTextInput"] > div > div, [data-testid="stTextArea"] > div > div { background:var(--card); border-color:var(--border); border-radius:10px; }
[data-testid="stTabs"] [role="tab"] { min-height:44px; color:var(--muted); } [data-testid="stTabs"] [aria-selected="true"] { color:var(--coral); }
.brand { padding:1rem .4rem 1.25rem; border-bottom:1px solid rgba(255,255,255,.15); margin-bottom:.8rem; }
.brand-mark, .kicker, .meta { color:#F19A7F; font-family:'DM Mono',monospace; font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; }
.brand-title { color:#fff; font-size:1.35rem; font-weight:700; margin-top:.45rem; line-height:1.05; }
.brand-copy { color:#B7CAC4; font-size:.78rem; margin-top:.6rem; }
.sidebar-footer { border:1px solid rgba(255,255,255,.15); border-radius:12px; padding:.8rem; margin-top:1.2rem; background:rgba(255,255,255,.04); }
.sidebar-footer small { color:#AFC4BE; font-family:'DM Mono',monospace; } .sidebar-footer strong { display:block; color:#fff; margin-top:.2rem; }
.page-header { border-bottom:1px solid var(--border); padding-bottom:1.4rem; margin-bottom:1.5rem; }
.page-header-title { font-size:3.1rem; font-weight:700; line-height:1.02; } .page-header-copy { color:var(--muted); max-width:760px; margin-top:.65rem; line-height:1.5; }
.rule { display:flex; gap:.4rem; margin-top:1rem; } .rule span { height:4px; border-radius:4px; } .rule span:first-child { width:52px; background:var(--coral); } .rule span:last-child { width:18px; background:var(--teal); }
.card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1rem 1.1rem; box-shadow:0 8px 24px rgba(30,42,42,.06); }
.source-card { border-left:4px solid var(--teal); } .generated-card { border-left:4px solid var(--coral); }
.ai-kicker { color:var(--coral); font-family:'DM Mono',monospace; font-size:.65rem; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.4rem; }
.chip { display:inline-block; border:1px solid var(--border); border-radius:999px; padding:.25rem .55rem; margin:.15rem .2rem .15rem 0; color:var(--muted); font-family:'DM Mono',monospace; font-size:.68rem; }
.status-row { padding:.45rem .65rem; border-bottom:1px solid var(--border); } .status-row:last-child { border-bottom:0; }
.transcript-scroll { max-height:560px; overflow-y:auto; padding-right:.35rem; }
.active-segment { background:#F4D8CC; border-radius:8px; padding:.35rem; }
@media (max-width:900px) { [data-testid="stMainBlockContainer"] { padding:1.5rem 1rem 3rem; } .page-header-title { font-size:2.3rem; } h1 { font-size:2.35rem; } }
</style>
"""


def setup_page(title: str, icon: str = "📚") -> None:
    """Configure one page and install the shared visual system."""

    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    _render_sidebar()


def configure_page(title: str, icon: str) -> None:
    """Backward-compatible alias used by all existing pages."""

    setup_page(title, icon)


def _render_sidebar() -> None:
    lectures = ready_lectures()
    history = sorted(
        load_lectures().items(),
        key=lambda item: str(item[1].get("created_at", "")),
        reverse=True,
    )
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">FIELD NOTES / RAG</div>'
            '<div class="brand-title">Lecture<br>Workbench</div>'
            '<div class="brand-copy">A calm desk for turning long lectures into searchable study material.</div></div>',
            unsafe_allow_html=True,
        )
        for group, links in NAVIGATION.items():
            st.markdown(f"**{group}**")
            for label, path, number in links:
                st.page_link(path, label=f"{number}  {label}")
        if history:
            with st.expander(f"Lecture library · {len(history)}", expanded=True):
                for lecture_id, lecture in history:
                    title = escape(str(lecture.get("title", lecture_id)))
                    status = str(lecture.get("status", "unknown")).lower()
                    icon = {"ready": "✓", "failed": "×", "processing": "●"}.get(status, "○")
                    active = lecture_id == st.session_state.get("ui_active_lecture_id")
                    label = f"{icon} {'• ' if active else ''}{title[:30]}"
                    if status == "ready":
                        if st.button(label, key=f"sidebar-open-{lecture_id}", use_container_width=True):
                            st.session_state["ui_active_lecture_id"] = lecture_id
                            st.switch_page("pages/2_Lecture_Player.py")
                    else:
                        st.caption(f"{icon} {title}")
        title = next(iter(lectures.values())).get("title", "No lecture loaded") if lectures else "No lecture loaded"
        state = "Ready to study" if lectures else "Upload a lecture to begin"
        st.markdown(
            f'<div class="sidebar-footer"><small>ACTIVE LECTURE</small><strong>{escape(str(title))}</strong>'
            f'<small>{escape(state)}</small></div>',
            unsafe_allow_html=True,
        )


def ready_lectures() -> dict[str, dict]:
    lectures = {
        key: value
        for key, value in load_lectures().items()
        if value.get("status") == "ready" and value.get("index")
    }
    return dict(
        sorted(
            lectures.items(),
            key=lambda item: str(item[1].get("created_at", "")),
            reverse=True,
        )
    )


def lecture_selector(lectures: dict[str, dict], *, key: str = "ui_lecture_selector") -> str:
    """Select a lecture while keeping the newest or explicitly active one selected."""

    options = list(lectures)
    active_id = st.session_state.get("ui_active_lecture_id")
    index = options.index(active_id) if active_id in options else 0
    selected = st.selectbox(
        "Lecture",
        options,
        index=index,
        format_func=lambda item: lectures[item].get("title", item),
        key=key,
    )
    st.session_state["ui_active_lecture_id"] = selected
    return selected


def require_ready_lectures() -> dict[str, dict]:
    lectures = ready_lectures()
    if not lectures:
        empty_state("No lecture is ready yet", "Process an audio, video, PDF, PPTX, or YouTube source to unlock the study desk.", "Upload a lecture", "pages/1_Upload_and_Process.py")
        st.stop()
    return lectures


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="page-header"><div class="kicker">{escape(kicker)}</div>'
        f'<div class="page-header-title">{escape(title)}</div><div class="page-header-copy">{escape(description)}</div>'
        '<div class="rule"><span></span><span></span></div></div>',
        unsafe_allow_html=True,
    )


def lecture_strip(lecture: dict[str, Any], *, show_player_link: bool = True) -> None:
    title = escape(str(lecture.get("title", "Untitled lecture")))
    source = escape(str(lecture.get("source_type", "source")).upper())
    chunks = lecture.get("index", {}).get("chunk_count", "—")
    st.markdown(f'<div class="card"><strong>{title}</strong><span class="chip">{source}</span><span class="chip">✓ Ready</span><span class="chip">{chunks} chunks</span></div>', unsafe_allow_html=True)
    if show_player_link and st.button("Back to player", key=f"back-player-{lecture.get('lecture_id')}"):
        st.switch_page("pages/2_Lecture_Player.py")


def empty_state(title: str, text: str, button_label: str | None = None, page: str | None = None) -> None:
    st.markdown(f'<div class="card"><h3>{escape(title)}</h3><p>{escape(text)}</p></div>', unsafe_allow_html=True)
    if button_label and page and st.button(button_label, type="primary", key=f"empty-{button_label}"):
        st.switch_page(page)


def stage_tracker(stages: Iterable[str], active: int | None = None, failed: int | None = None) -> None:
    rows = []
    for index, label in enumerate(stages):
        state = "failed" if failed == index else "active" if active == index else "done" if active is not None and index < active else "pending"
        icon = {"done": "✓", "active": "●", "failed": "!", "pending": "○"}[state]
        rows.append(f'<div class="status-row"><strong>{icon}</strong> {escape(label)} <span class="meta">{state}</span></div>')
    st.markdown(f'<div class="card">{"".join(rows)}</div>', unsafe_allow_html=True)


def source_chip(timestamp: str | None = None, slide: int | None = None, *, key: str, navigate_to: str | None = None) -> None:
    label = f"Watch {timestamp}" if timestamp else f"Slide {slide}"
    if timestamp and slide:
        label = f"Watch {timestamp} · slide {slide}"
    if st.button(label, key=key):
        if timestamp:
            minutes, seconds = timestamp.split(":")
            st.session_state["player_time"] = int(minutes) * 60 + int(seconds)
        if navigate_to:
            st.session_state["ui_navigate_to"] = navigate_to


def stat_card(label: str, value: str | int, detail: str = "") -> None:
    st.markdown(f'<div class="card"><div class="meta">{escape(label)}</div><h2>{escape(str(value))}</h2><small>{escape(detail)}</small></div>', unsafe_allow_html=True)


def callout(kind: str, text: str) -> None:
    icons = {"success": "✓", "warning": "!", "error": "×", "info": "i"}
    st.info(f"{icons.get(kind, 'i')}  {text}") if kind == "info" else st.warning(f"{icons.get(kind, '!')}  {text}") if kind == "warning" else st.error(f"{icons.get(kind, '×')}  {text}") if kind == "error" else st.success(f"{icons.get(kind, '✓')}  {text}")


def badge(text: str, kind: str = "info") -> None:
    st.markdown(f'<span class="chip">{escape(text)}</span>', unsafe_allow_html=True)


def fmt_ts(seconds: float | int | None) -> str:
    total = max(0, int(seconds or 0))
    return f"{total // 60:02d}:{total % 60:02d}"


def section_header(kicker: str, title: str, description: str = "") -> None:
    st.markdown(f'<div class="kicker">{escape(kicker)}</div><h3>{escape(title)}</h3>{f"<p>{escape(description)}</p>" if description else ""}', unsafe_allow_html=True)
