"""Chat interface for grounded lecture Q&A."""

from __future__ import annotations

import streamlit as st

from core.qa import QAError, QAResponse, answer_question
from core.ui import configure_page, empty_state, lecture_selector, lecture_strip, page_header, require_ready_lectures, source_chip


configure_page("Ask Lecture", "?")
page_header("04 / ASK", "Ask the lecture", "Search the indexed source conversationally. Every supported answer points back to its evidence.")
lectures = require_ready_lectures()
lecture_id = lecture_selector(lectures)
lecture_strip(lectures[lecture_id], show_player_link=False)
st.success("✓ Grounded in this lecture")

all_messages = st.session_state.setdefault("qa_messages", {})
messages = all_messages.setdefault(lecture_id, [])
if not messages:
    st.markdown("Try a question")
    examples = ["What is the main idea of this lecture?", "Explain the second important concept.", "Summarize the section around 05:30."]
    cols = st.columns(3)
    for col, example in zip(cols, examples):
        if col.button(example, key=f"example-{example}"):
            st.session_state["ui_example_question"] = example

for message_index, message in enumerate(messages):
    with st.chat_message(str(message["role"])):
        st.write(str(message["content"]))
        response = message.get("response")
        if isinstance(response, QAResponse):
            if response.grounded:
                for citation_index, citation in enumerate(response.citations):
                    source_chip(citation.timestamp, citation.slide, key=f"qa-source-{lecture_id}-{message_index}-{citation_index}")
            else:
                st.warning("! This answer is not supported by the indexed lecture context.")

question = st.chat_input("Ask something about this lecture") or st.session_state.pop("ui_example_question", None)
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the lecture..."):
                response = answer_question(lecture_id, question)
            st.write(response.answer)
            if response.grounded:
                st.caption("Sources")
                for index, citation in enumerate(response.citations):
                    source_chip(citation.timestamp, citation.slide, key=f"new-qa-source-{lecture_id}-{index}")
            else:
                st.warning("! Not in lecture: the retrieved context did not support this answer.")
            messages.append({"role": "assistant", "content": response.answer, "response": response})
        except QAError as exc:
            st.error(f"Could not answer this question: {exc}")
            messages.append({"role": "assistant", "content": f"Q&A error: {exc}"})
