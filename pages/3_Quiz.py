"""Adaptive lecture quiz interface."""

from __future__ import annotations

import streamlit as st

from core.quiz import QuizError, answer_is_correct, generate_quiz, load_quiz
from core.ui import configure_page, empty_state, lecture_selector, lecture_strip, page_header, require_ready_lectures, source_chip, stat_card


configure_page("Quiz", "Q")
page_header("03 / PRACTICE", "Adaptive quiz", "Practice what the lecture actually says, then use misses to shape your next round.")
lectures = require_ready_lectures()
lecture_id = lecture_selector(lectures)
lecture_strip(lectures[lecture_id], show_player_link=False)

with st.container(border=True):
    st.markdown('<div class="kicker">QUIZ SETTINGS</div>', unsafe_allow_html=True)
    settings = st.columns(3)
    count = settings[0].slider("Questions", 1, 20, 5)
    difficulty = settings[1].selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
    bloom_level = settings[2].selectbox("Bloom level", ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"], index=1)
    weak_concepts = st.session_state.setdefault("quiz_weak_concepts", {}).setdefault(lecture_id, [])
    if weak_concepts:
        st.caption("Focus from your last attempt: " + ", ".join(weak_concepts))
    if st.button("Generate quiz", type="primary"):
        try:
            with st.spinner("Generating and verifying lecture questions..."):
                st.session_state["active_quiz"] = generate_quiz(lecture_id, count=count, difficulty=difficulty, bloom_level=bloom_level, weak_concepts=weak_concepts)
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_result"] = None
        except QuizError as exc:
            st.error(f"Could not generate quiz: {exc}")

quiz = st.session_state.get("active_quiz") or load_quiz(lecture_id, count, difficulty, bloom_level)
if quiz is None:
    empty_state("No quiz loaded", "Choose settings and generate a grounded quiz to begin.")
    st.stop()

st.progress(len(st.session_state.get("quiz_answers", {})) / len(quiz.questions), text="Answer progress")
answers = st.session_state.setdefault("quiz_answers", {})
for index, question in enumerate(quiz.questions):
    with st.container(border=True):
        st.markdown(f'<div class="kicker">QUESTION {index + 1:02d}</div>', unsafe_allow_html=True)
        st.subheader(question.question)
        st.caption(f"{question.difficulty.title()} · {question.bloom_level} · {question.concept}")
        if question.type in {"mcq", "true_false"}:
            saved_val = answers.get(index)
            opt_idx = question.options.index(saved_val) if saved_val in question.options else None
            user_choice = st.radio(
                "Answer",
                question.options,
                index=opt_idx,
                key=f"answer-{lecture_id}-{index}",
                label_visibility="collapsed",
            )
            if user_choice is not None:
                answers[index] = user_choice
        elif question.type == "fill_blank":
            answers[index] = st.text_input("Answer", key=f"answer-{lecture_id}-{index}")
        else:
            answers[index] = st.text_area("Answer", key=f"answer-{lecture_id}-{index}")
        source_chip(question.source_reference.timestamp, question.source_reference.slide, key=f"quiz-source-{lecture_id}-{index}")

if st.button("Submit answers", type="primary"):
    unanswered = sum(not str(answers.get(index, "")).strip() for index in range(len(quiz.questions)))
    if unanswered:
        st.warning(f"! {unanswered} question(s) are unanswered. They will be marked incorrect.")
    score, missed, results = 0, [], []
    for index, question in enumerate(quiz.questions):
        correct = answer_is_correct(question, answers.get(index, ""))
        results.append(correct)
        score += int(correct)
        if not correct:
            missed.append(question.concept)
    st.session_state["quiz_result"] = {"score": score, "results": results, "missed": missed}
    st.session_state["quiz_weak_concepts"][lecture_id] = list(dict.fromkeys(missed))

result = st.session_state.get("quiz_result")
if result:
    st.divider()
    a, b = st.columns(2)
    with a: stat_card("Score", f"{result['score']} / {len(quiz.questions)}", "Keep going; misses become your next focus.")
    with b: stat_card("Weak concepts", len(result["missed"]), "Review the source before retrying.")
    for index, question in enumerate(quiz.questions):
        if result["results"][index]:
            st.success(f"✓ Question {index + 1}: correct")
        else:
            st.error(f"× Question {index + 1}: incorrect")
            with st.expander("See explanation"):
                st.write(question.explanation)
                st.write(f"Correct answer: {question.correct_answer}")
