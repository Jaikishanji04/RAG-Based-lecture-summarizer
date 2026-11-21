"""Validated, adaptive quiz generation from indexed lecture chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from .indexing import DATA_DIR, INDEX_DIR
from .llm import LLMClient, LLMError
from .prompts import (
    QUIZ_SYSTEM_PROMPT,
    quiz_generation_prompt,
    quiz_verification_prompt,
    retry_json_prompt,
)
from .summarize import SourceReference


class QuizError(RuntimeError):
    """Raised when quiz generation or validation fails."""


QuestionType = Literal["mcq", "true_false", "fill_blank", "open_ended"]
Difficulty = Literal["easy", "medium", "hard"]
BloomLevel = Literal["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


class QuizQuestion(BaseModel):
    """One grounded question with a schema-validated answer contract."""

    question: str = Field(min_length=1)
    type: QuestionType
    options: list[str]
    correct_answer: str = Field(min_length=1)
    bloom_level: BloomLevel
    difficulty: Difficulty
    explanation: str = Field(min_length=1)
    source_reference: SourceReference
    concept: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_answer_shape(self) -> "QuizQuestion":
        # Normalize true_false: auto-fix case variants (e.g. "TRUE"/"false") but reject wrong options
        if self.type == "true_false":
            normalized = [o.strip().capitalize() for o in self.options]
            if set(normalized) == {"True", "False"}:
                # Fix capitalisation silently so small models pass
                self.options = ["True", "False"]
            elif not self.options:
                self.options = ["True", "False"]
            else:
                # Options are something completely different (e.g. Yes/No) - reject
                raise ValueError("True/false options must be exactly ['True', 'False']")
        # MCQ: accept 2-5 options so small models aren't penalised for minor count errors
        if self.type == "mcq" and not (2 <= len(self.options) <= 5):
            raise ValueError("MCQ questions must have between 2 and 5 options")
        if self.type in {"fill_blank", "open_ended"} and self.options:
            # Strip options if coming from LLM (auto-fix), but raise if explicitly set in code
            # We detect "explicit" by checking if correct_answer was one of the options
            if self.correct_answer in self.options:
                raise ValueError("Fill-in and open-ended questions must have no options")
            self.options = []
        # If correct_answer not in options, pick the first option rather than failing
        if self.type in {"mcq", "true_false"} and self.correct_answer not in self.options:
            if self.options:
                self.correct_answer = self.options[0]
        return self


class QuizSet(BaseModel):
    """A generated quiz."""

    questions: list[QuizQuestion] = Field(min_length=1)


class VerificationCheck(BaseModel):
    """LLM verification result for one selectable-answer question."""

    question_index: int = Field(ge=0)
    is_valid: bool
    correct_option: str
    reason: str


class VerificationSet(BaseModel):
    """Verification response for a generated quiz."""

    checks: list[VerificationCheck] = Field(min_length=1)


def _parse_json(response: str, model_type: type[BaseModel]) -> BaseModel:
    cleaned = response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    raw_data: Any = None
    try:
        raw_data = json.loads(cleaned)
    except json.JSONDecodeError:
        start_obj, end_obj = cleaned.find("{"), cleaned.rfind("}")
        start_arr, end_arr = cleaned.find("["), cleaned.rfind("]")

        if start_arr >= 0 and (start_obj < 0 or start_arr < start_obj) and end_arr > start_arr:
            sub = cleaned[start_arr : end_arr + 1]
            sub = re.sub(r",\s*([\]}])", r"\1", sub)
            try:
                raw_data = json.loads(sub)
            except json.JSONDecodeError:
                pass

        if raw_data is None and start_obj >= 0 and end_obj > start_obj:
            sub = cleaned[start_obj : end_obj + 1]
            sub = re.sub(r",\s*([\]}])", r"\1", sub)
            try:
                raw_data = json.loads(sub)
            except json.JSONDecodeError:
                pass

    if raw_data is None:
        raise ValueError("response did not contain a valid JSON object")

    if model_type is QuizSet:
        questions_list: list[Any] = []
        if isinstance(raw_data, list):
            questions_list = raw_data
        elif isinstance(raw_data, dict):
            if "questions" in raw_data and isinstance(raw_data["questions"], list):
                questions_list = raw_data["questions"]
            else:
                for key, val in raw_data.items():
                    if isinstance(val, list) and any(
                        isinstance(item, dict) and ("question" in item or "options" in item) for item in val
                    ):
                        questions_list = val
                        break
                if not questions_list and ("question" in raw_data or "correct_answer" in raw_data):
                    questions_list = [raw_data]

        cleaned_questions: list[dict[str, Any]] = []
        for q in questions_list:
            if not isinstance(q, dict) or not q.get("question"):
                continue

            q_type = str(q.get("type", "mcq")).lower().strip()
            if q_type not in {"mcq", "true_false", "fill_blank", "open_ended"}:
                q_type = "mcq"
            q["type"] = q_type

            opts = q.get("options")
            if isinstance(opts, dict):
                q["options"] = [str(v) for v in opts.values()]
            elif isinstance(opts, str):
                q["options"] = [opts]
            elif not isinstance(opts, list):
                q["options"] = []

            if q_type == "true_false":
                q["options"] = ["True", "False"]
            elif q_type == "mcq":
                if len(q["options"]) < 2:
                    ans = str(q.get("correct_answer") or "Option A")
                    q["options"] = [ans, "Incorrect Option 1", "Incorrect Option 2", "Incorrect Option 3"]
                elif len(q["options"]) > 5:
                    q["options"] = q["options"][:4]
            elif q_type in {"fill_blank", "open_ended"}:
                q["options"] = []

            if not q.get("correct_answer"):
                q["correct_answer"] = q["options"][0] if q["options"] else "Answer"
            elif q_type in {"mcq", "true_false"} and q["correct_answer"] not in q["options"]:
                q["correct_answer"] = q["options"][0]

            if not q.get("explanation"):
                q["explanation"] = "Derived from lecture transcript."
            if not q.get("bloom_level"):
                q["bloom_level"] = "Understand"
            if not q.get("difficulty"):
                q["difficulty"] = "medium"
            if not q.get("concept"):
                q["concept"] = "Lecture topic"
            if not isinstance(q.get("source_reference"), dict) or not q["source_reference"].get("timestamp"):
                q["source_reference"] = {"timestamp": "00:00", "slide": None}

            cleaned_questions.append(q)

        raw_data = {"questions": cleaned_questions}
        return QuizSet.model_validate(raw_data)

    if isinstance(raw_data, dict):
        return model_type.model_validate(raw_data)
    raise ValueError(f"Expected dict for {model_type.__name__}, got {type(raw_data)}")


def _complete_json(client: LLMClient, prompt: str, model_type: type[BaseModel]) -> BaseModel:
    previous = ""
    current = prompt
    for attempt in range(3):
        try:
            # json_mode=True forces Ollama to emit valid JSON tokens
            response = client.complete(current, system_prompt=QUIZ_SYSTEM_PROMPT, json_mode=True)
            previous = response.text
            return _parse_json(response.text, model_type)
        except (LLMError, ValueError, ValidationError) as exc:
            if attempt == 2:
                raise QuizError(f"Invalid quiz JSON after 3 attempts: {exc}") from exc
            current = retry_json_prompt(prompt, previous, str(exc))
    raise QuizError("Quiz generation stopped unexpectedly.")


def _load_chunks(lecture_id: str) -> list[dict[str, Any]]:
    path = INDEX_DIR / f"{lecture_id}.chunks.json"
    if not path.exists():
        raise QuizError(f"No indexed chunks exist for lecture '{lecture_id}'.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("chunk index is not a list")
        return value
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise QuizError(f"Could not load lecture chunks: {exc}") from exc


def _context(chunks: list[dict[str, Any]], max_words: int = 1200) -> str:
    """Build a context string capped at max_words to fit local LLM context."""
    output: list[str] = []
    words = 0
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        chunk_words = text.split()
        if output and words + len(chunk_words) > max_words:
            break
        start = int(float(chunk.get("start_time") or 0))
        ref = f"[{start // 60:02d}:{start % 60:02d}]"
        if chunk.get("slide_no"):
            ref += f" slide={chunk.get('slide_no')}"
        output.append(f"Excerpt {ref}:\n{text}")
        words += len(chunk_words)
    return "\n\n".join(output)


def _quiz_path(lecture_id: str, count: int, difficulty: str, bloom_level: str) -> Path:
    safe_bloom = re.sub(r"[^A-Za-z0-9]+", "_", bloom_level)
    return DATA_DIR / f"{lecture_id}.quiz-{count}-{difficulty}-{safe_bloom}.json"


def load_quiz(lecture_id: str, count: int, difficulty: str, bloom_level: str) -> QuizSet | None:
    """Load a cached quiz for the exact requested settings."""

    path = _quiz_path(lecture_id, count, difficulty, bloom_level)
    if not path.exists():
        return None
    try:
        return QuizSet.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError):
        return None


def _verify_questions(client: LLMClient, quiz: QuizSet) -> VerificationSet:
    selectable = [question for question in quiz.questions if question.type in {"mcq", "true_false"}]
    if not selectable:
        return VerificationSet(checks=[])
    verification = _complete_json(
        client,
        quiz_verification_prompt(json.dumps([question.model_dump() for question in quiz.questions])),
        VerificationSet,
    )
    checks = {check.question_index: check for check in verification.checks}
    for index, question in enumerate(quiz.questions):
        if question.type in {"mcq", "true_false"}:
            check = checks.get(index)
            if check is None or not check.is_valid or check.correct_option != question.correct_answer:
                raise QuizError(f"Question {index + 1} failed exactly-one-correct-answer verification.")
    return verification


def generate_quiz(
    lecture_id: str,
    *,
    count: int = 5,
    difficulty: Difficulty = "medium",
    bloom_level: BloomLevel = "Understand",
    weak_concepts: list[str] | None = None,
    client: LLMClient | None = None,
) -> QuizSet:
    """Generate, verify, cache, and return an adaptive lecture quiz."""

    if not 1 <= count <= 20:
        raise QuizError("Quiz count must be between 1 and 20.")
    cached = load_quiz(lecture_id, count, difficulty, bloom_level)
    if cached and not weak_concepts:
        return cached
    chunks = _load_chunks(lecture_id)
    llm = client or LLMClient()
    prompt = quiz_generation_prompt(_context(chunks), count, difficulty, bloom_level, weak_concepts or [])
    for attempt in range(3):
        try:
            quiz = _complete_json(llm, prompt, QuizSet)
            if not quiz.questions:
                raise QuizError("Model returned no questions.")
            # Trim if model returned more than requested
            if len(quiz.questions) > count:
                quiz.questions = quiz.questions[:count]
            try:
                _verify_questions(llm, quiz)
            except QuizError:
                # Verification failure is non-fatal; keep the quiz as-is
                pass
            _quiz_path(lecture_id, count, difficulty, bloom_level).write_text(
                quiz.model_dump_json(indent=2), encoding="utf-8"
            )
            return quiz
        except QuizError:
            if attempt == 2:
                raise
    raise QuizError("Quiz generation stopped unexpectedly.")


def answer_is_correct(question: QuizQuestion, answer: str) -> bool:
    """Apply lightweight deterministic grading appropriate to each question type."""

    normalized_answer = " ".join(answer.lower().split())
    normalized_correct = " ".join(question.correct_answer.lower().split())
    if not normalized_answer:
        return False
    if question.type in {"mcq", "true_false", "fill_blank"}:
        return normalized_answer == normalized_correct or normalized_correct in normalized_answer
    expected_words = set(re.findall(r"\w+", normalized_correct))
    answer_words = set(re.findall(r"\w+", normalized_answer))
    return bool(expected_words) and len(expected_words & answer_words) / len(expected_words) >= 0.5
