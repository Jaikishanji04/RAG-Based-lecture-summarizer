import pytest
from pydantic import ValidationError

from core.qa import QAResponse, _context
from core.retrieval import RetrievalResult
from core.summarize import SourceReference


def test_ungrounded_answer_has_no_citations() -> None:
    response = QAResponse(answer="not in lecture", citations=[], grounded=False)
    assert response.answer == "not in lecture"


def test_grounded_answer_requires_citation() -> None:
    with pytest.raises(ValidationError):
        QAResponse(answer="The answer", citations=[], grounded=True)


def test_context_exposes_only_retrieved_source_references() -> None:
    result = RetrievalResult(
        chunk_id="chunk-1",
        text="Retrieval combines lexical and dense search.",
        score=0.9,
        metadata={"start_time": 72, "slide_no": None},
    )
    context, allowed = _context([result])
    assert "timestamp=01:12" in context
    assert ("01:12", None) in allowed
