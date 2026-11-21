import pytest
from pydantic import ValidationError

from core.summarize import FinalSummary, _parse_json_response, render_bullet
from core.prompts import concept_extraction_prompt, final_summary_prompt


def _summary_payload() -> dict:
    bullet = {"text": "Grounded point", "source_reference": {"timestamp": "00:12", "slide": None}}
    return {
        "tldr": [bullet, bullet, bullet],
        "key_points": [bullet],
        "detailed_notes": [bullet],
    }


def test_summary_json_requires_three_cited_tldr_lines() -> None:
    summary = _parse_json_response(__import__("json").dumps(_summary_payload()), FinalSummary)
    assert len(summary.tldr) == 3
    assert render_bullet(summary.tldr[0]).endswith("[00:12]")


def test_summary_json_rejects_invalid_timestamp() -> None:
    payload = _summary_payload()
    payload["tldr"][0]["source_reference"]["timestamp"] = "unknown"
    with pytest.raises(ValidationError):
        _parse_json_response(__import__("json").dumps(payload), FinalSummary)


def test_generation_prompts_require_real_timestamps_and_plain_words() -> None:
    final_prompt = final_summary_prompt("SECTION INPUT 1: summary")
    concept_prompt = concept_extraction_prompt("SOURCE [01:24] slide=None\nRetrieval finds context.")
    assert "Never output 00:00, MM:SS" in final_prompt
    assert "Never output the literal 00:00 or MM:SS" in concept_prompt
    assert "plain words" in final_prompt
