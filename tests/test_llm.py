"""Smoke test for the configured local Ollama model."""

import os

import httpx
import pytest

from core.llm import LLMClient, LLMConfig, LLMError


@pytest.mark.integration
def test_ollama_smoke_request() -> None:
    """Call Ollama once and verify that the adapter normalizes its response."""

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        health = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
        health.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"Ollama is not reachable: {exc}")

    models = {item.get("name") for item in health.json().get("models", [])}
    configured_model = os.getenv("OLLAMA_MODEL", "llama3")
    if configured_model not in models:
        pytest.skip(
            f"Ollama is reachable, but model '{configured_model}' is not installed. "
            f"Available models: {sorted(models)}"
        )

    client = LLMClient(LLMConfig.from_environment())
    try:
        response = client.complete("Reply with exactly: smoke test passed")
    except LLMError as exc:
        pytest.fail(str(exc))
    assert response.text
    assert response.model


def test_huggingface_client_validation() -> None:
    """Verify that Hugging Face client requires an API token."""
    config_missing_token = LLMConfig(provider="huggingface", hf_token=None)
    client = LLMClient(config_missing_token)
    with pytest.raises(LLMError, match="HF_TOKEN is required"):
        client.complete("Hello")
