"""Unit and integration tests for the Hugging Face provider abstraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from core.llm_provider import (
    ChatMessage,
    ChatResponse,
    HuggingFaceProvider,
    ProviderConfig,
    ProviderError,
    ToolCall,
)
from core.llm import LLMClient, LLMConfig, LLMError


def test_provider_initialization_defaults() -> None:
    """Verify that ProviderConfig correctly picks up default models."""
    cfg = ProviderConfig(api_token="dummy_token")
    assert cfg.chat_model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.embed_model == "BAAI/bge-small-en-v1.5"
    assert "huggingface.co" in cfg.base_url


def test_provider_requires_token_when_calling_chat() -> None:
    """Calling chat without HF_TOKEN raises ProviderError."""
    provider = HuggingFaceProvider(ProviderConfig(api_token=""))
    with pytest.raises(ProviderError, match="HF_TOKEN is missing"):
        provider.chat([{"role": "user", "content": "Hello"}])


def test_provider_chat_mocked() -> None:
    """Mock the underlying client to verify chat response normalization."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Test answer"
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    provider = HuggingFaceProvider(ProviderConfig(api_token="test_token"))
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    provider._openai_client = mock_client

    response = provider.chat([{"role": "user", "content": "What is RAG?"}])
    assert response.content == "Test answer"
    assert response.model == "Qwen/Qwen2.5-7B-Instruct"
    assert not response.has_tool_calls


def test_provider_tool_calling_mocked() -> None:
    """Verify tool call extraction when the model requests function calls."""
    mock_tc = MagicMock()
    mock_tc.id = "call_123"
    mock_tc.function.name = "search_lecture"
    mock_tc.function.arguments = '{"query": "backpropagation"}'

    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_choice.message.tool_calls = [mock_tc]
    mock_choice.finish_reason = "tool_calls"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    provider = HuggingFaceProvider(ProviderConfig(api_token="test_token"))
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    provider._openai_client = mock_client

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_lecture",
                "description": "Search lecture notes",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    response = provider.chat_with_tools([{"role": "user", "content": "Find backpropagation"}], tools=tools)
    assert response.has_tool_calls
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "search_lecture"
    assert response.tool_calls[0].arguments == {"query": "backpropagation"}


def test_llm_client_delegates_to_provider() -> None:
    """Verify LLMClient complete() maps correctly to the provider."""
    client = LLMClient(LLMConfig(hf_token="test_token"))
    client.provider.chat = MagicMock(return_value=ChatResponse(content="Grounded summary", model="test_model"))

    res = client.complete("Summarize lecture")
    assert res.text == "Grounded summary"
    assert res.model == "test_model"
