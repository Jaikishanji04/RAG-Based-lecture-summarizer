"""Unified Hugging Face provider abstraction for Chat, Tool Calling, and Embeddings.

This module provides a single vendor-neutral interface so the rest of the application
never talks directly to a specific LLM vendor API.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised when the LLM/Embedding provider fails to execute a request."""


class ToolCall(BaseModel):
    """Normalized representation of a tool call requested by the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]


class ChatMessage(BaseModel):
    """Normalized chat message."""

    role: str  # 'system', 'user', 'assistant', 'tool'
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatResponse(BaseModel):
    """Normalized chat response from the model."""

    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    model: str
    finish_reason: Optional[str] = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ProviderConfig(BaseModel):
    """Configuration for inference and embedding providers."""

    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openrouter_api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    api_token: str = Field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    chat_model: str = Field(
        default_factory=lambda: os.getenv("HF_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    )
    embed_model: str = Field(
        default_factory=lambda: os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )
    timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
    )


class HuggingFaceProvider:
    """Unified provider implementing chat, tool calling, and embeddings."""

    def __init__(self, config: Optional[ProviderConfig] = None) -> None:
        self.config = config or ProviderConfig()
        self._openai_client: Any = None
        self._embed_pipeline: Any = None

    def get_active_provider_name(self) -> str:
        """Return the name of the currently configured LLM provider."""
        if self.config.groq_api_key:
            return "Groq (Free/Fast)"
        if self.config.openrouter_api_key:
            return "OpenRouter"
        if self.config.api_token:
            return "Hugging Face"
        return "None"

    def _get_client(self) -> Any:
        """Lazily initialize an OpenAI-compatible client pointing to the active provider."""
        if self._openai_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderError(
                    "The 'openai' client package is required to connect to the LLM router. Run 'py -m pip install openai'."
                ) from exc

            # Prioritize Groq if provided (100% free high-speed provider)
            if self.config.groq_api_key:
                self._openai_client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=self.config.groq_api_key,
                    timeout=self.config.timeout_seconds,
                )
                if self.config.chat_model == "Qwen/Qwen2.5-7B-Instruct":
                    self.config.chat_model = "llama-3.3-70b-versatile"
                return self._openai_client

            # Prioritize OpenRouter if provided
            if self.config.openrouter_api_key:
                self._openai_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.config.openrouter_api_key,
                    timeout=self.config.timeout_seconds,
                )
                return self._openai_client

            # Default to Hugging Face
            if not self.config.api_token:
                raise ProviderError(
                    "HF_TOKEN is missing. Please add HF_TOKEN or GROQ_API_KEY to your .env file."
                )

            # Ensure proper HF router base URL
            base_url = self.config.base_url
            if "hf-inference/v1" in base_url:
                base_url = "https://router.huggingface.co/v1"

            self._openai_client = OpenAI(
                base_url=base_url,
                api_key=self.config.api_token,
                timeout=self.config.timeout_seconds,
            )
        return self._openai_client

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        max_tokens: int = 2048,
        model: Optional[str] = None,
    ) -> ChatResponse:
        """Execute a standard chat completion."""
        client = self._get_client()
        target_model = model or self.config.chat_model
        temp = temperature if temperature is not None else self.config.temperature

        kwargs: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                # Some HF inference endpoints may not support response_format='json_object' natively
                if json_mode and "response_format" in str(exc).lower():
                    kwargs.pop("response_format", None)
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise

            choice = response.choices[0]
            raw_content = choice.message.content or ""
            return ChatResponse(
                content=raw_content.strip(),
                tool_calls=[],
                model=target_model,
                finish_reason=choice.finish_reason,
            )
        except Exception as exc:
            logger.exception("Hugging Face chat completion failed")
            raise ProviderError(f"Hugging Face chat request failed: {exc}") from exc

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: int = 2048,
        model: Optional[str] = None,
    ) -> ChatResponse:
        """Execute chat completion with tool/function definitions."""
        client = self._get_client()
        target_model = model or self.config.chat_model
        temp = temperature if temperature is not None else self.config.temperature

        kwargs: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temp,
            "max_tokens": max_tokens,
        }

        try:
            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message

            tool_calls: List[ToolCall] = []
            if getattr(message, "tool_calls", None):
                for tc in message.tool_calls:
                    fn_args = tc.function.arguments
                    if isinstance(fn_args, str):
                        try:
                            fn_args = json.loads(fn_args)
                        except json.JSONDecodeError:
                            fn_args = {"raw": fn_args}
                    tool_calls.append(
                        ToolCall(
                            id=tc.id or f"call_{tc.function.name}",
                            name=tc.function.name,
                            arguments=fn_args if isinstance(fn_args, dict) else {},
                        )
                    )

            return ChatResponse(
                content=message.content,
                tool_calls=tool_calls,
                model=target_model,
                finish_reason=choice.finish_reason,
            )
        except Exception as exc:
            logger.exception("Hugging Face tool-calling chat failed")
            raise ProviderError(f"Hugging Face tool call request failed: {exc}") from exc

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized vector embeddings for a list of texts."""
        if not texts:
            return []

        # 1. Primary engine: Direct local PyTorch + Hugging Face Transformers (offline, no sklearn DLL dependencies)
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            if self._embed_pipeline is None:
                tok = AutoTokenizer.from_pretrained(self.config.embed_model)
                mod = AutoModel.from_pretrained(self.config.embed_model)
                mod.eval()
                self._embed_pipeline = (tok, mod)

            tok, mod = self._embed_pipeline
            inputs = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                out = mod(**inputs)
                if hasattr(out, "last_hidden_state"):
                    emb = out.last_hidden_state[:, 0]
                else:
                    emb = out[0][:, 0]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                return emb.cpu().tolist()
        except Exception as tf_exc:
            logger.debug("Local transformers embedding failed, trying SentenceTransformers: %s", tf_exc)

        # 2. Secondary fallback: SentenceTransformer
        try:
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer(self.config.embed_model)
            embeddings = st_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            return embeddings.tolist()
        except Exception as exc:
            logger.warning("SentenceTransformer embedding fallback triggered: %s", exc)

        # 3. Tertiary fallback: Direct Hugging Face Inference API
        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(
                token=self.config.api_token, timeout=self.config.timeout_seconds
            )
            raw_embeds = client.feature_extraction(texts, model=self.config.embed_model)
            import numpy as np

            arr = np.array(raw_embeds)
            if arr.ndim == 3:  # (batch, seq, dim) -> mean pool
                arr = np.mean(arr, axis=1)
            norms = np.linalg.norm(arr, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
            return arr.tolist()
        except Exception as api_exc:
            logger.exception("Embedding extraction failed on all backends")
            raise ProviderError(f"Failed to generate embeddings: {api_exc}") from api_exc


# Global default instance
_default_provider: Optional[HuggingFaceProvider] = None


def get_provider() -> HuggingFaceProvider:
    """Return the shared global HuggingFaceProvider instance."""
    global _default_provider
    if _default_provider is None:
        _default_provider = HuggingFaceProvider()
    return _default_provider
