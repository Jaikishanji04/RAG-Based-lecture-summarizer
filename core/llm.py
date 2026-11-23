"""Hugging Face language-model client wrapper used across the application."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os

from .llm_provider import HuggingFaceProvider, ProviderConfig, ProviderError, get_provider

logger = logging.getLogger(__name__)
load_dotenv()


class LLMError(RuntimeError):
    """Raised when the LLM provider cannot complete a request."""


class LLMConfig(BaseModel):
    """Runtime configuration loaded from environment variables."""

    provider: str = "huggingface"
    hf_chat_model: str = Field(
        default_factory=lambda: os.getenv("HF_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    )
    hf_embed_model: str = Field(
        default_factory=lambda: os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    )
    hf_token: Optional[str] = Field(
        default_factory=lambda: os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )
    timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
    )

    @classmethod
    def from_environment(cls) -> "LLMConfig":
        """Build validated configuration from `.env` values."""
        return cls()


@dataclass
class LLMResponse:
    """Normalized response returned by an LLM provider."""

    text: str
    model: str


class LLMClient:
    """Application adapter that delegates to the unified HuggingFaceProvider."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig.from_environment()
        provider_cfg = ProviderConfig(
            api_token=self.config.hf_token or "",
            chat_model=self.config.hf_chat_model,
            embed_model=self.config.hf_embed_model,
            temperature=self.config.temperature,
            timeout_seconds=self.config.timeout_seconds,
        )
        self.provider = HuggingFaceProvider(provider_cfg)

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate a response for a prompt using Hugging Face."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        should_use_json = (
            json_mode
            or (bool(system_prompt) and "JSON" in system_prompt)
            or ("as JSON" in prompt or "into JSON" in prompt or "Return JSON" in prompt)
        )

        try:
            resp = self.provider.chat(
                messages,
                temperature=self.config.temperature,
                json_mode=should_use_json,
            )
            return LLMResponse(text=resp.content or "", model=resp.model)
        except ProviderError as exc:
            logger.exception("LLM completion failed")
            raise LLMError(str(exc)) from exc
