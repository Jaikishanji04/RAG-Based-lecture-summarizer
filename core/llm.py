"""Provider-agnostic language-model client used by the application."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel
import os


logger = logging.getLogger(__name__)
load_dotenv()


class LLMError(RuntimeError):
    """Raised when an LLM provider cannot complete a request."""


class LLMConfig(BaseModel):
    """Runtime configuration loaded from environment variables."""

    provider: str = "ollama"
    model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    hf_model: str = "Qwen/Qwen2.5-7B-Instruct"
    hf_token: str | None = None
    temperature: float = 0.2
    timeout_seconds: float = 300.0

    @classmethod
    def from_environment(cls) -> "LLMConfig":
        """Build validated configuration from `.env` values."""

        provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
        model = os.getenv("OLLAMA_MODEL", "llama3").strip()
        return cls(
            provider=provider,
            model=model,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            hf_model=os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct").strip(),
            hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or None,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "300")),
        )


@dataclass
class LLMResponse:
    """Normalized response returned by an LLM provider."""

    text: str
    model: str


class LLMClient:
    """Small provider adapter for Ollama, OpenAI, and Hugging Face chat APIs."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_environment()
        if self.config.provider not in {"ollama", "openai", "huggingface"}:
            raise ValueError("LLM_PROVIDER must be 'ollama', 'openai', or 'huggingface'.")
        self._openai_client: Any = None
        self._hf_client: Any = None

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate a response for a prompt using the configured provider."""

        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.config.provider == "ollama":
            return self._complete_ollama(prompt, system_prompt, json_mode=json_mode)
        if self.config.provider == "huggingface":
            return self._complete_huggingface(prompt, system_prompt, json_mode=json_mode)
        return self._complete_openai(prompt, system_prompt, json_mode=json_mode)

    def _complete_ollama(
        self,
        prompt: str,
        system_prompt: str | None,
        *,
        json_mode: bool = False,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        should_use_json = (
            json_mode
            or (bool(system_prompt) and "JSON" in system_prompt)
            or ("as JSON" in prompt or "into JSON" in prompt or "Return JSON" in prompt)
        )

        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": num_ctx,
            },
        }
        if should_use_json:
            payload["format"] = "json"

        try:
            response = httpx.post(
                f"{self.config.ollama_base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content")
            if not isinstance(content, str) or not content.strip():
                raise LLMError("Ollama returned an empty response.")
            return LLMResponse(text=content.strip(), model=self.config.model)
        except (httpx.HTTPError, ValueError, LLMError) as exc:
            logger.exception("Ollama request failed")
            detail = response.text if "response" in locals() else str(exc)
            raise LLMError(f"Ollama request failed: {detail}") from exc

    def _complete_openai(
        self,
        prompt: str,
        system_prompt: str | None,
        *,
        json_mode: bool = False,
    ) -> LLMResponse:
        if not self.config.openai_api_key:
            raise LLMError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("Install the 'openai' package to use the OpenAI provider.") from exc
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self.config.openai_api_key)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        should_use_json = (
            json_mode
            or (bool(system_prompt) and "JSON" in system_prompt)
            or ("as JSON" in prompt or "into JSON" in prompt or "Return JSON" in prompt)
        )
        kwargs: dict[str, Any] = {
            "model": self.config.openai_model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if should_use_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._openai_client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content:
                raise LLMError("OpenAI returned an empty response.")
            return LLMResponse(text=content.strip(), model=self.config.openai_model)
        except Exception as exc:
            logger.exception("OpenAI request failed")
            raise LLMError(f"OpenAI request failed: {exc}") from exc

    def _complete_huggingface(
        self,
        prompt: str,
        system_prompt: str | None,
        *,
        json_mode: bool = False,
    ) -> LLMResponse:
        if not self.config.hf_token:
            raise LLMError("HF_TOKEN is required when LLM_PROVIDER=huggingface.")
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise LLMError("Install 'huggingface_hub' to use the Hugging Face provider.") from exc

        if self._hf_client is None:
            self._hf_client = InferenceClient(
                token=self.config.hf_token,
                timeout=self.config.timeout_seconds,
            )

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        should_use_json = (
            json_mode
            or (bool(system_prompt) and "JSON" in system_prompt)
            or ("as JSON" in prompt or "into JSON" in prompt or "Return JSON" in prompt)
        )

        kwargs: dict[str, Any] = {
            "model": self.config.hf_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": 2048,
        }
        if should_use_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            try:
                response = self._hf_client.chat.completions.create(**kwargs)
            except Exception as exc:
                if should_use_json and "response_format" in str(exc).lower():
                    kwargs.pop("response_format", None)
                    response = self._hf_client.chat.completions.create(**kwargs)
                else:
                    raise
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise LLMError("Hugging Face returned an empty response.")
            return LLMResponse(text=content.strip(), model=self.config.hf_model)
        except Exception as exc:
            logger.exception("Hugging Face request failed")
            raise LLMError(f"Hugging Face request failed: {exc}") from exc
