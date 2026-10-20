"""
LLM Provider Abstraction.

Provides pluggable integration for:
1. Offline / Local demonstration generator (default zero-cost mode, no API key required).
2. OpenAI API adapter (reads OPENAI_API_KEY from environment).
3. Google Gemini API adapter (reads GEMINI_API_KEY from environment).

Strictly ensures:
- Never hardcodes or logs secrets.
- Real token counts and latency recorded when available.
- Gracefully handles missing credentials, network failures, timeouts, and rate limits.
- Explicitly distinguishes provider output from local deterministic fallbacks.
"""

from __future__ import annotations
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import requests


@dataclass
class LLMResponse:
    """Standardized response from an LLM provider."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    model_name: str = "offline_local"
    provider_name: str = "offline"
    is_mock: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "model_name": self.model_name,
            "provider_name": self.provider_name,
            "is_mock": self.is_mock,
            "error": self.error,
        }


class BaseLLMProvider(ABC):
    """Abstract interface for all model adapters."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 500) -> LLMResponse:
        """Generate text from prompt."""
        pass


class OfflineLocalProvider(BaseLLMProvider):
    """Local demonstration generator. Works offline without API keys or paid services."""

    def __init__(self, model_name: str = "offline-rule-demonstrator"):
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 500) -> LLMResponse:
        start_time = time.perf_counter()
        
        # Approximate tokens based on whitespace and punctuation
        prompt_tokens = max(len(prompt.split()) * 4 // 3, 1)

        lower_p = prompt.lower()
        
        # Intelligent canned responses for demonstration benchmark questions
        if "mount everest" in lower_p or "highest mountain" in lower_p:
            answer = "Mount Everest is Earth's highest mountain above sea level, officially measured at 8,848.86 meters (29,031.7 feet)."
        elif "speed of light" in lower_p:
            answer = "The speed of light in vacuum is exactly 299,792,458 meters per second."
        elif "apollo 11" in lower_p or "moon landing" in lower_p or "neil armstrong" in lower_p:
            answer = "Apollo 11 landed astronauts Neil Armstrong and Buzz Aldrin on the Moon on July 20, 1969. Armstrong stepped onto the surface on July 21, 1969."
        elif "alan turing" in lower_p:
            answer = "Alan Turing was a pioneering British mathematician and computer scientist who formulated the concept of the universal Turing Machine in 1936."
        elif "rag" in lower_p or "retrieval-augmented" in lower_p:
            answer = "Retrieval-Augmented Generation (RAG) enhances language models by retrieving external document passages before generating answers."
        elif "einstein" in lower_p and "nobel" in lower_p:
            answer = "Albert Einstein was awarded the 1921 Nobel Prize in Physics for his discovery of the law of the photoelectric effect, not for the theory of relativity."
        elif "capital of france" in lower_p:
            answer = "The capital of France is Paris."
        else:
            # General draft placeholder
            answer = (
                f"Direct response to: '{prompt[:60]}...'. "
                "This answer was produced by the local offline demonstration adapter. "
                "Configure an OPENAI_API_KEY or GEMINI_API_KEY in .env to use live neural models."
            )

        latency = (time.perf_counter() - start_time) * 1000.0
        completion_tokens = max(len(answer.split()) * 4 // 3, 1)

        return LLMResponse(
            text=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency,
            model_name=self.model_name,
            provider_name="offline",
            is_mock=True,
            error=None,
        )


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API adapter using standard HTTP requests."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: float = 15.0):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 500) -> LLMResponse:
        start_time = time.perf_counter()

        if not self.api_key or self.api_key.startswith("your_"):
            return LLMResponse(
                text="",
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                model_name=self.model,
                provider_name="openai",
                is_mock=False,
                error="Missing or invalid OPENAI_API_KEY. Configure a valid key in .env or environment variables."
            )

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            latency = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                p_tok = usage.get("prompt_tokens", 0)
                c_tok = usage.get("completion_tokens", 0)
                return LLMResponse(
                    text=content,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=p_tok + c_tok,
                    latency_ms=latency,
                    model_name=self.model,
                    provider_name="openai",
                    is_mock=False,
                    error=None
                )
            elif resp.status_code == 401:
                return LLMResponse(
                    text="",
                    latency_ms=latency,
                    model_name=self.model,
                    provider_name="openai",
                    is_mock=False,
                    error="OpenAI Authentication failed (HTTP 401): Invalid API Key."
                )
            elif resp.status_code == 429:
                return LLMResponse(
                    text="",
                    latency_ms=latency,
                    model_name=self.model,
                    provider_name="openai",
                    is_mock=False,
                    error="OpenAI Rate limit or quota exceeded (HTTP 429)."
                )
            else:
                return LLMResponse(
                    text="",
                    latency_ms=latency,
                    model_name=self.model,
                    provider_name="openai",
                    is_mock=False,
                    error=f"OpenAI API Error (HTTP {resp.status_code}): {resp.text[:150]}"
                )
        except requests.Timeout:
            return LLMResponse(
                text="",
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                model_name=self.model,
                provider_name="openai",
                is_mock=False,
                error=f"OpenAI request timed out after {self.timeout} seconds."
            )
        except Exception as e:
            return LLMResponse(
                text="",
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                model_name=self.model,
                provider_name="openai",
                is_mock=False,
                error=f"Network or request error: {str(e)}"
            )


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API adapter via REST."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: float = 15.0):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 500) -> LLMResponse:
        start_time = time.perf_counter()

        if not self.api_key or self.api_key.startswith("your_"):
            return LLMResponse(
                text="",
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                model_name=self.model,
                provider_name="gemini",
                is_mock=False,
                error="Missing or invalid GEMINI_API_KEY. Configure a valid key in .env or environment variables."
            )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System Instruction: {system_prompt}"}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.2
            }
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            latency = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                candidate_text = data["candidates"][0]["content"]["parts"][0]["text"]
                usage = data.get("usageMetadata", {})
                p_tok = usage.get("promptTokenCount", len(prompt.split()))
                c_tok = usage.get("candidatesTokenCount", len(candidate_text.split()))
                return LLMResponse(
                    text=candidate_text,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=p_tok + c_tok,
                    latency_ms=latency,
                    model_name=self.model,
                    provider_name="gemini",
                    is_mock=False,
                    error=None
                )
            else:
                return LLMResponse(
                    text="",
                    latency_ms=latency,
                    model_name=self.model,
                    provider_name="gemini",
                    is_mock=False,
                    error=f"Gemini API Error (HTTP {resp.status_code}): {resp.text[:150]}"
                )
        except requests.Timeout:
            return LLMResponse(
                text="",
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                model_name=self.model,
                provider_name="gemini",
                is_mock=False,
                error=f"Gemini request timed out after {self.timeout} seconds."
            )
        except Exception as e:
            return LLMResponse(
                text="",
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                model_name=self.model,
                provider_name="gemini",
                is_mock=False,
                error=f"Network or request error: {str(e)}"
            )


def get_llm_provider(preference: Optional[str] = None) -> BaseLLMProvider:
    """Factory creating the appropriate provider based on config or explicit preference."""
    pref = (preference or os.getenv("LLM_PROVIDER", "offline")).lower()
    
    if pref == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if key and not key.startswith("your_"):
            return OpenAIProvider()
    elif pref == "gemini":
        key = os.getenv("GEMINI_API_KEY", "")
        if key and not key.startswith("your_"):
            return GeminiProvider()

    # Default to transparent local offline provider
    return OfflineLocalProvider()
