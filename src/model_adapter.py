"""
Model Adapter & Draft Generation (Phase 2).

Provides modular LLM integration supporting:
- Pluggable providers: Offline Local, OpenAI, Google Gemini, and local Ollama.
- Prompt templates and explicit prompt versioning.
- Configurable generation settings (temperature, top_p, max_tokens).
- Extraction of model-reported confidence (treated explicitly as an experimental feature, NOT ground truth).
- Comprehensive metadata logging: model ID, prompt version, generation settings, latency, token counts, and error tracking.
"""

from __future__ import annotations
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import requests


@dataclass
class GenerationSettings:
    """Configurable generation parameters for LLM draft generation."""
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 500
    prompt_version: str = "v1.0-standard"
    elicit_confidence: bool = False
    stop_sequences: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelDraft:
    """Standardized draft generation output and metadata."""
    text: str
    model_id: str
    provider: str
    prompt_version: str
    generation_settings: Dict[str, Any]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    
    # Model self-reported confidence (0.0 to 1.0)
    # CRITICAL: Treated as an uncalibrated hypothesis feature to evaluate, NOT ground truth!
    model_reported_confidence: Optional[float] = None
    confidence_is_ground_truth: bool = False
    
    is_mock: bool = False
    error: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model_id": self.model_id,
            "provider": self.provider,
            "prompt_version": self.prompt_version,
            "generation_settings": self.generation_settings,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "model_reported_confidence": self.model_reported_confidence,
            "confidence_is_ground_truth": False,
            "is_mock": self.is_mock,
            "error": self.error,
            "raw_metadata": self.raw_metadata,
        }


class PromptManager:
    """Manages versioned prompt templates for draft generation."""

    PROMPT_TEMPLATES = {
        "v1.0-standard": {
            "direct": "Question: {query}\n\nAnswer:",
            "rag": "You are a factual assistant. Answer the question using strictly the retrieved evidence below.\n\nEvidence:\n{evidence}\n\nQuestion: {query}\n\nAnswer:",
            "clarify": "The user query is ambiguous or underspecified: '{query}'. Formulate a polite, structured question asking for clarification of the missing context, criteria, or options.",
        },
        "v2.0-concise": {
            "direct": "Provide a concise, direct, factual answer to: {query}\nAnswer:",
            "rag": "Context:\n{evidence}\n\nUsing only the context above, succinctly answer: {query}\nAnswer:",
            "clarify": "Briefly ask the user to clarify: '{query}'",
        },
        "v1.1-with-confidence-elicitation": {
            "direct": "Question: {query}\n\nProvide your answer followed by your estimated self-confidence on a scale from 0.0 to 1.0 in the format: Confidence: [0.0-1.0]\n\nAnswer:",
            "rag": "Evidence:\n{evidence}\n\nQuestion: {query}\n\nProvide your answer grounded in the evidence, followed by your estimated confidence in the format: Confidence: [0.0-1.0]\n\nAnswer:",
            "clarify": "Ask for clarification for query: '{query}'. Also state Confidence: 0.5",
        }
    }

    @classmethod
    def get_prompt(
        cls,
        task_type: str,
        query: str,
        evidence: str = "",
        prompt_version: str = "v1.0-standard"
    ) -> str:
        """Render prompt template with variables."""
        version_dict = cls.PROMPT_TEMPLATES.get(prompt_version, cls.PROMPT_TEMPLATES["v1.0-standard"])
        template = version_dict.get(task_type, version_dict["direct"])
        return template.format(query=query, evidence=evidence)

    @classmethod
    def extract_confidence(cls, text: str) -> tuple[str, Optional[float]]:
        """
        Parse self-reported confidence if elicited in prompt.
        Treats extracted score strictly as an uncalibrated feature.
        """
        conf_match = re.search(r"confidence:\s*\[?([0-1](?:\.\d+)?)\]?", text, re.IGNORECASE)
        if not conf_match:
            conf_match = re.search(r"confidence:\s*([0-1](?:\.\d+)?)", text, re.IGNORECASE)
            
        if conf_match:
            try:
                score = float(conf_match.group(1))
                clean_text = re.sub(r"confidence:\s*\[?([0-1](?:\.\d+)?)\]?", "", text, flags=re.IGNORECASE).strip()
                return clean_text, max(0.0, min(score, 1.0))
            except ValueError:
                pass
        return text, None


class BaseModelAdapter(ABC):
    """Abstract interface for all model adapters."""

    def __init__(self, model_id: str, provider_name: str):
        self.model_id = model_id
        self.provider_name = provider_name

    @abstractmethod
    def generate_draft(
        self,
        query: str,
        task_type: str = "direct",
        evidence: str = "",
        settings: Optional[GenerationSettings] = None
    ) -> ModelDraft:
        """Generate draft answer with metadata."""
        pass


class OfflineLocalAdapter(BaseModelAdapter):
    """
    Local demonstration model adapter.
    Runs 100% locally with zero external API calls or costs.
    Provides reproducible benchmark responses, prompt versioning, and token tracking.
    """

    def __init__(self, model_id: str = "offline-rule-demonstrator"):
        super().__init__(model_id=model_id, provider_name="offline")

    def generate_draft(
        self,
        query: str,
        task_type: str = "direct",
        evidence: str = "",
        settings: Optional[GenerationSettings] = None
    ) -> ModelDraft:
        start_time = time.perf_counter()
        cfg = settings or GenerationSettings()
        prompt = PromptManager.get_prompt(task_type, query, evidence, cfg.prompt_version)
        prompt_tokens = max(len(prompt.split()) * 4 // 3, 1)

        lower_q = query.lower()
        lower_ev = evidence.lower()

        # Generate response based on query topic, grounded in evidence if available
        if "mount everest" in lower_q or "elevation" in lower_q or "highest mountain" in lower_q:
            answer = "Mount Everest has an official elevation of 8,848.86 meters (29,031.7 feet), established in the 2020 joint survey by Nepal and China."
            self_conf = 0.95
        elif "speed of light" in lower_q:
            answer = "The speed of light in vacuum is defined as exactly 299,792,458 meters per second."
            self_conf = 0.99
        elif "apollo 11" in lower_q or "moon landing" in lower_q or "neil armstrong" in lower_q:
            answer = "Apollo 11 landed on the Moon on July 20, 1969, with Neil Armstrong and Buzz Aldrin."
            self_conf = 0.98
        elif "einstein" in lower_q or "photoelectric" in lower_q or "nobel" in lower_q:
            answer = "Albert Einstein received the 1921 Nobel Prize in Physics for his discovery of the law of the photoelectric effect, not for the theory of relativity."
            self_conf = 0.95
        elif "alan turing" in lower_q or "turing machine" in lower_q:
            answer = "Alan Turing introduced the universal Turing Machine in 1936, establishing theoretical computing."
            self_conf = 0.95
        elif "photosynthesis" in lower_q:
            answer = "Photosynthesis is the biological process by which green plants and certain other organisms transform light energy into chemical energy, converting water, carbon dioxide, and sunlight into oxygen and energy-rich organic compounds."
            self_conf = 0.85
        elif evidence:
            # Generic grounded answer using first passage
            first_sentence = evidence.split("\n")[0].replace("[", "").split("]: ")[-1].strip()
            answer = f"Based on the retrieved evidence: {first_sentence}"
            self_conf = 0.85
        else:
            answer = (
                f"Direct response to '{query[:60]}...'. "
                "Generated by local offline demonstration adapter. "
                "Configure an OPENAI_API_KEY or GEMINI_API_KEY in .env to use live models."
            )
            self_conf = 0.50

        # If confidence elicitation requested, append it
        if cfg.elicit_confidence or "confidence" in cfg.prompt_version:
            answer += f"\nConfidence: {self_conf:.2f}"

        clean_text, parsed_conf = PromptManager.extract_confidence(answer)
        final_conf = parsed_conf if parsed_conf is not None else (self_conf if cfg.elicit_confidence else None)

        latency = (time.perf_counter() - start_time) * 1000.0
        completion_tokens = max(len(clean_text.split()) * 4 // 3, 1)

        return ModelDraft(
            text=clean_text,
            model_id=self.model_id,
            provider=self.provider_name,
            prompt_version=cfg.prompt_version,
            generation_settings=cfg.to_dict(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency,
            model_reported_confidence=final_conf,
            confidence_is_ground_truth=False,
            is_mock=True,
            error=None,
            raw_metadata={"simulation_source": "offline_demonstrator"}
        )


class OpenAIAdapter(BaseModelAdapter):
    """OpenAI API adapter using HTTP requests with full generation settings."""

    def __init__(self, api_key: Optional[str] = None, model_id: Optional[str] = None, timeout: float = 20.0):
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        model = model_id or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        super().__init__(model_id=model, provider_name="openai")
        self.api_key = key
        self.timeout = timeout

    def generate_draft(
        self,
        query: str,
        task_type: str = "direct",
        evidence: str = "",
        settings: Optional[GenerationSettings] = None
    ) -> ModelDraft:
        start_time = time.perf_counter()
        cfg = settings or GenerationSettings()
        prompt = PromptManager.get_prompt(task_type, query, evidence, cfg.prompt_version)

        if not self.api_key or self.api_key.startswith("your_"):
            return ModelDraft(
                text="",
                model_id=self.model_id,
                provider=self.provider_name,
                prompt_version=cfg.prompt_version,
                generation_settings=cfg.to_dict(),
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                error="Missing or placeholder OPENAI_API_KEY. Configure a valid key in .env.",
                is_mock=False,
            )

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "You are a factual, concise research assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        }
        if cfg.stop_sequences:
            payload["stop"] = cfg.stop_sequences

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            latency = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                p_tok = usage.get("prompt_tokens", 0)
                c_tok = usage.get("completion_tokens", 0)

                clean_text, parsed_conf = PromptManager.extract_confidence(raw_text)

                return ModelDraft(
                    text=clean_text,
                    model_id=self.model_id,
                    provider=self.provider_name,
                    prompt_version=cfg.prompt_version,
                    generation_settings=cfg.to_dict(),
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=p_tok + c_tok,
                    latency_ms=latency,
                    model_reported_confidence=parsed_conf,
                    confidence_is_ground_truth=False,
                    is_mock=False,
                    error=None,
                    raw_metadata={"finish_reason": data["choices"][0].get("finish_reason")}
                )
            else:
                return ModelDraft(
                    text="",
                    model_id=self.model_id,
                    provider=self.provider_name,
                    prompt_version=cfg.prompt_version,
                    generation_settings=cfg.to_dict(),
                    latency_ms=latency,
                    error=f"OpenAI API Error (HTTP {resp.status_code}): {resp.text[:150]}",
                    is_mock=False,
                )
        except requests.Timeout:
            return ModelDraft(
                text="",
                model_id=self.model_id,
                provider=self.provider_name,
                prompt_version=cfg.prompt_version,
                generation_settings=cfg.to_dict(),
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                error=f"OpenAI request timed out after {self.timeout}s",
                is_mock=False,
            )
        except Exception as e:
            return ModelDraft(
                text="",
                model_id=self.model_id,
                provider=self.provider_name,
                prompt_version=cfg.prompt_version,
                generation_settings=cfg.to_dict(),
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                error=f"Network or request error: {str(e)}",
                is_mock=False,
            )


class GeminiAdapter(BaseModelAdapter):
    """Google Gemini API adapter via REST."""

    def __init__(self, api_key: Optional[str] = None, model_id: Optional[str] = None, timeout: float = 20.0):
        key = api_key or os.getenv("GEMINI_API_KEY", "")
        model = model_id or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        super().__init__(model_id=model, provider_name="gemini")
        self.api_key = key
        self.timeout = timeout

    def generate_draft(
        self,
        query: str,
        task_type: str = "direct",
        evidence: str = "",
        settings: Optional[GenerationSettings] = None
    ) -> ModelDraft:
        start_time = time.perf_counter()
        cfg = settings or GenerationSettings()
        prompt = PromptManager.get_prompt(task_type, query, evidence, cfg.prompt_version)

        if not self.api_key or self.api_key.startswith("your_"):
            return ModelDraft(
                text="",
                model_id=self.model_id,
                provider=self.provider_name,
                prompt_version=cfg.prompt_version,
                generation_settings=cfg.to_dict(),
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                error="Missing or placeholder GEMINI_API_KEY. Configure a valid key in .env.",
                is_mock=False,
            )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": cfg.max_tokens,
                "temperature": cfg.temperature,
                "topP": cfg.top_p,
            }
        }
        if cfg.stop_sequences:
            payload["generationConfig"]["stopSequences"] = cfg.stop_sequences

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            latency = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                usage = data.get("usageMetadata", {})
                p_tok = usage.get("promptTokenCount", len(prompt.split()))
                c_tok = usage.get("candidatesTokenCount", len(raw_text.split()))

                clean_text, parsed_conf = PromptManager.extract_confidence(raw_text)

                return ModelDraft(
                    text=clean_text,
                    model_id=self.model_id,
                    provider=self.provider_name,
                    prompt_version=cfg.prompt_version,
                    generation_settings=cfg.to_dict(),
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=p_tok + c_tok,
                    latency_ms=latency,
                    model_reported_confidence=parsed_conf,
                    confidence_is_ground_truth=False,
                    is_mock=False,
                    error=None,
                    raw_metadata={"finish_reason": data["candidates"][0].get("finishReason")}
                )
            else:
                return ModelDraft(
                    text="",
                    model_id=self.model_id,
                    provider=self.provider_name,
                    prompt_version=cfg.prompt_version,
                    generation_settings=cfg.to_dict(),
                    latency_ms=latency,
                    error=f"Gemini API Error (HTTP {resp.status_code}): {resp.text[:150]}",
                    is_mock=False,
                )
        except requests.Timeout:
            return ModelDraft(
                text="",
                model_id=self.model_id,
                provider=self.provider_name,
                prompt_version=cfg.prompt_version,
                generation_settings=cfg.to_dict(),
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                error=f"Gemini request timed out after {self.timeout}s",
                is_mock=False,
            )
        except Exception as e:
            return ModelDraft(
                text="",
                model_id=self.model_id,
                provider=self.provider_name,
                prompt_version=cfg.prompt_version,
                generation_settings=cfg.to_dict(),
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                error=f"Network or request error: {str(e)}",
                is_mock=False,
            )


class OllamaAdapter(BaseModelAdapter):
    """Local Ollama adapter for offline open-weights neural model execution."""

    def __init__(self, host: str = "http://localhost:11434", model_id: str = "llama3.2:3b", timeout: float = 30.0):
        super().__init__(model_id=model_id, provider_name="ollama")
        self.host = host
        self.timeout = timeout

    def generate_draft(
        self,
        query: str,
        task_type: str = "direct",
        evidence: str = "",
        settings: Optional[GenerationSettings] = None
    ) -> ModelDraft:
        start_time = time.perf_counter()
        cfg = settings or GenerationSettings()
        prompt = PromptManager.get_prompt(task_type, query, evidence, cfg.prompt_version)

        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "num_predict": cfg.max_tokens,
            }
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            latency = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                raw_text = data.get("response", "")
                p_tok = data.get("prompt_eval_count", len(prompt.split()))
                c_tok = data.get("eval_count", len(raw_text.split()))

                clean_text, parsed_conf = PromptManager.extract_confidence(raw_text)

                return ModelDraft(
                    text=clean_text,
                    model_id=self.model_id,
                    provider=self.provider_name,
                    prompt_version=cfg.prompt_version,
                    generation_settings=cfg.to_dict(),
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=p_tok + c_tok,
                    latency_ms=latency,
                    model_reported_confidence=parsed_conf,
                    confidence_is_ground_truth=False,
                    is_mock=False,
                    error=None,
                    raw_metadata={"total_duration": data.get("total_duration")}
                )
            else:
                return ModelDraft(
                    text="",
                    model_id=self.model_id,
                    provider=self.provider_name,
                    prompt_version=cfg.prompt_version,
                    generation_settings=cfg.to_dict(),
                    latency_ms=latency,
                    error=f"Ollama server returned HTTP {resp.status_code}",
                    is_mock=False,
                )
        except requests.ConnectionError:
            return ModelDraft(
                text="",
                model_id=self.model_id,
                provider=self.provider_name,
                prompt_version=cfg.prompt_version,
                generation_settings=cfg.to_dict(),
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                error=f"Could not connect to local Ollama server at {self.host}. Ensure Ollama is running.",
                is_mock=False,
            )
        except Exception as e:
            return ModelDraft(
                text="",
                model_id=self.model_id,
                provider=self.provider_name,
                prompt_version=cfg.prompt_version,
                generation_settings=cfg.to_dict(),
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                error=f"Ollama error: {str(e)}",
                is_mock=False,
            )


def get_model_adapter(
    provider_name: Optional[str] = None,
    model_id: Optional[str] = None,
) -> BaseModelAdapter:
    """Factory creating the configured model adapter."""
    prov = (provider_name or os.getenv("LLM_PROVIDER", "offline")).lower()

    if prov == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if key and not key.startswith("your_"):
            return OpenAIAdapter(api_key=key, model_id=model_id)
    elif prov == "gemini":
        key = os.getenv("GEMINI_API_KEY", "")
        if key and not key.startswith("your_"):
            return GeminiAdapter(api_key=key, model_id=model_id)
    elif prov == "ollama":
        return OllamaAdapter(model_id=model_id or "llama3.2:3b")

    # Default: offline local demonstration adapter
    return OfflineLocalAdapter(model_id=model_id or "offline-rule-demonstrator")
