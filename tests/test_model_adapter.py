"""
Unit tests for Phase 2 Model Adapter & Draft Generation.
Verifies prompt versioning, generation settings, model-reported confidence extraction,
local offline adapter, remote API adapters (OpenAI, Gemini, Ollama), and error resilience.
"""

import unittest
from unittest.mock import patch, MagicMock
import requests

from src.model_adapter import (
    PromptManager,
    GenerationSettings,
    ModelDraft,
    OfflineLocalAdapter,
    OpenAIAdapter,
    GeminiAdapter,
    OllamaAdapter,
    get_model_adapter,
)


class TestModelAdapterPhase2(unittest.TestCase):

    def test_prompt_manager_versioning(self):
        # Test standard prompt template
        p_v1 = PromptManager.get_prompt("direct", "What is 2+2?", prompt_version="v1.0-standard")
        self.assertIn("Question: What is 2+2?", p_v1)

        # Test concise template
        p_v2 = PromptManager.get_prompt("direct", "What is 2+2?", prompt_version="v2.0-concise")
        self.assertIn("Provide a concise, direct, factual answer", p_v2)

        # Test confidence elicitation template
        p_conf = PromptManager.get_prompt("direct", "What is 2+2?", prompt_version="v1.1-with-confidence-elicitation")
        self.assertIn("estimated self-confidence", p_conf)

    def test_extract_confidence_parsing(self):
        # Valid confidence in response
        raw_text = "The elevation is 8,848.86 meters.\nConfidence: 0.95"
        clean, score = PromptManager.extract_confidence(raw_text)
        self.assertEqual(score, 0.95)
        self.assertNotIn("Confidence: 0.95", clean)

        # Bracketed confidence
        raw_text_bracket = "Answer text.\nConfidence: [0.82]"
        clean2, score2 = PromptManager.extract_confidence(raw_text_bracket)
        self.assertEqual(score2, 0.82)

        # Text without confidence
        raw_text_none = "Normal answer without any score."
        clean3, score3 = PromptManager.extract_confidence(raw_text_none)
        self.assertIsNone(score3)
        self.assertEqual(clean3, raw_text_none)

    def test_offline_local_adapter_draft_generation(self):
        adapter = OfflineLocalAdapter()
        settings = GenerationSettings(
            temperature=0.3,
            max_tokens=300,
            prompt_version="v1.1-with-confidence-elicitation",
            elicit_confidence=True
        )
        draft = adapter.generate_draft(
            query="What is the speed of light in vacuum?",
            task_type="direct",
            settings=settings
        )

        self.assertIsInstance(draft, ModelDraft)
        self.assertIn("299,792,458", draft.text)
        self.assertTrue(draft.is_mock)
        self.assertFalse(draft.confidence_is_ground_truth)  # Safety check: must not be treated as ground truth
        self.assertIsNotNone(draft.model_reported_confidence)
        self.assertEqual(draft.prompt_version, "v1.1-with-confidence-elicitation")
        self.assertEqual(draft.generation_settings["temperature"], 0.3)
        self.assertGreater(draft.total_tokens, 0)
        self.assertGreaterEqual(draft.latency_ms, 0.0)

    def test_offline_local_adapter_rag_draft(self):
        adapter = OfflineLocalAdapter()
        draft = adapter.generate_draft(
            query="What is Mount Everest elevation?",
            task_type="rag",
            evidence="Mount Everest elevation is 8,848.86 meters according to the 2020 survey."
        )
        self.assertIn("8,848.86", draft.text)
        self.assertGreater(draft.total_tokens, 0)

    def test_openai_adapter_missing_key(self):
        adapter = OpenAIAdapter(api_key="")
        draft = adapter.generate_draft(query="Test query")
        self.assertIsNotNone(draft.error)
        self.assertIn("Missing or placeholder OPENAI_API_KEY", draft.error)

    @patch("requests.post")
    def test_openai_adapter_success_mock(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {"content": "The capital of France is Paris.\nConfidence: 0.99"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10}
        }
        mock_post.return_value = mock_resp

        adapter = OpenAIAdapter(api_key="sk-mock-valid-key")
        settings = GenerationSettings(temperature=0.1, max_tokens=100)
        draft = adapter.generate_draft(query="Capital of France?", settings=settings)

        self.assertIsNone(draft.error)
        self.assertEqual(draft.text, "The capital of France is Paris.")
        self.assertEqual(draft.model_reported_confidence, 0.99)
        self.assertFalse(draft.confidence_is_ground_truth)
        self.assertEqual(draft.prompt_tokens, 15)
        self.assertEqual(draft.completion_tokens, 10)
        self.assertEqual(draft.total_tokens, 25)

    @patch("requests.post")
    def test_openai_adapter_timeout(self, mock_post):
        mock_post.side_effect = requests.Timeout("Request timed out")
        adapter = OpenAIAdapter(api_key="sk-mock-valid-key", timeout=1.0)
        draft = adapter.generate_draft(query="Test timeout")
        self.assertIsNotNone(draft.error)
        self.assertIn("timed out", draft.error)

    def test_gemini_adapter_missing_key(self):
        adapter = GeminiAdapter(api_key="")
        draft = adapter.generate_draft(query="Test query")
        self.assertIsNotNone(draft.error)
        self.assertIn("Missing or placeholder GEMINI_API_KEY", draft.error)

    @patch("requests.post")
    def test_gemini_adapter_success_mock(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {"parts": [{"text": "Universal constant is c.\nConfidence: 0.90"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 12}
        }
        mock_post.return_value = mock_resp

        adapter = GeminiAdapter(api_key="AIza-mock-valid-key")
        draft = adapter.generate_draft(query="Universal constant c?")

        self.assertIsNone(draft.error)
        self.assertEqual(draft.text, "Universal constant is c.")
        self.assertEqual(draft.model_reported_confidence, 0.90)
        self.assertEqual(draft.total_tokens, 32)

    @patch("requests.post")
    def test_ollama_adapter_connection_error(self, mock_post):
        mock_post.side_effect = requests.ConnectionError("Connection refused")
        adapter = OllamaAdapter(host="http://localhost:99999")
        draft = adapter.generate_draft(query="Test local model")
        self.assertIsNotNone(draft.error)
        self.assertIn("Could not connect to local Ollama server", draft.error)

    @patch("requests.post")
    def test_ollama_adapter_success_mock(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": "Local llama output response.\nConfidence: 0.88",
            "prompt_eval_count": 18,
            "eval_count": 8,
            "total_duration": 123456
        }
        mock_post.return_value = mock_resp

        adapter = OllamaAdapter()
        draft = adapter.generate_draft(query="Test prompt")
        self.assertIsNone(draft.error)
        self.assertEqual(draft.text, "Local llama output response.")
        self.assertEqual(draft.model_reported_confidence, 0.88)
        self.assertEqual(draft.total_tokens, 26)

    def test_factory_fallback(self):
        # Default with no keys must be offline local adapter
        adapter = get_model_adapter("offline")
        self.assertIsInstance(adapter, OfflineLocalAdapter)


if __name__ == "__main__":
    unittest.main()
