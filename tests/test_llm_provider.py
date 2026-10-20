"""
Unit tests for LLM Provider Abstraction.
Tests offline local demonstration fallback, missing API keys handling,
network failure/timeout handling, and provider metadata.
"""

import unittest
from unittest.mock import patch, MagicMock
import requests
from src.llm_provider import OfflineLocalProvider, OpenAIProvider, GeminiProvider, get_llm_provider


class TestLLMProvider(unittest.TestCase):

    def test_offline_local_provider(self):
        provider = OfflineLocalProvider()
        resp = provider.generate("What is the speed of light in vacuum?")
        self.assertTrue(resp.is_mock)
        self.assertEqual(resp.provider_name, "offline")
        self.assertIn("299,792,458", resp.text)
        self.assertGreater(resp.total_tokens, 0)
        self.assertIsNone(resp.error)

    def test_openai_missing_api_key(self):
        provider = OpenAIProvider(api_key="")
        resp = provider.generate("Test prompt")
        self.assertIsNotNone(resp.error)
        self.assertIn("Missing or invalid OPENAI_API_KEY", resp.error)

    def test_gemini_missing_api_key(self):
        provider = GeminiProvider(api_key="")
        resp = provider.generate("Test prompt")
        self.assertIsNotNone(resp.error)
        self.assertIn("Missing or invalid GEMINI_API_KEY", resp.error)

    @patch("requests.post")
    def test_openai_timeout_handling(self, mock_post):
        mock_post.side_effect = requests.Timeout("Connection timed out")
        provider = OpenAIProvider(api_key="sk-fake-key-for-testing")
        resp = provider.generate("Test prompt")
        self.assertIsNotNone(resp.error)
        self.assertIn("timed out", resp.error)

    @patch("requests.post")
    def test_openai_401_handling(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_post.return_value = mock_resp

        provider = OpenAIProvider(api_key="sk-invalid-test-key")
        resp = provider.generate("Test prompt")
        self.assertIsNotNone(resp.error)
        self.assertIn("401", resp.error)

    def test_default_factory_returns_offline_when_no_credentials(self):
        # When environment has no active keys, factory defaults to offline
        provider = get_llm_provider("offline")
        self.assertIsInstance(provider, OfflineLocalProvider)


if __name__ == "__main__":
    unittest.main()
