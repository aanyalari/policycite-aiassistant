import unittest
from unittest.mock import patch

import llm_provider


class LLMProviderTests(unittest.TestCase):
    @patch("langchain_ollama.ChatOllama")
    def test_attribution_uses_model_separate_from_answer_generation(self, chat):
        with (
            patch.object(llm_provider, "LLM_PROVIDER", "ollama"),
            patch.object(llm_provider, "OLLAMA_CHAT_MODEL", "llama3.2:1b"),
            patch.object(llm_provider, "ATTRIBUTION_PROVIDER", "ollama"),
            patch.object(llm_provider, "ATTRIBUTION_MODEL", "qwen2.5:3b"),
        ):
            llm_provider.get_chat_llm()
            llm_provider.get_attribution_llm()

        self.assertEqual(chat.call_args_list[0].kwargs["model"], "llama3.2:1b")
        self.assertEqual(chat.call_args_list[1].kwargs["model"], "qwen2.5:3b")

    def test_unknown_attribution_provider_is_rejected(self):
        with patch.object(llm_provider, "ATTRIBUTION_PROVIDER", "unknown"):
            with self.assertRaisesRegex(ValueError, "ATTRIBUTION_PROVIDER"):
                llm_provider.get_attribution_llm()


if __name__ == "__main__":
    unittest.main()
