from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.generation import llm


def test_gemini_provider_uses_prompt_and_configured_model(monkeypatch):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        text="Grounded answer. [act.pdf, page 7]"
    )
    genai_client = Mock(return_value=client)
    monkeypatch.setattr(llm, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "GEMINI_MODEL", "gemini-test")
    monkeypatch.setattr(llm.genai, "Client", genai_client)

    result = llm.generate_answer("Use the retrieved context.")

    assert result == "Grounded answer. [act.pdf, page 7]"
    genai_client.assert_called_once_with(api_key="test-key")
    client.models.generate_content.assert_called_once()
    call = client.models.generate_content.call_args.kwargs
    assert call["model"] == "gemini-test"
    assert call["contents"] == "Use the retrieved context."
    assert call["config"].temperature == 0.1


def test_gemini_provider_requires_api_key(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "GEMINI_API_KEY", "")

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is missing"):
        llm.generate_answer("prompt")
