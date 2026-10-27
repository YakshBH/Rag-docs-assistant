from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.generation import llm


def test_ollama_provider_passes_output_token_limit(monkeypatch):
    response = Mock()
    response.json.return_value = {"response": "Ollama answer."}
    post = Mock(return_value=response)
    monkeypatch.setattr(llm, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm, "LLM_MAX_OUTPUT_TOKENS", 256)
    monkeypatch.setattr(llm.requests, "post", post)

    result = llm.generate_answer("Use the retrieved context.")

    assert result == "Ollama answer."
    post.assert_called_once()
    assert post.call_args.kwargs["json"]["options"] == {"num_predict": 256}


def test_gemini_provider_uses_prompt_and_configured_model(monkeypatch):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        text="Grounded answer. [act.pdf, page 7]"
    )
    genai_client = Mock(return_value=client)
    monkeypatch.setattr(llm, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "GEMINI_MODEL", "gemini-test")
    monkeypatch.setattr(llm, "LLM_MAX_OUTPUT_TOKENS", 256)
    monkeypatch.setattr(llm.genai, "Client", genai_client)

    result = llm.generate_answer("Use the retrieved context.")

    assert result == "Grounded answer. [act.pdf, page 7]"
    genai_client.assert_called_once_with(api_key="test-key")
    client.models.generate_content.assert_called_once()
    call = client.models.generate_content.call_args.kwargs
    assert call["model"] == "gemini-test"
    assert call["contents"] == "Use the retrieved context."
    assert call["config"].temperature == 0.1
    assert call["config"].max_output_tokens == 256


def test_gemini_provider_requires_api_key(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "GEMINI_API_KEY", "")

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is missing"):
        llm.generate_answer("prompt")


def test_groq_provider_uses_openai_compatible_client(monkeypatch):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Groq answer."))]
    )
    openai_client = Mock(return_value=client)
    monkeypatch.setattr(llm, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm, "GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(llm, "LLM_MAX_OUTPUT_TOKENS", 256)
    monkeypatch.setattr(llm, "OpenAI", openai_client)

    result = llm.generate_answer("Use the retrieved context.")

    assert result == "Groq answer."
    openai_client.assert_called_once_with(
        api_key="test-key",
        base_url="https://api.groq.com/openai/v1",
    )
    client.chat.completions.create.assert_called_once_with(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": "Use the retrieved context."}],
        temperature=0.1,
        max_tokens=256,
    )


def test_groq_provider_requires_api_key(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "")

    with pytest.raises(RuntimeError, match="GROQ_API_KEY is missing"):
        llm.generate_answer("prompt")
