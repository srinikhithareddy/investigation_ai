from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.config import Settings
from app.llm.client import LLMClient, LLMError


def test_settings_read_gemini_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", " test-key ")
    monkeypatch.setenv("GEMINI_MODEL", " test-model ")

    configured = Settings()

    assert configured.gemini_api_key == "test-key"
    assert configured.gemini_model == "test-model"


def test_blank_model_uses_default(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", " ")

    assert Settings().gemini_model == "gemini-2.5-flash"


def test_missing_api_key_has_clear_error():
    client = LLMClient(api_key=" ")

    with pytest.raises(LLMError, match="GEMINI_API_KEY is not set"):
        client._get_client()


def test_complete_uses_google_genai_request_shape():
    generate_content = Mock(return_value=SimpleNamespace(text="  answer  "))
    client = LLMClient(api_key="test-key", model="test-model")
    client._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )

    result = client.complete(
        system="system instructions",
        user="user request",
        max_tokens=42,
        temperature=0.4,
    )

    assert result == "answer"
    assert generate_content.call_args.kwargs == {
        "model": "test-model",
        "contents": "user request",
        "config": {
            "system_instruction": "system instructions",
            "temperature": 0.4,
            "max_output_tokens": 42,
        },
    }


def test_provider_error_is_redacted():
    generate_content = Mock(side_effect=RuntimeError("secret-test-key in error"))
    client = LLMClient(api_key="secret-test-key", model="test-model")
    client._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )

    with pytest.raises(LLMError, match="Gemini request failed") as error:
        client.complete(system="system", user="request")

    assert "secret-test-key" not in str(error.value)
    assert error.value.__cause__ is None