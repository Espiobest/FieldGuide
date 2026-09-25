import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from fieldguide.agents import Draft
from fieldguide.providers import resolve_model, structured_model


def test_ollama_uses_local_endpoint_and_structured_output(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    model = Mock()
    messages = []
    model.with_structured_output.return_value = RunnableLambda(lambda value: messages.extend(value))
    constructor = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "langchain_ollama", SimpleNamespace(ChatOllama=constructor))
    result = structured_model(Draft, "qwen3:4b", "ollama")
    options = constructor.call_args.kwargs
    assert options["base_url"] == "http://127.0.0.1:11434"
    assert options["client_kwargs"]["trust_env"] is False
    assert options["reasoning"] is False
    model.with_structured_output.assert_called_once_with(Draft, method="json_schema")
    result.invoke(ChatPromptTemplate.from_messages([("human", "Question")]).invoke({}))
    assert "answerable" in messages[0].content
    assert messages[-1].content == "Question"


def test_provider_models_have_separate_defaults(monkeypatch):
    monkeypatch.setenv("FIELDGUIDE_MODEL", "gemini-custom")
    monkeypatch.setenv("FIELDGUIDE_OLLAMA_MODEL", "qwen3:4b")
    assert resolve_model(provider="ollama") == "qwen3:4b"
    assert resolve_model(provider="gemini") == "gemini-custom"
    assert resolve_model("override", "ollama") == "override"


@pytest.mark.parametrize("model", ["qwen3:cloud", "host/model"])
def test_cloud_models_not_accepted_as_local(model):
    with pytest.raises(ValueError, match="locally downloaded"):
        structured_model(Draft, model, "ollama")


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="Unknown provider"):
        resolve_model(provider="other")
