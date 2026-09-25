"""Structured model clients for Gemini and local Ollama inference."""

import json
import os


def resolve_model(model: str | None = None, provider: str = "gemini") -> str:
    if provider == "ollama":
        name = model or os.getenv("FIELDGUIDE_OLLAMA_MODEL", "qwen3:4b")
        if "cloud" in name.lower() or "/" in name:
            raise ValueError("Use a locally downloaded Ollama model, such as qwen3:4b.")
        return name
    if provider != "gemini":
        raise ValueError(f"Unknown provider: {provider}")
    return model or os.getenv("FIELDGUIDE_MODEL", "gemini-2.5-flash")


def structured_model(schema, model: str | None = None, provider: str = "gemini"):
    name = resolve_model(model, provider)
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ValueError(
                'Install local model support with pip install -e ".[local]".'
            ) from None
        llm = ChatOllama(
            model=name,
            base_url="http://127.0.0.1:11434",
            temperature=0,
            reasoning=False,
            num_ctx=16384,
            num_predict=4096,
            keep_alive="10m",
            client_kwargs={"timeout": 180, "trust_env": False},
        )
        from langchain_core.messages import SystemMessage
        from langchain_core.runnables import RunnableLambda

        instruction = "Return JSON matching this schema: " + json.dumps(schema.model_json_schema())
        prepare = RunnableLambda(
            lambda prompt: [SystemMessage(content=instruction)] + prompt.to_messages()
        )
        return prepare | llm.with_structured_output(schema, method="json_schema")

    from langchain_google_genai import ChatGoogleGenerativeAI

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Set GEMINI_API_KEY or use --provider ollama for local inference.")
    llm = ChatGoogleGenerativeAI(
        model=name,
        api_key=key,
        timeout=60,
        max_retries=2,
        vertexai=False,
    )
    return llm.with_structured_output(schema, method="json_schema").bind(
        automatic_function_calling={"disable": True}
    )
