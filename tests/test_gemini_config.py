import json

from google.genai import models, types

from fieldguide.agents import make_chains
from fieldguide.evaluate import make_judge


def test_structured_chains_disable_automatic_function_calling(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    responses = iter(
        [
            {"answerable": False, "claims": []},
            {"approved": False, "answers_question": False, "checks": [], "feedback": "No evidence"},
            {"faithfulness": 0, "correctness": 0, "reason": "No evidence"},
        ]
    )
    configs = []

    def generate(self, *, model, contents, config):
        configs.append(config)
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model", parts=[types.Part(text=json.dumps(next(responses)))]
                    ),
                    finish_reason="STOP",
                )
            ]
        )

    monkeypatch.setattr(models.Models, "generate_content", generate)
    answerer, verifier = make_chains()
    answerer.invoke({"question": "Unknown?", "context": "[]", "feedback": ""})
    verifier.invoke({"question": "Unknown?", "context": "[]", "draft": "{}"})
    make_judge().invoke({"question": "Unknown?", "reference": "", "answer": "", "context": "[]"})
    assert len(configs) == 3
    assert all(config.automatic_function_calling.disable for config in configs)
    assert all(config.response_mime_type == "application/json" for config in configs)
