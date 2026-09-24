import pytest

from fieldguide.agents import (
    ABSTENTION,
    Claim,
    ClaimCheck,
    Draft,
    Evidence,
    GroundedQA,
    Verdict,
    require_api_permission,
)

SOURCES = [
    {
        "id": "S1",
        "text": "Use a 1 m quadrat. Record NA for missing values.",
        "source": "demo.md",
        "page": None,
        "chunk_id": "abc",
        "score": 0.8,
    }
]


class FakeChain:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def draft(text="Use a 1 m quadrat.", quote="Use a 1 m quadrat.", source="S1"):
    return Draft(
        answerable=True,
        claims=[Claim(text=text, evidence=[Evidence(source_id=source, quote=quote)])],
    )


def verdict(approved=True, checks=None):
    return Verdict(
        approved=approved,
        answers_question=True,
        checks=checks
        if checks is not None
        else [ClaimCheck(claim_index=0, supported=approved, reason="Check the specified size.")],
        feedback="Use the size given in the source.",
    )


def test_supported_answer_includes_only_used_sources():
    qa = GroundedQA(FakeChain(draft()), FakeChain(verdict()))
    extra = {**SOURCES[0], "id": "S2"}
    result = qa.answer("What size?", SOURCES + [extra])
    assert result.status == "verified"
    assert result.text == "Use a 1 m quadrat. [S1]"
    assert result.sources == SOURCES


@pytest.mark.parametrize("bad", [draft(source="S99"), draft(quote="Invented quotation")])
def test_bad_citation_retries_before_verifier(bad):
    answerer = FakeChain(bad, draft())
    verifier = FakeChain(verdict())
    result = GroundedQA(answerer, verifier).answer("What size?", SOURCES)
    assert result.status == "verified" and result.attempts == 2
    assert len(verifier.calls) == 1
    assert "Stricter retry" in answerer.calls[1]["feedback"]


def test_real_quote_does_not_bypass_semantic_verification():
    wrong = draft(text="Use a 9 m quadrat.")
    verifier = FakeChain(verdict(False), verdict(False))
    result = GroundedQA(FakeChain(wrong, wrong), verifier).answer("What size?", SOURCES)
    assert result.status == "abstained" and result.text == ABSTENTION
    assert not result.claims and not result.sources
    assert "9 m" not in result.text


def test_rejected_answer_can_be_corrected():
    answerer = FakeChain(draft(text="Use a 9 m quadrat."), draft())
    result = GroundedQA(answerer, FakeChain(verdict(False), verdict())).answer(
        "What size?", SOURCES
    )
    assert result.status == "verified" and result.attempts == 2


@pytest.mark.parametrize(
    "checks", [[], [ClaimCheck(claim_index=1, supported=True, reason="Wrong claim index")]]
)
def test_missing_verifier_checks_fail_closed(checks):
    result = GroundedQA(
        FakeChain(draft(), draft()), FakeChain(verdict(checks=checks), verdict(checks=checks))
    ).answer("What size?", SOURCES)
    assert result.status == "abstained"


def test_no_context_never_calls_model():
    result = GroundedQA(FakeChain(), FakeChain()).answer("Unknown?", [])
    assert result.status == "abstained" and result.attempts == 0


def test_model_abstention_never_returns_unverified_claims():
    result = GroundedQA(FakeChain(Draft(answerable=False, claims=[])), FakeChain()).answer(
        "Unknown?", SOURCES
    )
    assert result.status == "abstained"


def test_api_failure_does_not_return_a_draft():
    with pytest.raises(RuntimeError):
        GroundedQA(FakeChain(draft()), FakeChain(RuntimeError("offline"))).answer(
            "What size?", SOURCES
        )


def test_private_index_needs_explicit_api_permission(monkeypatch):
    monkeypatch.delenv("FIELDGUIDE_ALLOW_PRIVATE_API", raising=False)
    with pytest.raises(ValueError, match="private"):
        require_api_permission({"public": False})
    require_api_permission({"public": True})
    require_api_permission({"public": False}, allow_private=True)
