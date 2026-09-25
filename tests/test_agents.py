import pytest

from fieldguide.agents import (
    ABSTENTION,
    Claim,
    ClaimCheck,
    Draft,
    Evidence,
    GroundedQA,
    Verdict,
    evidence_errors,
    require_api_permission,
    verdict_passes,
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
    assert result.abstention_reason == "verification_rejected"
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
    assert result.abstention_reason == "no_context"


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


def test_pdf_spacing_does_not_reject_matching_evidence():
    sources = [{**SOURCES[0], "text": "Use a 1 m quadrat . Record NA for missing values ."}]
    verifier = FakeChain(verdict())
    result = GroundedQA(FakeChain(draft()), verifier).answer("What size?", sources)
    assert result.status == "verified"
    assert len(verifier.calls) == 1


@pytest.mark.parametrize(
    "source,quote",
    [
        ("Use a 1 m quadrat .", "Use a 9 m quadrat."),
        ("Do not estimate values .", "Do estimate values."),
        ("Record 1 . 5 m .", "Record 1.5 m."),
        ("Record 1 .5 m .", "Record 1.5 m."),
        ("Record 1,5 m .", "Record 1.5 m."),
    ],
)
def test_quote_normalization_preserves_meaning(source, quote):
    assert evidence_errors(draft(quote=quote), [{**SOURCES[0], "text": source}])


def test_quote_failure_explains_abstention():
    result = GroundedQA(
        FakeChain(draft(quote="Invented"), draft(quote="Invented")), FakeChain()
    ).answer("What size?", SOURCES)
    assert result.abstention_reason == "invalid_citations"


def test_approval_is_derived_without_redundant_model_flag():
    response = {
        "approved": False,
        "answers_question": True,
        "checks": [{"claim_index": 0, "supported": True, "reason": "Explicit in source."}],
        "feedback": "All claims are supported.",
    }
    assert "approved" not in Verdict.model_json_schema()["properties"]
    result = GroundedQA(FakeChain(draft()), FakeChain(response)).answer("What size?", SOURCES)
    assert result.status == "verified"
    assert result.diagnostics[-1]["approved_by_checks"] is True
    assert "approved" not in result.diagnostics[-1]["verdict"]


@pytest.mark.parametrize("indices,count", [([], 0), ([0, 0], 2), ([0], 2), ([0, 2], 2)])
def test_approval_requires_nonempty_complete_unique_checks(indices, count):
    response = verdict(
        checks=[ClaimCheck(claim_index=i, supported=True, reason="Supported") for i in indices]
    )
    assert not verdict_passes(response, count)


def test_supported_but_irrelevant_answer_is_rejected():
    response = verdict().model_copy(update={"answers_question": False})
    result = GroundedQA(FakeChain(draft()), FakeChain(response), max_attempts=1).answer(
        "What units are water levels recorded in?", SOURCES
    )
    assert result.status == "abstained"
    assert result.diagnostics[-1]["approved_by_checks"] is False


def test_rejected_historical_generalization_still_abstains():
    sources = [{**SOURCES[0], "text": "Sampling occurred in August 2024."}]
    proposal = draft(text="Sample every August.", quote=sources[0]["text"])
    rejected = Verdict(
        answers_question=True,
        checks=[
            ClaimCheck(
                claim_index=0,
                supported=False,
                reason="A dated event does not establish an annual requirement.",
            )
        ],
        feedback="Keep the observation's date and past tense.",
    )
    result = GroundedQA(FakeChain(proposal), FakeChain(rejected), max_attempts=1).answer(
        "When should sampling occur?", sources
    )
    assert result.status == "abstained"
    assert not result.claims
    assert result.diagnostics[-1]["approved_by_checks"] is False


@pytest.mark.parametrize("recheck_supported", [True, False])
def test_reduced_answer_requires_fresh_verification(recheck_supported):
    proposal = Draft(answerable=True, claims=[draft().claims[0], draft(text="Use 9 m.").claims[0]])
    mixed = verdict(
        checks=[
            ClaimCheck(claim_index=0, supported=True, reason="Correct size"),
            ClaimCheck(claim_index=1, supported=False, reason="Wrong size"),
        ]
    )
    verifier = FakeChain(mixed, verdict(recheck_supported))
    result = GroundedQA(FakeChain(proposal), verifier, max_attempts=1).answer("What size?", SOURCES)
    assert result.status == ("verified" if recheck_supported else "abstained")
    assert "9 m" not in result.text
    assert len(verifier.calls) == 2
    assert result.diagnostics[-1]["stage"] == "verify_reduced_answer"
    assert result.diagnostics[-1]["kept_claim_indices"] == [0]


def test_reduced_answer_cannot_omit_required_part_of_question():
    proposal = Draft(answerable=True, claims=[draft().claims[0], draft(text="Use 9 m.").claims[0]])
    mixed = verdict(
        checks=[
            ClaimCheck(claim_index=0, supported=True, reason="Size is supported"),
            ClaimCheck(claim_index=1, supported=False, reason="Does not answer units"),
        ]
    )
    incomplete = verdict().model_copy(update={"answers_question": False})
    result = GroundedQA(FakeChain(proposal), FakeChain(mixed, incomplete), max_attempts=1).answer(
        "What size quadrat and what water level units?", SOURCES
    )
    assert result.status == "abstained"


def test_duplicate_checks_do_not_trigger_reduced_answer():
    proposal = Draft(answerable=True, claims=[draft().claims[0], draft(text="Use 9 m.").claims[0]])
    malformed = verdict(
        checks=[
            ClaimCheck(claim_index=0, supported=True, reason="Supported"),
            ClaimCheck(claim_index=0, supported=False, reason="Conflicting duplicate"),
        ]
    )
    verifier = FakeChain(malformed)
    result = GroundedQA(FakeChain(proposal), verifier, max_attempts=1).answer("What size?", SOURCES)
    assert result.status == "abstained"
    assert len(verifier.calls) == 1
