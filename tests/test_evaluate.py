import json

from fieldguide.agents import Answer, Claim, Evidence
from fieldguide.evaluate import run_evaluation, summary, token_f1


class Index:
    def search(self, question, k):
        return [{"source": "demo.md"}]


def test_retrieval_eval_does_not_claim_generation_scores(tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [{"id": "a", "question": "Units?", "reference": "cm", "expected_sources": ["demo.md"]}]
        ),
        encoding="utf-8",
    )
    frame, _ = run_evaluation(Index(), cases)
    assert frame.iloc[0].source_recall == 1.0
    assert "faithfulness" not in frame and "correctness" not in frame
    assert "retrieval_only" in summary(frame)


def test_unanswerable_cases_do_not_inflate_faithfulness(tmp_path):
    class QA:
        def answer(self, question, sources):
            return Answer(
                question=question,
                status="abstained",
                text="unknown",
                attempts=1,
                sources=[],
                retrieved=sources,
            )

    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {
                    "id": "a",
                    "question": "Unknown?",
                    "reference": "unknown",
                    "expected_sources": [],
                    "answerable": False,
                }
            ]
        ),
        encoding="utf-8",
    )
    frame, _ = run_evaluation(Index(), cases, qa=QA())
    assert bool(frame.iloc[0].behavior_correct)
    assert "faithfulness" not in frame


def test_token_f1_counts_repetition():
    assert token_f1("cm cm", "cm") < 1
    assert token_f1("1 m", "1 m") == 1


def test_generation_errors_count_as_failures(tmp_path):
    class QA:
        def answer(self, question, sources):
            raise RuntimeError("unavailable")

    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [{"id": "a", "question": "Units?", "reference": "cm", "expected_sources": ["demo.md"]}]
        ),
        encoding="utf-8",
    )
    frame, _ = run_evaluation(Index(), cases, qa=QA())
    assert frame.iloc[0].status == "error"
    assert not bool(frame.iloc[0].behavior_correct)
    assert "correctness" not in frame
    assert frame.iloc[0].reference_f1 == 0.0


def test_quote_validity_checks_evidence_instead_of_status(tmp_path):
    class Corpus:
        def search(self, question, k):
            return [{"id": "S1", "source": "demo.md", "text": "Record units in cm."}]

    class QA:
        def answer(self, question, sources):
            return Answer(
                question=question,
                status="verified",
                text="Use m.",
                attempts=1,
                sources=sources,
                retrieved=sources,
                claims=[Claim(text="Use m.", evidence=[Evidence(source_id="S1", quote="Use m.")])],
            )

    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [{"id": "a", "question": "Units?", "reference": "cm", "expected_sources": ["demo.md"]}]
        ),
        encoding="utf-8",
    )
    frame, _ = run_evaluation(Corpus(), cases, qa=QA())
    assert frame.iloc[0].quote_validity == 0.0
