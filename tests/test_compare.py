import json
from pathlib import Path

import pandas as pd
import pytest

from fieldguide.compare import (
    compare_retrieval,
    comparison_summary_frame,
    normalize,
    summarize_comparison,
)


class Index:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, question, *, k, mode):
        if mode == "dense":
            raise RuntimeError("Do not include private exception text in reports.")
        return self.chunks[:k]


def write_cases(tmp_path, cases):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return path


def case(**overrides):
    return {
        "id": "v", "question": "What units?", "expected_sources": ["a.md", "b.md"],
        "expected_evidence": [
            {"source": "a.md", "text": "Record centimeters."},
            {"source": "b.md", "text": "Use NA."},
        ], **overrides,
    }


def test_passage_metrics_distinguish_right_document_from_right_text(tmp_path):
    index = Index([
        {"source": "a.md", "text": "Introduction"},
        {"source": "a.md", "text": "Please RECORD\n centimeters. Now."},
        {"source": "b.md", "text": "Introduction"},
    ])
    frame, details = compare_retrieval(
        {"original": index}, write_cases(tmp_path, [case()]), modes=["hybrid"]
    )
    row = frame.iloc[0]
    assert row.source_recall == 1
    assert row.source_mrr == 1
    assert row.evidence_recall == 0.5
    assert row.evidence_mrr == 0.5
    assert len(details[0]["retrieved"]) == 3


def test_wrong_source_and_split_excerpts_are_not_passage_hits(tmp_path):
    index = Index([
        {"source": "b.md", "text": "Record centimeters."},
        {"source": "a.md", "text": "Record"},
        {"source": "a.md", "text": "centimeters."},
    ])
    frame, _ = compare_retrieval(
        {"test": index}, write_cases(tmp_path, [case()]), modes=["lexical"]
    )
    assert frame.iloc[0].evidence_recall == 0
    assert frame.iloc[0].evidence_mrr == 0


def test_summary_exposes_errors_and_excludes_unanswerable_scores(tmp_path):
    cases = write_cases(tmp_path, [case(), case(id="u", answerable=False)])
    frame, details = compare_retrieval({"test": Index([])}, cases, modes=["dense", "hybrid"])
    summary = comparison_summary_frame(frame).set_index("mode")
    assert summary.loc["dense", "errors"] == 2
    assert summary.loc["dense", "evidence_recall_n"] == 0
    assert pd.isna(summary.loc["dense", "evidence_recall"])
    assert summary.loc["hybrid", "cases"] == 2
    assert summary.loc["hybrid", "evidence_recall_n"] == 1
    assert summary.loc["hybrid", "evidence_recall"] == 0
    assert details[0]["error"] == "RuntimeError"
    assert "private exception" not in str(details)
    assert "exclude errors" in summarize_comparison(frame)


@pytest.mark.parametrize("cases", [
    [case(), case()],
    [case(expected_evidence=[{"source": "unknown.md", "text": "No."}])],
    [case(expected_evidence=[{"source": "a.md", "text": "   "}])],
    [],
])
def test_invalid_labels_fail_before_search(tmp_path, cases):
    with pytest.raises(ValueError):
        compare_retrieval({"test": Index([])}, write_cases(tmp_path, cases))


def test_source_only_cases_have_no_evidence_score(tmp_path):
    frame, _ = compare_retrieval(
        {"test": Index([])}, write_cases(tmp_path, [case(expected_evidence=[])]),
        modes=["hybrid"],
    )
    assert frame.iloc[0].source_recall == 0
    assert pd.isna(frame.iloc[0].evidence_recall)


def test_public_gold_is_verbatim_and_all_cases_have_passage_labels():
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "eval/retrieval_questions.json").read_text(encoding="utf-8"))
    assert len(cases) == 11
    for item in cases:
        assert item["expected_evidence"]
        for evidence in item["expected_evidence"]:
            corpus = (root / "sample_corpus" / evidence["source"]).read_text(encoding="utf-8")
            assert normalize(evidence["text"]) in normalize(corpus)
