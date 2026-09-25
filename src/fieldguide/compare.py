"""Compare retrieval settings against source and passage relevance labels."""

import json
import time
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, model_validator

METRICS = ("source_recall", "source_mrr", "evidence_recall", "evidence_mrr")
MODES = ("dense", "lexical", "hybrid")


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


class ExpectedEvidence(BaseModel):
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)


class RetrievalCase(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_sources: list[str]
    expected_evidence: list[ExpectedEvidence] = Field(default_factory=list)
    answerable: bool = True

    @model_validator(mode="after")
    def validate_labels(self):
        if not self.question.strip() or not self.id.strip():
            raise ValueError("Question and ID must not be blank.")
        if self.answerable and not self.expected_sources:
            raise ValueError("Answerable cases need expected sources.")
        if len(set(self.expected_sources)) != len(self.expected_sources):
            raise ValueError("Expected sources must be unique.")
        for evidence in self.expected_evidence:
            if evidence.source not in self.expected_sources or not evidence.text.strip():
                raise ValueError("Evidence needs a nonblank excerpt and an expected source.")
        labels = [(e.source, normalize(e.text)) for e in self.expected_evidence]
        if len(labels) != len(set(labels)):
            raise ValueError("Expected evidence must be unique.")
        return self


def compare_retrieval(
    indexes: dict, cases_path: Path, *, k: int = 3, modes=MODES, rerank_model=None
):
    """Return case-level measurements and retrieved passages; never invoke an LLM.

    Evidence recall requires each gold excerpt to occur within one retrieved chunk
    from its labeled source. MRR uses the first relevant chunk's reciprocal rank.
    Errors and unanswerable cases have missing relevance scores, not zero scores.
    """
    modes = tuple(modes)
    if not indexes or k < 1 or not modes or any(mode not in MODES for mode in modes):
        raise ValueError("Provide indexes, a positive k, and dense/lexical/hybrid modes.")
    if len(set(modes)) != len(modes):
        raise ValueError("Retrieval modes must be unique.")
    cases = [
        RetrievalCase.model_validate(item)
        for item in json.loads(Path(cases_path).read_text(encoding="utf-8"))
    ]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Retrieval cases must be nonempty with unique IDs.")
    rows, details = [], []
    for name, index in indexes.items():
        for mode in modes:
            for case in cases:
                start = time.perf_counter()
                row = dict.fromkeys(METRICS)
                label = mode + "+rerank" if rerank_model else mode
                row.update(index=name, mode=label, k=k, id=case.id, answerable=case.answerable)
                detail = {"index": name, "mode": mode, "k": k, "case": case.model_dump()}
                try:
                    options = {"k": k, "mode": mode}
                    if rerank_model:
                        options["rerank_model"] = rerank_model
                    retrieved = index.search(case.question, **options)
                    detail["retrieved"] = retrieved
                    row["status"] = "ok"
                    if case.answerable:
                        expected = set(case.expected_sources)
                        ranks = [
                            rank
                            for rank, chunk in enumerate(retrieved, 1)
                            if chunk["source"] in expected
                        ]
                        found = {chunk["source"] for chunk in retrieved}
                        row["source_recall"] = len(found & expected) / len(expected)
                        row["source_mrr"] = 1 / min(ranks) if ranks else 0.0
                        if case.expected_evidence:
                            evidence_ranks = [
                                next(
                                    (
                                        rank
                                        for rank, chunk in enumerate(retrieved, 1)
                                        if chunk["source"] == evidence.source
                                        and normalize(evidence.text) in normalize(chunk["text"])
                                    ),
                                    None,
                                )
                                for evidence in case.expected_evidence
                            ]
                            matches = [rank for rank in evidence_ranks if rank is not None]
                            row["evidence_recall"] = len(matches) / len(evidence_ranks)
                            row["evidence_mrr"] = 1 / min(matches) if matches else 0.0
                except Exception as exc:
                    row.update(dict.fromkeys(METRICS))
                    row.update(status="error", error=type(exc).__name__)
                    detail["error"] = type(exc).__name__
                row["seconds"] = time.perf_counter() - start
                rows.append(row)
                details.append(detail)
    return pd.DataFrame(rows), details


def comparison_summary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return grouped means with denominators so failures cannot hide behind averages."""
    rows = []
    for (name, mode, k), group in frame.groupby(["index", "mode", "k"], sort=False):
        row = {
            "index": name,
            "mode": mode,
            "k": k,
            "cases": len(group),
            "errors": int((group.status == "error").sum()),
            "answerable": int(group.answerable.sum()),
            "mean_seconds": group.seconds.mean(),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[metric] = values.mean() if len(values) else float("nan")
            row[f"{metric}_n"] = len(values)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_comparison(frame: pd.DataFrame) -> str:
    summary = comparison_summary_frame(frame)
    return summary.to_string(index=False, float_format=lambda value: f"{value:.3f}") + (
        "\n\nRelevance means exclude errors and unanswerable cases; *_n shows each denominator."
        "\nEvidence metrics require labeled excerpts within individual retrieved chunks."
        "\nTimings include query embedding and retrieval, but exclude index/model loading."
    )
