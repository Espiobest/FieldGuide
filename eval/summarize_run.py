"""Headline metrics, per-category counts and 3-fold spread from one eval run's scores.csv."""

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

FOLDS = 3


def mean_n(series):
    values = series.dropna()
    return {"mean": float(values.mean()) if len(values) else None, "n": int(len(values))}


def metrics(frame):
    answerable = frame[frame.answerable]
    unanswerable = frame[~frame.answerable]
    verified = frame[frame.status == "verified"]
    return {
        "cases": int(len(frame)),
        "errors": int((frame.status == "error").sum()),
        "verified_rate_answerable": {
            "k": int((answerable.status == "verified").sum()),
            "n": int(len(answerable)),
        },
        "abstention_rate_unanswerable": {
            "k": int((unanswerable.status == "abstained").sum()),
            "n": int(len(unanswerable)),
        },
        "behavior_correct": mean_n(frame.behavior_correct.astype(float)),
        "source_recall": mean_n(frame.source_recall),
        "answer_source_recall": mean_n(frame.answer_source_recall),
        "quote_validity": mean_n(frame.quote_validity),
        "faithfulness_verified": mean_n(verified.faithfulness),
        "correctness_answerable": mean_n(answerable.correctness),
        "correctness_verified_answerable": mean_n(
            answerable[answerable.status == "verified"].correctness
        ),
        "reference_f1_answerable": mean_n(answerable.reference_f1),
        "mean_seconds": mean_n(frame.seconds),
    }


def fold_frames(frame, cases):
    order = sorted(cases, key=lambda c: (c["category"], c["id"]))
    fold_of = {case["id"]: position % FOLDS for position, case in enumerate(order)}
    fold = frame.id.map(fold_of)
    return [frame[fold == number] for number in range(FOLDS)]


def spread(folds, extract):
    values = [extract(metrics(fold)) for fold in folds]
    values = [v for v in values if v is not None and v == v]
    if not values:
        return None
    return {
        "per_fold": values,
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
    }


def rate(entry):
    return entry["k"] / entry["n"] if entry["n"] else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.run / "scores.csv")
    for column in ("faithfulness", "correctness", "answer_source_recall", "quote_validity"):
        if column not in frame:
            frame[column] = float("nan")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    categories = {case["id"]: case["category"] for case in cases}
    frame["category"] = frame.id.map(categories)
    report = {"overall": metrics(frame), "by_category": {}, "folds": {}}
    for name, group in frame.groupby("category"):
        report["by_category"][name] = {
            "n": int(len(group)),
            "verified": int((group.status == "verified").sum()),
            "behavior_correct": mean_n(group.behavior_correct.astype(float)),
            "correctness": mean_n(group.correctness),
        }
    folds = fold_frames(frame, cases)
    report["folds"]["sizes"] = [int(len(fold)) for fold in folds]
    for label, extract in {
        "verified_rate_answerable": lambda m: rate(m["verified_rate_answerable"]),
        "abstention_rate_unanswerable": lambda m: rate(m["abstention_rate_unanswerable"]),
        "behavior_correct": lambda m: m["behavior_correct"]["mean"],
        "source_recall": lambda m: m["source_recall"]["mean"],
        "faithfulness_verified": lambda m: m["faithfulness_verified"]["mean"],
        "correctness_answerable": lambda m: m["correctness_answerable"]["mean"],
    }.items():
        report["folds"][label] = spread(folds, extract)
    print(json.dumps(report, indent=2))
    (args.run / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
