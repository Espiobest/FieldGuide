"""Matched retrieval ablation: dense vs hybrid, reranker off vs on, over one case file."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from fieldguide.rerank import DEFAULT_RERANKER
from fieldguide.store import LocalIndex

CONFIGS = {
    "dense": {"mode": "dense"},
    "hybrid": {"mode": "hybrid"},
    "dense+rerank": {"mode": "dense", "rerank_model": DEFAULT_RERANKER},
    "hybrid+rerank": {"mode": "hybrid", "rerank_model": DEFAULT_RERANKER},
}


def score(index, cases, k, options):
    rows = []
    for case in cases:
        expected = case["expected_sources"]
        start = time.perf_counter()
        results = index.search(case["question"], k=k, **options)
        seconds = time.perf_counter() - start
        ranks = [
            next((r for r, s in enumerate(results, 1) if s["source"] == name), None)
            for name in expected
        ]
        found = [r for r in ranks if r]
        rows.append(
            {
                "id": case["id"],
                "source_recall": len(found) / len(expected),
                "source_mrr": 1 / min(found) if found else 0.0,
                "seconds": seconds,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    index = LocalIndex.load(args.index)
    cases = [
        c for c in json.loads(args.cases.read_text(encoding="utf-8")) if c["answerable"]
    ]
    index.search(cases[0]["question"], k=args.k, **CONFIGS["hybrid+rerank"])
    report = {"k": args.k, "n_answerable": len(cases), "index_count": index.manifest["count"]}
    for name, options in CONFIGS.items():
        rows = score(index, cases, args.k, options)
        report[name] = {
            "source_recall": statistics.mean(r["source_recall"] for r in rows),
            "source_mrr": statistics.mean(r["source_mrr"] for r in rows),
            "mean_seconds": statistics.mean(r["seconds"] for r in rows),
            "cases": rows,
        }
        print(
            f"{name:14s} recall={report[name]['source_recall']:.3f} "
            f"mrr={report[name]['source_mrr']:.3f} n={len(rows)} k={args.k}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
