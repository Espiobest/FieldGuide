"""Local cross-encoder scoring of retrieved query/passage pairs."""

import os
from functools import lru_cache

import numpy as np

from fieldguide.ingest import _units

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _load_model(name: str, offline: bool):
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        name,
        device="cpu",
        trust_remote_code=False,
        local_files_only=offline,
        max_length=512,
    )


class LocalReranker:
    def __init__(self, model: str = DEFAULT_RERANKER):
        self.model = model

    def rerank(self, question: str, sources: list[dict], k: int = 5) -> list[dict]:
        if not question.strip() or k < 1:
            raise ValueError("Provide a nonempty question and a positive retrieval count.")
        if not sources:
            return []
        offline = any(
            os.environ.get(name, "").upper() in {"1", "ON", "YES", "TRUE"}
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        )
        model = _load_model(self.model, offline)
        pairs, owners = [], []
        for position, source in enumerate(sources):
            text = source["text"]
            units = _units(text, 450)
            windows = [
                text[a : units[min(i + 1, len(units) - 1)][1]] for i, (a, _) in enumerate(units)
            ] or [text]
            for window in dict.fromkeys(windows):
                pairs.append((question, window))
                owners.append(position)
        window_scores = np.asarray(
            model.predict(
                pairs,
                batch_size=16,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=float,
        )
        if window_scores.shape == (len(pairs), 1):
            window_scores = window_scores[:, 0]
        if window_scores.shape != (len(pairs),) or not np.isfinite(window_scores).all():
            raise ValueError("Reranker must return one finite relevance score per passage.")
        scores = np.full(len(sources), -np.inf)
        for owner, score in zip(owners, window_scores, strict=True):
            scores[owner] = max(scores[owner], score)
        order = sorted(range(len(sources)), key=lambda i: (-scores[i], i))[:k]
        return [
            {
                **sources[position],
                "id": f"S{rank}",
                "retrieval_score": sources[position].get(
                    "retrieval_score", sources[position].get("score")
                ),
                "retrieval_score_kind": sources[position].get(
                    "retrieval_score_kind", sources[position].get("score_kind")
                ),
                "score": float(scores[position]),
                "score_kind": "reranker",
                "rank": rank,
            }
            for rank, position in enumerate(order, 1)
        ]

    __call__ = rerank


@lru_cache(maxsize=2)
def get_reranker(model: str = DEFAULT_RERANKER) -> LocalReranker:
    return LocalReranker(model)
