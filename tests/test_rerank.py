import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fieldguide.rerank import DEFAULT_RERANKER, LocalReranker, _load_model


@pytest.fixture
def encoder(monkeypatch):
    _load_model.cache_clear()
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    model = Mock()
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        SimpleNamespace(CrossEncoder=factory))
    yield factory, model
    _load_model.cache_clear()


def sources():
    return [
        {"id": "S1", "text": "Record plot ID", "source": "vegetation.pdf", "page": 2,
         "chunk_id": "a", "score": 0.7, "score_kind": "cosine"},
        {"id": "S2", "text": "Record water level", "source": "water.pdf", "page": 3,
         "chunk_id": "b", "score": 0.5, "score_kind": "cosine"},
    ]


def test_reranks_preserves_evidence_and_original_scores(encoder):
    factory, model = encoder
    model.predict.return_value = [-2.0, 4.0]
    original = sources()
    result = LocalReranker()("What water reading?", original, k=1)
    assert result == [{**original[1], "id": "S1", "retrieval_score": 0.5,
                       "retrieval_score_kind": "cosine", "score": 4.0,
                       "score_kind": "reranker", "rank": 1}]
    assert original == sources()
    factory.assert_called_once_with(DEFAULT_RERANKER, device="cpu", trust_remote_code=False,
                                    local_files_only=False, max_length=512)
    model.predict.assert_called_once_with(
        [("What water reading?", row["text"]) for row in original],
        batch_size=16, show_progress_bar=False, convert_to_numpy=True,
    )


def test_lazy_cached_model_and_offline_mode(encoder, monkeypatch):
    factory, model = encoder
    reranker = LocalReranker("local/model")
    factory.assert_not_called()
    assert reranker("question", []) == []
    factory.assert_not_called()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model.predict.return_value = [1, 1]
    assert [r["chunk_id"] for r in reranker("question", sources())] == ["a", "b"]
    LocalReranker("local/model")("question", sources())
    factory.assert_called_once_with("local/model", device="cpu", trust_remote_code=False,
                                    local_files_only=True, max_length=512)


def test_scores_answer_windows_but_returns_full_original_passage(encoder):
    rows = sources()
    rows[1]["text"] = "Unrelated setup. Record the reading. Return after one year."
    def score(pairs, **kwargs):
        return [5.0 if "after one year" in passage else -1.0 for _, passage in pairs]
    encoder[1].predict.side_effect = score
    result = LocalReranker()("When to return?", rows, k=1)
    assert result[0]["chunk_id"] == "b"
    assert result[0]["text"] == rows[1]["text"]
    assert result[0]["score"] == 5.0
    pairs = encoder[1].predict.call_args.args[0]
    assert len(pairs) > len(rows)


@pytest.mark.parametrize("scores", [[float("nan"), 1], [float("inf"), 1], [1],
                                  [[1, 2], [3, 4]]])
def test_rejects_invalid_model_scores(encoder, scores):
    encoder[1].predict.return_value = scores
    with pytest.raises(ValueError, match="finite relevance score"):
        LocalReranker()("question", sources())


@pytest.mark.parametrize("question,k", [("", 1), ("  ", 1), ("question", 0)])
def test_rejects_invalid_query(encoder, question, k):
    with pytest.raises(ValueError, match="nonempty question"):
        LocalReranker()(question, sources(), k)
    encoder[0].assert_not_called()
