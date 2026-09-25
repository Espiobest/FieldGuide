import faiss
import numpy as np
import pytest
from langchain_core.documents import Document

from fieldguide.store import LocalIndex


class Embedder:
    def embed_query(self, question):
        return [1.0, 0.0]


@pytest.fixture
def index():
    documents = [
        Document(page_content="Record plant cover at each quadrat.",
                 metadata={"source": "vegetation.md", "chunk_id": "plant"}),
        Document(page_content="Record station elevation relative to NAVD88.",
                 metadata={"source": "water.md", "chunk_id": "datum"}),
        Document(page_content="Review the field notebook after sampling.",
                 metadata={"source": "vegetation.md", "chunk_id": "review"}),
    ]
    vectors = np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype="float32")
    store = faiss.IndexFlatIP(2)
    store.add(vectors)
    return LocalIndex(store, documents, {}, Embedder())


def test_dense_default_preserves_cosine(index):
    result = index.search("NAVD88", k=1)[0]
    assert result["chunk_id"] == "plant"
    assert result["score"] == pytest.approx(1.0)
    assert result["score_kind"] == "cosine"


def test_lexical_recovers_exact_term_without_embeddings(index):
    index.embedder = None
    results = index.search("NAVD88", mode="lexical")
    assert [r["chunk_id"] for r in results] == ["datum"]
    assert results[0]["score_kind"] == "bm25"
    assert results[0]["dense_score"] is None
    assert index.search("nonexistentword", mode="lexical") == []


def test_hybrid_promotes_document_supported_by_both_rankings(index):
    result = index.search("NAVD88", mode="hybrid", k=1)[0]
    assert result["chunk_id"] == "datum"
    assert result["score_kind"] == "rrf"
    assert result["dense_rank"] == 2
    assert result["lexical_rank"] == 1
    assert result["score"] == pytest.approx(1 / 62 + 1 / 61)


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
def test_source_filter_limits_rankings(index, mode):
    results = index.search("Record", k=1, source="WATER", mode=mode)
    assert results[0]["source"] == "water.md"
    if mode != "lexical":
        assert results[0]["dense_rank"] == 1
    assert index.search("Record", source="absent", mode=mode) == []


def test_lexical_statistics_reused(index):
    index.search("plant", mode="lexical")
    cached = index._lexical_corpus
    index.search("station", mode="lexical")
    assert index._lexical_corpus is cached


def test_unknown_mode_rejected(index):
    with pytest.raises(ValueError, match="mode"):
        index.search("plant", mode="unknown")
