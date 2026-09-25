from types import SimpleNamespace

from langchain_core.documents import Document

from fieldguide.audit import audit_index, format_audit


def make_index(documents, embedder=None):
    return SimpleNamespace(documents=documents, embedder=embedder,
                           manifest={"embedding_model": "test-model"})


def test_audit_detects_duplicates_missing_provenance_and_pages():
    docs = [
        Document(page_content="same", metadata={"source": "a.pdf", "page": 1,
                 "source_sha256": "abc", "chunk_method": "sentence-v1"}),
        Document(page_content="same", metadata={"source": "a.pdf"}),
        Document(page_content="longer text", metadata={"source": "b.md"}),
    ]
    report = audit_index(make_index(docs))
    assert report["chunk_count"] == 3
    assert report["source_count"] == 2
    assert report["duplicate_chunks"] == 1
    assert report["pdf_chunks_missing_pages"] == 1
    assert report["characters"] == {"min": 4, "median": 4, "p95": 11, "max": 11}
    assert report["sources"][0]["sha256"] == ["abc"]
    assert report["sources"][0]["chunks_missing_hash"] == 1
    assert "not checked" in format_audit(report)


def test_audit_counts_actual_token_lengths_without_truncation():
    def tokenize(texts, **kwargs):
        assert kwargs == {"truncation": False, "padding": False, "add_special_tokens": True}
        return {"input_ids": [[0] * (len(text.split()) + 2) for text in texts]}

    embedder = SimpleNamespace(_client=SimpleNamespace(tokenizer=tokenize, max_seq_length=5))
    docs = [Document(page_content=text) for text in ["one two three", "one two three four"]]
    report = audit_index(make_index(docs, embedder))
    assert report["truncation"] == {
        "status": "checked", "max_tokens": 5, "chunks_exceeding_limit": 1,
        "max_observed_tokens": 6,
    }
    assert "1 chunks exceed 5 tokens" in format_audit(report)


def test_empty_index_audit_is_well_defined():
    report = audit_index(make_index([]))
    assert report["characters"] == {"min": 0, "median": 0, "p95": 0, "max": 0}
    assert report["sources"] == []
    assert "Chunks: 0" in format_audit(report)
