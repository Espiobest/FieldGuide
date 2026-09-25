from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from fieldguide.token_budget import fit_embedding_window


@pytest.fixture(autouse=True)
def no_context_prefix(monkeypatch):
    monkeypatch.setattr("fieldguide.token_budget.context_prefix", lambda document: "")


class CharacterTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text)))}


def test_embedding_window_preserves_text_and_provenance():
    text = "Do not record zero. Record 25 centimeters. Preserve the original reading."
    doc = Document(
        page_content=text,
        metadata={
            "source": "sop.pdf",
            "page": 2,
            "page_end": 3,
            "chunk_id": "parent",
            "start_index": 100,
            "section": "Measurements",
        },
    )
    embedder = SimpleNamespace(
        _client=SimpleNamespace(
            tokenizer=CharacterTokenizer(),
            max_seq_length=32,
        )
    )
    chunks, limit = fit_embedding_window([doc], embedder)
    assert limit == 32 and len(chunks) > 1
    assert all(len(chunk.page_content) <= 30 for chunk in chunks)
    assert " ".join(chunk.page_content for chunk in chunks) == text
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)
    assert all(chunk.metadata["parent_chunk_id"] == "parent" for chunk in chunks)
    assert all(chunk.metadata["page_end"] == 3 for chunk in chunks)
    assert chunks[0].metadata["start_index"] == 100


def test_no_split_within_window_or_without_tokenizer():
    docs = [Document(page_content="Short", metadata={"chunk_id": "original"})]
    assert fit_embedding_window(docs, object()) == (docs, None)
    embedder = SimpleNamespace(
        _client=SimpleNamespace(
            tokenizer=CharacterTokenizer(),
            max_seq_length=32,
        )
    )
    assert fit_embedding_window(docs, embedder) == (docs, 32)


def test_short_sentence_does_not_force_next_sentence_to_be_cut():
    sentences = [
        "Note this.",
        "Record the station identifier and observer initials in the field notebook.",
    ]
    doc = Document(page_content=" ".join(sentences), metadata={"chunk_id": "parent"})
    embedder = SimpleNamespace(
        _client=SimpleNamespace(tokenizer=CharacterTokenizer(), max_seq_length=80),
    )
    chunks, _ = fit_embedding_window([doc], embedder)
    assert [chunk.page_content for chunk in chunks] == sentences


def test_oversized_sentence_is_split_without_losing_text():
    text = "Record " + "the long field measurement " * 15 + "now."
    doc = Document(page_content=text, metadata={"chunk_id": "parent"})
    embedder = SimpleNamespace(
        _client=SimpleNamespace(tokenizer=CharacterTokenizer(), max_seq_length=50),
    )
    chunks, _ = fit_embedding_window([doc], embedder)
    assert " ".join(chunk.page_content for chunk in chunks) == text
    assert all(len(chunk.page_content) <= 48 for chunk in chunks)


def test_context_prefix_counts_toward_embedding_window(monkeypatch):
    prefix = "Survey: "
    monkeypatch.setattr("fieldguide.token_budget.context_prefix", lambda document: prefix)
    text = "Record the plot. Record the plants. Record the cover."
    doc = Document(page_content=text, metadata={"chunk_id": "parent"})
    embedder = SimpleNamespace(
        _client=SimpleNamespace(tokenizer=CharacterTokenizer(), max_seq_length=32),
    )
    chunks, _ = fit_embedding_window([doc], embedder)
    assert " ".join(chunk.page_content for chunk in chunks) == text
    assert all(len(prefix + chunk.page_content) <= 30 for chunk in chunks)


def test_context_prefix_cannot_consume_entire_window(monkeypatch):
    monkeypatch.setattr("fieldguide.token_budget.context_prefix", lambda document: "x" * 30)
    doc = Document(page_content="Record cover.", metadata={"chunk_id": "parent"})
    embedder = SimpleNamespace(
        _client=SimpleNamespace(tokenizer=CharacterTokenizer(), max_seq_length=32),
    )
    with pytest.raises(ValueError, match="context exceeds"):
        fit_embedding_window([doc], embedder)
