import pytest
from langchain_core.documents import Document

from fieldguide.ingest import chunk_documents


def page(text, number=1, source="training.pdf"):
    return Document(page_content=text, metadata={"source": source, "page": number})


def test_sentence_boundaries_and_whole_sentence_overlap():
    sentences = [f"Observation {i} records the species and its cover." for i in range(8)]
    chunks = chunk_documents([page(" ".join(sentences))], size=150, overlap=55)
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 150 for chunk in chunks)
    assert all(chunk.page_content.endswith(".") for chunk in chunks)
    assert all(chunk.page_content.startswith("Observation") for chunk in chunks)
    assert sentences[2] in chunks[0].page_content and sentences[2] in chunks[1].page_content
    assert all(any(sentence in chunk.page_content for chunk in chunks) for sentence in sentences)


def test_pdf_sentence_continuation_preserves_page_range():
    chunks = chunk_documents([
        page("Record the species and"), page("its percent cover. Record NA for missing data.", 2),
    ])
    assert len(chunks) == 1
    assert chunks[0].page_content.startswith("Record the species and its percent cover.")
    assert chunks[0].metadata["page"] == 1
    assert chunks[0].metadata["page_end"] == 2


@pytest.mark.parametrize("second", [page("Other text.", 3), page("Other text.", 2, "other.pdf")])
def test_pdf_gaps_and_sources_are_not_joined(second):
    assert len(chunk_documents([page("First text."), second])) == 2


def test_long_sentence_and_unbroken_word_are_bounded_without_text_loss():
    text = "measurement " * 40 + "x" * 220
    chunks = chunk_documents([page(text)], size=100, overlap=0)
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)
    assert "".join(chunk.page_content.replace(" ", "") for chunk in chunks) == text.replace(" ", "")


def test_layout_normalization_keeps_words_numbers_and_negation():
    chunks = chunk_documents([page("Do not\nrecord 1.5 as zero.\n\n- Record NA.\n- Explain why.")])
    assert "Do not record 1.5 as zero." in chunks[0].page_content
    assert "\n- Explain why." in chunks[0].page_content


def test_page_range_does_not_include_previous_page_for_later_chunk():
    chunks = chunk_documents([
        page("First observation requires recording all species present in this marked plot."),
        page("Second observation requires recording all species present in another plot.", 2),
    ], size=100, overlap=0)
    assert [chunk.metadata["page"] for chunk in chunks] == [1, 2]
    assert [chunk.metadata["page_end"] for chunk in chunks] == [1, 2]


def test_recursive_baseline_and_invalid_strategy():
    chunks = chunk_documents([page("Repeated observation. " * 20)], strategy="recursive")
    assert chunks[0].metadata["chunk_id"]
    with pytest.raises(ValueError, match="strategy"):
        chunk_documents([], strategy="unknown")
