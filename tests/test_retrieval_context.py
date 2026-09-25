from langchain_core.documents import Document

from fieldguide.retrieval_text import infer_source, retrieval_text


def doc(source, text="Use a quadrat."):
    return Document(page_content=text, metadata={"source": source, "section": "Field recording"})


def test_explicit_document_name_is_resolved_but_general_topic_is_not():
    docs = [doc("Vegetation SOP_submit_2025.pdf"), doc("Bird monitoring plan.pdf")]
    assert (
        infer_source("According to the vegetation SOP, what do I record?", docs)
        == docs[0].metadata["source"]
    )
    assert infer_source("What do I record during vegetation monitoring?", docs) is None


def test_ambiguous_revision_is_not_silently_selected():
    docs = [doc("Vegetation SOP_2024.pdf"), doc("Vegetation SOP_2025.pdf")]
    assert infer_source("What does the vegetation SOP require?", docs) is None


def test_retrieval_context_does_not_change_quotable_text():
    document = doc("Vegetation SOP.pdf")
    value = retrieval_text(document)
    assert "Vegetation SOP" in value and "Field recording" in value
    assert value.endswith(document.page_content)
    assert document.page_content == "Use a quadrat."
