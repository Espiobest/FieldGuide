"""Inspect local index quality without calling a language model."""

from collections import Counter, defaultdict
from math import ceil
from statistics import median

from fieldguide.retrieval_text import retrieval_text


def audit_index(index) -> dict:
    documents = index.documents
    lengths = sorted(len(doc.page_content) for doc in documents)
    sources = defaultdict(list)
    for doc in documents:
        sources[doc.metadata.get("source", "unknown")].append(doc)
    report = {
        "chunk_count": len(documents),
        "source_count": len(sources),
        "sources": [
            {
                "source": source,
                "chunks": len(docs),
                "sha256": sorted(
                    {d.metadata["source_sha256"] for d in docs if d.metadata.get("source_sha256")}
                ),
                "chunks_missing_hash": sum(not d.metadata.get("source_sha256") for d in docs),
            }
            for source, docs in sorted(sources.items())
        ],
        "characters": {
            "min": lengths[0] if lengths else 0,
            "median": median(lengths) if lengths else 0,
            "p95": lengths[ceil(len(lengths) * 0.95) - 1] if lengths else 0,
            "max": lengths[-1] if lengths else 0,
        },
        "duplicate_chunks": len(documents) - len({d.page_content for d in documents}),
        "chunk_methods": dict(
            sorted(
                Counter(
                    d.metadata.get("chunk_method", "legacy-recursive") for d in documents
                ).items()
            )
        ),
        "pdf_chunks_missing_pages": sum(
            str(d.metadata.get("source", "")).lower().endswith(".pdf")
            and not isinstance(d.metadata.get("page"), int)
            for d in documents
        ),
        "embedding_model": index.manifest.get("embedding_model", "unknown"),
        "ingestion_warnings": index.manifest.get("ingestion_warnings"),
        "truncation": {"status": "unavailable"},
    }
    client = getattr(index.embedder, "_client", None)
    tokenizer = getattr(client, "tokenizer", None)
    limit = getattr(client, "max_seq_length", None)
    if callable(tokenizer) and isinstance(limit, int) and limit > 0:
        token_lengths = []
        for start in range(0, len(documents), 64):
            encoded = tokenizer(
                [
                    retrieval_text(d) if index.manifest.get("retrieval_context") else d.page_content
                    for d in documents[start : start + 64]
                ],
                truncation=False,
                padding=False,
                add_special_tokens=True,
            )
            token_lengths.extend(len(ids) for ids in encoded["input_ids"])
        report["truncation"] = {
            "status": "checked",
            "max_tokens": limit,
            "chunks_exceeding_limit": sum(length > limit for length in token_lengths),
            "max_observed_tokens": max(token_lengths, default=0),
        }
    return report


def format_audit(report: dict) -> str:
    sizes = report["characters"]
    lines = [
        f"Chunks: {report['chunk_count']} | Sources: {report['source_count']}",
        f"Embedding model: {report['embedding_model']}",
        "Chunk characters: " + ", ".join(f"{key}={value:g}" for key, value in sizes.items()),
        f"Duplicate chunks (exact text): {report['duplicate_chunks']}",
        f"PDF chunks missing page citations: {report['pdf_chunks_missing_pages']}",
        "Chunk methods: "
        + ", ".join(f"{method}={count}" for method, count in report["chunk_methods"].items()),
    ]
    tokens = report["truncation"]
    warnings = report.get("ingestion_warnings")
    if warnings is None:
        lines.append("Extraction warnings: unavailable in this older index.")
    else:
        lines.append(f"Extraction warnings: {len(warnings)}")
        lines.extend(f"  {warning}" for warning in warnings)
    if tokens["status"] == "checked":
        lines.append(
            f"Embedding truncation risk: {tokens['chunks_exceeding_limit']} chunks exceed "
            f"{tokens['max_tokens']} tokens (largest: {tokens['max_observed_tokens']})."
        )
    else:
        lines.append("Embedding truncation: not checked; load the embedding model to measure.")
    lines.append("Source provenance:")
    for source in report["sources"]:
        hashes = ", ".join(source["sha256"]) or "unavailable"
        lines.append(
            f"  {source['source']}: {source['chunks']} chunks; SHA-256: {hashes}; "
            f"missing hash: {source['chunks_missing_hash']}"
        )
    return "\n".join(lines)
