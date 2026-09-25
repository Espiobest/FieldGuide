"""Document context for retrieval, separate from quotable passage text."""

import re
from pathlib import PurePosixPath


def context_prefix(document):
    metadata = document.metadata
    name = PurePosixPath(metadata.get("source", "")).stem.replace("_", " ")
    parts = [name]
    for field in ("title", "section"):
        value = metadata.get(field)
        if value and value not in parts:
            parts.append(value)
    context = " | ".join(parts)[:240]
    return context + "\n" if context else ""


def retrieval_text(document):
    return context_prefix(document) + document.page_content


def infer_source(question, documents):
    """Resolve an explicit, unambiguous document name; never infer a topic as a source."""
    normalized = " ".join(re.findall(r"\w+", question.casefold()))
    matches = set()
    for document in documents:
        source = document.metadata["source"]
        stem = PurePosixPath(source).stem
        # Underscores commonly separate a document name from revision/date metadata.
        name = stem.split("_")[0]
        words = re.findall(r"[a-z]+", name.casefold())
        if len(words) < 2 or "sop" not in words:
            continue
        phrase = " ".join(words)
        if f" {phrase} " in f" {normalized} ":
            matches.add(source)
    return next(iter(matches)) if len(matches) == 1 else None
