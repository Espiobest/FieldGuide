"""Keep every indexed passage inside the embedding model's input limit."""

import hashlib

from langchain_core.documents import Document

from fieldguide.ingest import _units
from fieldguide.retrieval_text import context_prefix


def fit_embedding_window(documents, embedder):
    client = getattr(embedder, "_client", None)
    tokenizer = getattr(client, "tokenizer", None)
    limit = getattr(client, "max_seq_length", None)
    if tokenizer is None or not isinstance(limit, int):
        return documents, None
    budget = limit - tokenizer.num_special_tokens_to_add(pair=False)
    if budget < 8:
        raise ValueError("Embedding model input window is too small for document retrieval.")
    output = []
    for document in documents:
        text = document.page_content
        prefix = context_prefix(document)
        prefix_tokens = tokenizer(prefix, add_special_tokens=False, truncation=False)
        if len(prefix_tokens["input_ids"]) >= budget:
            raise ValueError("Document context exceeds the embedding model input window.")
        tokens = tokenizer(prefix + text, add_special_tokens=False, truncation=False)
        if len(tokens["input_ids"]) <= budget:
            output.append(document)
            continue
        start = 0
        sentence_ends = [end for _, end in _units(text, len(text))]
        while start < len(text):
            low, high = start + 1, len(text)
            stop = start
            while low <= high:
                middle = (low + high) // 2
                count = len(tokenizer(
                    prefix + text[start:middle], add_special_tokens=False, truncation=False,
                )["input_ids"])
                if count <= budget:
                    stop, low = middle, middle + 1
                else:
                    high = middle - 1
            if stop <= start:
                raise ValueError("Cannot fit source text into embedding model input window.")
            if stop < len(text):
                endings = [end for end in sentence_ends if start < end <= stop]
                if endings:
                    stop = endings[-1]
                else:
                    space = text.rfind(" ", start + 1, stop)
                    if space > start:
                        stop = space
            piece = text[start:stop].strip()
            metadata = {**document.metadata, "parent_chunk_id": document.metadata["chunk_id"]}
            metadata["start_index"] = document.metadata.get("start_index", 0) + start
            metadata["end_index"] = document.metadata.get("start_index", 0) + stop
            metadata["chunk_method"] = document.metadata.get("chunk_method", "recursive") + "+token"
            identity = f"{metadata['parent_chunk_id']}:{start}:{piece}"
            metadata["chunk_id"] = hashlib.sha256(identity.encode()).hexdigest()[:16]
            if piece:
                output.append(Document(page_content=piece, metadata=metadata))
            start = stop
            while start < len(text) and text[start].isspace():
                start += 1
    return output, limit
