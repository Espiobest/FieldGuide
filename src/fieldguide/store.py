"""Local dense, BM25, and reciprocal-rank-fused document retrieval."""

import hashlib
import json
import math
import os
import re
import uuid
from collections import Counter
from functools import cached_property
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document

DEFAULT_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"


def embeddings(model: str = DEFAULT_EMBEDDING):
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from langchain_huggingface import HuggingFaceEmbeddings
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": "cpu", "trust_remote_code": False},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


class LocalIndex:
    def __init__(self, index, documents, manifest, embedder):
        self.index = index
        self.documents = documents
        self.manifest = manifest
        self.embedder = embedder

    @classmethod
    def build(
        cls,
        documents: list[Document],
        folder: Path,
        *,
        model: str = DEFAULT_EMBEDDING,
        public: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        if not documents:
            raise ValueError("Cannot build an index without document chunks.")
        embedder = embeddings(model)
        vectors = np.asarray(
            embedder.embed_documents([doc.page_content for doc in documents]), dtype="float32"
        )
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        folder.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        vector_name, docs_name = f"{generation}.faiss", f"{generation}.json"
        faiss.write_index(index, str(folder / vector_name))
        payload = json.dumps(
            [{"text": d.page_content, "metadata": d.metadata} for d in documents],
            ensure_ascii=False,
        ).encode("utf-8")
        (folder / docs_name).write_bytes(payload)
        manifest = {
            "version": 1,
            "embedding_model": model,
            "dimension": vectors.shape[1],
            "count": len(documents),
            "public": public,
            "vectors": vector_name,
            "documents": docs_name,
            "documents_sha256": hashlib.sha256(payload).hexdigest(),
            "vectors_sha256": hashlib.sha256((folder / vector_name).read_bytes()).hexdigest(),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_methods": sorted({
                doc.metadata.get("chunk_method", "legacy-recursive") for doc in documents
            }),
        }
        temporary = folder / f"{generation}.manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(folder / "manifest.json")
        return cls(index, documents, manifest, embedder)

    @classmethod
    def load(cls, folder: Path, *, embed: bool = True):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != 1:
            raise ValueError("Unsupported index version. Rebuild the index.")
        for key in ("documents", "vectors"):
            name = manifest[key]
            if Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Invalid index filename.")
            content = (folder / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != manifest[f"{key}_sha256"]:
                raise ValueError("Index checksum mismatch. Rebuild the index.")
        rows = json.loads((folder / manifest["documents"]).read_text(encoding="utf-8"))
        documents = [Document(page_content=r["text"], metadata=r["metadata"]) for r in rows]
        index = faiss.read_index(str(folder / manifest["vectors"]))
        if (
            index.ntotal != len(documents)
            or len(documents) != manifest["count"]
            or index.d != manifest["dimension"]
        ):
            raise ValueError("Index and document metadata do not match. Rebuild the index.")
        return cls(
            index, documents, manifest, embeddings(manifest["embedding_model"]) if embed else None
        )

    @cached_property
    def _lexical_corpus(self):
        terms = [Counter(self._tokens(doc.page_content)) for doc in self.documents]
        frequencies = Counter(term for counts in terms for term in counts)
        lengths = [sum(counts.values()) for counts in terms]
        average = sum(lengths) / max(len(lengths), 1)
        return terms, frequencies, lengths, average

    @staticmethod
    def _tokens(text):
        return re.findall(r"[^\W_]+", text.casefold())

    def _lexical_scores(self, question, eligible):
        terms, frequencies, lengths, average = self._lexical_corpus
        query = set(self._tokens(question))
        count = len(terms)
        scores = {}
        for position in eligible:
            score = 0.0
            for term in query:
                frequency = terms[position][term]
                if not frequency:
                    continue
                inverse = math.log(1 + (count - frequencies[term] + 0.5)
                                   / (frequencies[term] + 0.5))
                normalization = 1.5 * (0.25 + 0.75 * lengths[position] / (average or 1))
                score += inverse * frequency * 2.5 / (frequency + normalization)
            if score > 0:
                scores[position] = score
        return scores

    def search(
        self, question: str, k: int = 5, source: str | None = None, *, mode: str = "dense"
    ):
        if not question.strip() or k < 1:
            raise ValueError("Provide a nonempty question and a positive retrieval count.")
        if mode not in {"dense", "lexical", "hybrid"}:
            raise ValueError("Retrieval mode must be dense, lexical, or hybrid.")
        if mode != "lexical" and self.embedder is None:
            raise ValueError("Load the index with embeddings enabled before searching.")
        eligible = {
            i for i, doc in enumerate(self.documents)
            if not source or source.casefold() in doc.metadata["source"].casefold()
        }
        if not eligible:
            return []
        dense = {}
        lexical = {}
        if mode != "lexical":
            vector = np.asarray([self.embedder.embed_query(question)], dtype="float32")
            faiss.normalize_L2(vector)
            scores, positions = self.index.search(vector, self.index.ntotal)
            dense = {
                int(position): float(score)
                for score, position in zip(scores[0], positions[0], strict=True)
                if position in eligible
            }
        if mode != "dense":
            lexical = self._lexical_scores(question, eligible)
        dense_order = sorted(dense, key=lambda i: (-dense[i], i))
        lexical_order = sorted(lexical, key=lambda i: (-lexical[i], i))
        pool_size = max(20, k * 4)
        dense_rank = {position: rank for rank, position in
                      enumerate(dense_order[:pool_size], 1)}
        lexical_rank = {position: rank for rank, position in
                        enumerate(lexical_order[:pool_size], 1)}
        if mode == "hybrid":
            ranking_scores = {
                position: sum(1 / (60 + ranks[position])
                              for ranks in (dense_rank, lexical_rank) if position in ranks)
                for position in dense_rank.keys() | lexical_rank.keys()
            }
        else:
            ranking_scores = dense if mode == "dense" else lexical
        order = sorted(ranking_scores, key=lambda i: (-ranking_scores[i], i))[:k]
        results = []
        for position in order:
            doc = self.documents[position]
            results.append({
                **doc.metadata,
                "id": f"S{len(results) + 1}",
                "text": doc.page_content,
                "score": ranking_scores[position],
                "score_kind": {"dense": "cosine", "lexical": "bm25", "hybrid": "rrf"}[mode],
                "dense_score": dense.get(position),
                "lexical_score": lexical.get(position),
                "dense_rank": dense_rank.get(position),
                "lexical_rank": lexical_rank.get(position),
            })
        return results
