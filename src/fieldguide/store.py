"""Local cosine retrieval using normalized embeddings and a FAISS index."""

import hashlib
import json
import os
import uuid
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document

DEFAULT_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"


def embeddings(model: str = DEFAULT_EMBEDDING):
    from langchain_huggingface import HuggingFaceEmbeddings

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
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

    def search(self, question: str, k: int = 5, source: str | None = None):
        if not question.strip() or k < 1:
            raise ValueError("Provide a nonempty question and a positive retrieval count.")
        if self.embedder is None:
            raise ValueError("Load the index with embeddings enabled before searching.")
        if self.index.ntotal == 0:
            return []
        vector = np.asarray([self.embedder.embed_query(question)], dtype="float32")
        faiss.normalize_L2(vector)
        count = self.index.ntotal if source else min(k, self.index.ntotal)
        scores, positions = self.index.search(vector, count)
        results = []
        for score, position in zip(scores[0], positions[0], strict=True):
            if position < 0:
                continue
            doc = self.documents[position]
            if source and source.casefold() not in doc.metadata["source"].casefold():
                continue
            results.append(
                {
                    "id": f"S{len(results) + 1}",
                    "text": doc.page_content,
                    **doc.metadata,
                    "score": float(score),
                }
            )
            if len(results) == k:
                break
        return results
