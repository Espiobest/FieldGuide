"""Read documents and preserve source locations through chunking."""

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED = {".pdf", ".docx", ".md", ".txt"}


def read_documents(folder: Path) -> tuple[list[Document], list[str]]:
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError(f"Document folder does not exist: {folder}")
    documents, warnings = [], []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(folder):
            continue
        relative = path.relative_to(folder).as_posix()
        if any(part.startswith(".") for part in path.relative_to(folder).parts):
            continue
        try:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                from pypdf import PdfReader

                pages = [page.extract_text() or "" for page in PdfReader(path).pages]
            elif suffix == ".docx":
                from docx import Document as WordDocument
                from docx.table import Table

                doc = WordDocument(path)
                blocks = []
                for block in doc.iter_inner_content():
                    if isinstance(block, Table):
                        blocks.extend(" | ".join(c.text for c in row.cells) for row in block.rows)
                    else:
                        blocks.append(block.text)
                pages = ["\n".join(blocks)]
            else:
                pages = [path.read_text(encoding="utf-8-sig")]
            for page, text in enumerate(pages, 1):
                text = text.replace("\x00", "").strip()
                if not text:
                    warnings.append(f"{relative}, page {page}: no text; may need OCR")
                    continue
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": relative,
                            "page": page if suffix == ".pdf" else None,
                        },
                    )
                )
        except Exception as exc:
            warnings.append(f"{relative}: could not read ({type(exc).__name__})")
    if not documents:
        raise ValueError("No readable documents found. Use text PDFs, DOCX, Markdown, or TXT.")
    return documents, warnings


def chunk_documents(
    documents: list[Document], size: int = 1000, overlap: int = 150
) -> list[Document]:
    if size < 100 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be >= 100 and 0 <= overlap < size.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        identity = (
            f"{chunk.metadata['source']}:{chunk.metadata['page']}:"
            f"{chunk.metadata['start_index']}:{chunk.page_content}"
        )
        chunk.metadata["chunk_id"] = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return chunks
