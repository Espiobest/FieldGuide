"""Exchange page and chunk records with the optional Spark ingestion notebook."""

import hashlib
import json
from pathlib import Path, PurePosixPath

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fieldguide.ingest import read_documents


class PageRecord(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    source: str
    page: int | None = Field(default=None, ge=1)
    text: str = Field(min_length=1)

    @field_validator("source")
    @classmethod
    def relative_source(cls, value):
        if (
            not value
            or PurePosixPath(value).is_absolute()
            or "\\" in value
            or ":" in value
            or ".." in PurePosixPath(value).parts
        ):
            raise ValueError("Source must be a relative document path.")
        return value

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value):
        if not value.strip():
            raise ValueError("Text must not be blank.")
        return value


class ChunkRecord(PageRecord):
    start_index: int = Field(ge=0)
    chunk_id: str
    chunk_size: int = Field(ge=100)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_chunk(self):
        if self.chunk_overlap >= self.chunk_size or len(self.text) > self.chunk_size:
            raise ValueError("Invalid chunk length or overlap.")
        identity = f"{self.source}:{self.page}:{self.start_index}:{self.text}"
        expected = hashlib.sha256(identity.encode()).hexdigest()[:16]
        if self.chunk_id != expected:
            raise ValueError("Chunk ID does not match its source, position, and text.")
        return self


def export_pages(folder: Path, output: Path) -> tuple[int, list[str]]:
    documents, warnings = read_documents(folder)
    rows = [
        PageRecord(source=d.metadata["source"], page=d.metadata["page"], text=d.page_content)
        for d in documents
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")
    return len(rows), warnings


def import_chunks(path: Path) -> tuple[list[Document], int, int]:
    documents, ids, settings = [], set(), set()
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = ChunkRecord.model_validate(json.loads(line))
            except ValueError:
                raise ValueError(f"Invalid chunk record on line {line_number}.") from None
            if row.chunk_id in ids:
                raise ValueError(f"Duplicate chunk ID on line {line_number}.")
            ids.add(row.chunk_id)
            settings.add((row.chunk_size, row.chunk_overlap))
            documents.append(
                Document(
                    page_content=row.text,
                    metadata={
                        "source": row.source,
                        "page": row.page,
                        "start_index": row.start_index,
                        "chunk_id": row.chunk_id,
                        "chunk_method": "spark-fixed-character-v1",
                    },
                )
            )
    if not documents or len(settings) != 1:
        raise ValueError("Expected nonempty chunks with consistent size and overlap.")
    size, overlap = settings.pop()
    return documents, size, overlap
