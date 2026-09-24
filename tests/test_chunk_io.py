import hashlib
import json
from pathlib import Path

import pytest

from fieldguide.chunk_io import export_pages, import_chunks


def record(text="Example observation.", page=None):
    identity = f"notes.md:{page}:0:{text}"
    return dict(
        source="notes.md",
        page=page,
        text=text,
        start_index=0,
        chunk_id=hashlib.sha256(identity.encode()).hexdigest()[:16],
        chunk_size=1000,
        chunk_overlap=150,
    )


def test_export_preserves_text_and_does_not_overwrite(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    (folder / "notes.md").write_text("Example observation.", encoding="utf-8")
    target = tmp_path / "pages.jsonl"
    count, warnings = export_pages(folder, target)
    assert count == 1 and not warnings
    assert json.loads(target.read_text()) == {
        "source": "notes.md",
        "page": None,
        "text": "Example observation.",
    }
    with pytest.raises(FileExistsError):
        export_pages(folder, target)


def test_import_preserves_citation_and_settings(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(record(page=3)), encoding="utf-8")
    docs, size, overlap = import_chunks(path)
    assert (size, overlap) == (1000, 150)
    assert docs[0].metadata["page"] == 3
    assert docs[0].metadata["source"] == "notes.md"
    assert docs[0].page_content == "Example observation."


@pytest.mark.parametrize(
    "change",
    [
        {"chunk_id": "invalid"},
        {"text": "altered"},
        {"source": "../private.md"},
        {"page": 0},
        {"start_index": -1},
        {"chunk_overlap": 1000},
        {"text": " "},
    ],
)
def test_invalid_records_are_rejected(tmp_path, change):
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps({**record(), **change}), encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        import_chunks(path)


def test_duplicate_chunks_rejected(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text((json.dumps(record()) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        import_chunks(path)


def test_empty_import_rejected(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="nonempty"):
        import_chunks(path)


def test_notebook_has_valid_python_and_no_saved_outputs():
    notebook = json.loads(Path("notebooks/databricks_ingestion.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            compile("".join(cell["source"]), "notebook", "exec")
