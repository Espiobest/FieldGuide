import json
from unittest.mock import Mock

from fieldguide.cli import parser, run


def test_comparison_reuses_embedding_model_and_writes_reports(tmp_path, monkeypatch):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{
        "id": "one", "question": "What units?", "expected_sources": ["sop.md"],
        "expected_evidence": [{"source": "sop.md", "text": "Use centimeters."}],
    }]))
    indexes = [Mock(manifest={"embedding_model": "test"}) for _ in range(2)]
    for index in indexes:
        index.search.return_value = [{"source": "sop.md", "text": "Use centimeters."}]
    loader = Mock(side_effect=indexes)
    embedder = Mock()
    monkeypatch.setattr("fieldguide.store.LocalIndex.load", loader)
    monkeypatch.setattr("fieldguide.store.embeddings", embedder)
    output = tmp_path / "reports"
    args = parser().parse_args([
        "compare-retrieval", "--indexes", "first", "second", "--cases", str(cases),
        "--output", str(output),
    ])
    assert run(args) == 0
    embedder.assert_called_once_with("test")
    assert all(call.kwargs == {"embed": False} for call in loader.call_args_list)
    report = json.loads((output / "details.json").read_text())
    assert len(report["results"]) == 6
    assert (output / "scores.csv").is_file()
    assert "evidence_recall" in (output / "summary.txt").read_text()


def test_lexical_search_does_not_load_embeddings(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text("{}")
    index = Mock()
    index.search.return_value = []
    loader = Mock(return_value=index)
    monkeypatch.setattr("fieldguide.store.LocalIndex.load", loader)
    args = parser().parse_args([
        "search", "plant", "--retrieval", "lexical", "--index", str(tmp_path),
    ])
    assert run(args) == 0
    loader.assert_called_once_with(tmp_path, embed=False)
    index.search.assert_called_once_with("plant", k=5, source=None, mode="lexical")
