from unittest.mock import Mock

from fieldguide.agents import Answer
from fieldguide.cli import parser, run


def test_chat_reuses_index_and_agents(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text('{"public": true}')
    index = Mock(manifest={"public": True})
    index.search.return_value = []
    load = Mock(return_value=index)
    monkeypatch.setattr("fieldguide.store.LocalIndex.load", load)
    chains = Mock(return_value=(Mock(), Mock()))
    monkeypatch.setattr("fieldguide.agents.make_chains", chains)
    qa = Mock()
    qa.answer.return_value = Answer(
        question="q", status="abstained", text="Unknown", attempts=0, sources=[], retrieved=[]
    )
    factory = Mock(return_value=qa)
    monkeypatch.setattr("fieldguide.agents.GroundedQA", factory)
    inputs = iter(["", "First question?", "/search local question", "Second question?", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    args = parser().parse_args(["chat", "--index", str(tmp_path)])
    assert run(args) == 0
    assert load.call_count == chains.call_count == factory.call_count == 1
    assert index.search.call_count == 3
    assert qa.answer.call_count == 2


def test_private_chat_local_search_needs_no_api_permission(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text('{"public": false}')
    index = Mock(manifest={"public": False})
    index.search.return_value = []
    monkeypatch.setattr("fieldguide.store.LocalIndex.load", Mock(return_value=index))
    chains = Mock(side_effect=AssertionError("Must not call Gemini"))
    monkeypatch.setattr("fieldguide.agents.make_chains", chains)
    inputs = iter(["Private question?", "/search local question", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert run(parser().parse_args(["chat", "--index", str(tmp_path)])) == 0
    chains.assert_not_called()
    index.search.assert_called_once()


def test_chat_preserves_session_after_api_error(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text('{"public": true}')
    index = Mock(manifest={"public": True})
    index.search.return_value = []
    monkeypatch.setattr("fieldguide.store.LocalIndex.load", Mock(return_value=index))
    monkeypatch.setattr("fieldguide.agents.make_chains", Mock(return_value=(Mock(), Mock())))
    qa = Mock()
    qa.answer.side_effect = RuntimeError("rate limited")
    monkeypatch.setattr("fieldguide.agents.GroundedQA", Mock(return_value=qa))
    inputs = iter(["Question?", "/search local question", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert run(parser().parse_args(["chat", "--index", str(tmp_path)])) == 0
    assert index.search.call_count == 2
