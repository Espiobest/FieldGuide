from fieldguide.cli import parser


def test_interactive_commands_default_to_hybrid_retrieval():
    root = parser()
    for name, args in (
        ("search", ["question"]),
        ("ask", ["question"]),
        ("chat", []),
    ):
        parsed = root.parse_args([name, *args])
        assert parsed.retrieval == "hybrid"


def test_evaluation_defaults_to_hybrid_retrieval():
    assert parser().parse_args(["eval"]).retrieval == "hybrid"
