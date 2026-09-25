"""Command-line entry points."""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="FieldGuide: grounded field-document Q&A")
    commands = root.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Index a document folder locally")
    ingest.add_argument("folder", type=Path)
    ingest.add_argument("--index", type=Path, default=Path("index/private"))
    ingest.add_argument("--public", action="store_true", help="Declare this corpus public")
    ingest.add_argument("--chunk-size", type=int, default=1000)
    ingest.add_argument("--overlap", type=int, default=150)
    ingest.add_argument("--chunking", choices=["sentence", "recursive"], default="sentence")
    ingest.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    export = commands.add_parser("export-pages", help="Export document pages for Spark ingestion")
    export.add_argument("folder", type=Path)
    export.add_argument("--output", type=Path, required=True)
    imported = commands.add_parser("import-chunks", help="Index chunks exported from Spark")
    imported.add_argument("file", type=Path)
    imported.add_argument("--index", type=Path, default=Path("index/imported"))
    imported.add_argument("--public", action="store_true")
    imported.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    for name, help_text in (
        ("search", "Retrieve chunks locally without an API"),
        ("ask", "Return a verified answer with citations"),
        ("chat", "Keep the model loaded for repeated questions and local searches"),
    ):
        command = commands.add_parser(name, help=help_text)
        if name != "chat":
            command.add_argument("question")
        command.add_argument("--index", type=Path, default=Path("index/sample"))
        command.add_argument("--k", type=int, default=5)
        command.add_argument("--source", help="Filter by source filename substring")
        command.add_argument("--retrieval", choices=["dense", "lexical", "hybrid"], default="dense")
        command.add_argument("--json", action="store_true")
        if name in {"ask", "chat"}:
            command.add_argument("--model")
            command.add_argument("--provider", choices=["gemini", "ollama"], default="gemini")
            command.add_argument("--allow-private-api", action="store_true")
            command.add_argument("--show-context", action="store_true")
    evaluate = commands.add_parser("eval", help="Run fixed evaluation cases")
    evaluate.add_argument("--cases", type=Path, default=Path("eval/sample_questions.json"))
    evaluate.add_argument("--index", type=Path, default=Path("index/sample"))
    evaluate.add_argument("--output", type=Path, default=Path("reports/eval"))
    evaluate.add_argument("--retrieval-only", action="store_true")
    evaluate.add_argument("--no-judge", action="store_true")
    evaluate.add_argument("--allow-private-api", action="store_true")
    evaluate.add_argument("--model")
    evaluate.add_argument("--provider", choices=["gemini", "ollama"], default="gemini")
    evaluate.add_argument("--judge-model")
    evaluate.add_argument("--k", type=int, default=5)
    compare = commands.add_parser("compare-retrieval", help="Compare local indexes without an LLM")
    compare.add_argument("--indexes", nargs="+", type=Path, required=True)
    compare.add_argument("--cases", type=Path, default=Path("eval/retrieval_questions.json"))
    compare.add_argument("--k", type=int, default=3)
    compare.add_argument("--output", type=Path, default=Path("reports/retrieval"))
    overview = commands.add_parser("overview", help="Cluster chunks into local topic groups")
    overview.add_argument("--index", type=Path, default=Path("index/sample"))
    overview.add_argument("--clusters", type=int, default=4)
    for command in commands.choices.values():
        command.add_argument(
            "--offline",
            action="store_true",
            help="Load cached embeddings without contacting Hugging Face",
        )
    return root


def print_sources(sources: list[dict], show_context: bool = False):
    for source in sources:
        page = f", page {source['page']}" if source.get("page") else ""
        if source.get("page_end") and source["page_end"] != source.get("page"):
            page += f"–{source['page_end']}"
        print(
            f"[{source['id']}] {source['source']}{page} | chunk {source['chunk_id']} "
            f"| {source.get('score_kind', 'similarity')} {source['score']:.3f}"
        )
        if show_context:
            print(source["text"] + "\n")


def run(args) -> int:
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from fieldguide.store import LocalIndex

    if args.command == "compare-retrieval":
        from fieldguide.compare import compare_retrieval, summarize_comparison
        from fieldguide.store import embeddings

        if len(set(args.indexes)) != len(args.indexes):
            raise ValueError("Provide distinct index paths.")
        indexes = {str(path): LocalIndex.load(path, embed=False) for path in args.indexes}
        embedders = {}
        for index in indexes.values():
            model = index.manifest["embedding_model"]
            if model not in embedders:
                embedders[model] = embeddings(model)
            index.embedder = embedders[model]
        frame, details = compare_retrieval(indexes, args.cases, k=args.k)
        args.output.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output / "scores.csv", index=False)
        (args.output / "details.json").write_text(
            json.dumps(
                {
                    "k": args.k,
                    "indexes": {name: index.manifest for name, index in indexes.items()},
                    "results": details,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        report = summarize_comparison(frame)
        print(report)
        (args.output / "summary.txt").write_text(report, encoding="utf-8")
        return int((frame.status == "error").any())

    if args.command == "export-pages":
        from fieldguide.chunk_io import export_pages

        count, warnings = export_pages(args.folder, args.output)
        print(f"Exported {count} page records to {args.output}. No files were uploaded.")
        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        return 0
    if args.command == "import-chunks":
        from fieldguide.chunk_io import import_chunks

        documents, size, overlap = import_chunks(args.file)
        LocalIndex.build(
            documents,
            args.index,
            model=args.embedding_model,
            public=args.public,
            chunk_size=size,
            chunk_overlap=overlap,
        )
        print(f"Indexed {len(documents)} imported chunks into {args.index}.")
        return 0
    if args.command == "ingest":
        from fieldguide.ingest import chunk_documents, read_documents

        documents, warnings = read_documents(args.folder)
        chunks = chunk_documents(documents, args.chunk_size, args.overlap, strategy=args.chunking)
        LocalIndex.build(
            chunks,
            args.index,
            model=args.embedding_model,
            public=args.public,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
        )
        print(
            f"Indexed {len(chunks)} chunks from "
            f"{len({d.metadata['source'] for d in documents})} documents into {args.index}."
        )
        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        return 0
    if not (args.index / "manifest.json").is_file():
        raise ValueError(f"No index at {args.index}. Run fieldguide ingest first.")
    if args.command == "ask" or (args.command == "eval" and not args.retrieval_only):
        from fieldguide.agents import require_api_permission

        manifest = json.loads((args.index / "manifest.json").read_text(encoding="utf-8"))
        if args.provider == "gemini":
            require_api_permission(manifest, args.allow_private_api)
    need_embeddings = args.command != "overview" and getattr(args, "retrieval", None) != "lexical"
    index = LocalIndex.load(args.index, embed=need_embeddings)
    if args.command == "chat":
        return chat(index, args)
    if args.command == "overview":
        from fieldguide.overview import corpus_overview

        print(corpus_overview(index, args.clusters).to_string(index=False))
        return 0
    if args.command == "eval":
        from fieldguide.agents import GroundedQA, make_chains
        from fieldguide.evaluate import make_judge, run_evaluation, summary
        from fieldguide.providers import resolve_model

        qa = (
            None
            if args.retrieval_only
            else GroundedQA(*make_chains(args.model, provider=args.provider))
        )
        judge = (
            None
            if args.retrieval_only or args.no_judge
            else make_judge(args.judge_model or args.model, provider=args.provider)
        )
        frame, details = run_evaluation(index, args.cases, qa=qa, judge=judge, k=args.k)
        args.output.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output / "scores.csv", index=False)
        (args.output / "details.json").write_text(
            json.dumps(
                {
                    "index": index.manifest,
                    "provider": args.provider,
                    "model": resolve_model(args.model, args.provider),
                    "judge_model": resolve_model(args.judge_model or args.model, args.provider),
                    "judge_enabled": judge is not None,
                    "k": args.k,
                    "results": details,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        report = summary(frame)
        print(report)
        (args.output / "summary.txt").write_text(report, encoding="utf-8")
        return 1 if (frame.status == "error").any() else 0
    sources = index.search(args.question, k=args.k, source=args.source, mode=args.retrieval)
    if args.command == "search":
        if args.json:
            print(json.dumps(sources, indent=2, ensure_ascii=False))
        else:
            print_sources(sources, show_context=True)
        return 0
    from fieldguide.agents import GroundedQA, make_chains

    result = GroundedQA(*make_chains(args.model, provider=args.provider)).answer(
        args.question, sources
    )
    return print_answer(result, args)


def print_answer(result, args) -> int:
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(f"Status: {result.status} | attempts: {result.attempts}\n\n{result.text}\n")
        if result.status == "abstained":
            reasons = {
                "no_context": "No chunks matched the selected source.",
                "insufficient_evidence": "The answerer found insufficient evidence in the chunks.",
                "invalid_citations": "The draft's evidence quotes did not match the cited chunks.",
                "verification_rejected": "The verifier did not approve every claim in the answer.",
            }
            print(f"Reason: {reasons.get(result.abstention_reason, 'Evidence checks failed.')}")
            print("Use --show-context to inspect retrieval or --source to select a specific SOP.")
            if args.show_context:
                print("\nRetrieved context (not a verified answer):")
                print_sources(result.retrieved, show_context=True)
        else:
            print_sources(result.sources, show_context=args.show_context)
    return 0 if result.status == "verified" else 2


def chat(index, args) -> int:
    from fieldguide.agents import GroundedQA, make_chains, require_api_permission

    qa = None
    print("Ready. Ask a complete question, /search <question> for local retrieval, or /exit.")
    print("Questions are independent; previous answers are not sent as conversation history.")
    while True:
        try:
            question = input("fieldguide> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"/exit", "/quit"}:
            return 0
        if not question:
            continue
        if question == "/search":
            print("Usage: /search <question>")
            continue
        local = question.startswith("/search ")
        try:
            if not local:
                if args.provider == "gemini":
                    require_api_permission(index.manifest, args.allow_private_api)
                if qa is None:
                    qa = GroundedQA(*make_chains(args.model, provider=args.provider))
            query = question[len("/search ") :].strip() if local else question
            sources = index.search(query, k=args.k, source=args.source, mode=args.retrieval)
            if local:
                print_sources(sources, show_context=True)
            else:
                print_answer(qa.answer(query, sources), args)
        except KeyboardInterrupt:
            print("Question interrupted. The index is still loaded.")
        except Exception as exc:
            print(
                f"Question failed ({type(exc).__name__}). Check API quota, credentials, "
                "and private API permission. /search still works without Gemini."
            )


def main():
    load_dotenv(Path.cwd() / ".env")
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    args = parser().parse_args()
    try:
        code = run(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:
        print(
            f"Error ({type(exc).__name__}): operation failed. Check your model, API key, "
            "quota, network connection, or Ollama server/model availability.",
            file=sys.stderr,
        )
        code = 1
    sys.exit(code)
