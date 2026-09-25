# FieldGuide

FieldGuide is a document Q&A CLI for salt-marsh researchers and intern training. It retrieves passages from a local collection of field-survey SOPs, drafts an answer, and checks each claim against its cited evidence before returning it. The ingester accepts any document folder, so other teams can use the same tool with their own procedures.

The public demo contains two short, fictional training documents. They are software test fixtures, not field instructions. Real MassMarsh and other organizations' SOPs belong in the private, gitignored `data/` directory; they are not distributed with this repository.

## How it works

```text
Document folder (PDF, DOCX, Markdown, TXT)
                  |
          Parse and chunk locally
                  |
    HuggingFace embeddings on PyTorch
                  |
          Local FAISS index
                  |
Question --> retrieve relevant chunks
                  |
       Answerer (Gemini or local Ollama)
                  |
     Check source IDs and exact quotes
                  |
       Verifier (Gemini or local Ollama)
                  |
         all claims supported?
           /              \
         yes               no
          |                 |
  Answer + citations   Retry with feedback
                            |
                    Verify again or abstain
```

The answerer and verifier are separate chains with separate prompts and structured outputs. They use the same configured Gemini model by default; they are distinct roles, not independently trained models. Verification checks unsupported claims, altered quantities, missing qualifications, and incompatible organization or revision scopes. Python computes approval from complete claim checks and whether the answer addresses the question. A second attempt receives stricter feedback. If it still contains both supported and rejected claims, the supported subset gets one fresh verification, including whether the shorter answer still answers the question. Otherwise the CLI abstains. Model judgments can still be wrong; verification is not a correctness guarantee.

Evidence quotes must match the cited chunk, allowing whitespace differences and PDF extraction spaces before sentence punctuation. Source citations include the relative filename, stable chunk ID, and PDF page when available. Similarity scores describe retrieval similarity, not answer confidence. Verification reduces unsupported output but does not guarantee correctness; researchers should review the cited SOP before applying a procedure.

## Install

Use Python 3.11. From the repository root, on PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,overview]"
Copy-Item .env.example .env
```

On macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,overview]'
cp .env.example .env
```

For Gemini answers and live evaluation, set `GEMINI_API_KEY` in `.env`. `GOOGLE_API_KEY` is also accepted. `FIELDGUIDE_MODEL` defaults to `gemini-2.5-flash`; `--model` overrides it for a command. Gemini remains the default provider. Local ingestion, search, clustering, and retrieval-only evaluation need no API key. The embedding model downloads on first use, then runs locally on CPU; PyTorch makes the initial installation substantial. The `dev` extra supplies tests and linting; `overview` supplies scikit-learn clustering.

`requirements.lock` records the versions used for local validation. To reproduce those versions, add `-c requirements.lock` to the editable install command. The lock was captured on Windows with Python 3.11; installation on other platforms is covered separately by CI.

## Run the public demo

```bash
fieldguide ingest sample_corpus --index index/sample --public
fieldguide search "What should a trainee do with an unidentified plant?"
fieldguide ask "What should a trainee do with an unidentified plant?"
fieldguide ask "How big is the training quadrat?" --source vegetation --show-context
fieldguide overview --index index/sample --clusters 3
```

`python -m fieldguide` also works in place of `fieldguide`. Use `--json` with `ask` or `search` for machine-readable results. `--k` sets the number of retrieved chunks, and `--source` restricts retrieval to filenames containing the supplied text.

### Compare chunking and retrieval

Ingestion uses structure-aware splitting by default (`--chunking structure`), preserving
explicit Markdown/DOCX headings and sentence boundaries, with whole-sentence overlap where it fits.
Consecutive PDF pages can share a chunk; citations retain the start and end pages.
`--chunk-size` and `--overlap` are character limits, not token counts. Long sentences still
require a bounded fallback split. PDF tables, headers, and multi-column layouts may need
extraction cleanup; sentence splitting does not repair their reading order.
PDF line-leading decimal section numbers (such as `2.2`) are recognized; other PDF headings
are not inferred. Source filenames and available section headings
are included in retrieval inputs but kept separate from the quotable passage text. Oversized
chunks are split again to fit the embedding model's actual token limit, including this context.

Keep separate indexes to compare against the earlier recursive splitter:

```powershell
fieldguide ingest sample_corpus --index index/sample-recursive --public --chunking recursive --chunk-size 850 --offline
fieldguide ingest sample_corpus --index index/sample-sentence --public --chunking sentence --chunk-size 850 --offline
fieldguide compare-retrieval --indexes index/sample-recursive index/sample-sentence --k 3 --offline
```

The comparison runs dense similarity, lexical BM25, and hybrid reciprocal rank fusion on
the same questions without an LLM. It writes per-case scores and retrieved passages under
`reports/retrieval/`. Source recall measures whether the right documents appear. Evidence
recall measures whether the labeled excerpts appear in the retrieved chunks; reciprocal
rank measures how early a relevant result appears. These measure retrieval, not answer
correctness. Missing labels are unscored, and errors are reported separately. The tiny
public corpus is a smoke test, not a performance estimate for researchers' SOPs.

Use `--retrieval hybrid` with `search`, `ask`, or `chat` to combine semantic matches with
exact terms. `--retrieval lexical` needs no embedding model. Dense remains the default;
choose a mode using held-out questions from your own corpus. Scores labeled `cosine`,
`bm25`, and `rrf` have different scales and are not answer confidence.

```powershell
fieldguide ingest data --index index/private-sentence --chunking sentence --chunk-size 850 --offline
fieldguide search "What must be recorded at each vegetation plot?" --index index/private-sentence --retrieval hybrid --k 3 --offline
```

For a private benchmark, put cases in `data/retrieval_questions.json`, following
`eval/retrieval_questions.json`: each gold evidence entry needs a source path and an exact
excerpt from that source. Compare indexes built from the same document versions with
`--cases data/retrieval_questions.json`. Keep cases and reports private. A Databricks sample
index can join the public comparison, but cannot measure retrieval over private SOPs.
The supplied Spark notebook uses fixed character cuts; it does not use the local sentence
strategy. Existing indexes only change when rebuilt.

For local SOP questions, build a fresh index and optionally use a local cross-encoder reranker:

```powershell
fieldguide ingest data --index index/private-local --offline
fieldguide audit --index index/private-local --tokens --offline
fieldguide search "According to the vegetation SOP, what should be recorded?" --index index/private-local --retrieval hybrid --rerank
fieldguide ask "According to the vegetation SOP, what should be recorded?" --provider ollama --index index/private-local --retrieval hybrid --rerank --show-context --explain --offline
```

The first `--rerank` run downloads `cross-encoder/ms-marco-MiniLM-L-6-v2`; subsequent runs
can use `--offline`. Reranking runs on CPU and does not call an API. It scores up to
`max(20, 4*k)` candidate passages and returns the best `k`; scores are relevance scores,
not probabilities. Short overlapping sentence windows are scored within each chunk, and its
best window score determines its rank; the original chunk remains available as evidence.
Very long query/window pairs are capped at the reranker's 512-token window.
Compare with and without `--rerank` using `compare-retrieval` before adopting it for a corpus.
Both `eval` and interactive commands accept `--retrieval` and `--rerank`.

An unambiguous SOP name in a question (for example, "Vegetation SOP") restricts retrieval
to that document. If multiple revisions match, no revision is silently selected; use
`--source` with a distinctive filename substring. Broad topic questions can retrieve several
organizations' documents. `--show-context` identifies automatically selected documents.

`audit` reports source-file hashes, duplicate passages, page-reference coverage, and saved
extraction warnings. `--tokens` loads the embedding model to check its input limits. Older
indexes may lack provenance and extraction warnings. Scanned pages without text still need
local OCR before ingestion; this tool does not recover them automatically.

`--explain` shows verifier decisions and the computed `approved_by_checks` result. These
diagnostics are model judgments, not additional SOP instructions. Private reports, extracted
passages, and labeled evaluation cases belong under the gitignored directories.

For repeated questions, keep the embeddings and index in memory with an interactive session:

```bash
fieldguide chat --index index/private --source vegetation --allow-private-api --offline
```

Enter a complete question to get a verified answer, `/search <question>` for local retrieval without Gemini, or `/exit` to quit. Each question is independent; the session does not send conversation history. API failures leave the session open so local search remains available. Gemini questions still use API quota.

`--offline` skips Hugging Face network checks and loads cached model files; use it after the first successful model download. It only affects embedding downloads, not Gemini calls. A new CLI process still loads weights into memory once; `chat` reuses them for the session. See [Hugging Face offline mode](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhuboffline).

Example answer format using the fictional vegetation document:

```text
Question: How big is the training quadrat?
Status: verified | attempts: 1

Use a 1 m by 1 m quadrat at each marked training plot. [S1]

[S1] vegetation_training.md | chunk <chunk ID> | similarity <score>
```

If evidence is insufficient, the CLI returns: “I could not verify an answer from the retrieved documents. Check the source SOP.” The CLI also reports whether retrieval found no context, the answerer lacked evidence, citation matching failed, or the verifier rejected the answer. Use `--show-context` to inspect retrieved passages after an abstention, or `--source vegetation` to focus on a particular SOP. An `ask` command exits with code 0 for a verified answer, 2 for abstention, and 1 for an operational error.

## Use private SOPs

Place authorized local copies in `data/`, then build a separate index:

```bash
fieldguide ingest data --index index/private
fieldguide search "What should an intern record at each plot?" --index index/private
```

Ingestion recursively reads text PDFs, DOCX, Markdown, and UTF-8 text. Scanned PDFs require OCR outside this tool. Ingestion reports unreadable or empty documents. Default chunks are 1,000 characters with 150 characters of overlap; adjust with `--chunk-size` and `--overlap`. Rerun ingestion after changing documents. You can pass any folder path, including a folder outside the repository; use an index under `index/` to keep generated files ignored.

Private indexes are the default. `search` remains local. With the default Gemini provider, `ask` sends the question and retrieved excerpts to Google's API. Live evaluation also sends references and answer context to the judge. Only enable these calls when your organization permits that processing:

```bash
fieldguide ask "What should an intern record at each plot?" --index index/private --allow-private-api
```

The equivalent persistent setting is `FIELDGUIDE_ALLOW_PRIVATE_API=true` in `.env`. The `--public` ingestion flag declares that the corpus is public; it does not anonymize anything. LangChain tracing is disabled by the CLI.

`data/`, `index/`, `reports/`, and `.env` are gitignored. Indexes and evaluation reports contain source content and should be stored alongside the private corpus. Source metadata uses paths relative to the input folder.

## Use a local model with Ollama

Install [Ollama](https://ollama.com/download), keep its local server running, and install FieldGuide's optional local provider:

```powershell
python -m pip install -e ".[local]"
ollama pull qwen3:4b
fieldguide ask "How big is the training quadrat?" --provider ollama --index index/sample
```

`--provider ollama` uses the local server at `http://127.0.0.1:11434` for both the answerer and verifier. The default model is `qwen3:4b`; set `FIELDGUIDE_OLLAMA_MODEL` in `.env` or pass `--model` to select another installed local model. Gemini credentials and `--allow-private-api` are not needed for this provider. Private questions and retrieved excerpts stay on your computer.

For repeated private SOP questions and public evaluation:

```powershell
fieldguide chat --provider ollama --index index/private --source vegetation --offline
fieldguide eval --provider ollama --index index/sample --output reports/ollama --offline
```

Evaluation uses Ollama for the judge as well. Local generation uses no Gemini quota, but speed and answer quality depend on the model and hardware. The same citation checks, verification, and abstention rules apply. Inspect evaluation results and source passages before relying on answers for fieldwork.

Use `--offline` only after the embedding model has been downloaded; it controls embedding downloads, not which answer provider is selected. Ollama must already have the requested model downloaded.

## Evaluate

The fixed public suite in `eval/sample_questions.json` contains 14 cases: 11 answerable questions and 3 abstention cases, including an instruction to invent an unsupported answer. It covers quantities, missing values, cross-document attribution, and unsupported requests.

```bash
fieldguide eval --retrieval-only
fieldguide eval
fieldguide eval --no-judge --output reports/no-judge
fieldguide eval --judge-model gemini-2.5-flash --output reports/judged
```

Each run prints a pandas table and writes `scores.csv`, `details.json`, and `summary.txt` under the output folder (default `reports/eval`). Use separate output folders to preserve comparisons. A case error is recorded and causes a nonzero evaluation exit code; low scores alone do not.

| Metric | Meaning and denominator |
| --- | --- |
| Source recall | Fraction of expected filenames retrieved per case; mean excludes cases with no expected sources. Filename retrieval does not establish that the relevant passage was retrieved. |
| Behavior correct | Verified response for an answerable case, or abstention for an unanswerable case; errors count as failures. This checks response behavior, not factual correctness. |
| Reference F1 | Token overlap with the reference for answerable cases. Paraphrases can score poorly, and overlap does not establish grounding. |
| Quote validity | Exact evidence quote and source checks passed for verified answers. This is a mechanical check, not semantic faithfulness. |
| Faithfulness | Optional model judge score against cited context, for verified answers only. Abstentions do not inflate the mean. |
| Correctness | Optional judge score against the reference for verified answers; answerable abstentions and errors receive zero, expected abstentions receive one. |

Summary means report their own `n` because metrics have different denominators; inspect the case table and error count alongside every mean. `--retrieval-only` does not generate or verify answers and cannot measure faithfulness. `--no-judge` still runs the answerer and verifier but omits the separate evaluation judge. The judge defaults to the answering model, so its errors may be correlated with the answerer's. This small synthetic suite is a reproducible smoke evaluation, not evidence of performance on real SOPs.

For a private evaluation, author cases with the same JSON schema under `data/` and use `--cases data/questions.json --index index/private --output reports/private`. Private evaluation with Gemini requires the same API opt-in; `--provider ollama` keeps it local. Keep references scoped to the correct organization and SOP revision, and include unanswerable questions.

## Repository and checks

For optional PySpark ingestion in Databricks Free Edition, follow the [sample ingestion walkthrough](docs/databricks.md). It exports document pages, chunks them in a notebook, and imports the result into local FAISS without requiring Gemini calls.

```text
src/fieldguide/       Ingestion, retrieval, agents, evaluation, and CLI
sample_corpus/       Public fictional training documents
eval/                Fixed public evaluation cases
tests/               Unit tests with local fixtures and model doubles
data/                Private SOPs and evaluation cases (ignored)
index/               Local vectors and source chunks (ignored)
reports/             Evaluation outputs (ignored)
```

```bash
ruff check src tests
pytest -q
```

CI runs these checks without Gemini credentials, an Ollama server, or an embedding-model download. Unit tests cover verification and retrieval mechanics with test doubles; they do not replace a live model run or evaluation against the team's real SOPs.

Integration references: [LangChain's Gemini integration](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai), [local sentence-transformer embeddings](https://docs.langchain.com/oss/python/integrations/embeddings/sentence_transformers), and [Gemini structured output](https://ai.google.dev/gemini-api/docs/structured-output).
