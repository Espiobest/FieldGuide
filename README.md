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
          Answerer (Gemini / LangChain)
                  |
     Check source IDs and exact quotes
                  |
          Verifier (Gemini / LangChain)
                  |
         all claims supported?
           /              \
         yes               no
          |                 |
  Answer + citations   Retry with feedback
                            |
                    Verify again or abstain
```

The answerer and verifier are separate chains with separate prompts and structured outputs. They use the same configured Gemini model by default; they are distinct roles, not independently trained models. Verification rejects unsupported claims, altered quantities, missing qualifications, and incompatible organization or revision scopes. A second attempt receives stricter feedback. If verification still fails, the CLI abstains.

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

Set `GEMINI_API_KEY` in `.env` to enable answers and live evaluation. `GOOGLE_API_KEY` is also accepted. `FIELDGUIDE_MODEL` defaults to `gemini-2.5-flash`; `--model` overrides it for a command. Local ingestion, search, clustering, and retrieval-only evaluation need no API key. The embedding model downloads on first use, then runs locally on CPU; PyTorch makes the initial installation substantial. The `dev` extra supplies tests and linting; `overview` supplies scikit-learn clustering.

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

Private indexes are the default. `search` remains local, while `ask` sends the question and retrieved excerpts to Google's Gemini API. Live evaluation also sends references and answer context to the judge. Only enable these calls when your organization permits that processing:

```bash
fieldguide ask "What should an intern record at each plot?" --index index/private --allow-private-api
```

The equivalent persistent setting is `FIELDGUIDE_ALLOW_PRIVATE_API=true` in `.env`. The `--public` ingestion flag declares that the corpus is public; it does not anonymize anything. LangChain tracing is disabled by the CLI.

`data/`, `index/`, `reports/`, and `.env` are gitignored. Indexes and evaluation reports contain source content and should be stored alongside the private corpus. Source metadata uses paths relative to the input folder.

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
| Faithfulness | Optional Gemini judge score against cited context, for verified answers only. Abstentions do not inflate the mean. |
| Correctness | Optional judge score against the reference for verified answers; answerable abstentions and errors receive zero, expected abstentions receive one. |

Summary means report their own `n` because metrics have different denominators; inspect the case table and error count alongside every mean. `--retrieval-only` does not generate or verify answers and cannot measure faithfulness. `--no-judge` still runs the answerer and verifier but omits the separate evaluation judge. The judge defaults to the answering model, so its errors may be correlated with the answerer's. This small synthetic suite is a reproducible smoke evaluation, not evidence of performance on real SOPs.

For a private evaluation, author cases with the same JSON schema under `data/` and use `--cases data/questions.json --index index/private --output reports/private`. Live private evaluation requires the same API opt-in. Keep references scoped to the correct organization and SOP revision, and include unanswerable questions.

## Repository and checks

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

CI runs these checks without Gemini credentials or an embedding-model download. Unit tests cover verification and retrieval mechanics with test doubles; they do not replace a live Gemini run or evaluation against the team's real SOPs.

Integration references: [LangChain's Gemini integration](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai), [local sentence-transformer embeddings](https://docs.langchain.com/oss/python/integrations/embeddings/sentence_transformers), and [Gemini structured output](https://ai.google.dev/gemini-api/docs/structured-output).
