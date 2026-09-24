# Databricks sample ingestion

This optional path chunks public sample documents with PySpark in Databricks Free Edition, stores the chunks in Delta, and imports them into local FAISS. Embeddings and Q&A continue to run through the existing CLI. The ordinary `ingest` command needs no Databricks account.

## Export locally

From your activated FieldGuide environment:

```powershell
fieldguide export-pages sample_corpus --output reports/databricks/sample_pages.jsonl
```

The output contains document text, relative filenames, and page numbers (null for Markdown). The command does not upload anything and refuses to overwrite an existing file. Use a new output filename for another export. Only upload the fictional sample data for this demonstration; private SOPs and extracted text stay local.

## Run in Databricks

1. In your Free Edition workspace, import [databricks_ingestion.ipynb](../notebooks/databricks_ingestion.ipynb) using the Workspace import option. Select serverless compute.
2. Run the first code cell. It creates a `fieldguide` schema and `demo_files` volume under your current catalog. If that catalog is not writable, set `CATALOG` to the name of a writable catalog visible in Catalog Explorer.
3. Copy the volume path printed by the cell. In Catalog Explorer, open that volume and upload the local `reports/databricks/sample_pages.jsonl` file. Keep its filename `sample_pages.jsonl`.
4. Run the remaining cells in order. The notebook checks input records, creates overlapping chunks with Spark SQL functions, checks IDs, and saves a new Delta table for each run.
5. The final cell prints a `chunks_<run_id>.jsonl` path. Download that file through Catalog Explorer and save it locally as `reports/databricks/chunks.jsonl`.

Spark uses fixed character windows of 1,000 characters with 150 characters of overlap. This differs from the local ingester's paragraph-aware splitting and can split words. PDF page locations remain intact. The single-file download is intended for small demonstrations; use partitioned output for large datasets.

## Import and search locally

```powershell
fieldguide import-chunks reports/databricks/chunks.jsonl --index index/databricks-sample --public --offline
fieldguide search "How big is the training quadrat?" --index index/databricks-sample --offline
fieldguide eval --retrieval-only --index index/databricks-sample --output reports/databricks-retrieval --offline
```

`--offline` requires the embedding model to be cached already. Omit it for the initial model download. These commands do not call Gemini. Import validates chunk IDs, relative source paths, duplicate records, and consistent chunk settings before creating the index. Imported indexes default to private; use `--public` only for public material.

The exported data and generated indexes remain gitignored. Commit notebook source without saved outputs. A completed demonstration consists of a successful Databricks notebook run followed by local retrieval from its downloaded output; local tests alone do not establish that the cloud run succeeded.

References: [Free Edition](https://docs.databricks.com/aws/en/getting-started/free-edition), [uploading and downloading volume files](https://docs.databricks.com/aws/en/volumes/volume-files).
