# Part 107 & Remote ID Drone FAQ RAG

A zero-framework Retrieval-Augmented Generation (RAG) assistant for navigating U.S. FAA small unmanned aircraft regulations—built with pure Python (`numpy`, `rank-bm25`, `requests`, and `pypdf`) rather than high-level orchestration wrappers.

## Data Sources (357 Indexed Chunks)
- **14 CFR Part 107 & Part 89 (119 chunks):** Fetched from the official eCFR Versioner XML API and parsed at section (`§ 107.xx` / `§ 89.xx`) and paragraph boundaries with subpart breadcrumbs and extracted cross-references.
- **FAA Advisory Circular AC 107-2A (238 chunks):** Extracted from the official FAA PDF with sliding-window overlap and page citations.

## Architecture
1. **Ingestion (`src/ingest/`):** Structure-aware parsers normalize CFR sections and AC 107-2A pages into `data/processed/chunks.jsonl` with source type, citation, section metadata, and official URLs.
2. **Indexing (`src/index/build_index.py`):** Embeds all 357 chunks with `gemini-embedding-001` (`RETRIEVAL_DOCUMENT`, 3072 dimensions) into a normalized NumPy matrix (`data/processed/embeddings.npy`), with automatic rate-limit handling.
3. **Hybrid Retrieval (`src/rag/retriever.py`):** Combines dense cosine similarity (`numpy`) with lexical keyword search (`BM25Okapi`) using **Reciprocal Rank Fusion (RRF)** so both natural-language scenarios and exact rule numbers (`107.51`, `FRIA`) match reliably.
4. **Grounded Generation (`src/rag/pipeline.py`):** Queries Gemini REST endpoints with retry/fallback handling and prompts the model to answer strictly from retrieved excerpts with inline CFR/AC citations—or refuse when the regulations do not cover the question.

## Evaluation & Tests

Evaluated on a 12-question test suite (`data/eval/part107_eval.json`) covering Part 107, Part 89, and out-of-scope edge cases:

| Metric | Score |
| :--- | :--- |
| **Retrieval Recall@5** | 10 / 10 (100%) |
| **Citation Accuracy** | 10 / 10 (100%) |
| **Refusal Guardrails** | 2 / 2 (100%) |
| **Offline Unit Tests (`pytest`)** | 17 passed |
| **Linter (`ruff check .`)** | Passed |

## Project Structure

    part107-rag/
    ├── app.py                  # Interactive CLI Q&A loop
    ├── pyproject.toml          # ruff & pytest configuration
    ├── requirements.txt
    ├── data/
    │   ├── eval/part107_eval.json
    │   └── processed/          # chunks.jsonl + embeddings.npy
    ├── src/
    │   ├── config.py           # Centralized paths, models, and retry settings
    │   ├── ingest/             # eCFR XML & AC 107-2A PDF fetchers/parsers
    │   ├── index/              # Embedding index builder
    │   ├── rag/                # Hybrid RRF retriever & grounded pipeline
    │   └── eval/               # Recall@5, citation, and refusal evaluator
    └── tests/                  # Offline unit tests (no API calls required)

## Quickstart

1. Create and activate a virtual environment (Python 3.10+):

        python3 -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt

2. Add your Gemini API key to `.env`:

        echo "GEMINI_API_KEY=your_api_key_here" > .env

3. Run the interactive Q&A CLI (pre-built chunks and embeddings are already included):

        python app.py

4. (Optional) Rebuild data and embeddings from scratch:

        python -m src.ingest.fetch_ecfr && python -m src.ingest.parse_cfr
        python -m src.ingest.fetch_ac && python -m src.ingest.parse_ac
        python -m src.index.build_index

## Running Tests & Evaluations

    pytest                        # 17 offline unit tests
    ruff check .                  # Lint checks
    python -m src.eval.evaluate   # End-to-end RAG evaluation (uses GEMINI_API_KEY)

## Disclaimer
This project is an educational engineering portfolio piece and is not official FAA or legal guidance. Always verify current regulations and airspace authorizations (such as LAANC) before flying.
