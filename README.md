# Part 107 & Remote ID Drone FAQ RAG

**[Live Demo](https://YOUR_APP_URL.streamlit.app)** | **[GitHub CI](https://github.com/macd3412/part107-rag/actions)**

A zero-framework Retrieval-Augmented Generation (RAG) assistant for navigating U.S. FAA small unmanned aircraft regulations—built with pure Python (`numpy`, `rank-bm25`, `requests`, and `pypdf`) rather than high-level orchestration wrappers.

## Why Part 107? (Domain Context)
As a drone pilot navigating FAA rules in the field, regulation questions rarely map cleanly to a single paragraph. Pilots ask scenario-based questions—like combining night lighting (`§ 107.29`), operations over people (`Subpart D`), and Remote ID (`Part 89`)—where a hallucinated answer creates real airspace and compliance risk. This project pairs binding legal text with plain-English FAA guidance and refuses to guess when a question falls outside federal rules.

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

## What Failed & How I Fixed It
- **Naive chunking broke legal exceptions:** Fixed-size character splits severed CFR subsections right before *"except as provided in..."* clauses. Switched to parsing eCFR `<DIV8>` section boundaries directly and prepending `[14 CFR § 107.xx | Subpart]` breadcrumbs to every chunk.
- **SDK client-closed errors on Python 3.15:** `google-genai`'s underlying `httpx2` client threw `RuntimeError: Cannot send a request, as the client has been closed` during retries. Refactored both `retriever.py` and `pipeline.py` to direct `requests` REST calls with explicit timeouts and backoff.
- **Free-tier embedding `429` limits:** Indexing all 357 chunks in one burst hit the 100 requests/minute cap at chunk 101. Added automatic `429` detection and a 62-second pause in `build_index.py` so full rebuilds finish unattended.
- **Model latency spikes (`503` / timeouts):** `gemini-3.5-flash` occasionally timed out under load. Added ordered fallback routing with `gemini-3.5-flash-lite` as primary, cutting eval runtime with zero timeouts.

## Project Structure

    part107-rag/
    ├── app.py                  # Interactive CLI Q&A loop
    ├── streamlit_app.py        # Web UI with clickable eCFR & FAA citations
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

3. Run the web UI or CLI (pre-built chunks and embeddings are already included):

        streamlit run streamlit_app.py
        # or: python app.py

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
