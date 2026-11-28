# Part 107 Field Ops

**[Live Demo](https://part107-rag-crroj772gwmg67r7zyjfsh.streamlit.app)** | **[GitHub CI](https://github.com/macd3412/part107-rag/actions)**

A zero-framework Retrieval-Augmented Generation (RAG) assistant for navigating U.S. FAA small unmanned aircraft regulations—built with pure Python (`numpy`, `rank-bm25`, `requests`, and `pypdf`) rather than high-level orchestration wrappers.

## Why Part 107? & Estimated Business Impact
As a drone pilot navigating FAA rules in the field, compliance questions rarely map to a single paragraph. Pilots ask scenario-based questions—combining night lighting (`§ 107.29`), operations over people (`Subpart D`), and Remote ID (`Part 89`)—where guessing creates real airspace and enforcement risk.

- **Measured Speedup:** Manual lookups across eCFR and 100-page FAA PDFs take ~10 minutes; this agent averages **2.35 seconds** across 150 benchmark queries (~9.96 minutes saved per lookup).
- **Team Time Savings (20-pilot team, 50 weeks/yr):** **~83 hours/year** saved at 0.5 lookups/pilot/week, or **~166 hours/year** (~4 workweeks) at 1 lookup/pilot/week.
- **Compliance Risk Reduction:** Verified **91% recall** and **88% out-of-scope refusal** across 150 unseen questions, helping teams avoid FAA civil penalties (up to $20,000+ per violation) and certificate suspension.

## Data Sources (375 Indexed Chunks)
- **14 CFR Part 107, Part 89 & Cross-Referenced Sections (137 chunks):** Fetched from the official eCFR Versioner XML API (including targeted sections `§ 48.15`, `§ 61.56`, `§ 91.17`, and `§§ 91.137–91.145`) and parsed at section and paragraph boundaries with subpart breadcrumbs and extracted cross-references.
- **FAA Advisory Circular AC 107-2A (238 chunks):** Extracted from the official FAA PDF with sliding-window overlap and page citations.

## Architecture
1. **Ingestion (`src/ingest/`):** Structure-aware parsers normalize CFR sections and AC 107-2A pages into `data/processed/chunks.jsonl` with source type, citation, section metadata, and official URLs.
2. **Incremental Indexing (`src/index/build_index.py`):** Embeds chunks with `gemini-embedding-001` (`RETRIEVAL_DOCUMENT`, 3072 dimensions) into a normalized NumPy matrix (`data/processed/embeddings.npy`), reusing cached vectors and pausing automatically on rate limits.
3. **Source-Balanced Hybrid Retrieval (`src/rag/retriever.py`):** Combines dense cosine similarity (`numpy`) with lexical keyword search (`BM25Okapi`) via **Reciprocal Rank Fusion (RRF)**, reserving 4 CFR slots + 1 AC 107-2A slot in Top-5 so verbose PDF guidance never crowds out binding law.
4. **Single-Pass & Agentic RAG (`src/rag/pipeline.py`, `src/rag/agent.py`):** Supports fast single-pass Q&A and a multi-step tool-calling loop (`search_regulations`) that rewrites casual phrasing (like *"I moved"* -> *"change of address"*) into formal FAA terms, strictly grounded to retrieved excerpts.

## Evaluation & Tests

### 50-Question Development Benchmark (3-Stage Progression)
| Pipeline Iteration | Retrieval Recall | Citation Accuracy | Refusal Guardrails |
| :--- | :--- | :--- | :--- |
| **1. Baseline Hybrid RRF** | 68% (32/47) | 74% (35/47) | 100% (3/3) |
| **2. Source-Balanced (4 CFR + 1 AC)** | 85% (40/47) | 87% (41/47) | 100% (3/3) |
| **3. Agentic Multi-Step RAG** | **98% (46/47)** | **96% (45/47)** | **100% (3/3)** |

### 150-Question Unseen Expansion Benchmark (Agentic RAG)
Tested on 150 additional questions spanning **Easy / Medium / Hard** difficulties and **24 out-of-scope traps**:
- **Retrieval Recall:** `115/126 (91%)` · **Citation Accuracy:** `111/126 (88%)`
- **Refusal Guardrails:** `21/24 (88%)` *(up from 33% after tightening prompt against pretraining leaks)*
- **Avg Lookup Time:** `2.35s`
- **By Difficulty:** Easy `35/40 (88%)` · Medium `46/51 (90%)` · Hard `30/35 (86%)`
- **Offline Unit Tests (`pytest`):** `17 passed` · **Linter (`ruff check .`):** Passed

## What Failed & How I Fixed It
- **Naive chunking broke legal exceptions:** Fixed-size character splits severed CFR subsections right before *"except as provided in..."* clauses. Switched to parsing eCFR `<DIV8>` section boundaries directly and prepending `[14 CFR § 107.xx | Subpart]` breadcrumbs.
- **AC 107-2A crowded out binding CFR sections:** Adding 238 verbose PDF chunks dropped Top-5 Recall to 68% because AC pages filled the slots. Adding source-balanced selection (4 CFR + 1 AC) jumped Recall to 85%.
- **Vocabulary mismatches on casual phrasing:** Queries like *"I moved"* or *"fly for fun"* missed legal titles (`Change of address`, `Applicability`). Adding an agentic `search_regulations` tool-calling loop lifted 50-question Recall to 98%.
- **Pretraining leakage on out-of-scope aviation rules:** On the 150-question benchmark, the agent initially answered non-corpus questions (like manned `§ 91.155` or National Park rules) from pretraining memory (33% refusal). Tightening `SYSTEM_PROMPT` to forbid external citations raised refusal accuracy to 88%.
- **SDK client-closed errors on Python 3.15:** `google-genai`'s underlying `httpx2` client threw `RuntimeError: Cannot send a request, as the client has been closed` during retries. Refactored both `retriever.py` and `pipeline.py` to direct `requests` REST calls with explicit timeouts and backoff.

## Project Structure

    part107-rag/
    ├── app.py                  # Interactive CLI Q&A loop
    ├── streamlit_app.py        # Web UI with clickable citations & latency timer
    ├── pyproject.toml          # ruff & pytest configuration
    ├── requirements.txt
    ├── data/
    │   ├── eval/part107_eval.json
    │   └── processed/          # chunks.jsonl + embeddings.npy
    ├── src/
    │   ├── config.py           # Centralized paths, models, and retry settings
    │   ├── ingest/             # eCFR XML & AC 107-2A PDF fetchers/parsers
    │   ├── index/              # Incremental embedding index builder
    │   ├── rag/                # Hybrid RRF retriever, pipeline, & agent loop
    │   └── eval/               # Recall, citation, refusal, & latency evaluator
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
