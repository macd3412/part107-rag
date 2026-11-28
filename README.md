# Part 107 Field Ops

**[Live Demo](https://part107-drone-faq.streamlit.app)** | **[GitHub CI](https://github.com/macd3412/part107-rag/actions)**

A zero-framework **Agentic RAG + Live Preflight Copilot** for U.S. FAA small unmanned aircraft operations—built with pure Python (`numpy`, `rank-bm25`, `requests`, and `pypdf`). Combines **14 CFR Part 107 / Part 89 / AC 107-2A** retrieval with **live GPS TFR boundary checks** (`tfr.faa.gov`) and **live METAR aviation weather** (`aviationweather.gov`).

## Why Part 107 Field Ops? & Estimated Impact
As a drone pilot navigating FAA rules in the field, compliance questions rarely map to a single paragraph. Pilots ask scenario-based questions combining night lighting (`§ 107.29`), operations over people (`Subpart D`), Remote ID (`Part 89`), active TFRs (`§ 107.47`), and weather minimums (`§ 107.51`).

- **Latency vs. Estimated Manual Lookup:** Measured average query time is **2.35 seconds** across 150 benchmark queries, compared to an **estimated** ~10-minute manual lookup across eCFR, 100-page FAA PDFs, TFR maps, and METARs (~9.96 minutes estimated savings per lookup).
- **Estimated Team Time Savings (20-pilot team, 50 weeks/yr):** **~83 hours/year** at 0.5 lookups/pilot/week, or **~166 hours/year** (~4 workweeks) at 1 lookup/pilot/week (based on the 10-minute manual baseline assumption).
- **Compliance Risk Reduction:** Verified **91% recall**, **88% citation accuracy**, and **88% out-of-scope refusal** on 150 unseen questions—reducing the risk of misreading or missing applicable rules before flight.

## Data Sources & Live Tools
- **14 CFR Part 107, Part 89 & Cross-Referenced Sections (137 chunks):** Official eCFR XML (`§ 48.15`, `§ 61.56`, `§ 91.17`, `§§ 91.137–91.145`) parsed at section boundaries with subpart breadcrumbs.
- **FAA Advisory Circular AC 107-2A (238 chunks):** Extracted from the official FAA PDF (**375 total chunks**).
- **Live FAA TFR Geospatial Tool (`src/tools/tfr.py`):** Queries `tfr.faa.gov` GeoJSON polygons (`TFR:V_TFR_LOC`) with point-in-polygon and Haversine distance math (30-mile radius, state fallback).
- **Live Aviation Weather Tool (`src/tools/weather.py`):** Queries `aviationweather.gov` METAR JSON by ICAO code or nearest GPS stations against `§ 107.51(c)–(d)` minimums.

## Evaluation & Tests

| Pipeline Iteration (50-Question Dev Set) | Retrieval Recall | Citation Accuracy | Refusal Guardrails |
| :--- | :--- | :--- | :--- |
| **1. Baseline Hybrid RRF** | 68% (32/47) | 74% (35/47) | 100% (3/3) |
| **2. Source-Balanced (4 CFR + 1 AC)** | 85% (40/47) | 87% (41/47) | 100% (3/3) |
| **3. Agentic Multi-Step RAG** | **98% (46/47)** | **96% (45/47)** | **100% (3/3)** |

**150-Question Unseen Expansion Benchmark (Agentic RAG):**  
The 150-question expansion set comprises 120 in-scope questions + 6 boundary out-of-scope questions that map to a Part 107 exclusion/rule (`Q183`, `Q184`, `Q194`, `Q195`, `Q196`, `Q200`, scored on citing that section = **126** section-graded items) and **24** pure out-of-scope refusal checks:
- **Retrieval Recall:** `115/126 (91%)` · **Citation Accuracy:** `111/126 (88%)`
- **Refusal Guardrails:** `21/24 (88%)` *(up from 33% after tightening prompt against pretraining leaks)*
- **Measured Avg Lookup Time:** `2.35s` · **By Difficulty:** Easy `88%`, Medium `90%`, Hard `86%`
- **Analyzing the 3 Refusal Misses (`Q171`, `Q180`, `Q185`):** `Q180` (HOA bans) actually refused using *"do not specifically address"* (missed our string-match list), `Q171` (5.8 GHz FCC) surfaced a brief frequency note inside `AC 107-2A`, and `Q185` (gun/flamethrower) cited `§ 107.36` hazmat without noting that federal weaponization statutes sit outside Part 107. Next fix: use an LLM-as-judge refusal grader and add a partial-coverage instruction for questions that touch Part 107 but depend on external statutes.
- **Automated Checks:** `pytest` unit tests + `ruff check .` clean via GitHub Actions CI.

## Quickstart & Running Tests

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    echo "GEMINI_API_KEY=your_api_key_here" > .env
    streamlit run streamlit_app.py

    pytest                        # Offline unit tests (including TFR & METAR mocks)
    ruff check .                  # Lint checks
    python -m src.eval.evaluate   # End-to-end RAG evaluation (uses GEMINI_API_KEY)

## Disclaimer
Educational engineering portfolio project only—not official FAA or legal guidance. Always verify current regulations, official NOTAMs, weather, and airspace authorizations (such as LAANC) before flying.
