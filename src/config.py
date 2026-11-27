"""Central configuration for the Part 107 RAG project.

Collects the file-system layout, Gemini model names, and shared REST settings
in one place so the ingestion, indexing, retrieval, and evaluation modules all
agree on the same paths and models.
"""

from pathlib import Path

# --- File-system layout -----------------------------------------------------
# Paths are anchored to the project root (two levels up from this file) so the
# modules work regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
EVAL_PATH = EVAL_DIR / "part107_eval.json"

# --- Gemini models ----------------------------------------------------------
# Embedding model used for both document indexing and query retrieval.
EMBED_MODEL = "gemini-embedding-001"

# Generation models tried in order; the first one that responds wins. Listing
# several gives graceful fallback when a model is busy (429/503) or missing.
GENERATION_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]

# --- REST API settings ------------------------------------------------------
# We call the Gemini REST API directly with `requests` rather than the
# `google-genai` client, which raises client-closed errors on Python 3.15.
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
REQUEST_TIMEOUT = 45
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 3
