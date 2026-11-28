"""Hybrid dense + lexical retriever over the Part 107 / Part 89 corpus.

Combines cosine similarity over Gemini embeddings with BM25 lexical scores via
Reciprocal Rank Fusion (RRF). Query embeddings are fetched from the Gemini REST
API with `requests` (retry + timeout), matching `src/rag/pipeline.py`, to avoid
the `google-genai` / `httpx2` client-closed errors seen on Python 3.15.
"""

import json
import os
import re
import time

import numpy as np
import requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from src import config

load_dotenv()


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric/dotted tokens for BM25."""
    return re.findall(r"[a-z0-9.]+", text.lower())


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], k: int = 60, top_k: int = 5
) -> list[int]:
    """Fuse several ranked index lists into one via Reciprocal Rank Fusion.

    Each ranked list contains chunk indices ordered best-first. An item's fused
    score is the sum of ``1 / (k + rank + 1)`` across the lists it appears in.

    Args:
        ranked_lists: One list of chunk indices per retriever, best-first.
        k: RRF damping constant; larger values flatten rank contributions.
        top_k: Number of fused indices to return.

    Returns:
        The ``top_k`` chunk indices sorted by descending fused score.
    """
    scores: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, idx in enumerate(ranking):
            scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:top_k]


class HybridRetriever:
    """Retrieve relevant regulation chunks using dense + BM25 fusion."""

    def __init__(
        self,
        chunks: list[dict] | None = None,
        embeddings: np.ndarray | None = None,
        api_key: str | None = None,
    ) -> None:
        """Build the retriever.

        Args:
            chunks: Pre-loaded chunk dicts. Defaults to reading
                ``config.CHUNKS_PATH``. Injectable for offline testing.
            embeddings: Normalized document embedding matrix. Defaults to
                reading ``config.EMBEDDINGS_PATH``. Injectable for testing.
            api_key: Gemini API key. Defaults to the ``GEMINI_API_KEY`` env var;
                only required when :meth:`search` performs a live embed call.
        """
        if chunks is None:
            chunks = [
                json.loads(line)
                for line in config.CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if embeddings is None:
            embeddings = np.load(config.EMBEDDINGS_PATH)

        self.chunks = chunks
        self.embeddings = embeddings
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query via the Gemini REST API, returning a unit vector.

        Retries transient 429/503 responses and timeouts up to
        ``config.MAX_RETRIES`` times before giving up.
        """
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set; cannot embed query.")

        url = f"{config.GEMINI_API_BASE}/models/{config.EMBED_MODEL}:embedContent"
        payload = {
            "model": f"models/{config.EMBED_MODEL}",
            "content": {"parts": [{"text": query}]},
            "taskType": "RETRIEVAL_QUERY",
        }

        for attempt in range(config.MAX_RETRIES):
            try:
                resp = requests.post(
                    url,
                    params={"key": self.api_key},
                    json=payload,
                    timeout=config.REQUEST_TIMEOUT,
                )
            except requests.exceptions.Timeout:
                if attempt == config.MAX_RETRIES - 1:
                    raise
                continue

            if resp.status_code == 200:
                values = resp.json()["embedding"]["values"]
                vec = np.array(values, dtype=np.float32)
                return vec / max(float(np.linalg.norm(vec)), 1e-12)
            if resp.status_code in {429, 503}:
                time.sleep(20 if resp.status_code == 429 else 4)
                continue
            resp.raise_for_status()

        raise RuntimeError("Gemini embedding API is temporarily unavailable.")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the ``top_k`` chunks most relevant to ``query``.

        Dense cosine-similarity and BM25 rankings are fused with
        :func:`reciprocal_rank_fusion`, balancing binding CFR sections with
        advisory circular guidance when both source types exist.
        """
        q_vec = self.embed_query(query)

        vec_scores = self.embeddings @ q_vec
        vec_ranks = np.argsort(-vec_scores)

        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_ranks = np.argsort(-bm25_scores)

        fused = reciprocal_rank_fusion(
            [vec_ranks.tolist(), bm25_ranks.tolist()], top_k=len(self.chunks)
        )

        cfr = [i for i in fused if self.chunks[i].get("source_type") == "cfr_regulation"]
        ac = [i for i in fused if self.chunks[i].get("source_type") == "advisory_circular"]
        if cfr and ac and top_k >= 2:
            chosen = cfr[: top_k - 1] + ac[:1]
            if len(chosen) < top_k:
                seen = set(chosen)
                chosen.extend([i for i in fused if i not in seen][: top_k - len(chosen)])
            return [self.chunks[i] for i in chosen]

        return [self.chunks[i] for i in fused[:top_k]]
