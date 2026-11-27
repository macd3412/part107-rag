import json
import re
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi

load_dotenv()

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMB_PATH = Path("data/processed/embeddings.npy")
EMBED_MODEL = "gemini-embedding-001"

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9.]+", text.lower())

class HybridRetriever:
    def __init__(self) -> None:
        self.chunks = [
            json.loads(line)
            for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.embeddings = np.load(EMB_PATH)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.client = genai.Client()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        resp = self.client.models.embed_content(
            model=EMBED_MODEL,
            contents=[query],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        q_vec = np.array(resp.embeddings[0].values, dtype=np.float32)
        q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-12)

        vec_scores = self.embeddings @ q_vec
        vec_ranks = np.argsort(-vec_scores)

        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_ranks = np.argsort(-bm25_scores)

        rrf: dict[int, float] = {}
        for rank, idx in enumerate(vec_ranks):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(bm25_ranks):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (60 + rank + 1)

        top_indices = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
        return [self.chunks[i] for i in top_indices]
