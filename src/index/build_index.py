"""Build the document embedding matrix from the chunked corpus.

This is a one-off offline indexing step run with the `google-genai` client;
query-time retrieval uses the REST API (see `src/rag/retriever.py`).
"""

import json
import time

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from src import config

load_dotenv()

BATCH_SIZE = 25

def main() -> None:
    chunks = [json.loads(line) for line in config.CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = [c["text"] for c in chunks]

    client = genai.Client()
    all_vecs: list[list[float]] = []

    i = 0
    while i < len(texts):
        batch = texts[i : i + BATCH_SIZE]
        print(f"Embedding chunks {i + 1}–{i + len(batch)} of {len(texts)}...")
        try:
            resp = client.models.embed_content(
                model=config.EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            all_vecs.extend([e.values for e in resp.embeddings])
            i += BATCH_SIZE
        except errors.ClientError as e:
            if e.code == 429:
                print("Hit free-tier 100/min limit — waiting 62 seconds before continuing...")
                time.sleep(62)
            else:
                raise

    matrix = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-12, None)

    np.save(config.EMBEDDINGS_PATH, matrix)
    print(f"Saved embeddings {matrix.shape} to {config.EMBEDDINGS_PATH}")

if __name__ == "__main__":
    main()
