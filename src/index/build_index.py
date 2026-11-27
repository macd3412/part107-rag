"""Build and incrementally update the normalized document embedding matrix."""

import json
import time

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from src import config

load_dotenv()

BATCH_SIZE = 25
IDS_PATH = config.PROCESSED_DIR / "embedded_ids.json"


def main() -> None:
    """Embed any uncached chunks and save aligned vectors to ``config.EMBEDDINGS_PATH``."""
    chunks = [
        json.loads(line)
        for line in config.CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    cached: dict[str, np.ndarray] = {}
    if IDS_PATH.exists() and config.EMBEDDINGS_PATH.exists():
        old_ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
        old_matrix = np.load(config.EMBEDDINGS_PATH)
        if len(old_ids) == len(old_matrix):
            cached = dict(zip(old_ids, old_matrix, strict=True))

    missing = [c for c in chunks if c["chunk_id"] not in cached]
    print(f"Reusing {len(chunks) - len(missing)} cached embeddings; {len(missing)} new to embed.")

    if missing:
        client = genai.Client()
        texts = [c["text"] for c in missing]
        new_vecs: list[list[float]] = []
        i = 0
        while i < len(texts):
            batch = texts[i : i + BATCH_SIZE]
            print(f"Embedding new chunks {i + 1}-{i + len(batch)} of {len(texts)}...")
            try:
                resp = client.models.embed_content(
                    model=config.EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                new_vecs.extend([e.values for e in resp.embeddings])
                i += BATCH_SIZE
            except errors.ClientError as e:
                if e.code == 429:
                    print("Hit free-tier limit — waiting 62 seconds...")
                    time.sleep(62)
                else:
                    raise

        new_mat = np.array(new_vecs, dtype=np.float32)
        norms = np.linalg.norm(new_mat, axis=1, keepdims=True)
        new_mat = new_mat / np.clip(norms, 1e-12, None)
        for c, vec in zip(missing, new_mat, strict=True):
            cached[c["chunk_id"]] = vec

    matrix = np.stack([cached[c["chunk_id"]] for c in chunks]).astype(np.float32)
    np.save(config.EMBEDDINGS_PATH, matrix)
    IDS_PATH.write_text(json.dumps([c["chunk_id"] for c in chunks]), encoding="utf-8")
    print(f"Saved {matrix.shape} to {config.EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
