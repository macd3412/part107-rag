"""Retrieval-augmented generation pipeline for Part 107 / Part 89 questions.

Retrieves regulation chunks with :class:`HybridRetriever`, then answers via the
Gemini REST API using `requests` with ordered model fallback and retries.
"""

import os
import time

import requests
from dotenv import load_dotenv

from src import config
from src.rag.retriever import HybridRetriever

load_dotenv()

SYSTEM_PROMPT = """You are a helpful FAA Part 107 & Remote ID (Part 89) assistant.
Answer the user's question using ONLY the provided regulation excerpts.
Always cite the exact CFR section(s) (e.g., 14 CFR § 107.51(b)) in your answer.
If the provided excerpts do not contain enough information to answer safely, say so clearly and do not guess."""


class Part107RAG:
    """End-to-end retrieve-then-generate pipeline backed by Gemini."""

    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.api_key = os.environ["GEMINI_API_KEY"]

    def ask(self, question: str, top_k: int = 5) -> tuple[str, list[dict]]:
        """Answer ``question`` from retrieved chunks, returning (answer, chunks)."""
        chunks = self.retriever.search(question, top_k=top_k)
        context = "\n\n---\n\n".join(
            f"Source: {c['citation']} ({c['url']})\n{c['text']}" for c in chunks
        )
        prompt = f"Regulations:\n{context}\n\nQuestion: {question}"

        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }

        for model in config.GENERATION_MODELS:
            url = f"{config.GEMINI_API_BASE}/models/{model}:generateContent"
            for _attempt in range(config.MAX_RETRIES):
                try:
                    resp = requests.post(
                        url,
                        params={"key": self.api_key},
                        json=payload,
                        timeout=config.REQUEST_TIMEOUT,
                    )
                except requests.exceptions.Timeout:
                    print(f"{model} timed out — trying next...")
                    break

                if resp.status_code == 200:
                    data = resp.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    return text, chunks
                if resp.status_code in {429, 503}:
                    print(f"{model} busy ({resp.status_code}) — retrying...")
                    time.sleep(config.RETRY_BACKOFF_SECONDS)
                elif resp.status_code == 404:
                    break
                else:
                    resp.raise_for_status()

        raise RuntimeError("All Gemini models are temporarily unavailable. Try again in a minute.")
