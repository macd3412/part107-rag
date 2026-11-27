import os
import time
import requests
from dotenv import load_dotenv
from src.rag.retriever import HybridRetriever

load_dotenv()

SYSTEM_PROMPT = """You are a helpful FAA Part 107 & Remote ID (Part 89) assistant.
Answer the user's question using ONLY the provided regulation excerpts.
Always cite the exact CFR section(s) (e.g., 14 CFR § 107.51(b)) in your answer.
If the provided excerpts do not contain enough information to answer safely, say so clearly and do not guess."""

MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.8-flash"]

class Part107RAG:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.api_key = os.environ["GEMINI_API_KEY"]

    def ask(self, question: str, top_k: int = 5) -> tuple[str, list[dict]]:
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

        for model in MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            for attempt in range(2):
                try:
                    resp = requests.post(
                        url,
                        params={"key": self.api_key},
                        json=payload,
                        timeout=45,
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
                    time.sleep(3)
                elif resp.status_code == 404:
                    break
                else:
                    resp.raise_for_status()

        raise RuntimeError("All Gemini models are temporarily unavailable. Try again in a minute.")
