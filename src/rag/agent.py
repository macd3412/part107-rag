"""Agentic tool-calling RAG loop using the Gemini REST API."""

import os
import time

import requests
from dotenv import load_dotenv

from src import config
from src.rag.retriever import HybridRetriever

load_dotenv()

SYSTEM_PROMPT = (
    "You are an FAA Part 107 & Remote ID (Part 89) assistant. "
    "Call search_regulations 1 to 2 times to gather relevant excerpts. If the first "
    "search misses, rewrite your query using formal FAA terms (e.g., 'change of "
    "address' instead of 'moved', 'recreational' instead of 'fly for fun'). "
    "Once you have relevant excerpts, stop searching and write a clear answer "
    "citing exact CFR/AC sections. If not covered, state that clearly."
)

TOOLS = [
    {
        "functionDeclarations": [
            {
                "name": "search_regulations",
                "description": (
                    "Search 14 CFR Part 107, Part 89, referenced Parts 48/61/91, "
                    "and FAA AC 107-2A. Use specific regulatory keywords."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "query": {
                            "type": "STRING",
                            "description": "Search query using FAA regulatory terms.",
                        }
                    },
                    "required": ["query"],
                },
            }
        ]
    }
]


class AgenticPart107RAG:
    """Multi-step tool-calling RAG agent over FAA Part 107 regulations."""

    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.api_key = os.environ["GEMINI_API_KEY"]

    def _call_gemini(self, contents: list[dict], use_tools: bool = True) -> dict:
        payload: dict = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.2},
        }
        if use_tools:
            payload["tools"] = TOOLS

        for model in config.GENERATION_MODELS:
            url = f"{config.GEMINI_API_BASE}/models/{model}:generateContent"
            for _ in range(2):
                try:
                    resp = requests.post(
                        url,
                        params={"key": self.api_key},
                        json=payload,
                        timeout=config.REQUEST_TIMEOUT,
                    )
                except requests.exceptions.Timeout:
                    break
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in {429, 503}:
                    time.sleep(config.RETRY_BACKOFF_SECONDS)
                elif resp.status_code == 404:
                    break
                else:
                    resp.raise_for_status()
        raise RuntimeError("All Gemini models are temporarily unavailable.")

    def ask(
        self, question: str, max_steps: int = 3, verbose: bool = True
    ) -> tuple[str, list[dict]]:
        """Run a tool-calling loop and guarantee a synthesized final answer."""
        contents: list[dict] = [{"role": "user", "parts": [{"text": question}]}]
        seen_ids: set[str] = set()
        collected: list[dict] = []

        for step in range(max_steps):
            data = self._call_gemini(contents, use_tools=True)
            candidate = data["candidates"][0]["content"]
            parts = candidate.get("parts", [])
            calls = [p["functionCall"] for p in parts if "functionCall" in p]

            if not calls:
                text = "\n".join(p["text"] for p in parts if "text" in p)
                return text, collected

            contents.append(candidate)
            response_parts = []
            for call in calls:
                query = call["args"]["query"]
                if verbose:
                    print(f"  [Step {step + 1}] search_regulations({query!r})")
                hits = self.retriever.search(query, top_k=5)
                for h in hits:
                    if h["chunk_id"] not in seen_ids:
                        seen_ids.add(h["chunk_id"])
                        collected.append(h)

                snippet = "\n\n".join(
                    f"Source: {h['citation']} ({h['url']})\n{h['text']}" for h in hits
                )
                response_parts.append({
                    "functionResponse": {
                        "name": call["name"],
                        "response": {"results": snippet},
                    }
                })
            contents.append({"role": "user", "parts": response_parts})

        # Fallback: force final answer from collected excerpts without tools
        context = "\n\n---\n\n".join(
            f"Source: {c['citation']} ({c['url']})\n{c['text']}" for c in collected
        )
        final = self._call_gemini(
            [{"role": "user", "parts": [{"text": f"Regulations:\n{context}\n\nQuestion: {question}"}]}],
            use_tools=False,
        )
        final_parts = final["candidates"][0]["content"].get("parts", [])
        text = "\n".join(p["text"] for p in final_parts if "text" in p)
        return text, collected
