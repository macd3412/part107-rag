"""Agentic tool-calling RAG loop using the Gemini REST API."""

import os
import time

import requests
from dotenv import load_dotenv

from src import config
from src.rag.retriever import HybridRetriever
from src.tools.tfr import check_active_tfrs
from src.tools.weather import check_aviation_weather

load_dotenv()

SYSTEM_PROMPT = (
    "You are an FAA Part 107 & Remote ID (Part 89) assistant. "
    "Call search_regulations 1 to 2 times to gather relevant excerpts, and call "
    "check_active_tfrs whenever the user asks about active TFRs, a specific state/city, "
    "or whether they can fly at a location right now, and call check_aviation_weather whenever the user asks about current weather, wind, visibility, or clouds. If the first search misses, "
    "rewrite your query using formal FAA terms (e.g., 'change of address' instead of "
    "'moved', 'recreational' instead of 'fly for fun'). "
    "Answer ONLY from the retrieved excerpts and tool outputs, and ALWAYS cite the "
    "exact CFR, AC, or NOTAM identifier from those results. "
    "Do NOT answer from outside memory or cite any regulation, state law, "
    "national park rule, or hardware spec not in the tool outputs. "
    "If the tool outputs do not directly answer the question, state clearly that the "
    "provided regulations do not cover this topic."
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
            },
            {
                "name": "check_active_tfrs",
                "description": (
                    "Check live active and scheduled FAA Temporary Flight Restrictions "
                    "(TFRs) from tfr.faa.gov filtered by 2-letter US state code and/or keyword."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "state": {
                            "type": "STRING",
                            "description": "2-letter US state code (e.g., 'MA', 'TX'), or empty.",
                        },
                        "keyword": {
                            "type": "STRING",
                            "description": "Optional city or keyword filter (e.g., 'Boston'), or empty.",
                        },
                        "lat": {"type": "NUMBER", "description": "Optional latitude if user GPS is provided."},
                        "lon": {"type": "NUMBER", "description": "Optional longitude if user GPS is provided."},
                    },
                    "required": [],
                },
            },
            {
                "name": "check_aviation_weather",
                "description": (
                    "Fetch live aviation METAR weather (visibility, clouds, wind) "
                    "by 4-letter ICAO airport code or user GPS (lat, lon)."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "station_id": {"type": "STRING", "description": "Optional 4-letter ICAO code (e.g., KBOS)."},
                        "lat": {"type": "NUMBER", "description": "Optional latitude from user GPS."},
                        "lon": {"type": "NUMBER", "description": "Optional longitude from user GPS."},
                    },
                    "required": [],
                },
            },
        ]
    }
]


class AgenticPart107RAG:
    """Multi-step tool-calling RAG agent over FAA Part 107 and live TFRs."""

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
                name = call["name"]
                args = call.get("args", {})
                if name == "check_active_tfrs":
                    state = str(args.get("state", ""))
                    keyword = str(args.get("keyword", ""))
                    lat = args.get("lat")
                    lon = args.get("lon")
                    if verbose:
                        print(f"  [Step {step + 1}] check_active_tfrs(state={state!r}, lat={lat}, lon={lon})")
                    result_text = check_active_tfrs(state=state, keyword=keyword, lat=lat, lon=lon)
                    collected.append({
                        "chunk_id": f"tfr_{state}_{keyword}",
                        "citation": f"FAA Live TFRs ({state or 'US'})",
                        "section_title": "Active & Scheduled NOTAMs",
                        "url": "https://tfr.faa.gov/",
                        "text": result_text,
                    })
                elif name == "check_aviation_weather":
                    sid = str(args.get("station_id", ""))
                    lat = args.get("lat")
                    lon = args.get("lon")
                    if verbose:
                        print(f"  [Step {step + 1}] check_aviation_weather(station_id={sid!r}, lat={lat}, lon={lon})")
                    result_text = check_aviation_weather(station_id=sid, lat=lat, lon=lon)
                    collected.append({
                        "chunk_id": f"metar_{sid}_{lat}_{lon}",
                        "citation": f"Live METAR ({sid or 'Nearest'})",
                        "section_title": "AviationWeather.gov Observation",
                        "url": "https://aviationweather.gov/",
                        "text": result_text,
                    })
                else:
                    query = str(args.get("query", question))
                    if verbose:
                        print(f"  [Step {step + 1}] search_regulations({query!r})")
                    hits = self.retriever.search(query, top_k=5)
                    for h in hits:
                        if h["chunk_id"] not in seen_ids:
                            seen_ids.add(h["chunk_id"])
                            collected.append(h)
                    result_text = "\n\n".join(
                        f"Source: {h['citation']} ({h['url']})\n{h['text']}" for h in hits
                    )

                response_parts.append({
                    "functionResponse": {
                        "name": name,
                        "response": {"results": result_text},
                    }
                })
            contents.append({"role": "user", "parts": response_parts})

        context = "\n\n---\n\n".join(
            f"Source: {c['citation']} ({c['url']})\n{c['text']}" for c in collected
        )
        final = self._call_gemini(
            [{"role": "user", "parts": [{"text": f"Context:\n{context}\n\nQuestion: {question}"}]}],
            use_tools=False,
        )
        final_parts = final["candidates"][0]["content"].get("parts", [])
        text = "\n".join(p["text"] for p in final_parts if "text" in p)
        return text, collected
