"""Run the Part 107 evaluation suite with LLM-as-a-judge refusal grading."""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

from src import config
from src.rag.agent import AgenticPart107RAG
from src.rag.pipeline import Part107RAG

load_dotenv()

RESULTS_PATH = Path("data/eval/eval_results.json")
REFUSAL_PHRASES = [
    "do not contain",
    "does not contain",
    "not covered",
    "do not cover",
    "does not cover",
    "does not address",
    "do not address",
    "do not specifically address",
    "does not specifically address",
    "cannot answer",
    "does not mention",
    "not enough information",
    "do not provide",
    "does not provide",
    "no requirement",
    "outside",
    "do not require",
    "does not require",
    "no mandate",
    "no part 108",
    "do not regulate",
    "does not regulate",
    "do not specify",
    "does not specify",
]


def judge_refusal(question: str, answer: str, api_key: str | None = None) -> tuple[bool, str]:
    """Use Gemini at temperature=0 to grade whether an out-of-scope answer refused cleanly."""
    key = api_key or os.environ["GEMINI_API_KEY"]
    prompt = (
        "You are grading an FAA Part 107 RAG assistant on an out-of-scope question.\n"
        f"Question: {question}\n"
        f"Assistant Answer: {answer}\n\n"
        "Did the assistant appropriately refuse or clarify that the provided FAA Part 107 / "
        "Part 89 regulations do not cover the core out-of-scope topic, without inventing "
        "external laws, non-corpus CFR sections, pricing, or hardware specs?\n"
        "Reply with exactly one line starting with 'PASS: ' or 'FAIL: ' and a brief reason."
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0},
    }
    for model in config.GENERATION_MODELS:
        url = f"{config.GEMINI_API_BASE}/models/{model}:generateContent"
        try:
            resp = requests.post(
                url, params={"key": key}, json=payload, timeout=config.REQUEST_TIMEOUT
            )
        except requests.exceptions.Timeout:
            continue
        if resp.status_code == 200:
            verdict = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            return verdict.upper().startswith("PASS"), verdict
        if resp.status_code in {429, 503}:
            time.sleep(config.RETRY_BACKOFF_SECONDS)
        elif resp.status_code == 404:
            continue
        else:
            resp.raise_for_status()
    raise RuntimeError("LLM refusal judge is temporarily unavailable.")


def main() -> None:
    """Evaluate retrieval recall, citation accuracy, refusal guardrails, and latency."""
    parser = argparse.ArgumentParser(description="Evaluate Part 107 RAG")
    parser.add_argument("--mode", choices=["agent", "single"], default="agent")
    parser.add_argument("--judge", choices=["llm", "keyword"], default="llm")
    parser.add_argument(
        "--only-refusals", action="store_true", help="Run only out-of-scope refusal cases"
    )
    args = parser.parse_args()

    cases = json.loads(config.EVAL_PATH.read_text(encoding="utf-8"))
    if args.only_refusals:
        cases = [c for c in cases if c.get("should_refuse", False)]

    rag = AgenticPart107RAG() if args.mode == "agent" else Part107RAG()

    in_scope_total = 0
    recall_hits = 0
    citation_hits = 0
    refusal_total = 0
    refusal_hits = 0
    by_diff: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    results: list[dict] = []
    latencies: list[float] = []

    for case in cases:
        qid = case.get("id", "?")
        q = case["question"]
        expected = case.get("expected_sections") or (
            [case["expected_section"]] if case.get("expected_section") else []
        )
        should_refuse = case.get("should_refuse", False)
        diff = case.get("difficulty", "Unknown")

        t0 = time.perf_counter()
        if args.mode == "agent":
            answer, chunks = rag.ask(q, verbose=False)
        else:
            answer, chunks = rag.ask(q, top_k=5)
        latencies.append(time.perf_counter() - t0)
        retrieved = [c["section_number"] for c in chunks]

        if should_refuse:
            refusal_total += 1
            if args.judge == "llm":
                refused, reason = judge_refusal(q, answer)
            else:
                refused = any(p in answer.lower() for p in REFUSAL_PHRASES)
                reason = "keyword match"
            if refused:
                refusal_hits += 1
            print(f"[{qid}] REFUSAL {'PASS' if refused else 'FAIL'} ({diff}) | {q} -> {reason}")
            results.append(
                {
                    "id": qid,
                    "status": "PASS" if refused else "FAIL",
                    "judge_reason": reason,
                    "question": q,
                    "answer": answer,
                }
            )
        else:
            in_scope_total += 1
            recalled = any(s in retrieved for s in expected)
            cited = any(s in answer for s in expected)
            if recalled:
                recall_hits += 1
            if cited:
                citation_hits += 1
            by_diff[diff][1] += 1
            if recalled and cited:
                by_diff[diff][0] += 1
            print(
                f"[{qid}] Recall={'OK' if recalled else 'MISS'} "
                f"Cite={'OK' if cited else 'MISS'} "
                f"({diff}, expected {expected}) | {q}"
            )
            results.append(
                {
                    "id": qid,
                    "recalled": recalled,
                    "cited": cited,
                    "expected": expected,
                    "retrieved": retrieved,
                    "question": q,
                    "answer": answer,
                }
            )
        time.sleep(0.3)

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n=== {len(cases)}-Question Evaluation Summary ({args.mode}, judge={args.judge}) ===")
    if in_scope_total:
        print(f"Retrieval Recall : {recall_hits}/{in_scope_total} ({recall_hits / in_scope_total:.0%})")
        print(f"Citation Accuracy: {citation_hits}/{in_scope_total} ({citation_hits / in_scope_total:.0%})")
    if refusal_total:
        print(f"Refusal Guardrail: {refusal_hits}/{refusal_total} ({refusal_hits / refusal_total:.0%})")
    avg_lat = sum(latencies) / len(latencies)
    print(f"Avg Lookup Time  : {avg_lat:.2f}s")
    if by_diff:
        print("\nBy Difficulty (Recall + Cite both OK):")
        for diff in ["Easy", "Medium", "Hard"]:
            if diff in by_diff:
                ok, tot = by_diff[diff]
                print(f"  {diff:6s}: {ok}/{tot} ({ok / tot:.0%})")
    print(f"\nDetailed results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
