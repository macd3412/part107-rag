"""Run the 50-question Part 107 evaluation suite."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from src import config
from src.rag.agent import AgenticPart107RAG
from src.rag.pipeline import Part107RAG

RESULTS_PATH = Path("data/eval/eval_results.json")
REFUSAL_PHRASES = [
    "do not contain",
    "does not contain",
    "not covered",
    "do not cover",
    "does not cover",
    "does not address",
    "do not address",
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
]


def main() -> None:
    """Evaluate retrieval recall, citation accuracy, and refusal guardrails."""
    parser = argparse.ArgumentParser(description="Evaluate Part 107 RAG")
    parser.add_argument(
        "--mode", choices=["agent", "single"], default="agent", help="Pipeline mode"
    )
    args = parser.parse_args()

    cases = json.loads(config.EVAL_PATH.read_text(encoding="utf-8"))
    rag = AgenticPart107RAG() if args.mode == "agent" else Part107RAG()

    in_scope_total = 0
    recall_hits = 0
    citation_hits = 0
    refusal_total = 0
    refusal_hits = 0
    by_diff: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    results: list[dict] = []

    for case in cases:
        qid = case.get("id", "?")
        q = case["question"]
        expected = case.get("expected_sections") or (
            [case["expected_section"]] if case.get("expected_section") else []
        )
        should_refuse = case.get("should_refuse", False)
        diff = case.get("difficulty", "Unknown")

        if args.mode == "agent":
            answer, chunks = rag.ask(q, verbose=False)
        else:
            answer, chunks = rag.ask(q, top_k=5)
        retrieved = [c["section_number"] for c in chunks]

        if should_refuse:
            refusal_total += 1
            refused = any(p in answer.lower() for p in REFUSAL_PHRASES)
            if refused:
                refusal_hits += 1
            print(f"[{qid}] REFUSAL {'PASS' if refused else 'FAIL'} ({diff}) | {q}")
            results.append(
                {
                    "id": qid,
                    "status": "PASS" if refused else "FAIL",
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
        time.sleep(5.0)

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n=== 50-Question Evaluation Summary ({args.mode}) ===")
    print(
        f"Retrieval Recall : {recall_hits}/{in_scope_total} "
        f"({recall_hits / in_scope_total:.0%})"
    )
    print(
        f"Citation Accuracy: {citation_hits}/{in_scope_total} "
        f"({citation_hits / in_scope_total:.0%})"
    )
    print(
        f"Refusal Guardrail: {refusal_hits}/{refusal_total} "
        f"({refusal_hits / refusal_total:.0%})"
    )
    print("\nBy Difficulty (Recall + Cite both OK):")
    for diff in ["Easy", "Medium", "Hard"]:
        if diff in by_diff:
            ok, tot = by_diff[diff]
            print(f"  {diff:6s}: {ok}/{tot} ({ok / tot:.0%})")
    print(f"\nDetailed results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
