import json
import time
from pathlib import Path
from src.rag.pipeline import Part107RAG

EVAL_PATH = Path("data/eval/part107_eval.json")

def main() -> None:
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    rag = Part107RAG()

    in_scope_total = 0
    recall_hits = 0
    citation_hits = 0
    refusal_total = 0
    refusal_hits = 0

    for idx, case in enumerate(cases, start=1):
        q = case["question"]
        expected = case["expected_section"]
        should_refuse = case["should_refuse"]

        answer, chunks = rag.ask(q, top_k=5)
        retrieved_sections = [c["section_number"] for c in chunks]

        if should_refuse:
            refusal_total += 1
            lower = answer.lower()
            refused = any(
                phrase in lower
                for phrase in ["do not contain", "not covered", "cannot answer", "does not mention", "not enough information", "do not provide"]
            )
            if refused:
                refusal_hits += 1
            status = "PASS" if refused else "FAIL"
            print(f"[{idx:02d}] REFUSAL {status} | {q}")
        else:
            in_scope_total += 1
            recalled = expected in retrieved_sections
            cited = expected in answer
            if recalled:
                recall_hits += 1
            if cited:
                citation_hits += 1
            print(f"[{idx:02d}] Recall={'OK' if recalled else 'MISS'} Cite={'OK' if cited else 'MISS'} (expected § {expected}) | {q}")

        time.sleep(1.5)

    print("\n=== Evaluation Summary ===")
    print(f"Retrieval Recall@5 : {recall_hits}/{in_scope_total} ({recall_hits / in_scope_total:.0%})")
    print(f"Citation Accuracy  : {citation_hits}/{in_scope_total} ({citation_hits / in_scope_total:.0%})")
    print(f"Refusal Guardrails : {refusal_hits}/{refusal_total} ({refusal_hits / refusal_total:.0%})")

if __name__ == "__main__":
    main()
