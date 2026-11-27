import json
import re
from pathlib import Path

import openpyxl

from src.config import EVAL_PATH

CANDIDATES = [
    Path("part107_test_set.xlsx"),
    Path.home() / "Downloads" / "part107_test_set.xlsx",
    Path.home() / "Desktop" / "part107_test_set.xlsx",
]
SEC_RE = re.compile(r"\b(?:107|89)\.\d+\b")

def find_xlsx() -> Path:
    for path in CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find part107_test_set.xlsx in current folder, ~/Downloads, or ~/Desktop."
    )

def main() -> None:
    xlsx_path = find_xlsx()
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["Questions"]

    cases: list[dict] = []
    for r in range(2, ws.max_row + 1):
        qid = ws.cell(r, 1).value
        if not qid:
            continue
        topic = ws.cell(r, 2).value or ""
        difficulty = ws.cell(r, 3).value or ""
        qtype = ws.cell(r, 4).value or ""
        question = ws.cell(r, 5).value or ""
        expected_answer = ws.cell(r, 6).value or ""
        citation_raw = str(ws.cell(r, 7).value or "")
        sections = sorted(
            set(SEC_RE.findall(citation_raw)),
            key=lambda s: [int(x) for x in s.split(".")],
        )
        should_refuse = (qtype.strip().lower() == "out of scope") and (
            len(sections) == 0 or qid == "Q50"
        )
        cases.append({
            "id": qid,
            "topic": topic,
            "difficulty": difficulty,
            "question_type": qtype,
            "question": question,
            "expected_answer": expected_answer,
            "expected_sections": sections,
            "should_refuse": should_refuse,
        })

    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Imported {len(cases)} questions from {xlsx_path} -> {EVAL_PATH}")

if __name__ == "__main__":
    main()
