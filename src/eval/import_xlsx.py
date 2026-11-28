"""Convert the Part 107 Excel test set into ``data/eval/part107_eval.json``."""

import json
import re
from pathlib import Path

import openpyxl

from src import config

CANDIDATES = [
    Path("part107_test_set_expansion_150.xlsx"),
    Path.home() / "Downloads" / "part107_test_set_expansion_150.xlsx",
    Path.home() / "Desktop" / "part107_test_set_expansion_150.xlsx",
    Path("part107_test_set.xlsx"),
    Path.home() / "Downloads" / "part107_test_set.xlsx",
    Path.home() / "Desktop" / "part107_test_set.xlsx",
]
SEC_RE = re.compile(r"\b(?:107|89|48|61|91)\.\d+\b")


def find_xlsx() -> Path:
    """Locate the Part 107 Excel test set in common local folders."""
    for path in CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find part107_test_set_expansion_150.xlsx or part107_test_set.xlsx."
    )


def main() -> None:
    """Parse all questions from Excel and write ``config.EVAL_PATH``."""
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
        notes = str(ws.cell(r, 9).value or "")
        sections = sorted(
            set(SEC_RE.findall(citation_raw)),
            key=lambda s: [int(x) for x in s.split(".")],
        )
        is_oos = qtype.strip().lower() == "out of scope"
        should_refuse = is_oos and (
            len(sections) == 0
            or qid == "Q50"
            or "refus" in notes.lower()
            or "outside" in notes.lower()
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

    config.EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.EVAL_PATH.write_text(
        json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    refusals = sum(1 for c in cases if c["should_refuse"])
    print(
        f"Imported {len(cases)} questions ({refusals} refusal checks) "
        f"from {xlsx_path.name} -> {config.EVAL_PATH}"
    )


if __name__ == "__main__":
    main()
