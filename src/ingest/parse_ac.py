"""Parse the AC 107-2A advisory-circular PDF into overlapping page chunks."""

import json
import re

from pypdf import PdfReader

from src import config

PDF_PATH = config.RAW_DIR / "ac_107_2a.pdf"
AC_URL = "https://www.faa.gov/regulations_policies/advisory_circulars/index.cfm/go/document.information/documentID/1038977"
CROSS_REF_RE = re.compile(r"(?:§+\s*\d+\.\d+[a-z0-9()]*|part\s+\d+)", re.IGNORECASE)

MAX_CHARS = 1500
OVERLAP = 200

def clean_page_text(raw: str) -> str:
    """Collapse runs of whitespace in extracted PDF text to single spaces."""
    return re.sub(r"\s+", " ", raw).strip()

def split_with_overlap(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Split ``text`` into windows of ``max_chars`` that overlap by ``overlap``."""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return parts

def main() -> None:
    reader = PdfReader(PDF_PATH)
    ac_chunks: list[dict] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = clean_page_text(page.extract_text() or "")
        # Skip nearly empty pages and dotted Table of Contents pages
        if len(text) < 120 or text.count(".....") >= 3:
            continue

        breadcrumb = f"[FAA AC 107-2A — Small UAS Advisory Circular | Page {page_num}]"
        for idx, part in enumerate(split_with_overlap(text), start=1):
            full_text = f"{breadcrumb}\n{part}"
            refs = sorted(set(CROSS_REF_RE.findall(part)))
            ac_chunks.append({
                "chunk_id": f"ac107_2a_p{page_num}_{idx}",
                "source_type": "advisory_circular",
                "citation": f"FAA AC 107-2A, p. {page_num}",
                "part": "AC 107-2A",
                "subpart": "Advisory Circular",
                "section_number": f"p.{page_num}",
                "section_title": f"AC 107-2A Page {page_num}",
                "cross_references": refs,
                "url": AC_URL,
                "text": full_text,
            })

    existing = [
        json.loads(line)
        for line in config.CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cfr_only = [c for c in existing if c.get("source_type") != "advisory_circular"]
    combined = cfr_only + ac_chunks

    with config.CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for chunk in combined:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Added {len(ac_chunks)} AC 107-2A chunks (total in chunks.jsonl: {len(combined)})")

if __name__ == "__main__":
    main()
