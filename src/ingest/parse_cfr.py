"""Parse eCFR XML files into section-aware JSONL chunks."""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from src import config
from src.ingest.fetch_ecfr import TARGETS

CROSS_REF_RE = re.compile(r"(?:§+\s*\d+\.\d+[a-z0-9()]*|part\s+\d+)", re.IGNORECASE)
MAX_CHARS = 1600


def clean_text(elem: ET.Element) -> str:
    """Extract and whitespace-normalize all text inside an XML element."""
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()


def _chunk_section(
    section: ET.Element, part_num: str, subpart_title: str
) -> list[dict]:
    sec_num = section.attrib.get("N", "").strip()
    sec_head_el = section.find("HEAD")
    sec_head = clean_text(sec_head_el) if sec_head_el is not None else f"§ {sec_num}"
    paragraphs = [clean_text(p) for p in section.findall(".//P") if clean_text(p)]
    if not paragraphs:
        return []

    breadcrumb = f"[14 CFR {sec_head} | {subpart_title}]"
    groups: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for p in paragraphs:
        if current and current_len + len(p) > MAX_CHARS:
            groups.append(current)
            current = [p]
            current_len = len(p)
        else:
            current.append(p)
            current_len += len(p)
    if current:
        groups.append(current)

    out: list[dict] = []
    for idx, group in enumerate(groups, start=1):
        body = "\n".join(group)
        full_text = f"{breadcrumb}\n{body}"
        refs = sorted(set(CROSS_REF_RE.findall(body)))
        suffix = f"_p{idx}" if len(groups) > 1 else ""
        out.append({
            "chunk_id": f"14cfr_{sec_num.replace('.', '_')}{suffix}",
            "source_type": "cfr_regulation",
            "citation": f"14 CFR § {sec_num}",
            "part": part_num,
            "subpart": subpart_title,
            "section_number": sec_num,
            "section_title": sec_head,
            "cross_references": refs,
            "url": f"https://www.ecfr.gov/current/title-14/part-{part_num}/section-{sec_num}",
            "text": full_text,
        })
    return out


def parse_part(xml_path: Path, part_num: str) -> list[dict]:
    """Parse an eCFR XML file into section-level regulation chunks."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    chunks: list[dict] = []

    subparts = root.findall(".//DIV6")
    if subparts:
        for subpart in subparts:
            subpart_head_el = subpart.find("HEAD")
            subpart_title = (
                clean_text(subpart_head_el) if subpart_head_el is not None else ""
            )
            for section in subpart.findall(".//DIV8"):
                chunks.extend(_chunk_section(section, part_num, subpart_title))
    else:
        fallback_subpart = f"14 CFR Part {part_num} (Referenced)"
        sections = [root] if root.tag == "DIV8" else root.findall(".//DIV8")
        for section in sections:
            chunks.extend(_chunk_section(section, part_num, fallback_subpart))

    return chunks


def main() -> None:
    """Parse all downloaded CFR XML files and merge with AC 107-2A chunks."""
    cfr_chunks: list[dict] = []
    seen_ids: set[str] = set()
    for name, part_num, _ in TARGETS:
        path = config.RAW_DIR / f"{name}.xml"
        if path.exists():
            for chunk in parse_part(path, part_num):
                if chunk["chunk_id"] not in seen_ids:
                    seen_ids.add(chunk["chunk_id"])
                    cfr_chunks.append(chunk)

    ac_chunks: list[dict] = []
    if config.CHUNKS_PATH.exists():
        existing = [
            json.loads(line)
            for line in config.CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ac_chunks = [c for c in existing if c.get("source_type") == "advisory_circular"]

    combined = cfr_chunks + ac_chunks
    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for chunk in combined:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(cfr_chunks)} CFR chunks + {len(ac_chunks)} AC chunks "
        f"(total: {len(combined)})"
    )


if __name__ == "__main__":
    main()
