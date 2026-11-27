import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/chunks.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CROSS_REF_RE = re.compile(r"(?:§+\s*\d+\.\d+[a-z0-9()]*|part\s+\d+)", re.IGNORECASE)
MAX_CHARS = 1600

def clean_text(elem: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()

def parse_part(xml_path: Path, part_num: str) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    chunks: list[dict] = []

    for subpart in root.findall(".//DIV6"):
        subpart_head_el = subpart.find("HEAD")
        subpart_title = clean_text(subpart_head_el) if subpart_head_el is not None else ""

        for section in subpart.findall(".//DIV8"):
            sec_num = section.attrib.get("N", "").strip()
            sec_head_el = section.find("HEAD")
            sec_head = clean_text(sec_head_el) if sec_head_el is not None else f"§ {sec_num}"
            paragraphs = [clean_text(p) for p in section.findall(".//P") if clean_text(p)]
            if not paragraphs:
                continue

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

            for idx, group in enumerate(groups, start=1):
                body = "\n".join(group)
                full_text = f"{breadcrumb}\n{body}"
                refs = sorted(set(CROSS_REF_RE.findall(body)))
                suffix = f"_p{idx}" if len(groups) > 1 else ""
                chunks.append({
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
    return chunks

def main() -> None:
    all_chunks = []
    for part_num, filename in [("107", "part107.xml"), ("89", "part89.xml")]:
        path = RAW_DIR / filename
        if path.exists():
            all_chunks.extend(parse_part(path, part_num))

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_chunks)} CFR chunks to {OUT_PATH}")

if __name__ == "__main__":
    main()
