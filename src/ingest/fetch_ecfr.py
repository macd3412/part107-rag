"""Download 14 CFR Part 107, Part 89, and cross-referenced sections from eCFR."""

from datetime import date

import requests

from src import config

DATES = [date.today().isoformat(), "2026-11-01"]

TARGETS: list[tuple[str, str, str | None]] = [
    ("part107", "107", None),
    ("part89", "89", None),
    ("sec48_15", "48", "48.15"),
    ("sec61_56", "61", "61.56"),
    ("sec91_17", "91", "91.17"),
    ("sec91_137", "91", "91.137"),
    ("sec91_138", "91", "91.138"),
    ("sec91_139", "91", "91.139"),
    ("sec91_141", "91", "91.141"),
    ("sec91_143", "91", "91.143"),
    ("sec91_145", "91", "91.145"),
]


def fetch_target(part_num: str, section: str | None = None) -> bytes:
    """Fetch eCFR XML for a full part or a single section."""
    headers = {"Accept-Encoding": "gzip"}
    query = f"part={part_num}" + (f"&section={section}" if section else "")
    for d in DATES:
        url = f"https://www.ecfr.gov/api/versioner/v1/full/{d}/title-14.xml?{query}"
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.content
    resp.raise_for_status()
    return b""


def main() -> None:
    """Download all configured eCFR XML targets into ``config.RAW_DIR``."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, part_num, section in TARGETS:
        print(f"Downloading {name} from eCFR...")
        try:
            content = fetch_target(part_num, section)
        except requests.HTTPError as exc:
            print(f"  Skipped {name} ({exc})")
            continue
        out_path = config.RAW_DIR / f"{name}.xml"
        out_path.write_bytes(content)
        print(f"  Saved {out_path.name} ({len(content):,} bytes)")


if __name__ == "__main__":
    main()
