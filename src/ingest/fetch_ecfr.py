from datetime import date
from pathlib import Path
import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATES = [date.today().isoformat(), "2026-11-01"]

def fetch_part(part_num: str) -> bytes:
    headers = {"Accept-Encoding": "gzip"}
    for d in DATES:
        url = f"https://www.ecfr.gov/api/versioner/v1/full/{d}/title-14.xml?part={part_num}"
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.content
    resp.raise_for_status()
    return b""

def main() -> None:
    for name, part_num in [("part107", "107"), ("part89", "89")]:
        print(f"Downloading {name} from eCFR...")
        content = fetch_part(part_num)
        out_path = RAW_DIR / f"{name}.xml"
        out_path.write_bytes(content)
        print(f"  Saved {out_path} ({len(content):,} bytes)")

if __name__ == "__main__":
    main()
