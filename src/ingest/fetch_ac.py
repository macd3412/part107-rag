"""Download the FAA AC 107-2A advisory-circular PDF into the raw data dir."""

import requests

from src import config

OUT_PATH = config.RAW_DIR / "ac_107_2a.pdf"
config.RAW_DIR.mkdir(parents=True, exist_ok=True)

URLS = [
    "https://www.faa.gov/documentLibrary/media/Advisory_Circular/Editorial_Update_AC_107-2A.pdf",
    "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_107-2A.pdf",
]

def main() -> None:
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in URLS:
        print(f"Trying {url}...")
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code == 200 and resp.content[:4] == b"%PDF":
            OUT_PATH.write_bytes(resp.content)
            print(f"Saved {OUT_PATH} ({len(resp.content):,} bytes)")
            return
    raise RuntimeError("Could not download AC 107-2A PDF from FAA.")

if __name__ == "__main__":
    main()
