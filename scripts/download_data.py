"""Download the exact source snapshot without republishing raw observations."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
source = json.loads((ROOT / "docs/upstream.json").read_text())
target = ROOT / "data/raw/weatherAUS.csv"
with urlopen(source["data_url"], timeout=60) as response:
    data = response.read()
expected = source.get("data_sha256")
actual = hashlib.sha256(data).hexdigest()
if expected and expected != actual:
    raise ValueError("Downloaded dataset does not match the recorded SHA-256")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(data)
print(f"Saved {target.name}: {len(data):,} bytes; SHA-256 {actual}")
