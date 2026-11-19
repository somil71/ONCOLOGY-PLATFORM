"""Verify pinned RCC-AID v2 metadata before considering annotated renal CT data.

Downloads only the small metadata CSV and TCIA manifest, not images. The v2
authors provide contrast-phase labels and report corrected failed conversions.
Overlap with our CDR/RNA/WSI cases must still be measured locally.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from tools.extract_ct_features import publish


FILES = {
    "metadata.csv": "7297e5ea8201eddb0c3aae55c1068952",
    "manifest.tcia": "adfa32c16ef9287711cab0702083f3e7",
}
RECORD = "https://zenodo.org/records/20719257"


def main():
    output = Path("data/rcc-aid/v2")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, expected in FILES.items():
        target = output / name
        payload = target.read_bytes() if target.is_file() else None
        if payload is None or hashlib.md5(payload).hexdigest() != expected:
            request = Request(
                f"{RECORD}/files/{name}?download=1",
                headers={
                    "User-Agent": "CancerFusion research metadata access",
                },
            )
            with urlopen(request, timeout=120) as response:
                payload = response.read(2 * 1024 * 1024 + 1)
        if len(payload) > 2 * 1024 * 1024:
            raise ValueError("Metadata download exceeds the explicit 2 MiB limit.")
        if hashlib.md5(payload).hexdigest() != expected:
            raise ValueError(f"Published RCC-AID checksum mismatch for {name}.")
        publish(target, payload)
        records.append(
            {
                "name": name,
                "bytes": len(payload),
                "md5": expected,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    report = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "record": RECORD,
        "version": "2.0.0",
        "files": records,
        "metadata_license": "CC BY 3.0; source TCIA attribution required",
        "annotations_license": "CC BY 4.0; cite de Boer, Häntze et al. and original TCIA collections",
        "scope": "Metadata verification only; no RCC-AID image/label archive downloaded or cohort admitted.",
    }
    publish(output / "download-provenance.json", json.dumps(report, indent=2).encode())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
