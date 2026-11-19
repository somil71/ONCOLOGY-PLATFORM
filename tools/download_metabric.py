"""Retrieve checksum-verified public METABRIC Datahub files (Git LFS aware)."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression", action="store_true")
    parser.add_argument(
        "--revision",
        default="master",
        help="Optional exact Datahub Git commit; saved LFS object hashes pin the default inputs",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Explicitly refresh saved upstream LFS pointers",
    )
    args = parser.parse_args()
    out = Path(__file__).resolve().parents[1] / "data/metabric"
    out.mkdir(exist_ok=True)
    pin = out / "upstream-revision.json"
    revision = args.revision
    files = ["data_clinical_patient.txt", "data_clinical_sample.txt"]
    if args.expression:
        files += ["data_mrna_illumina_microarray.txt"]
    records = []
    for name in files:
        raw = (
            "https://raw.githubusercontent.com/cBioPortal/datahub/"
            + revision
            + "/public/brca_metabric/"
            + name
        )
        saved_pointer = out / (name + ".lfs-pointer")
        pointer = (
            saved_pointer.read_bytes()
            if saved_pointer.exists() and not args.refresh
            else urlopen(
                Request(raw, headers={"User-Agent": "CancerFusion-research"}),
                timeout=60,
            ).read()
        )
        if not pointer.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(
                "Expected a Git LFS pointer; review upstream release before continuing"
            )
        fields = dict(line.split(" ", 1) for line in pointer.decode().splitlines()[1:])
        expected = fields["oid"].split(":", 1)[1]
        size = int(fields["size"])
        target = out / name
        cached = (
            target.exists()
            and target.stat().st_size == size
            and hashlib.sha256(target.read_bytes()).hexdigest() == expected
        )
        url = (
            "https://media.githubusercontent.com/media/cBioPortal/datahub/"
            + revision
            + "/public/brca_metabric/"
            + name
        )
        if not cached:
            partial = target.with_suffix(".partial")
            digest = hashlib.sha256()
            count = 0
            with (
                urlopen(
                    Request(url, headers={"User-Agent": "CancerFusion-research"}),
                    timeout=180,
                ) as response,
                partial.open("wb") as stream,
            ):
                while True:
                    b = response.read(1024 * 1024)
                    if not b:
                        break
                    stream.write(b)
                    digest.update(b)
                    count += len(b)
            if count != size or digest.hexdigest() != expected:
                raise ValueError("METABRIC checksum or size mismatch")
            partial.replace(target)
        (out / (name + ".lfs-pointer")).write_bytes(pointer)
        records.append(
            {
                "file": name,
                "url": url,
                "sha256": expected,
                "bytes": size,
                "verified_cached": cached,
            }
        )
        print(name, size, "verified", flush=True)
    path = out / "download-provenance.json"
    old = json.loads(path.read_text()) if path.exists() else {"files": []}
    merged = {x["file"]: x for x in old["files"]}
    merged.update({x["file"]: x for x in records})
    pin.write_text(
        json.dumps(
            {
                "repository": "https://github.com/cBioPortal/datahub",
                "revision_requested": revision,
                "version_pin": "Saved Git LFS SHA256 object IDs; commit SHA not resolved for master",
                "files": records,
            },
            indent=2,
        )
    )
    path.write_text(
        json.dumps(
            {
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "source": "cBioPortal/Datahub public brca_metabric",
                "upstream_revision_requested": revision,
                "files": list(merged.values()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
