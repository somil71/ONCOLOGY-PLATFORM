"""Recover missing public RNA sources through documented GDC compressed batches.

https://docs.gdc.cancer.gov/API/Users_Guide/Downloading_Files/
Archive member paths are never extracted. Only expected UUIDs and checksummed
payloads are promoted to final local source paths.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
from urllib.request import Request, urlopen

from tools.download_primary_rna import verified_download


def valid(item, path):
    return (
        path.exists()
        and path.stat().st_size == item["file_size"]
        and hashlib.md5(path.read_bytes()).hexdigest() == item["md5sum"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument(
        "--cancer",
        choices=["BRCA", "KIRC"],
        default="BRCA",
        help="Keep cancer cohorts in separate data folders and outputs",
    )
    args = parser.parse_args()
    folder = (
        Path(__file__).resolve().parents[1]
        / f"data/tcga-{args.cancer.lower()}/rna-primary"
    )
    manifest = json.loads((folder / "selected-manifest.json").read_text())
    missing = [
        x for x in manifest["files"] if not valid(x, folder / (x["file_id"] + ".tsv"))
    ]
    print("Missing source files before compressed recovery:", len(missing), flush=True)
    results = []
    for offset in range(0, len(missing), args.batch_size):
        batch = missing[offset : offset + args.batch_size]
        expected = {x["file_id"]: x for x in batch}
        if len(batch) == 1:
            # GDC may return raw TSV instead of tar.gz for a one-file request.
            # Use the independently verified resumable single-file route.
            record = verified_download(batch[0], folder)
            results.append(record)
            print(record, flush=True)
            continue
        request = Request(
            "https://api.gdc.cancer.gov/data",
            data=json.dumps({"ids": list(expected)}).encode(),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "CancerFusion-research",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                with tarfile.open(fileobj=response, mode="r|gz") as archive:
                    for member in archive:
                        if not member.isfile():
                            continue
                        identity = next(
                            (
                                part
                                for part in member.name.split("/")
                                if part in expected
                            ),
                            None,
                        )
                        if identity is None:
                            continue
                        item = expected[identity]
                        if member.size != item["file_size"]:
                            raise ValueError("Archive member size mismatch")
                        stream = archive.extractfile(member)
                        payload = stream.read(item["file_size"] + 1)
                        if (
                            len(payload) != item["file_size"]
                            or hashlib.md5(payload).hexdigest() != item["md5sum"]
                        ):
                            raise ValueError("Archive member checksum mismatch")
                        target = folder / (identity + ".tsv")
                        partial = folder / (identity + ".bulkpart")
                        partial.write_bytes(payload)
                        partial.replace(target)
                        record = {
                            "Patient_ID": item["Patient_ID"],
                            "file_id": identity,
                            "status": "verified_bulk_download",
                        }
                        results.append(record)
                        print(record, flush=True)
        except Exception as exc:
            results.append(
                {
                    "batch_ids": list(expected),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print("Batch failed:", type(exc).__name__, str(exc), flush=True)
    remaining = [
        x["file_id"]
        for x in manifest["files"]
        if not valid(x, folder / (x["file_id"] + ".tsv"))
    ]
    report = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "route": "POST https://api.gdc.cancer.gov/data compressed tar.gz batches; GET for one-file batches",
        "selected_total": len(manifest["files"]),
        "verified_total": len(manifest["files"]) - len(remaining),
        "requested": len(missing),
        "remaining_ids": remaining,
        "results": results,
    }
    (folder / "bulk-recovery-audit.json").write_text(json.dumps(report, indent=2))
    print("Remaining sources:", len(remaining), flush=True)


if __name__ == "__main__":
    main()
