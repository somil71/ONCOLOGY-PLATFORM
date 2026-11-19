"""Download deterministic primary-tumour GDC RNA files with checksum validation.

One file per case, choosing the lexicographically smallest file UUID among
Primary Tumor-only candidates; no outcome values enter selection. Original
counts and selected source UUIDs remain available for alternate rules.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import shutil
from pathlib import Path
import time
from urllib.request import Request, urlopen


def choose_primary_files(metadata, allowed_ids=None):
    grouped = {}
    for item in metadata:
        cases = item.get("cases", [])
        if len(cases) != 1:
            continue
        case = cases[0]
        kinds = {s.get("sample_type") for s in case.get("samples", [])}
        if kinds != {"Primary Tumor"}:
            continue
        patient = case["submitter_id"]
        if allowed_ids is not None and patient not in allowed_ids:
            continue
        grouped.setdefault(patient, []).append(item)
    return [
        {
            "Patient_ID": patient,
            "candidate_files": len(items),
            **min(items, key=lambda x: x["file_id"]),
        }
        for patient, items in sorted(grouped.items())
    ]


def file_md5(path):
    value = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def verified_download(item, folder, suffix=".tsv", minimum_free_bytes=None):
    target = folder / (item["file_id"] + suffix)

    def valid(path):
        return (
            path.exists()
            and path.stat().st_size == item["file_size"]
            and file_md5(path) == item["md5sum"]
        )

    if valid(target):
        return {
            "Patient_ID": item["Patient_ID"],
            "file_id": item["file_id"],
            "status": "verified_cached",
        }
    partial = target.with_suffix(".partial")
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            offset = offset if 0 < offset < item["file_size"] else 0
            headers = {"User-Agent": "CancerFusion-research"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urlopen(
                Request(
                    "https://api.gdc.cancer.gov/data/" + item["file_id"],
                    headers=headers,
                ),
                timeout=120,
            ) as response:
                status = getattr(response, "status", 200)
                if offset and status == 206:
                    content_range = response.headers.get("Content-Range", "").strip()
                    # GDC has returned both RFC-style and unitless range headers.
                    if content_range.lower().startswith("bytes "):
                        content_range = content_range[6:]
                    if (
                        content_range
                        != f"{offset}-{item['file_size'] - 1}/{item['file_size']}"
                    ):
                        raise ValueError(
                            "Server returned an unexpected resume byte range"
                        )
                    mode = "ab"
                elif status == 200:
                    mode = "wb"  # Server may ignore Range; restart without appending duplicate bytes.
                else:
                    raise ValueError(f"Unexpected GDC download status: {status}")
                with partial.open(mode) as stream:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        if (
                            minimum_free_bytes is not None
                            and shutil.disk_usage(folder).free - len(chunk)
                            < minimum_free_bytes
                        ):
                            raise ValueError(
                                "Download would cross the configured free-disk reserve"
                            )
                        if stream.tell() + len(chunk) > item["file_size"]:
                            raise ValueError(
                                "Server payload exceeds the selected source size"
                            )
                        stream.write(chunk)
            if not valid(partial):
                raise ValueError("Downloaded size or GDC MD5 mismatch")
            partial.replace(target)
            return {
                "Patient_ID": item["Patient_ID"],
                "file_id": item["file_id"],
                "status": "verified_download",
            }
        except Exception as exc:
            if attempt == 2:
                return {
                    "Patient_ID": item["Patient_ID"],
                    "file_id": item["file_id"],
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            time.sleep(2 * (attempt + 1))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=32,
        help="0 downloads all selected primary-tumour files",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--cancer",
        choices=["BRCA", "KIRC"],
        default="BRCA",
        help="Keep cancer cohorts in separate data folders and outputs",
    )
    args = parser.parse_args()
    metadata = json.loads(
        (
            root / f"data/tcga-{args.cancer.lower()}/inventory/rna-metadata.json"
        ).read_text()
    )
    import pandas as pd

    clinical = pd.read_csv(
        root / f"data/processed/tcga_{args.cancer.lower()}_clinical.csv"
    )
    allowed = set(clinical.Patient_ID)
    selected = choose_primary_files(metadata, allowed)
    coverage = clinical.assign(
        has_selected_primary_rna=clinical.Patient_ID.isin(
            {x["Patient_ID"] for x in selected}
        )
    )
    coverage.to_csv(
        root / f"data/processed/tcga_{args.cancer.lower()}_rna_selection_coverage.csv",
        index=False,
    )
    folder = root / f"data/tcga-{args.cancer.lower()}/rna-primary"
    folder.mkdir(exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "Primary Tumor-only sample type; one file per OS-eligible case, lexicographically first UUID; selection independent of outcome values",
        "candidate_count": len(selected),
        "selected_bytes": sum(x["file_size"] for x in selected),
        "files": selected,
    }
    (folder / "selected-manifest.json").write_text(json.dumps(manifest, indent=2))
    requested = selected[: args.limit] if args.limit else selected
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(verified_download, item, folder) for item in requested
        ]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 25 == 0:
                print(f"Completed {len(results)}/{len(requested)} files", flush=True)
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "requested": len(requested),
        "verified": sum(x["status"].startswith("verified") for x in results),
        "failed": sum(x["status"] == "failed" for x in results),
        "results": results,
    }
    (folder / "download-audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()
