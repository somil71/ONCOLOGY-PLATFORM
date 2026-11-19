"""Acquire outcome-blind KIRC CT candidates and extract frozen slice features.

Volumes pass technical QC. Baseline timing and tumour visibility remain
unreviewed; output is explicitly retrospective engineering research data.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import stat
import time
from types import SimpleNamespace
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zipfile

import pandas as pd

from src.ct_cohort import POLICY_VERSION, ranked_candidates, verify_volume
from tools.extract_ct_features import extract, publish, sha256


def write_json(path, value):
    publish(path, json.dumps(value, indent=2, allow_nan=False).encode())


def cache_size(folder):
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def safe_extract(archive, destination, byte_limit):
    """Check names, total sizes and CRCs before extracting DICOM inputs."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) > 10000 or sum(i.file_size for i in entries) > byte_limit:
            raise ValueError("CT archive exceeds the extraction budget.")
        seen = set()
        for entry in entries:
            target = (destination / entry.filename).resolve()
            if (Path(entry.filename).is_absolute() or not target.is_relative_to(destination)
                    or stat.S_ISLNK(entry.external_attr >> 16) or target in seen):
                raise ValueError("Unsafe or duplicate CT ZIP entry.")
            seen.add(target)
        bad = bundle.testzip()
        if bad:
            raise ValueError(f"CT ZIP CRC failed: {bad}")
        bundle.extractall(destination)


def download_series(entry, output, limit_bytes, reserve_bytes, total_budget):
    """Only resume acknowledged ranges; verify complete archive integrity."""
    uid = entry["SeriesInstanceUID"]
    archive = output / (uid + ".zip")
    audit = archive.with_suffix(".json")
    if archive.exists() and audit.exists():
        original = json.loads(audit.read_text())
        if original.get("series_uid") != uid or original.get("sha256") != sha256(archive):
            raise ValueError("Cached CT archive identity/hash mismatch.")
        if not zipfile.is_zipfile(archive):
            raise ValueError("Cached CT download is not a ZIP.")
        return archive, original
    url = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?" + urlencode({"SeriesInstanceUID": uid})
    partial = archive.with_suffix(".zip.partial")
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= limit_bytes:
                raise ValueError("Partial CT archive exceeds its byte limit.")
            headers = {"User-Agent": "CancerFusion retrospective imaging research"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urlopen(Request(url, headers=headers), timeout=120) as response:
                status = getattr(response, "status", 200)
                if offset and status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("CT resume byte range differs from cached offset.")
                    mode = "ab"
                elif status == 200:
                    offset, mode = 0, "wb"
                else:
                    raise ValueError(f"Unexpected CT HTTP status {status}")
                with partial.open(mode) as stream:
                    while block := response.read(1024 * 1024):
                        offset += len(block)
                        if offset > limit_bytes or cache_size(output) + len(block) > total_budget:
                            raise ValueError("CT download exceeds series or archive budget.")
                        if shutil.disk_usage(output).free - len(block) < reserve_bytes:
                            raise ValueError("CT download violates the free-disk reserve.")
                        stream.write(block)
            if not zipfile.is_zipfile(partial):
                raise ValueError("CT endpoint did not deliver a complete ZIP.")
            with zipfile.ZipFile(partial) as bundle:
                if bundle.testzip():
                    raise ValueError("CT archive CRC check failed.")
            partial.replace(archive)
            record = {"series_uid": uid, "patient_id": entry["PatientID"], "url": url,
                      "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                      "bytes": archive.stat().st_size, "sha256": sha256(archive),
                      "validation": "ZIP CRC and pinned local SHA256; endpoint supplies no independent source checksum."}
            write_json(audit, record)
            return archive, record
        except (ValueError, OSError) as exc:
            if isinstance(exc, ValueError) or attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--max-series-per-case", type=int, default=3)
    parser.add_argument("--items", type=int, default=32)
    parser.add_argument("--archive-budget-gb", type=float, default=15)
    parser.add_argument("--raw-budget-gb", type=float, default=25)
    parser.add_argument("--max-series-gb", type=float, default=1.5)
    parser.add_argument("--disk-reserve-gb", type=float, default=75)
    parser.add_argument("--output", type=Path, default=Path("data/processed/kirc_ct_features"))
    parser.add_argument("--dependency-dir", action="append", default=[])
    args = parser.parse_args()
    if (args.max_cases < 1 or args.max_series_per_case < 1 or not 1 <= args.items <= 256
            or min(args.archive_budget_gb, args.raw_budget_gb, args.max_series_gb) <= 0
            or args.disk_reserve_gb < 0):
        raise ValueError("Invalid CT extraction limits.")
    import sys
    for directory in args.dependency_dir:
        sys.path.insert(0, str(Path(directory).resolve()))
    metadata_path = Path("data/separate-cancer-feasibility/TCGA-KIRC-ct-metadata.json")
    patient_path = Path("data/tcga-kirc/rna-primary/selected-manifest.json")
    allowed = [entry["Patient_ID"] for entry in json.loads(patient_path.read_text())["files"]]
    if len(allowed) != len(set(allowed)):
        raise ValueError("Duplicated patients in pinned RNA manifest.")
    phase_path = Path("data/rcc-aid/v2/metadata.csv")
    phases = {}
    if phase_path.exists():
        annotations = pd.read_csv(phase_path)
        if annotations.SeriesInstanceUID.duplicated().any():
            raise ValueError("Duplicate RCC-AID phase series.")
        phases = dict(zip(annotations.SeriesInstanceUID, annotations.phase_name))
    groups, rejected = ranked_candidates(json.loads(metadata_path.read_text()), allowed, phases)
    source_contract = {
        "metadata_sha256": sha256(metadata_path), "patient_manifest_sha256": sha256(patient_path),
        "phase_annotation_sha256": sha256(phase_path) if phase_path.exists() else None,
        "policy_version": POLICY_VERSION, "policy_sha256": sha256(Path("src/ct_cohort.py")),
        "extractor_sha256": sha256(Path("tools/extract_ct_features.py")), "items": args.items,
    }
    folder = Path("data/tcia-kirc/cohort")
    archives, raw = folder / "archives", folder / "dicom"
    for path in [folder, archives, raw, args.output]:
        path.mkdir(parents=True, exist_ok=True)
    plan = {
        "contract": source_contract, "scope": "Outcome-blind technical selection; clinical anatomy/timing unreviewed",
        "outcomes_used_in_selection": False, "candidate_patients": len(allowed),
        "patients_with_candidates": sum(bool(v) for v in groups.values()),
        "rejected_metadata_series": rejected, "candidates": groups,
    }
    write_json(folder / "selection-plan.json", plan)
    if not args.download:
        print(json.dumps({"status": "plan_only", "patients": len(groups), "with_candidates": plan["patients_with_candidates"]}))
        return
    records = []
    status = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "contract": source_contract,
        "status": "running", "clinical_review_complete": False, "prediction_time_eligible": False,
        "retrospective_exploration_only": True,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "patients": records,
    }
    status_path = folder / "acquisition-audit.json"
    write_json(status_path, status)
    for patient, candidates in list(groups.items())[:args.max_cases]:
        result = {"patient_id": patient, "status": "no_technical_volume", "attempts": []}
        records.append(result)
        target = args.output / patient
        audit_path = target / "ct_feature_audit.json"
        feature_path = target / "rad_features.npz"
        if feature_path.exists() and audit_path.exists():
            cached = json.loads(audit_path.read_text())
            if cached.get("cohort_contract") == source_contract and cached.get("features_sha256") == sha256(feature_path):
                result.update(status="verified_cached", series_uid=cached["series_uid"])
                write_json(status_path, status)
                continue
        for entry in candidates[:args.max_series_per_case]:
            attempt = {"series_uid": entry["SeriesInstanceUID"], "phase_label": entry["phase_label"]}
            result["attempts"].append(attempt)
            stage = "resource_preflight"
            try:
                expected = int(entry.get("FileSize", 0))
                if expected <= 0 or expected > args.max_series_gb * 1e9:
                    raise ValueError("Series metadata size is invalid or exceeds resource limit.")
                if (cache_size(raw) + expected > args.raw_budget_gb * 1e9
                        or shutil.disk_usage(folder).free - expected * 2 < args.disk_reserve_gb * 1e9):
                    raise ValueError("Insufficient raw-cache budget or disk reserve.")
                stage = "download"
                archive, download = download_series(entry, archives, int(args.max_series_gb * 1e9),
                                                    int(args.disk_reserve_gb * 1e9), int(args.archive_budget_gb * 1e9))
                destination = raw / patient / entry["SeriesInstanceUID"]
                stage = "archive_extraction"
                remaining = int(args.raw_budget_gb * 1e9 - cache_size(raw))
                safe_extract(archive, destination, min(int(args.max_series_gb * 1e9), remaining))
                stage = "volume_qc"
                ordered, qc = verify_volume(list(destination.rglob("*.dcm")), patient, entry["SeriesInstanceUID"])
                if len(ordered) != int(entry["ImageCount"]):
                    raise ValueError("Decoded CT count differs from pinned metadata.")
                target.mkdir(parents=True, exist_ok=True)
                selection = target / "selected-series.json"
                write_json(selection, entry)
                stage = "feature_encoding"
                extract(SimpleNamespace(items=args.items, batch_size=8, selection=selection,
                                        dicom_root=destination, output=target, threads=2,
                                        weights_root=Path("runtime/model-cache/checkpoints")))
                audit = json.loads(audit_path.read_text())
                audit.update(cohort_contract=source_contract, technical_volume_qc=qc,
                             phase_label=entry["phase_label"], phase_source=entry["phase_source"],
                             source_archive=download, clinical_review_complete=False,
                             retrospective_exploration_only=True)
                write_json(audit_path, audit)
                attempt.update(status="technical_qc_passed", qc=qc)
                result.update(status="verified_extracted", series_uid=entry["SeriesInstanceUID"],
                              phase_label=entry["phase_label"], feature_sha256=sha256(feature_path))
                break
            except Exception as exc:
                attempt.update(status="failed", failure_stage=stage, error=f"{type(exc).__name__}: {exc}")
            write_json(status_path, status)
        write_json(status_path, status)
        print(json.dumps({"patient": patient, "status": result["status"], "attempted": len(result["attempts"])}), flush=True)
    status.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                  verified_patients=sum(row["status"].startswith("verified") for row in records),
                  processed_patients=len(records), requested_patients=min(args.max_cases, len(groups)))
    write_json(status_path, status)
    print(json.dumps({key: status[key] for key in ["status", "verified_patients", "processed_patients"]}))


if __name__ == "__main__":
    main()
