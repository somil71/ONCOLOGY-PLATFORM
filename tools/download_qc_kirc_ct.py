"""Bounded sequential TCIA download and DICOM QC for locked KIRC CT series.

This produces an actual decoded-input availability audit, not an efficacy cohort.
It never chooses a new series during download and never transfers a mask from a
different series. Failed archives/series stay explicit with no partial admission.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import stat
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd

from src.ct_preprocessing import ct_hounsfield_pixels
from src.imaging_manifest import order_ct_series
from tools.extract_ct_features import publish, sha256

ROOT = Path(__file__).resolve().parents[1]
MIN_FREE_BYTES = 75 * 1_000_000_000
RAW_BUDGET_BYTES = 25 * 1_000_000_000


def atomic_json(path, value):
    publish(path, json.dumps(value, indent=2, allow_nan=False).encode())


def safe_extract(archive, destination, expected_images, expected_bytes):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        names, total = set(), 0
        for item in infos:
            rel = Path(item.filename)
            target = (destination / rel).resolve()
            mode = item.external_attr >> 16
            if (
                rel.is_absolute()
                or not target.is_relative_to(destination)
                or stat.S_ISLNK(mode)
            ):
                raise ValueError("Unsafe TCIA ZIP member path or symlink.")
            if item.filename in names:
                raise ValueError("Duplicate ZIP member.")
            names.add(item.filename)
            total += item.file_size
        if total > expected_bytes * 1.10 + 16_000_000:
            raise ValueError("ZIP expands substantially beyond TCIA metadata FileSize.")
        dcm = [i for i in infos if i.filename.casefold().endswith(".dcm")]
        if len(dcm) != expected_images:
            raise ValueError(
                f"Archive DICOM count {len(dcm)} differs from metadata {expected_images}."
            )
        zf.extractall(destination)
    paths = sorted(destination.rglob("*.dcm"))
    if len(paths) != expected_images:
        raise ValueError("Extracted DICOM count differs from selected metadata.")
    return paths, total


def download_series(item, archive, max_bytes):
    uid = item["SeriesInstanceUID"]
    url = (
        "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?"
        + urlencode({"SeriesInstanceUID": uid})
    )
    partial = archive.with_suffix(archive.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    request = Request(
        url,
        headers={
            "User-Agent": "CancerFusion KIRC CT QC research",
            "Range": f"bytes={offset}-",
        }
        if offset
        else {"User-Agent": "CancerFusion KIRC CT QC research"},
    )
    try:
        with urlopen(request, timeout=180) as response:
            status = getattr(response, "status", 200)
            if offset and status != 206:
                # Server ignored range; restart the partial as a fresh full response.
                offset = 0
            mode = "ab" if offset else "wb"
            size = offset
            with partial.open(mode) as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError("TCIA ZIP exceeded locked download budget.")
                    free = shutil.disk_usage(ROOT).free
                    if free - len(chunk) < MIN_FREE_BYTES:
                        raise RuntimeError(
                            "Stopping before free-space reserve is breached."
                        )
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
    except (HTTPError, URLError, TimeoutError, OSError):
        raise
    if not zipfile.is_zipfile(partial):
        raise ValueError("TCIA returned a non-ZIP payload.")
    final = archive
    os.replace(partial, final)
    return final, url


def qc_series(item, dicom_paths, phase_source):
    import pydicom

    ordered = order_ct_series(dicom_paths)
    if len(ordered) != int(item["ImageCount"]):
        raise ValueError(
            "Geometrically valid DICOM count differs from the locked selection."
        )
    datasets = []
    hu_min, hu_max = np.inf, -np.inf
    for path in ordered:
        ds = pydicom.dcmread(path)
        if (
            str(ds.PatientID) != item["PatientID"]
            or str(ds.SeriesInstanceUID) != item["SeriesInstanceUID"]
            or str(ds.StudyInstanceUID) != item["StudyInstanceUID"]
        ):
            raise ValueError(
                "DICOM patient/study/series identity differs from locked TCIA selection."
            )
        hu, valid = ct_hounsfield_pixels(ds)
        if not valid.any() or not np.isfinite(hu[valid]).all():
            raise ValueError("No valid finite HU pixels in a selected DICOM slice.")
        hu_min = min(hu_min, float(hu[valid].min()))
        hu_max = max(hu_max, float(hu[valid].max()))
        datasets.append(ds)
    if not (np.isfinite([hu_min, hu_max]).all() and hu_max > hu_min):
        raise ValueError("Series has degenerate Hounsfield range.")
    first = datasets[0]
    try:
        thickness = float(first.get("SliceThickness"))
        if not np.isfinite(thickness) or thickness <= 0:
            thickness = None
    except (TypeError, ValueError):
        thickness = None
    return {
        "status": "verified_decoded",
        "patient_id": item["PatientID"],
        "series_uid": item["SeriesInstanceUID"],
        "study_uid": item["StudyInstanceUID"],
        "metadata_phase": item["phase_label"],
        "metadata_phase_source": phase_source,
        "dicom_files": len(ordered),
        "identity_verified_all_slices": True,
        "geometry_verified": True,
        "pixel_decode_verified_all_slices": True,
        "hu_conversion_verified_all_slices": True,
        "hu_range": [hu_min, hu_max],
        "rows_columns": [int(first.Rows), int(first.Columns)],
        "pixel_spacing_mm": [float(x) for x in first.PixelSpacing],
        "slice_thickness_mm": thickness,
        "series_description": item.get("SeriesDescription", ""),
        "protocol_name": item.get("ProtocolName", ""),
        "body_part_examined": item.get("BodyPartExamined", ""),
        "acquisition_timing_verified": False,
        "note": "TCIA dates are deidentified; this is decoded CT availability/QC, not radiologist tumor-coverage approval.",
    }


def main():
    source = ROOT / "data/tcia-kirc/ct-cohort/selected-series.json"
    selection_file = json.loads(source.read_text(encoding="utf-8"))
    report = selection_file["report"]
    items = selection_file["files"]
    audit_path = ROOT / "data/tcia-kirc/ct-cohort/selection-audit.json"
    audit = json.loads(audit_path.read_text())
    if audit != report:
        raise ValueError("Selection report differs from the predownload audit.")
    if len(items) != report["selected_unique_patients"]:
        raise ValueError("Locked CT selection count mismatch.")
    out = ROOT / "data/tcia-kirc/ct-cohort"
    archive_dir, image_dir = out / "archives", out / "dicom"
    archive_dir.mkdir(exist_ok=True)
    image_dir.mkdir(exist_ok=True)
    journal_path = out / "download-qc-audit.json"
    prior = json.loads(journal_path.read_text()) if journal_path.exists() else {}
    results = {
        row["series_uid"]: row
        for row in prior.get("results", [])
        if row.get("status") == "verified_decoded"
    }
    failures = {row["series_uid"]: row for row in prior.get("failed_or_skipped", [])}
    start = time.perf_counter()

    def save(active=None, stage=None, status="running"):
        verified = [results[x] for x in sorted(results)]
        cli = pd.read_csv(ROOT / "data/processed/tcga_kirc_clinical.csv")
        included = cli[cli.Patient_ID.isin([x["patient_id"] for x in verified])]
        atomic_json(
            journal_path,
            {
                "started_utc": prior.get(
                    "started_utc", datetime.now(timezone.utc).isoformat()
                ),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "selection_sha256": sha256(source),
                "selection_audit_sha256": sha256(out / "selection-audit.json"),
                "locked_candidate_patients": report["candidate_patients"],
                "selected_ct_patients": report["selected_unique_patients"],
                "verified_decoded_ct_patients": len(verified),
                "verified_decoded_ct_deaths": int(included.event.sum()),
                "active_series_uid": active,
                "active_stage": stage,
                "raw_budget_bytes": RAW_BUDGET_BYTES,
                "minimum_free_bytes": MIN_FREE_BYTES,
                "free_bytes": shutil.disk_usage(ROOT).free,
                "elapsed_seconds_this_invocation": time.perf_counter() - start,
                "results": verified,
                "failed_or_skipped": [failures[x] for x in sorted(failures)],
            },
        )

    for item in items:
        uid = item["SeriesInstanceUID"]
        if uid in results:
            continue
        prior_failure = failures.get(uid, {})
        error = prior_failure.get("error", "")
        deterministic_geometry_failure = any(
            message in error
            for message in [
                "Duplicate or overlapping slice positions",
                "Nonuniform slice spacing",
                "Mixed DICOM orientation, dimensions or pixel spacing",
                "slice origins shift within the image plane",
            ]
        )
        if deterministic_geometry_failure:
            continue
        save(uid, "budget_check")
        used = (
            sum(p.stat().st_size for p in archive_dir.glob("*.zip"))
            + sum(p.stat().st_size for p in archive_dir.glob("*.partial"))
            + sum(p.stat().st_size for p in image_dir.rglob("*") if p.is_file())
        )
        if used + int(item["FileSize"]) > RAW_BUDGET_BYTES:
            failures[uid] = {
                "series_uid": uid,
                "patient_id": item["PatientID"],
                "reason": "locked raw archive budget exhausted",
            }
            save(uid, "budget_exhausted")
            break
        patient_dir = image_dir / item["PatientID"] / uid
        archive = archive_dir / f"{uid}.zip"
        extracted = list(patient_dir.rglob("*.dcm")) if patient_dir.exists() else []
        try:
            if not extracted:
                if not archive.exists():
                    save(uid, "download")
                    archive, url = download_series(
                        item, archive, int(item["FileSize"]) + 16_000_000
                    )
                else:
                    url = (
                        "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?"
                        + urlencode({"SeriesInstanceUID": uid})
                    )
                if not patient_dir.exists():
                    if (
                        shutil.disk_usage(ROOT).free - int(item["FileSize"])
                        < MIN_FREE_BYTES
                    ):
                        raise RuntimeError(
                            "Stopping before extraction would breach the free-space reserve."
                        )
                    patient_dir.mkdir(parents=True, exist_ok=True)
                    save(uid, "extract")
                    extracted, expanded = safe_extract(
                        archive,
                        patient_dir,
                        int(item["ImageCount"]),
                        int(item["FileSize"]),
                    )
                else:
                    expanded = sum(
                        p.stat().st_size for p in patient_dir.rglob("*") if p.is_file()
                    )
            else:
                url = "cached archive"
                expanded = sum(
                    p.stat().st_size for p in patient_dir.rglob("*") if p.is_file()
                )
            save(uid, "dicom_qc")
            qc = qc_series(item, extracted, item["phase_label_source"])
            qc.update(
                {
                    "archive_path": str(archive.relative_to(ROOT)),
                    "archive_bytes": archive.stat().st_size,
                    "archive_sha256": sha256(archive),
                    "expanded_bytes": expanded,
                    "download_url": url,
                    "selected_metadata_file_size": int(item["FileSize"]),
                    "source_tcia_metadata_sha256": report["source_sha256"].get(
                        "data/separate-cancer-feasibility/TCGA-KIRC-ct-metadata.json"
                    ),
                }
            )
            results[uid] = qc
            failures.pop(uid, None)
            # Retain verified DICOM and archive as provenance; budget guards both.
            save(uid, "verified")
        except Exception as exc:
            failures[uid] = {
                "series_uid": uid,
                "patient_id": item["PatientID"],
                "stage": "download_or_decode",
                "error": f"{type(exc).__name__}: {exc}",
                "partial_archive_bytes": (
                    archive.with_suffix(".zip.partial").stat().st_size
                    if archive.with_suffix(".zip.partial").exists()
                    else 0
                ),
                "attempted_utc": datetime.now(timezone.utc).isoformat(),
            }
            save(uid, "failed_series")
    failures = {
        row["series_uid"]: row
        for row in failures.values()
        if row["series_uid"] not in results
    }
    terminal_count = len(results) + len(failures)
    save(status="completed" if terminal_count >= len(items) else "running")
    print(
        json.dumps(
            {
                "status": "completed" if terminal_count >= len(items) else "running",
                "verified": len(results),
                "failed": len(failures),
                "free_gb": round(shutil.disk_usage(ROOT).free / 1e9, 2),
                "output": str(journal_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
