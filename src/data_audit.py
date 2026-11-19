"""Audit the Phase 2 source tables and the image loader's actual file selection."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit_outcomes(demographics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = {
        "status": "demographic_vital_status",
        "death": "demographic_days_to_death",
        "followup": "diagnoses_days_to_last_follow_up",
    }
    missing = [column for column in columns.values() if column not in demographics]
    summary = {"source_rows": len(demographics), "required_columns_missing": missing}
    status_column = columns["status"]
    if status_column in demographics:
        summary["raw_status_counts"] = {
            str(key): int(value)
            for key, value in demographics[status_column].value_counts(dropna=False).items()
        }
    if not missing:
        rows = demographics[["Patient_ID", *columns.values()]].copy()
        status = rows[status_column].astype("string").str.strip().str.casefold()
        rows["event"] = status.map({"alive": 0, "living": 0, "dead": 1, "deceased": 1})
        death = pd.to_numeric(rows[columns["death"]], errors="coerce")
        followup = pd.to_numeric(rows[columns["followup"]], errors="coerce")
        rows["time"] = death.where(rows["event"].eq(1), followup)
        rows["usable"] = rows["event"].notna() & np.isfinite(rows["time"]) & rows["time"].gt(0)
        rows["unmapped_status"] = rows["event"].isna()
        rows["invalid_duration"] = ~np.isfinite(rows["time"]) | rows["time"].le(0)
        rows.to_csv(output_dir / "outcome_rows.csv", index=False)
        summary.update({
            "usable_rows": int(rows["usable"].sum()),
            "usable_event_rows": int((rows["usable"] & rows["event"].eq(1)).sum()),
            "unmapped_status_rows": int(rows["unmapped_status"].sum()),
            "invalid_duration_rows": int(rows["invalid_duration"].sum()),
            "duplicate_patient_rows": int(rows["Patient_ID"].duplicated().sum()),
        })
    (output_dir / "outcome_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def audit_modality_files(cohort, image_root, rna_ids, clinical_ids, max_items, output_dir, image_manifest=None):
    """Replay the current extension-based routing and decoding without neural inference.

    The manifest describes loader behavior. It does not validate imaging modality identity.
    """
    from PIL import Image
    import pydicom

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rna_ids, clinical_ids = set(rna_ids), set(clinical_ids)
    cases, files = [], []
    for record in cohort.to_dict("records"):
        patient = str(record["Patient_ID"])
        # Patient identifiers are directory names, never paths supplied to the file system.
        if Path(patient).name != patient or patient in (".", "..") or "/" in patient or "\\" in patient:
            raise ValueError(f"Invalid patient directory identifier: {patient!r}")
        candidates = sorted(path for path in (Path(image_root) / patient).rglob("*") if path.is_file())
        if image_manifest is None:
            # Historical loader audit retained to explain archived runs only.
            wsi = [path for path in candidates if path.suffix in (".jpg", ".png")]
            rad = [path for path in candidates if path.suffix not in (".jpg", ".png")]
        else:
            from src.imaging_manifest import order_mr_series, select_evenly_spaced
            wsi = sorted(image_manifest.get((patient, 'wsi'), []))
            rad = order_mr_series(image_manifest.get((patient, 'rad'), []))
            wsi, rad = select_evenly_spaced(wsi, max_items), select_evenly_spaced(rad, max_items)
        counts = Counter()
        for modality, paths in (("wsi", wsi), ("rad", rad)):
            for path in paths[:max_items]:
                error = ""
                try:
                    if modality == "wsi":
                        with Image.open(path) as im:
                            im.convert("RGB").resize((224, 224)).load()
                    else:
                        array = pydicom.dcmread(str(path), force=True).pixel_array.astype(float)
                        array = (array - np.min(array)) / (np.max(array) - np.min(array) + 1e-8)
                        array = (array * 255).astype(np.uint8)
                        Image.fromarray(array if array.ndim == 2 else array[0]).convert("RGB").resize((224, 224)).load()
                    counts[modality] += 1
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                files.append({
                    "Patient_ID": patient, "loader_branch": modality,
                    "relative_path": path.relative_to(image_root).as_posix(),
                    "decoded": not error, "error": error,
                })
        cases.append({
            **record, "wsi_candidates": len(wsi), "rad_candidates": len(rad),
            "wsi_decoded": counts["wsi"], "rad_decoded": counts["rad"],
            "rna_available": patient in rna_ids, "cli_available": patient in clinical_ids,
            "raster_in_radiology_path": sum(
                any(word in path.as_posix().lower() for word in ("mri", "radiology", "dicom")) for path in wsi
            ),
        })
    frame = pd.DataFrame(cases)
    frame.to_csv(output_dir / "modality_manifest.csv", index=False)
    pd.DataFrame(files, columns=["Patient_ID", "loader_branch", "relative_path", "decoded", "error"]).to_csv(
        output_dir / "image_decode_audit.csv", index=False
    )
    partitions = {}
    for split, rows in frame.groupby("split"):
        partitions[split] = {
            "cases": len(rows), "events": int(rows["event"].sum()),
            "wsi_usable_cases": int(rows["wsi_decoded"].gt(0).sum()),
            "rad_usable_cases": int(rows["rad_decoded"].gt(0).sum()),
            "rna_available_cases": int(rows["rna_available"].sum()),
            "cli_available_cases": int(rows["cli_available"].sum()),
            "all_four_available_cases": int((rows["wsi_decoded"].gt(0) & rows["rad_decoded"].gt(0)
                                              & rows["rna_available"] & rows["cli_available"]).sum()),
        }
    summary = {
        "image_root": str(image_root), "max_items_tested_per_branch": max_items,
        "routing": "Reviewed modality manifest; geometric MR series ordering." if image_manifest is not None else "Historical loader assigns lowercase jpg/png to WSI and other files to RAD.",
        "interpretation": "Decode success does not establish true modality identity or feature quality.",
        "partitions": partitions,
    }
    (output_dir / "modality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
