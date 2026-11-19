"""Independently audit KIRC candidate identity/outcomes and evaluation feasibility.

Reads original CDR rows and saved public metadata. No model is fitted and the
illustrative split/fold counts are not a frozen scientific analysis partition.
Run again after image QC before committing an analysis plan.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.survival_metrics import CensoringDistribution


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    raw = path.read_bytes()
    if b"\0" in raw:
        raise ValueError(f"NUL-corrupted source: {path}")
    return json.loads(raw)


def sample_types(item):
    return sorted(
        {
            s.get("sample_type", "MISSING")
            for c in item["cases"]
            for s in c.get("samples", [])
        }
    )


def summary(frame, censor_reference=None):
    report = {"patients": len(frame), "deaths": int(frame.event.sum())}
    report["horizons_days"] = []
    censor = None
    if censor_reference is not None:
        censor = CensoringDistribution(censor_reference.time, censor_reference.event)
    for h in (365.0, 1095.0, 1825.0):
        cases = (frame.event == 1) & (frame.time <= h)
        controls = frame.time > h
        row = {
            "horizon": h,
            "observed_deaths_by_horizon": int(cases.sum()),
            "observed_beyond_horizon": int(controls.sum()),
            "censored_by_horizon": int(((frame.event == 0) & (frame.time <= h)).sum()),
            "followup_range_contains_horizon": bool(
                frame.time.min() <= h < frame.time.max()
            ),
            "sparse_counts_flag": bool(cases.sum() < 10 or controls.sum() < 10),
        }
        if censor is not None:
            try:
                row["reference_censor_survival"] = float(censor.at(h))
            except ValueError as exc:
                row["reference_censor_survival"] = None
                row["support_reason"] = str(exc)
        report["horizons_days"].append(row)
    return report


def illustrative_partitions(frame):
    frame = frame.reset_index(drop=True)
    train, holdout = train_test_split(
        np.arange(len(frame)), test_size=0.4, random_state=42, stratify=frame.event
    )
    val, test = train_test_split(
        holdout, test_size=0.5, random_state=42, stratify=frame.event.iloc[holdout]
    )
    holdout_counts = {
        name: summary(frame.iloc[ix], frame.iloc[train])
        for name, ix in {"train": train, "validation": val, "test": test}.items()
    }
    folds = []
    for fold, (development, heldout) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=42).split(frame, frame.event), 1
    ):
        folds.append(
            {
                "fold": fold,
                "development": summary(frame.iloc[development]),
                "heldout": summary(frame.iloc[heldout], frame.iloc[development]),
            }
        )
    return {
        "purpose": "Count diagnostics only, not final/frozen partitions; rerun after modality QC",
        "candidate_60_20_20_seed42": holdout_counts,
        "candidate_stratified_outer_five_fold_seed42": folds,
    }


def audit_timing(path, selections, clinical):
    """Receipt and patient procurement are distinct fields in the current GDC dictionary."""
    if not path.exists():
        return {
            "status": "not_available",
            "prediction_time_eligibility_established": False,
        }
    hits = read_json(path)["data"]["hits"]
    lookup = {
        item["file_id"]: (kind, item["Patient_ID"])
        for kind, files in selections.items()
        for item in files
    }
    if len(hits) != len(lookup) or {x["file_id"] for x in hits} != set(lookup):
        raise ValueError("Timing metadata does not cover exactly the selected files")
    endpoint = clinical.set_index("Patient_ID", verify_integrity=True)
    rows = []
    for item in hits:
        kind, patient = lookup[item["file_id"]]
        if len(item["cases"]) != 1 or item["cases"][0]["submitter_id"] != patient:
            raise ValueError("Timing/file case identity mismatch")
        case = item["cases"][0]
        for sample in case.get("samples", []):
            receipt = sample.get("days_to_collection")
            procurement = sample.get("days_to_sample_procurement")
            rows.append(
                {
                    "modality": kind,
                    "patient_id": patient,
                    "sample_id": sample.get("sample_id"),
                    "index_date": case.get("index_date"),
                    "days_to_bcr_receipt": receipt,
                    "days_to_patient_procurement": procurement,
                    "bcr_receipt_after_recorded_os_observation": receipt is not None
                    and float(receipt) > endpoint.loc[patient, "time"],
                }
            )
    groups = {}
    for kind in selections:
        records = [x for x in rows if x["modality"] == kind]
        groups[kind] = {
            "selected_files": len(selections[kind]),
            "sample_rows": len(records),
            "index_dates": dict(Counter(x["index_date"] for x in records)),
            "known_bcr_receipt_offsets": sum(
                x["days_to_bcr_receipt"] is not None for x in records
            ),
            "known_patient_procurement_offsets": sum(
                x["days_to_patient_procurement"] is not None for x in records
            ),
            "bcr_receipt_after_os_observation": int(
                sum(x["bcr_receipt_after_recorded_os_observation"] for x in records)
            ),
        }
    return {
        "status": "audited",
        "groups": groups,
        "dictionary_source": "https://raw.githubusercontent.com/NCI-GDC/gdcdictionary/develop/src/gdcdictionary/schemas/_terms.yaml",
        "definition_checked_date": "2026-11-18",
        "interpretation": "Current GDC days_to_collection is BCR/processing-center receipt, whereas days_to_sample_procurement is patient procedure/removal. Positive receipt or receipt after recorded death/followup does not prove future patient sampling. Unknown procurement cannot establish diagnosis-time availability. No automatic exclusion or landmark offset is justified by receipt alone.",
        "prediction_time_eligibility_established": False,
    }


def main():
    cdr_path = ROOT / "data/tcga-brca/TCGA-CDR-SupplementalTableS1.xlsx"
    clinical_path = ROOT / "data/processed/tcga_kirc_clinical.csv"
    rna_path = ROOT / "data/tcga-kirc/rna-primary/selected-manifest.json"
    wsi_path = ROOT / "data/tcga-kirc/wsi-diagnostic/selected-manifest.json"
    ct_path = ROOT / "data/separate-cancer-feasibility/TCGA-KIRC-ct-metadata.json"
    sources = [cdr_path, clinical_path, rna_path, wsi_path, ct_path]
    timing_path = (
        ROOT / "data/tcga-kirc/inventory/selected-file-sample-timing-metadata.json"
    )
    if timing_path.exists():
        sources.append(timing_path)
    # Parse directly rather than calling the cohort-preparation helper under audit.
    cdr = pd.read_excel(cdr_path, sheet_name="TCGA-CDR")
    cdr = cdr.loc[cdr.type.eq("KIRC")].set_index(
        "bcr_patient_barcode", verify_integrity=True
    )
    clinical = (
        pd.read_csv(clinical_path).sort_values("Patient_ID").reset_index(drop=True)
    )
    if clinical.Patient_ID.duplicated().any():
        raise ValueError("Duplicate patient in prepared KIRC clinical cohort")
    original = cdr.loc[clinical.Patient_ID]
    np.testing.assert_array_equal(original["OS"], clinical.event)
    np.testing.assert_array_equal(original["OS.time"], clinical.time)
    np.testing.assert_array_equal(
        original.age_at_initial_pathologic_diagnosis, clinical.age_years
    )
    if not (np.isfinite(clinical.time).all() and (clinical.time > 0).all()):
        raise ValueError("Invalid recorded OS duration")
    if not clinical.event.isin([0, 1]).all():
        raise ValueError("Invalid recorded OS event")
    rna, wsi, ct = (
        read_json(rna_path)["files"],
        read_json(wsi_path)["files"],
        read_json(ct_path),
    )
    by_kind = {"rna": rna, "wsi": wsi}
    mismatches = []
    for kind, files in by_kind.items():
        ids = [f["Patient_ID"] for f in files]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate selected {kind} patient")
        for item in files:
            if (
                len(item["cases"]) != 1
                or item["cases"][0]["submitter_id"] != item["Patient_ID"]
            ):
                raise ValueError(f"Selected {kind} case identity mismatch")
            if sample_types(item) != ["Primary Tumor"]:
                mismatches.append(
                    {
                        "kind": kind,
                        "patient_id": item["Patient_ID"],
                        "file_id": item["file_id"],
                        "file_name": item["file_name"],
                        "sample_types": sample_types(item),
                    }
                )
    primary_rna = {x["Patient_ID"] for x in rna if sample_types(x) == ["Primary Tumor"]}
    primary_wsi = {x["Patient_ID"] for x in wsi if sample_types(x) == ["Primary Tumor"]}
    ct_patients = {x["PatientID"] for x in ct if x.get("Modality") == "CT"}
    strict = clinical.loc[
        clinical.Patient_ID.isin(primary_rna & primary_wsi & ct_patients)
    ].copy()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent candidate feasibility and protocol audit; no model performance result",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "recorded_os_and_age_equal_original_cdr": True,
        "candidate_clinical": summary(clinical),
        "selected_sample_type_counts": {
            kind: dict(Counter(";".join(sample_types(x)) for x in files))
            for kind, files in by_kind.items()
        },
        "non_primary_selected_files": mismatches,
        "strict_primary_rna_wsi_ct_metadata_overlap": summary(strict),
        "removed_for_nonprimary_or_missing_metadata": clinical.loc[
            ~clinical.Patient_ID.isin(strict.Patient_ID), ["Patient_ID", "event"]
        ].to_dict("records"),
        "candidate_partition_diagnostics": illustrative_partitions(strict),
        "sample_timing_audit": audit_timing(timing_path, by_kind, clinical),
        "scientific_admission_gates": {
            "raw_all_modality_download_and_checksum_complete": False,
            "all_ct_series_protocol_tumor_visibility_and_geometry_qc_complete": False,
            "all_wsi_physical_scale_tissue_qc_complete": False,
            "image_sample_prediction_time_eligibility_established": False,
            "frozen_post_qc_patient_partitions": False,
            "four_input_survival_efficacy_established": False,
        },
        "sparse_flag_policy": "Fewer than10 observed deaths or10 observed beyond-horizon patients flags descriptive instability; this is an audit convention, not a universal statistical sufficiency rule.",
        "interpretation": "Metadata and a decoded pilot establish feasibility only. Additional/new primary tissue cannot automatically represent the primary diagnostic tumor. Deidentified CT dates do not establish timing relative to diagnosis.",
    }
    out = ROOT / "RUN REPORT/kirc-protocol-audit-20261118.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out),
                "candidate": report["candidate_clinical"],
                "strict_primary_overlap": report[
                    "strict_primary_rna_wsi_ct_metadata_overlap"
                ],
                "nonprimary_files": mismatches,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
