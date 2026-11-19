"""Outcome-blind TCIA series selection for the separate TCGA-KIRC study.

This transparent metadata selector is not radiologist approval. It selects at
most one renal/abdominal candidate per patient, rejects scout/chest/reformatted
series and pins exact SeriesInstanceUIDs before CT downloads. RCC-AID phase labels
are used only for their exact listed SeriesInstanceUID.
"""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from tools.extract_ct_features import publish, sha256

ROOT = Path(__file__).resolve().parents[1]


def phase_for(series, annotated):
    uid = series["SeriesInstanceUID"]
    if uid in annotated:
        return annotated[uid], "RCC-AID v2 exact-UID phase annotation"
    description = " ".join(
        str(series.get(key, ""))
        for key in ("SeriesDescription", "ProtocolName", "StudyDesc")
    ).casefold()
    if any(
        term in description
        for term in ("nephrographic", "portal venous", "venous phase", "venous_phase")
    ):
        return "venous", "TCIA description/protocol keyword"
    if any(term in description for term in ("delay", "delayed", "excretory")):
        return "delayed", "TCIA description/protocol keyword"
    if any(
        term in description
        for term in ("arterial", "corticomedullary", "cortical phase")
    ):
        return "arterial", "TCIA description/protocol keyword"
    if any(
        term in description
        for term in (
            "pre-contrast",
            "precontrast",
            "unenhanced",
            "noncontrast",
            "non-contrast",
        )
    ):
        return "unenhanced", "TCIA description/protocol keyword"
    return "unknown", "No phase label verified"


def candidate_reason(series):
    description = " ".join(
        str(series.get(key, ""))
        for key in ("SeriesDescription", "ProtocolName", "StudyDesc")
    ).casefold()
    if any(
        term in description
        for term in (
            "scout",
            "topogram",
            "localizer",
            "smart prep",
            "reformatted",
            "mpr",
            "mip",
            "minip",
        )
    ):
        return "scout/localizer/reformatted/derived series keyword"
    if any(term in description for term in ("lung", "thorax", "chest")):
        return "chest/thorax keyword; mixed chest-abdomen excluded for renal-image protocol"
    if not any(
        term in description
        for term in ("kidney", "renal", "abdomen", "abd/pel", "ab/pel", "a/p (renal)")
    ):
        return "no renal/abdominal protocol keyword"
    try:
        count = int(series.get("ImageCount", 0))
        size = int(series.get("FileSize", 0))
    except (ValueError, TypeError):
        return "invalid image count or size metadata"
    if count < 20:
        return "fewer than 20 images"
    if count > 250:
        return "over 250 images; possible mixed/derived volume requires manual review"
    if size <= 0:
        return "missing/nonpositive archive size"
    return None


def main():
    clinical_path = ROOT / "data/processed/tcga_kirc_clinical.csv"
    tcia_path = ROOT / "data/separate-cancer-feasibility/TCGA-KIRC-ct-metadata.json"
    rcc_path = ROOT / "data/rcc-aid/v2/metadata.csv"
    wsi_path = ROOT / "data/tcga-kirc/wsi-diagnostic/selected-manifest.json"
    rna_path = ROOT / "data/tcga-kirc/rna-primary/selected-manifest.json"
    clinical = pd.read_csv(clinical_path)
    if clinical.Patient_ID.duplicated().any():
        raise ValueError("Clinical IDs must be unique.")
    candidates = set(clinical.Patient_ID.astype(str))
    wsi_ids = {x["Patient_ID"] for x in json.loads(wsi_path.read_text())["files"]}
    rna_ids = {x["Patient_ID"] for x in json.loads(rna_path.read_text())["files"]}
    if not candidates <= wsi_ids or not candidates <= rna_ids:
        raise ValueError(
            "CT candidate universe must match the locked primary WSI/RNA cohort."
        )
    tcia = json.loads(tcia_path.read_text())
    if len({x["SeriesInstanceUID"] for x in tcia}) != len(tcia):
        raise ValueError("TCIA metadata contains duplicate series UIDs.")
    rcc = pd.read_csv(rcc_path)
    if rcc.SeriesInstanceUID.duplicated().any():
        raise ValueError("RCC-AID metadata contains duplicate exact series UIDs.")
    annotated = dict(zip(rcc.SeriesInstanceUID.astype(str), rcc.phase_name.astype(str)))
    tcia_by_uid = {x["SeriesInstanceUID"]: x for x in tcia}
    for row in rcc.itertuples():
        if (
            row.SeriesInstanceUID in tcia_by_uid
            and tcia_by_uid[row.SeriesInstanceUID]["PatientID"] != row.PatientID
        ):
            raise ValueError(
                "RCC-AID exact series UID maps to a different TCIA patient."
            )

    eligible, excluded = [], []
    for series in tcia:
        patient = str(series.get("PatientID", ""))
        if patient not in candidates:
            continue
        reason = candidate_reason(series)
        if reason:
            excluded.append(
                {
                    "patient_id": patient,
                    "series_uid": series.get("SeriesInstanceUID"),
                    "reason": reason,
                }
            )
            continue
        phase, phase_source = phase_for(series, annotated)
        if phase not in {"venous", "delayed", "arterial", "unenhanced", "unknown"}:
            raise ValueError(f"Unexpected phase value: {phase}")
        eligible.append(
            {**series, "phase_label": phase, "phase_label_source": phase_source}
        )

    phase_rank = {
        "venous": 0,
        "delayed": 1,
        "arterial": 2,
        "unenhanced": 3,
        "unknown": 4,
    }
    selected = {}
    for series in eligible:
        patient = series["PatientID"]
        key = (
            phase_rank[series["phase_label"]],
            -int(series["ImageCount"]),
            int(series["FileSize"]),
            series["SeriesInstanceUID"],
        )
        if patient not in selected or key < selected[patient][0]:
            selected[patient] = (key, series)
    records = [selected[x][1] for x in sorted(selected)]
    if len({x["PatientID"] for x in records}) != len(records):
        raise ValueError("Selection must contain at most one series per patient.")
    selected_frame = pd.DataFrame(
        [
            {
                "Patient_ID": x["PatientID"],
                "SeriesInstanceUID": x["SeriesInstanceUID"],
                "phase_label": x["phase_label"],
            }
            for x in records
        ]
    )
    joined = clinical.merge(
        selected_frame, on="Patient_ID", how="inner", validate="one_to_one"
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Outcome-blind TCIA KIRC CT metadata selection; no downloads or model results",
        "candidate_universe": "Locked primary-tumour RNA/WSI cohort; OS outcomes were not used for selection",
        "selection_algorithm": {
            "one_series_per_patient": True,
            "image_count_range": [20, 250],
            "required_keywords": ["kidney", "renal", "abdomen", "abd/pel", "ab/pel"],
            "excluded_keywords": [
                "scout",
                "topogram",
                "localizer",
                "smart prep",
                "reformatted",
                "MPR",
                "MIP",
                "MinIP",
                "lung",
                "thorax",
                "chest",
            ],
            "phase_preference": [
                "venous",
                "delayed",
                "arterial",
                "unenhanced",
                "unknown",
            ],
            "phase_semantics": "RCC-AID phase only by exact SeriesInstanceUID; otherwise keyword heuristic. Unknown retained and flagged.",
            "ties": "More images, then smaller TCIA archive size, then SeriesInstanceUID",
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [clinical_path, tcia_path, rcc_path, wsi_path, rna_path]
        },
        "candidate_patients": len(candidates),
        "eligible_metadata_series": len(eligible),
        "selected_unique_patients": len(records),
        "selected_os_deaths_postselection_audit_only": int(joined.event.sum()),
        "missing_ct_metadata_patients": len(
            candidates - {x["PatientID"] for x in records}
        ),
        "phase_counts": dict(Counter(x["phase_label"] for x in records)),
        "selected_series_bytes": sum(int(x["FileSize"]) for x in records),
        "selected_series": records,
        "excluded_candidate_series_count": len(excluded),
        "excluded_reason_counts": dict(Counter(x["reason"] for x in excluded)),
        "excluded_series_audit": excluded,
        "limitations": [
            "Keywords are not radiologist confirmation of renal/tumour coverage.",
            "The selected timepoint is not established relative to diagnosis; TCIA dates are deidentified.",
            "ImageCount does not prove valid single-volume DICOM geometry; downloaded series need header/pixel QC.",
            "Phase keywords are heuristics except exact RCC-AID v2 SeriesInstanceUID annotations.",
            "This is a predownload QC shortlist, not an admitted survival cohort.",
        ],
    }
    out_dir = ROOT / "data/tcia-kirc/ct-cohort"
    out_dir.mkdir(parents=True, exist_ok=True)
    publish(
        out_dir / "selected-series.json",
        json.dumps(
            {"report": report, "files": records}, indent=2, allow_nan=False
        ).encode(),
    )
    selected_outcomes = joined[
        ["Patient_ID", "time", "event", "SeriesInstanceUID", "phase_label"]
    ].sort_values("Patient_ID")
    publish(
        out_dir / "selected-patient-outcomes.csv",
        selected_outcomes.to_csv(index=False).encode(),
    )
    publish(
        out_dir / "selection-audit.json",
        json.dumps(report, indent=2, allow_nan=False).encode(),
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "candidate_patients",
                    "eligible_metadata_series",
                    "selected_unique_patients",
                    "selected_os_deaths_postselection_audit_only",
                    "missing_ct_metadata_patients",
                    "phase_counts",
                    "selected_series_bytes",
                    "excluded_reason_counts",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
