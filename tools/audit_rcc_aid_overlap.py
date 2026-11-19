"""Measure annotated RCC-AID overlap without admitting unverified images to training."""

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from tools.extract_ct_features import publish, sha256


def main():
    metadata = Path("data/rcc-aid/v2/metadata.csv")
    clinical = Path("data/processed/tcga_kirc_clinical.csv")
    tcia = Path("data/separate-cancer-feasibility/TCGA-KIRC-ct-metadata.json")
    frame, patients = pd.read_csv(metadata), pd.read_csv(clinical)
    if (
        frame.SeriesInstanceUID.isna().any()
        or frame.SeriesInstanceUID.duplicated().any()
    ):
        raise ValueError("Annotated series IDs must be nonempty and unique.")
    if patients.Patient_ID.duplicated().any():
        raise ValueError("Clinical patient IDs must be unique.")
    series = {item["SeriesInstanceUID"]: item for item in json.loads(tcia.read_text())}
    renal = frame[frame.Collection.eq("TCGA-KIRC")].copy()
    matched = renal[renal.PatientID.isin(patients.Patient_ID)].copy()
    for row in matched.itertuples():
        original = series.get(row.SeriesInstanceUID)
        if original is None or original["PatientID"] != row.PatientID:
            raise ValueError(
                "RCC-AID series/patient mapping differs from official TCIA metadata."
            )
    counts = {}
    for name, subset in [
        ("all_annotated_kirc", renal),
        ("matched_four_input_candidates", matched),
        *[
            (phase, matched[matched.phase_name.eq(phase)])
            for phase in sorted(matched.phase_name.unique())
        ],
    ]:
        outcomes = patients[patients.Patient_ID.isin(subset.PatientID)]
        counts[name] = {
            "series": len(subset),
            "patients": int(subset.PatientID.nunique()),
            "matched_cdr_patients": len(outcomes),
            "matched_os_deaths": int(outcomes.event.sum()),
            "tcia_metadata_uncompressed_bytes": int(subset.FileSize.sum()),
        }
    destination = Path("data/rcc-aid/v2/matched-kirc-series.csv")
    publish(destination, matched.to_csv(index=False).encode())
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://zenodo.org/records/20719257",
        "version": "2.0.0",
        "input_sha256": {
            str(path): sha256(path) for path in [metadata, clinical, tcia]
        },
        "counts": counts,
        "all_matched_series_patient_ids_verified": True,
        "phase_labels_source": "RCC-AID authors' phase_name column; not inferred from series descriptions",
        "cohort_admitted_for_efficacy": False,
        "limitations": [
            "Annotated subsets have few observed deaths and cannot substitute for a powered four-input study.",
            "Counts describe case/series metadata overlap, not completed image, segmentation or temporal QC.",
            "Masks apply to their exact SeriesInstanceUID. Never copy across another phase of the same patient.",
            "Image, label, identity, orientation and baseline-timing checks remain necessary before feature extraction.",
            "KIRC, KIRP and KICH remain distinct histologies; no pooled outcome analysis is performed.",
        ],
    }
    publish(
        Path("RUN REPORT/rcc-aid-overlap-20261118.json"),
        json.dumps(report, indent=2).encode(),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
