"""Count actual verified KIRC input features, separately from metadata overlap.

An available feature bag is not proof of acquisition before outcome origin,
adequate tumour coverage, model training, or clinical survival performance.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verified_image_bags(roots, modality, eligible, selected_wsi):
    filename, width = (
        ("wsi_features.npz", 768) if modality == "wsi" else ("rad_features.npz", 512)
    )
    bags = {}
    for folder in roots:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*/" + filename)):
            patient = path.parent.name
            if patient not in eligible:
                continue
            audit_path = (
                path.parent / "wsi_features.json"
                if modality == "wsi"
                else next(
                    (
                        path.parent / name
                        for name in ("rad_features.json", "ct_feature_audit.json")
                        if (path.parent / name).is_file()
                    ),
                    path.parent / "ct_feature_audit.json",
                )
            )
            if not audit_path.is_file():
                raise ValueError(f"Image bag lacks provenance audit: {path}")
            audit = json.loads(audit_path.read_bytes())
            expected_hash = audit.get("features_sha256", audit.get("feature_sha256"))
            if audit.get("patient_id") != patient or expected_hash != sha256(path):
                raise ValueError(f"Image identity/checksum mismatch: {path}")
            if modality == "wsi" and (
                audit.get("file_id") != selected_wsi[patient]["file_id"]
                or audit.get("source_md5") != selected_wsi[patient]["md5sum"]
            ):
                raise ValueError(
                    f"WSI bag source no longer matches primary-specimen selection: {patient}"
                )
            with np.load(path, allow_pickle=False) as d:
                values = d["features"]
                mask = d["valid_items"] if modality == "wsi" else d["item_mask"]
                if (
                    values.ndim != 2
                    or values.shape[1] != width
                    or not len(values)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(f"Invalid feature bag: {path}")
                if (
                    mask.shape != (len(values),)
                    or not np.isin(mask, [0, 1]).all()
                    or not mask.any()
                ):
                    raise ValueError(f"Invalid image item mask: {path}")
                if modality == "rad" and str(d["patient_id"]) != patient:
                    raise ValueError("CT archive patient mismatch")
                count = int(mask.sum())
            if patient in bags and bags[patient]["sha256"] != expected_hash:
                raise ValueError(
                    f"Ambiguous multiple {modality} feature contracts for {patient}"
                )
            bags[patient] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": expected_hash,
                "valid_items": count,
            }
    return bags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, action="append")
    args = parser.parse_args()
    roots = args.feature_root or [
        ROOT / "RUN REPORT/kirc-paired-pilot",
        ROOT / "data/processed/kirc_wsi_features",
        ROOT / "data/processed/kirc_ct_features",
    ]
    roots = [path.resolve() for path in roots]
    clinical_path = ROOT / "data/processed/tcga_kirc_clinical.csv"
    clinical = pd.read_csv(clinical_path)
    if clinical.Patient_ID.duplicated().any():
        raise ValueError("Duplicated clinical patient")
    ids = set(clinical.Patient_ID)
    rna_path = ROOT / "data/processed/tcga_kirc_rna_logtpm.npz"
    rna_audit = json.loads(rna_path.with_name("tcga_kirc_rna_audit.json").read_bytes())
    if not rna_audit["complete_selected_cohort"] or rna_audit["sha256"] != sha256(
        rna_path
    ):
        raise ValueError("RNA matrix provenance mismatch or incomplete cohort")
    with np.load(rna_path, allow_pickle=False) as d:
        rna_ids = d["patient_ids"].astype(str)
        if (
            len(set(rna_ids)) != len(rna_ids)
            or set(rna_ids) != ids
            or not np.isfinite(d["values"]).all()
        ):
            raise ValueError("RNA identity/value mismatch")
    selected = {
        item["Patient_ID"]: item
        for item in json.loads(
            (ROOT / "data/tcga-kirc/wsi-diagnostic/selected-manifest.json").read_bytes()
        )["files"]
    }
    wsi = verified_image_bags(roots, "wsi", ids, selected)
    rad = verified_image_bags(roots, "rad", ids, selected)
    frame = clinical[["Patient_ID", "time", "event"]].copy()
    frame["has_cli"] = np.isfinite(clinical.age_years)
    frame["has_rna"] = frame.Patient_ID.isin(set(rna_ids))
    for name, bags in [("wsi", wsi), ("rad", rad)]:
        frame["has_" + name] = frame.Patient_ID.isin(bags)
        frame[name + "_feature_path"] = frame.Patient_ID.map(
            {patient: item["path"] for patient, item in bags.items()}
        )
        frame[name + "_feature_sha256"] = frame.Patient_ID.map(
            {patient: item["sha256"] for patient, item in bags.items()}
        )
        frame[name + "_valid_items"] = (
            frame.Patient_ID.map(
                {patient: item["valid_items"] for patient, item in bags.items()}
            )
            .fillna(0)
            .astype(int)
        )
    frame["all_four_features_available"] = frame[
        ["has_cli", "has_rna", "has_wsi", "has_rad"]
    ].all(axis=1)
    out = ROOT / "data/processed/kirc_actual_input_availability.csv"
    frame.to_csv(out, index=False)
    complete = frame[frame.all_four_features_available]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": "TCGA-KIRC",
        "patients": len(frame),
        "observed_deaths": int(frame.event.sum()),
        "actual_feature_patients": {
            name: int(frame["has_" + name].sum())
            for name in ("cli", "rna", "wsi", "rad")
        },
        "actual_all_four_patients": len(complete),
        "actual_all_four_deaths": int(complete.event.sum()),
        "all_four_patient_ids": list(complete.Patient_ID),
        "clinical_sha256": sha256(clinical_path),
        "rna_sha256": rna_audit["sha256"],
        "availability_csv_sha256": sha256(out),
        "scope": "Verified feature availability only. This is not metadata overlap, acquisition-time eligibility, tumour localization, trained four-input model performance, or clinical validation.",
    }
    (out.with_suffix(".json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
