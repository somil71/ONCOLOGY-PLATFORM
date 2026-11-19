"""Join verified frozen image bags to explicit patient-level survival inputs.

Exports valid-item MEAN vectors for baseline models. No patch attention, image
training, cohort pooling, time-origin correction or efficacy claim is performed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.clinical_contract import select_reviewed_clinical_features
from src.image_feature_cohort import (
    checked_patient_id,
    file_hash,
    load_ct_bag,
    load_wsi_bag,
    safe_predictor_names,
)
from src.survival_models import validate_outcomes
from tools.run_survival_experiments import dump


def build(args):
    clinical = pd.read_csv(args.clinical)
    if clinical.Patient_ID.duplicated().any():
        raise ValueError("Clinical cohort contains duplicate patient identities")
    for patient in clinical.Patient_ID:
        checked_patient_id(patient)
    validate_outcomes(clinical.time, clinical.event)
    safe_predictor_names(args.clinical_columns)
    selected_clinical = select_reviewed_clinical_features(
        clinical, args.clinical_columns
    )
    if np.isinf(selected_clinical.to_numpy(float)).any():
        raise ValueError("Clinical predictors cannot contain infinities")
    frame = clinical[["Patient_ID", "time", "event", *args.clinical_columns]].copy()
    source_manifest_hash, selected_wsi = None, {}
    if args.wsi_root is not None:
        if args.wsi_source_manifest is None:
            raise ValueError(
                "WSI requires its explicit pinned GDC source-selection manifest"
            )
        wsi_manifest = json.loads(
            args.wsi_source_manifest.read_text(encoding="utf-8-sig")
        )
        source_manifest_hash = file_hash(args.wsi_source_manifest)
        records = wsi_manifest["files"]
        if len({r["Patient_ID"] for r in records}) != len(records) or len(
            {r["file_id"] for r in records}
        ) != len(records):
            raise ValueError(
                "GDC selection has duplicate patient or source-file assignments"
            )
        selected_wsi = {r["Patient_ID"]: r for r in records}

    modalities = {"cli": args.clinical_columns}
    available = pd.DataFrame(
        {
            "Patient_ID": clinical.Patient_ID,
            "cli_available": np.isfinite(selected_clinical.to_numpy(float)).any(1),
        }
    )
    provenance, rejected, fingerprints = [], [], {}
    for modality, root, width, feature_name, audit_name in [
        ("wsi", args.wsi_root, 768, "wsi_features.npz", "wsi_features.json"),
        ("rad", args.rad_root, 512, "rad_features.npz", "ct_feature_audit.json"),
    ]:
        if root is None:
            continue
        columns = [f"{modality}_mean_{i:04d}" for i in range(width)]
        safe_predictor_names(columns)
        matrix = np.full((len(clinical), width), np.nan, dtype=np.float32)
        statuses = []
        for index, patient in enumerate(clinical.Patient_ID):
            folder = root / patient
            feature_path, audit_path = folder / feature_name, folder / audit_name
            if not feature_path.exists() and not audit_path.exists():
                statuses.append("not_acquired_or_no_successful_extraction")
                continue
            try:
                if not feature_path.is_file() or not audit_path.is_file():
                    raise ValueError("Incomplete image artifact/audit pair")
                if modality == "wsi":
                    if patient not in selected_wsi:
                        raise ValueError(
                            "Feature patient absent from pinned GDC selection"
                        )
                    vector, record = load_wsi_bag(
                        feature_path,
                        audit_path,
                        patient,
                        selected_wsi[patient],
                        source_manifest_hash,
                    )
                else:
                    vector, record = load_ct_bag(feature_path, audit_path, patient)
                fingerprint = hashlib.sha256(
                    json.dumps(record["extraction_contract"], sort_keys=True).encode()
                ).hexdigest()
                if modality in fingerprints and fingerprints[modality] != fingerprint:
                    raise ValueError(
                        "Cohort mixes different encoder or preprocessing contracts"
                    )
                fingerprints[modality] = fingerprint
                matrix[index] = vector
                record.update(
                    {
                        "feature_file": str(feature_path.resolve()),
                        "audit_file": str(audit_path.resolve()),
                        "extraction_contract_sha256": fingerprint,
                    }
                )
                provenance.append(record)
                statuses.append("verified_technical_features")
            except (ValueError, KeyError, OSError) as exc:
                if not args.record_rejected_as_missing:
                    raise ValueError(f"{patient}/{modality}: {exc}") from exc
                rejected.append(
                    {"Patient_ID": patient, "modality": modality, "reason": str(exc)}
                )
                statuses.append("rejected_technical_artifact")
        frame = pd.concat(
            [frame, pd.DataFrame(matrix, columns=columns, index=frame.index)], axis=1
        )
        modalities[modality] = columns
        available[f"{modality}_available"] = np.isfinite(matrix).all(1)
        available[f"{modality}_status"] = statuses

    rna_contract = None
    if args.rna_npz:
        with np.load(args.rna_npz, allow_pickle=False) as rna:
            ids = rna["patient_ids"].astype(str)
            genes = rna["gene_ids"].astype(str)
            values = rna["values"]
            if len(set(ids)) != len(ids) or len(set(genes)) != len(genes):
                raise ValueError("RNA patient and gene identities must be unique")
            safe_predictor_names(genes.tolist())
            if (
                values.shape != (len(ids), len(genes))
                or not len(genes)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    "RNA matrix must have aligned finite patient/gene values"
                )
            available["rna_available"] = clinical.Patient_ID.isin(ids)
            rna_contract = {
                "path": str(args.rna_npz.resolve()),
                "sha256": file_hash(args.rna_npz),
                "matrix_patients": len(ids),
                "cohort_matched_patients": int(available.rna_available.sum()),
                "genes": len(genes),
                "gene_order_sha256": hashlib.sha256(
                    json.dumps(genes.tolist()).encode()
                ).hexdigest(),
                "training_requirement": "Training-only gene selection, imputation and scaling; use the same patient split ledger.",
            }
    availability_columns = [c for c in available if c.endswith("_available")]
    available["available_modality_count"] = available[availability_columns].sum(1)
    available["all_requested_modalities_available"] = available[
        availability_columns
    ].all(1)
    protocol = (
        json.loads(args.study_protocol.read_text()) if args.study_protocol else None
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "patient_features.csv", index=False)
    available.to_csv(output / "patient_availability.csv", index=False)
    dump(output / "modalities.json", modalities)
    dump(output / "image_feature_records.json", provenance)
    dump(output / "rejected_artifacts.json", rejected)
    complete = available.all_requested_modalities_available
    counts = {
        name.removesuffix("_available"): int(available[name].sum())
        for name in availability_columns
    }
    contract = {
        "schema": "cancerfusion.patient-feature-cohort.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": args.cohort,
        "representation": "Outcome-independent mean over valid frozen image embeddings; no patch attention. Fit downstream scalers using training patients only.",
        "patients": len(frame),
        "recorded_os_deaths": int(frame.event.sum()),
        "technical_modality_counts": counts,
        "complete_requested_inputs": {
            "patients": int(complete.sum()),
            "observed_deaths": int(frame.loc[complete, "event"].sum()),
        },
        "all_patient_ids_retained": True,
        "missing_image_vectors": "All NaN; never substituted as observed zeros. Downstream availability mask must be computed before imputation.",
        "clinical_columns": args.clinical_columns,
        "image_extraction_contract_sha256": fingerprints,
        "clinical_source": {
            "path": str(args.clinical.resolve()),
            "sha256": file_hash(args.clinical),
        },
        "wsi_selection_manifest": {
            "path": str(args.wsi_source_manifest.resolve()),
            "sha256": source_manifest_hash,
        }
        if args.wsi_source_manifest
        else None,
        "rna": rna_contract,
        "study_protocol": {
            "path": str(args.study_protocol.resolve()),
            "sha256": file_hash(args.study_protocol),
            "metadata": protocol,
        }
        if args.study_protocol
        else None,
        "admission": {
            "technical_features_exported": True,
            "prospective_prediction_eligible": False,
            "clinical_validation_established": False,
            "retrospective_exploration_only": True,
            "clinical_review_complete": False,
            "predictor_timing_not_established_by_this_adapter": True,
        },
        "rejected_artifact_policy": "Explicitly recorded as missing"
        if args.record_rejected_as_missing
        else "Fail on invalid/incomplete artifact",
        "rejected_artifacts": len(rejected),
        "implementation_sha256": {
            name: file_hash(Path(name))
            for name in [
                "src/image_feature_cohort.py",
                "src/clinical_contract.py",
                "tools/build_image_feature_cohort.py",
            ]
        },
        "artifacts_sha256": {
            p.name: file_hash(p) for p in output.iterdir() if p.is_file()
        },
        "runner_contract": {
            "input": "patient_features.csv",
            "modalities_json": "modalities.json",
            "rna_npz": rna_contract["path"] if rna_contract else None,
            "complete_cohort": "Optional: retain the locked patient split and export exclusions; never resplit to improve event counts.",
        },
    }
    dump(output / "cohort_contract.json", contract)
    print(
        json.dumps(
            {
                "output": str(output),
                "technical_modality_counts": counts,
                "complete_requested_inputs": contract["complete_requested_inputs"],
                "admission": contract["admission"],
            },
            indent=2,
        )
    )
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--clinical-columns", nargs="+", default=["age_years"])
    parser.add_argument(
        "--cohort",
        required=True,
        help="Explicit disease/cohort identifier; cohorts are never inferred or pooled",
    )
    parser.add_argument("--wsi-root", type=Path)
    parser.add_argument("--wsi-source-manifest", type=Path)
    parser.add_argument("--rad-root", type=Path)
    parser.add_argument("--rna-npz", type=Path)
    parser.add_argument("--study-protocol", type=Path)
    parser.add_argument("--record-rejected-as-missing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
