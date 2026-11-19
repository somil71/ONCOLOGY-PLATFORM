"""Synthetic source-to-vector and lock-integrity regressions; no patient result."""

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.image_cohort_lock import validate_image_cohort_lock
from tools.independently_validate_image_cohort import validate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def rejects(function):
    try:
        function()
    except (ValueError, AssertionError):
        return
    raise AssertionError("Invalid source/lock accepted")


def main():
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        clinical = pd.DataFrame(
            {
                "Patient_ID": ["A", "B"],
                "time": [100, 200],
                "event": [1, 0],
                "age_years": [60, 70],
            }
        )
        original = folder / "original.csv"
        clinical.to_csv(original, index=False)
        features = clinical.copy()
        modalities = {"cli": ["age_years"]}
        selected = [
            {
                "Patient_ID": "A",
                "file_id": "WSI-A",
                "md5sum": "a" * 32,
                "cases": [
                    {"submitter_id": "A", "samples": [{"sample_type": "Primary Tumor"}]}
                ],
            }
        ]
        write(folder / "wsi_selection.json", {"files": selected})
        records, fingerprints = [], {}
        for mode, width, maskname in (
            ("wsi", 768, "valid_items"),
            ("rad", 512, "item_mask"),
        ):
            values = np.stack(
                [np.ones(width), np.full(width, np.nan), np.full(width, 3)]
            ).astype("float32")
            source = folder / f"{mode}.npz"
            extras = {"series_uid": np.array("SERIES-A")} if mode == "rad" else {}
            np.savez(
                source,
                features=values,
                **{maskname: [1, 0, 1]},
                patient_id=np.array("A"),
                **extras,
            )
            extraction = {"encoder": "synthetic-fixture"}
            audit = {"patient_id": "A", "features_sha256": sha(source)}
            if mode == "wsi":
                audit.update(
                    configuration=extraction, status="verified_extracted_features"
                )
            else:
                audit.update(
                    **extraction,
                    series_uid="SERIES-A",
                    all_slice_identity_and_geometry_verified=True,
                )
            audit_path = folder / f"{mode}_audit.json"
            write(audit_path, audit)
            fingerprint = hashlib.sha256(
                json.dumps(extraction, sort_keys=True).encode()
            ).hexdigest()
            record = {
                "patient_id": "A",
                "modality": mode,
                "feature_file": str(source),
                "audit_file": str(audit_path),
                "features_sha256": sha(source),
                "audit_sha256": sha(audit_path),
                "valid_items": 2,
                "stored_items": 3,
                "extraction_contract": extraction,
                "extraction_contract_sha256": fingerprint,
            }
            record.update(
                {"source_file_id": "WSI-A", "source_md5": "a" * 32}
                if mode == "wsi"
                else {"series_uid": "SERIES-A"}
            )
            records.append(record)
            fingerprints[mode] = fingerprint
            modalities[mode] = [f"{mode}_mean_{n:04d}" for n in range(width)]
            features = pd.concat(
                [
                    features,
                    pd.DataFrame(
                        np.array([[2.0] * width, [np.nan] * width]),
                        columns=modalities[mode],
                    ),
                ],
                axis=1,
            )
        features.to_csv(folder / "patient_features.csv", index=False)
        write(folder / "modalities.json", modalities)
        write(folder / "image_feature_records.json", records)
        pd.DataFrame(
            {
                "Patient_ID": ["A", "B"],
                "cli_available": [True, True],
                "wsi_available": [True, False],
                "rad_available": [True, False],
                "rna_available": [True, True],
                "all_requested_modalities_available": [True, False],
            }
        ).to_csv(folder / "patient_availability.csv", index=False)
        rna = folder / "rna.npz"
        np.savez(
            rna,
            patient_ids=np.array(["A", "B"]),
            values=np.ones((2, 3), dtype="float32"),
        )
        protocol = folder / "protocol.json"
        write(protocol, {"scope": "synthetic fixture"})
        contract = {
            "schema": "cancerfusion.patient-feature-cohort.v1",
            "clinical_source": {"path": str(original), "sha256": sha(original)},
            "clinical_columns": ["age_years"],
            "wsi_selection_manifest": {
                "path": str(folder / "wsi_selection.json"),
                "sha256": sha(folder / "wsi_selection.json"),
            },
            "image_extraction_contract_sha256": fingerprints,
            "rna": {"path": str(rna), "sha256": sha(rna)},
            "complete_requested_inputs": {"patients": 1},
            "artifacts_sha256": {
                name: sha(folder / name)
                for name in [
                    "patient_features.csv",
                    "modalities.json",
                    "image_feature_records.json",
                    "patient_availability.csv",
                ]
            },
        }
        write(folder / "cohort_contract.json", contract)
        audit = validate(folder)
        assert audit["status"] == "passed" and audit["complete_input_patient_ids"] == [
            "A"
        ]
        assert audit["maximum_float32_csv_roundtrip_error"] == 0
        source_paths = {
            "patient_features": folder / "patient_features.csv",
            "modalities": folder / "modalities.json",
            "study_protocol": protocol,
            "rna_matrix": rna,
            "cohort_contract": folder / "cohort_contract.json",
            "independent_input_audit": folder / "independent_input_validation.json",
        }
        lock = {
            "schema": "cancerfusion.image-cohort-lock.v1",
            "decision": "locked_for_retrospective_complete_input_experiment",
            "prospective_prediction_eligible": False,
            "outcome_blind_qc_confirmed": True,
            "patient_fold_policy": "retain_original_locked_assignments_no_resplitting",
            "candidate_clinical_sha256": sha(original),
            "admitted_patient_ids": ["A"],
            "admitted_patients": 1,
            "admitted_os_deaths": 1,
            "sources": {
                k: {"path": str(v), "sha256": sha(v)} for k, v in source_paths.items()
            },
        }
        lock_path = folder / "lock.json"
        write(lock_path, lock)
        def call():
            return validate_image_cohort_lock(
                lock_path,
                folder / "patient_features.csv",
                folder / "modalities.json",
                protocol,
                rna,
                True,
            )
        assert call()["admitted_patient_ids"] == ["A"]
        lock["decision"] = "EXAMPLE_ONLY_NOT_AN_ADMISSION_LOCK"
        write(lock_path, lock)
        rejects(call)
        lock["decision"] = "locked_for_retrospective_complete_input_experiment"
        lock["admitted_patient_ids"] = ["B"]
        write(lock_path, lock)
        rejects(call)
        lock["admitted_patient_ids"] = ["A"]
        write(lock_path, lock)
        with (folder / "patient_features.csv").open("a") as stream:
            stream.write("\n")
        rejects(call)
        rejects(lambda: validate(folder))
    print(
        "PASS: independently reconstructed valid-item means ignore NaN padding; missing images remain NaN; exact joins/availability, root lock/source binding, draft/admitted-ID/tamper rejection. Synthetic fixtures only; no image model fitted."
    )


if __name__ == "__main__":
    main()
