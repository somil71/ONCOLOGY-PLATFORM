"""Adversarial patient-join, provenance and padded-item feature-cohort checks."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.image_feature_cohort import (
    file_hash,
    load_ct_bag,
    load_wsi_bag,
    safe_predictor_names,
    valid_item_mean,
)
from tools.build_image_feature_cohort import build
from tools.run_survival_experiments import locked_patient_assignments


def rejects(call):
    try:
        call()
    except (ValueError, KeyError):
        return
    raise AssertionError("Invalid image cohort input was accepted")


def main():
    values = np.array([[1.0, 3.0], [3.0, 7.0], [np.nan, np.inf]], dtype=np.float32)
    np.testing.assert_array_equal(valid_item_mean(values, [1, 1, 0]), [2.0, 5.0])
    changed = values.copy()
    changed[-1] = [-1e30, 1e30]
    np.testing.assert_array_equal(
        valid_item_mean(values, [1, 1, 0]), valid_item_mean(changed, [1, 1, 0])
    )
    for mask in [[0, 0, 0], [1, 0.5, 0], [1, np.nan, 0], [1, 1]]:
        rejects(lambda mask=mask: valid_item_mean(values, mask))
    rejects(lambda: valid_item_mean(values, [1, 1, 1]))
    rejects(lambda: valid_item_mean(np.ones((2, 3), dtype=int), [1, 1]))
    for columns in [
        ["wsi_last_contact_days_to"],
        ["rad_survival"],
        ["event"],
        ["Patient_ID"],
        ["x", "x"],
    ]:
        rejects(lambda columns=columns: safe_predictor_names(columns))
    runtime = Path("runtime").resolve()
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="feature-cohort-test-", dir=runtime
    ) as name:
        root = Path(name).resolve()
        assert root.is_relative_to(runtime)
        patient = "CASE-001"
        wsi = root / "wsi" / patient
        rad = root / "rad" / patient
        wsi.mkdir(parents=True)
        rad.mkdir(parents=True)
        source_record = {
            "Patient_ID": patient,
            "file_id": "fixture-source-001",
            "md5sum": "a" * 32,
        }
        manifest_path = root / "selected-manifest.json"
        manifest_path.write_text(json.dumps({"files": [source_record]}))
        source_hash = file_hash(manifest_path)
        cfg = {
            "source_manifest_sha256": source_hash,
            "weights": "fixture-frozen-encoder",
            "weight_sha256": "b" * 64,
            "feature_code_sha256": "c" * 64,
            "driver_code_sha256": "d" * 64,
            "transform": {"normalization": "fixture"},
            "target_mpp": 0.5,
            "max_patches": 128,
            "min_patches": 2,
            "seed": 42,
        }
        np.savez_compressed(
            wsi / "wsi_features.npz",
            features=np.full((2, 768), 0.25, dtype=np.float32),
            valid_items=np.ones(2, dtype=bool),
            coordinates_level0=np.array([[0, 0], [224, 0]]),
        )
        wsi_audit = {
            "patient_id": patient,
            "file_id": source_record["file_id"],
            "source_md5": source_record["md5sum"],
            "status": "verified_extracted_features",
            "features_sha256": file_hash(wsi / "wsi_features.npz"),
            "feature_shape": [2, 768],
            "configuration": cfg,
            "cache_key": hashlib.sha256(
                json.dumps(
                    {"source_md5": source_record["md5sum"], "config": cfg},
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            "qc": {
                "selected_patches": 2,
                "coordinates_level0": [[0, 0], [224, 0]],
                "source_mpp_x": 0.252,
                "source_mpp_y": 0.252,
                "target_mpp": 0.5,
            },
        }
        (wsi / "wsi_features.json").write_text(json.dumps(wsi_audit))
        pooled, record = load_wsi_bag(
            wsi / "wsi_features.npz",
            wsi / "wsi_features.json",
            patient,
            source_record,
            source_hash,
        )
        np.testing.assert_array_equal(pooled, np.full(768, 0.25))
        assert (
            record["valid_items"] == 2
            and record["prospective_prediction_eligible"] is False
        )
        rejects(
            lambda: load_wsi_bag(
                wsi / "wsi_features.npz",
                wsi / "wsi_features.json",
                "CASE-002",
                source_record,
                source_hash,
            )
        )
        rejects(
            lambda: load_wsi_bag(
                wsi / "wsi_features.npz",
                wsi / "wsi_features.json",
                patient,
                {**source_record, "file_id": "wrong"},
                source_hash,
            )
        )
        rejects(
            lambda: load_wsi_bag(
                wsi / "wsi_features.npz",
                wsi / "wsi_features.json",
                patient,
                source_record,
                "f" * 64,
            )
        )
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        np.savez_compressed(
            rad / "rad_features.npz",
            features=np.full((2, 512), 0.75, dtype=np.float32),
            item_mask=np.ones(2, dtype=bool),
            patient_id=np.asarray(patient),
            series_uid=np.asarray("1.2.3"),
            positions_mm=positions,
        )
        ct_audit = {
            "patient_id": patient,
            "series_uid": "1.2.3",
            "all_slice_identity_and_geometry_verified": True,
            "selection_sha256": "a" * 64,
            "weights_sha256": "b" * 64,
            "implementation_sha256": {"fixture.py": "c" * 64},
            "feature_shape": [2, 512],
            "selected_slices": 2,
            "features_sha256": file_hash(rad / "rad_features.npz"),
            "encoder": "fixture-frozen-encoder",
            "transform": "fixture",
            "hu_windows_center_width": [[40, 400]],
            "window_encoding": "fixture",
            "slices": [
                {"sha256": str(i + 1) * 64, "position_mm": list(p)}
                for i, p in enumerate(positions)
            ],
        }
        (rad / "ct_feature_audit.json").write_text(json.dumps(ct_audit))
        pooled, _ = load_ct_bag(
            rad / "rad_features.npz", rad / "ct_feature_audit.json", patient
        )
        np.testing.assert_array_equal(pooled, np.full(512, 0.75))
        rejects(
            lambda: load_ct_bag(
                rad / "rad_features.npz", rad / "ct_feature_audit.json", "CASE-002"
            )
        )
        clinical_path = root / "clinical.csv"
        pd.DataFrame(
            {
                "Patient_ID": [patient, "CASE-002"],
                "time": [100, 200],
                "event": [1, 0],
                "age_years": [60, np.nan],
            }
        ).to_csv(clinical_path, index=False)
        args = argparse.Namespace(
            clinical=clinical_path,
            clinical_columns=["age_years"],
            cohort="SYNTHETIC-ENGINEERING-FIXTURE",
            wsi_root=root / "wsi",
            wsi_source_manifest=manifest_path,
            rad_root=root / "rad",
            rna_npz=None,
            study_protocol=None,
            record_rejected_as_missing=False,
            output=root / "joined",
        )
        contract = build(args)
        table = pd.read_csv(args.output / "patient_features.csv")
        availability = pd.read_csv(args.output / "patient_availability.csv")
        assert contract["technical_modality_counts"] == {"cli": 1, "wsi": 1, "rad": 1}
        assert contract["complete_requested_inputs"] == {
            "patients": 1,
            "observed_deaths": 1,
        }
        assert not availability.loc[1, "cli_available"]
        assert table.loc[1].filter(regex="wsi_|rad_").isna().all()
        assert len(table) == 2, "Unacquired-image patients were silently dropped"
        args.clinical_columns = ["event"]
        args.output = root / "invalid-leakage"
        rejects(lambda: build(args))
        args.clinical_columns = ["age_years"]
        duplicated = pd.read_csv(clinical_path)
        duplicated.loc[1, "Patient_ID"] = patient
        duplicated.to_csv(clinical_path, index=False)
        rejects(lambda: build(args))
        bad_split = root / "split.csv"
        pd.DataFrame(
            {"Patient_ID": ["A", "A", "B"], "split": ["train", "validation", "test"]}
        ).to_csv(bad_split, index=False)
        rejects(lambda: locked_patient_assignments(bad_split, ["A", "B", "C"]))
        pd.DataFrame(
            {"Patient_ID": ["A", "B", "D"], "split": ["train", "validation", "test"]}
        ).to_csv(bad_split, index=False)
        rejects(lambda: locked_patient_assignments(bad_split, ["A", "B", "C"]))
        with (rad / "rad_features.npz").open("ab") as stream:
            stream.write(b"tamper")
        rejects(
            lambda: load_ct_bag(
                rad / "rad_features.npz", rad / "ct_feature_audit.json", patient
            )
        )
    print(
        "PASS: inactive-padding invariance, allmissing/binary-mask guards, patient/specimen/series identity, checksums, duplicate joins/splits, outcome leakage and honest missing clinical/image availability."
    )


if __name__ == "__main__":
    main()
