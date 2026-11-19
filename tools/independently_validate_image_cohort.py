"""Independently reconstruct frozen image-bag means and explicit patient joins.

This is a technical input audit. It does not authorize model fitting, choose a
cohort, infer acquisition timing, or establish clinical/imaging validity.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_strict_ct_layout(bag, audit):
    """Check the explicit padded128-item CT layout without reusing extraction code."""
    values, mask = bag["features"], bag["item_mask"]
    positions, normalized = bag["positions_mm"], bag["normalized_positions"]
    indices = bag["source_slice_indices"]
    total, retained = (
        int(audit["source_dicom_count"]),
        int(audit["retained_feature_items"]),
    )
    if values.shape != (128, 512) or mask.shape != (128,) or mask.dtype != np.bool_:
        raise ValueError(
            "Strict CT requires a128x512 float feature bag and boolean mask"
        )
    if (
        total < 20
        or retained != min(128, total)
        or not np.array_equal(mask, np.arange(128) < retained)
    ):
        raise ValueError("Strict CT source count/retention/right-padding mask mismatch")
    if positions.shape != (128, 3) or normalized.shape != (128, 1):
        raise ValueError("Strict CT physical/normalized positions are misaligned")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(positions).all()
        or not np.isfinite(normalized).all()
    ):
        raise ValueError("Strict CT bag contains nonfinite data")
    if (
        np.any(values[~mask] != 0)
        or np.any(positions[~mask] != 0)
        or np.any(normalized[~mask] != 0)
    ):
        raise ValueError(
            "Strict CT inactive padding must match its declared zero-padding contract"
        )
    expected = np.linspace(0, total - 1, retained, dtype=int)
    if (
        indices.shape != (retained,)
        or indices.dtype.kind not in "iu"
        or not np.array_equal(indices, expected)
    ):
        raise ValueError(
            "Strict CT slice indices are not the declared deterministic source selection"
        )
    if not np.array_equal(indices, np.asarray(audit["source_slice_indices"])):
        raise ValueError("Strict CT NPZ/source-index audit mismatch")
    source, selected = audit["all_source_dicom_sha256"], audit["slices"]
    if (
        len(source) != total
        or len(selected) != retained
        or len({x["sop_instance_uid"] for x in source}) != total
    ):
        raise ValueError("Strict CT source record count/SOP identities differ")
    if selected != [source[n] for n in expected]:
        raise ValueError(
            "Strict CT selected source records differ from declared source indices"
        )
    np.testing.assert_allclose(
        positions[mask],
        np.asarray([x["position_mm"] for x in selected]),
        atol=1e-8,
        rtol=0,
    )
    points = positions[mask]
    line = points[-1] - points[0]
    length2 = float(np.dot(line, line))
    if length2 <= 0:
        raise ValueError("Strict CT volume has no physical extent")
    fraction = ((points - points[0]) @ line) / length2
    if not (np.diff(fraction) > 0).all():
        raise ValueError("Strict CT physical source positions are not ordered/distinct")
    np.testing.assert_allclose(normalized[mask, 0], fraction, atol=2e-6, rtol=0)
    for name, expected_value in (
        ("selected_slices", retained),
        ("item_mask_true", retained),
    ):
        if audit[name] != expected_value:
            raise ValueError("Strict CT retained-item audit mismatch")
    if audit["feature_shape"] != [128, 512] or audit["padded_bag_shape"] != [128, 512]:
        raise ValueError("Strict CT audited padded shape mismatch")
    return {
        "status": "passed",
        "source_slices": total,
        "retained_items": retained,
        "normalized_positions_reconstructed": True,
        "source_indices_and_padding_reconstructed": True,
    }


def validate(folder):
    folder = Path(folder)
    contract_path = folder / "cohort_contract.json"
    contract = json.loads(contract_path.read_text())
    if contract["schema"] != "cancerfusion.patient-feature-cohort.v1":
        raise ValueError("Unsupported image feature contract")
    for name, expected in contract["artifacts_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError(f"Changed image-cohort artifact: {name}")
    original_path = Path(contract["clinical_source"]["path"])
    if sha(original_path) != contract["clinical_source"]["sha256"]:
        raise ValueError("Original clinical source changed")
    original = pd.read_csv(original_path)
    features = pd.read_csv(folder / "patient_features.csv")
    available = pd.read_csv(folder / "patient_availability.csv")
    modalities = json.loads((folder / "modalities.json").read_text())
    records = json.loads((folder / "image_feature_records.json").read_text())
    if any(f.Patient_ID.duplicated().any() for f in (original, features, available)):
        raise ValueError("Duplicate patient identity")
    ids = original.Patient_ID.tolist()
    if features.Patient_ID.tolist() != ids or available.Patient_ID.tolist() != ids:
        raise ValueError("Patient identity/order differs from original cohort")
    columns = ["time", "event", *contract["clinical_columns"]]
    np.testing.assert_allclose(
        features[columns], original[columns], atol=1e-12, rtol=0, equal_nan=True
    )
    for name in (
        "time",
        "event",
        "Patient_ID",
        "OS",
        "OS.time",
        "last_contact_days_to",
        "death_days_to",
    ):
        if any(name in cols for cols in modalities.values()):
            raise ValueError("Outcome/identity/followup predictor leakage")
    if modalities["cli"] != contract["clinical_columns"]:
        raise ValueError("Clinical predictor order changed")
    source_manifest = contract.get("wsi_selection_manifest")
    selected_wsi = {}
    if source_manifest:
        if sha(source_manifest["path"]) != source_manifest["sha256"]:
            raise ValueError("Pinned WSI selection changed")
        selection = json.loads(Path(source_manifest["path"]).read_text())["files"]
        if len({r["Patient_ID"] for r in selection}) != len(selection):
            raise ValueError("Repeated WSI source patient")
        selected_wsi = {r["Patient_ID"]: r for r in selection}
    observed, signatures = {}, {}
    strict_ct_checks = []
    maximum_roundtrip_error = 0.0
    for record in records:
        patient, mode = record["patient_id"], record["modality"]
        key = (patient, mode)
        if patient not in set(ids) or key in observed or mode not in {"wsi", "rad"}:
            raise ValueError("Ambiguous/unexpected image patient or modality")
        source, audit_path = Path(record["feature_file"]), Path(record["audit_file"])
        audit = json.loads(audit_path.read_text())
        if (
            sha(source) != record["features_sha256"]
            or sha(audit_path) != record["audit_sha256"]
        ):
            raise ValueError("Image feature/audit source digest changed")
        if (
            audit["patient_id"] != patient
            or audit["features_sha256"] != record["features_sha256"]
        ):
            raise ValueError("Image sidecar patient/hash mismatch")
        audit_contract = audit.get("configuration", {}) if mode == "wsi" else audit
        if any(
            audit_contract.get(k) != v for k, v in record["extraction_contract"].items()
        ):
            raise ValueError(
                "Exported extraction contract does not match the source audit"
            )
        if mode == "wsi" and audit.get("status") != "verified_extracted_features":
            raise ValueError("WSI extraction did not complete successfully")
        if (
            mode == "rad"
            and audit.get("all_slice_identity_and_geometry_verified") is not True
        ):
            raise ValueError("CT source audit has not verified identity/geometry")
        mask_name, width = ("valid_items", 768) if mode == "wsi" else ("item_mask", 512)
        with np.load(source, allow_pickle=False) as bag:
            values, mask = bag["features"], bag[mask_name]
            if values.ndim != 2 or values.shape[1] != width or values.dtype.kind != "f":
                raise ValueError("Invalid frozen feature dimensions/dtype")
            if (
                mask.shape != (len(values),)
                or not np.isin(mask, [0, 1]).all()
                or not mask.any()
            ):
                raise ValueError("Nonempty binary image mask required")
            valid = mask.astype(bool)
            if not np.isfinite(values[valid]).all():
                raise ValueError("Nonfinite observed image feature")
            pooled = np.mean(values[valid], axis=0, dtype=np.float64).astype(np.float32)
            if "patient_id" in bag and (
                bag["patient_id"].shape != ()
                or str(bag["patient_id"].item()) != patient
            ):
                raise ValueError("NPZ patient identity mismatch")
            if mode == "rad" and (
                str(bag["series_uid"].item()) != record["series_uid"]
                or record["series_uid"] != audit["series_uid"]
            ):
                raise ValueError("Exact CT series mismatch")
            if mode == "rad" and "normalized_positions" in bag:
                strict_ct_checks.append(
                    {"patient_id": patient, **validate_strict_ct_layout(bag, audit)}
                )
        if (
            int(mask.sum()) != record["valid_items"]
            or len(values) != record["stored_items"]
        ):
            raise ValueError("Image item counts differ from source masks")
        if mode == "wsi":
            original_wsi = selected_wsi[patient]
            if (
                original_wsi["file_id"] != record["source_file_id"]
                or original_wsi["md5sum"] != record["source_md5"]
            ):
                raise ValueError(
                    "WSI source specimen differs from selected primary slide"
                )
            cases = original_wsi["cases"]
            if (
                len(cases) != 1
                or cases[0]["submitter_id"] != patient
                or {s["sample_type"] for s in cases[0]["samples"]} != {"Primary Tumor"}
            ):
                raise ValueError(
                    "WSI does not link to the selected primary tumour case"
                )
        fingerprint = hashlib.sha256(
            json.dumps(record["extraction_contract"], sort_keys=True).encode()
        ).hexdigest()
        if (
            fingerprint != record["extraction_contract_sha256"]
            or fingerprint != contract["image_extraction_contract_sha256"][mode]
        ):
            raise ValueError("Image extraction/preprocessing contract mismatch")
        if mode in signatures and signatures[mode] != fingerprint:
            raise ValueError("Mixed image encoder/preprocessing contracts")
        signatures[mode] = fingerprint
        expected_columns = [f"{mode}_mean_{n:04d}" for n in range(width)]
        if modalities[mode] != expected_columns:
            raise ValueError("Pooled image feature order mismatch")
        actual = features.loc[
            features.Patient_ID.eq(patient), expected_columns
        ].to_numpy(float)[0]
        # Older float32 pandas exports print fewer decimals. New builders should
        # render float32 values through float64 columns for exact round trips.
        error = float(np.max(np.abs(actual - pooled.astype(float))))
        if not np.isfinite(actual).all() or error > 1e-6:
            raise ValueError(
                "Exported patient vector does not equal the actual valid-item bag mean"
            )
        maximum_roundtrip_error = max(maximum_roundtrip_error, error)
        observed[key] = True
    expected_available = {
        "cli": np.isfinite(original[modalities["cli"]].to_numpy(float)).any(1)
    }
    for mode in ("wsi", "rad"):
        if mode not in modalities:
            continue
        present = np.array([(patient, mode) in observed for patient in ids])
        values = features[modalities[mode]].to_numpy(float)
        if (
            not np.isnan(values[~present]).all()
            or not np.isfinite(values[present]).all()
        ):
            raise ValueError(
                "Absent images must be entirely NaN and observed images finite"
            )
        expected_available[mode] = present
    if contract.get("rna"):
        rna = contract["rna"]
        if sha(rna["path"]) != rna["sha256"]:
            raise ValueError("RNA matrix source changed")
        with np.load(rna["path"], allow_pickle=False) as data:
            rna_ids = data["patient_ids"].astype(str)
            if (
                len(set(rna_ids)) != len(rna_ids)
                or not np.isfinite(data["values"]).all()
            ):
                raise ValueError("RNA identity/value contract invalid")
            expected_available["rna"] = np.isin(ids, rna_ids)
    for mode, present in expected_available.items():
        np.testing.assert_array_equal(
            available[f"{mode}_available"].to_numpy(bool), present
        )
    complete = np.stack(list(expected_available.values())).all(0)
    np.testing.assert_array_equal(
        available.all_requested_modalities_available.to_numpy(bool), complete
    )
    if int(complete.sum()) != contract["complete_requested_inputs"]["patients"]:
        raise ValueError("Complete-input patient count differs")
    report = {
        "status": "passed",
        "scope": "Independent source/identity/valid-item-mean/availability audit only; no cohort admission or training authorization",
        "patients": len(ids),
        "recorded_os_deaths": int(original.event.sum()),
        "verified_image_bags": len(observed),
        "strict_ct_layout_checks": strict_ct_checks,
        "technical_available_counts": {
            k: int(v.sum()) for k, v in expected_available.items()
        },
        "complete_input_patient_ids": sorted(original.loc[complete, "Patient_ID"]),
        "complete_input_patients": int(complete.sum()),
        "complete_input_deaths": int(original.loc[complete, "event"].sum()),
        "maximum_float32_csv_roundtrip_error": maximum_roundtrip_error,
        "cohort_contract_sha256": sha(contract_path),
        "validator_sha256": sha(__file__),
        "prospective_prediction_eligible": False,
        "model_fitting_authorized_by_this_audit": False,
    }
    (folder / "independent_input_validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-root", type=Path, required=True)
    print(json.dumps(validate(parser.parse_args().cohort_root), indent=2))


if __name__ == "__main__":
    main()
