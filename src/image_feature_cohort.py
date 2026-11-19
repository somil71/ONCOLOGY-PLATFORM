"""Strict patient joins and valid-item mean pooling for frozen image features.

Pooling is outcome-independent. It is a vector baseline, not patch attention.
Clinical timing/admission remains separate from technical feature availability.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def checked_patient_id(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value
    ):
        raise ValueError(
            "Explicit patient ID must be nonempty and safe as one directory name"
        )
    return value


def checked_hash(value, algorithm="sha256"):
    length = 64 if algorithm == "sha256" else 32
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"Missing or malformed {algorithm} provenance digest")
    return value


def valid_item_mean(features, item_mask):
    """Pool finite valid rows only; inactive padding can contain arbitrary values."""
    features = np.asarray(features)
    item_mask = np.asarray(item_mask)
    if features.ndim != 2 or not features.shape[0] or not features.shape[1]:
        raise ValueError("A nonempty [items,features] image bag is required")
    if features.dtype.kind != "f":
        raise ValueError("Image embeddings must have floating-point dtype")
    if item_mask.shape != (len(features),) or not np.isin(item_mask, [0, 1]).all():
        raise ValueError(
            "One finite binary valid-item indicator per image item is required"
        )
    valid = item_mask.astype(bool)
    if not valid.any():
        raise ValueError(
            "An available image modality cannot contain an all-missing bag"
        )
    if not np.isfinite(features[valid]).all():
        raise ValueError("Active image features must be finite")
    return features[valid].mean(axis=0, dtype=np.float64).astype(np.float32)


def safe_predictor_names(columns):
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("A modality must declare nonempty unique predictor names")
    blocked = []
    for column in columns:
        name = str(column).casefold()
        if name in {
            "patient_id",
            "patientid",
            "time",
            "event",
            "os",
            "os.time",
            "os_time",
        } or any(
            token in name
            for token in [
                "last_contact",
                "follow_up",
                "followup",
                "death",
                "vital_status",
                "survival",
                "pfi",
                "dfi",
                "dss",
            ]
        ):
            blocked.append(column)
    if blocked:
        raise ValueError(
            f"Identifiers, outcomes and follow-up cannot be predictors: {blocked}"
        )


def _feature_audit(path, expected_patient, feature_path):
    audit = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if audit.get("patient_id") != expected_patient:
        raise ValueError("Image audit identity does not match the requested patient")
    if checked_hash(audit.get("features_sha256")) != file_hash(feature_path):
        raise ValueError("Image feature artifact checksum mismatch")
    return audit


def _bag(path, mask_key, expected_patient, dimensions):
    with np.load(path, allow_pickle=False) as data:
        if "features" not in data or mask_key not in data:
            raise ValueError(
                "The image bag requires features and an explicit valid-item mask"
            )
        features, mask = data["features"], data[mask_key]
        pooled = valid_item_mean(features, mask)
        if len(pooled) != dimensions:
            raise ValueError(
                "Frozen image encoder feature width differs from the study contract"
            )
        if "patient_id" in data:
            identity = data["patient_id"]
            if identity.shape != () or str(identity.item()) != expected_patient:
                raise ValueError(
                    "Embedded NPZ patient identity differs from the requested patient"
                )
        auxiliary = {
            name: data[name]
            for name in data.files
            if name not in {"features", mask_key}
        }
    return pooled, features.shape, mask.astype(bool), auxiliary


def load_wsi_bag(
    feature_path, audit_path, patient, source_record, source_manifest_sha256
):
    """Verify a GDC-selected specimen, extraction contract and frozen WSI bag."""
    checked_patient_id(patient)
    audit = _feature_audit(audit_path, patient, feature_path)
    if audit.get("status") != "verified_extracted_features":
        raise ValueError("WSI extraction audit is not successfully completed")
    if source_record.get("Patient_ID") != patient or source_record.get(
        "file_id"
    ) != audit.get("file_id"):
        raise ValueError(
            "WSI patient/specimen assignment differs from the pinned GDC selection"
        )
    source_md5 = checked_hash(audit.get("source_md5"), "md5")
    if source_record.get("md5sum") != source_md5:
        raise ValueError("WSI source digest differs from the pinned GDC selection")
    cfg = audit["configuration"]
    if cfg.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("WSI selection manifest changed after feature extraction")
    for key in ["weight_sha256", "feature_code_sha256", "driver_code_sha256"]:
        checked_hash(cfg.get(key))
    if not isinstance(cfg.get("transform"), dict) or not cfg.get("weights"):
        raise ValueError("Missing WSI encoder/preprocessing contract")
    expected_cache = hashlib.sha256(
        json.dumps({"source_md5": source_md5, "config": cfg}, sort_keys=True).encode()
    ).hexdigest()
    if audit.get("cache_key") != expected_cache:
        raise ValueError(
            "WSI cache key does not bind the recorded source and extraction contract"
        )
    pooled, shape, valid, auxiliary = _bag(feature_path, "valid_items", patient, 768)
    if list(shape) != audit.get("feature_shape"):
        raise ValueError("WSI feature shape differs from its extraction audit")
    qc = audit["qc"]
    if int(qc["selected_patches"]) != int(valid.sum()):
        raise ValueError("WSI valid-item count differs from selected-patch metadata")
    coordinates = auxiliary.get("coordinates_level0")
    if coordinates is None or coordinates.shape != (shape[0], 2):
        raise ValueError("WSI patch coordinates are missing or misaligned")
    if (
        not np.isfinite(coordinates[valid]).all()
        or (coordinates[valid] < 0).any()
        or len(np.unique(coordinates[valid], axis=0)) != int(valid.sum())
    ):
        raise ValueError(
            "Active WSI coordinates must be finite, nonnegative and distinct"
        )
    if not np.array_equal(coordinates[valid], np.asarray(qc["coordinates_level0"])):
        raise ValueError("WSI patch coordinates differ between the bag and its audit")
    if not all(
        np.isfinite(qc[key]) and qc[key] > 0
        for key in ["source_mpp_x", "source_mpp_y", "target_mpp"]
    ):
        raise ValueError("Recorded physical WSI resolution is required")
    if qc["target_mpp"] != cfg["target_mpp"]:
        raise ValueError("WSI physical preprocessing differs from its cache contract")
    extraction = {
        key: cfg[key]
        for key in [
            "weights",
            "weight_sha256",
            "transform",
            "target_mpp",
            "max_patches",
            "min_patches",
            "seed",
            "feature_code_sha256",
            "driver_code_sha256",
        ]
    }
    return pooled, {
        "patient_id": patient,
        "modality": "wsi",
        "source_file_id": audit["file_id"],
        "source_md5": source_md5,
        "features_sha256": audit["features_sha256"],
        "audit_sha256": file_hash(audit_path),
        "valid_items": int(valid.sum()),
        "stored_items": int(shape[0]),
        "extraction_contract": extraction,
        "selection_manifest_sha256": source_manifest_sha256,
        "physical_source_mpp": [qc["source_mpp_x"], qc["source_mpp_y"]],
        "technical_availability": True,
        "prospective_prediction_eligible": False,
        "clinical_timing_verified": False,
    }


def load_ct_bag(feature_path, audit_path, patient):
    """Verify the explicit patient/series identity and frozen CT slice bag."""
    checked_patient_id(patient)
    audit = _feature_audit(audit_path, patient, feature_path)
    if audit.get("all_slice_identity_and_geometry_verified") is not True:
        raise ValueError(
            "CT patient/series identity and physical ordering must be verified"
        )
    checked_hash(audit.get("selection_sha256"))
    checked_hash(audit.get("weights_sha256"))
    code_hashes = audit.get("implementation_sha256")
    if not isinstance(code_hashes, dict) or not code_hashes:
        raise ValueError("Missing CT implementation provenance")
    for value in code_hashes.values():
        checked_hash(value)
    pooled, shape, valid, auxiliary = _bag(feature_path, "item_mask", patient, 512)
    if list(shape) != audit.get("feature_shape") or int(
        audit["selected_slices"]
    ) != int(valid.sum()):
        raise ValueError("CT feature shape or valid-item count differs from its audit")
    identity = auxiliary.get("series_uid")
    if (
        identity is None
        or identity.shape != ()
        or str(identity.item()) != audit.get("series_uid")
    ):
        raise ValueError(
            "CT embedded series identity does not match the audited selection"
        )
    positions = auxiliary.get("positions_mm")
    slices = audit["slices"]
    if (
        positions is None
        or positions.shape != (shape[0], 3)
        or len(slices) != int(valid.sum())
    ):
        raise ValueError("CT source slices and physical positions are misaligned")
    if not np.isfinite(positions[valid]).all() or len(
        np.unique(positions[valid], axis=0)
    ) != int(valid.sum()):
        raise ValueError("CT valid physical positions must be finite and distinct")
    np.testing.assert_allclose(
        positions[valid],
        np.asarray([s["position_mm"] for s in slices]),
        rtol=0,
        atol=1e-8,
    )
    for source in slices:
        checked_hash(source.get("sha256"))
    extraction = {
        key: audit[key]
        for key in [
            "encoder",
            "weights_sha256",
            "transform",
            "hu_windows_center_width",
            "window_encoding",
            "implementation_sha256",
        ]
    }
    return pooled, {
        "patient_id": patient,
        "modality": "rad",
        "series_uid": audit["series_uid"],
        "source_slice_hashes": [s["sha256"] for s in slices],
        "selection_sha256": audit["selection_sha256"],
        "features_sha256": audit["features_sha256"],
        "audit_sha256": file_hash(audit_path),
        "valid_items": int(valid.sum()),
        "stored_items": int(shape[0]),
        "extraction_contract": extraction,
        "technical_availability": True,
        "prospective_prediction_eligible": False,
        "clinical_review_complete": audit.get("clinical_review_complete", False),
        "clinical_timing_verified": False,
    }
