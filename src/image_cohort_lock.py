"""Read-only admission gate for a root-locked retrospective image experiment.

This module never creates a lock or selects patients. Root must first finish the
outcome-blind availability/QC review and write the concrete hash-bound decision.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_image_cohort_lock(
    lock_path, input_path, modalities_path, protocol_path, rna_npz_path, complete_cohort
):
    if not complete_cohort:
        raise ValueError(
            "The locked primary image comparison requires identical complete-input patients"
        )
    lock_path = Path(lock_path)
    lock = json.loads(lock_path.read_text())
    if (
        lock.get("schema") != "cancerfusion.image-cohort-lock.v1"
        or lock.get("decision") != "locked_for_retrospective_complete_input_experiment"
    ):
        raise ValueError(
            "A concrete post-QC image cohort lock is required; templates/drafts cannot authorize fitting"
        )
    if (
        lock.get("prospective_prediction_eligible") is not False
        or lock.get("outcome_blind_qc_confirmed") is not True
    ):
        raise ValueError(
            "Image admission must explicitly remain retrospective with outcome-blind QC"
        )
    bindings = {
        "patient_features": input_path,
        "modalities": modalities_path,
        "study_protocol": protocol_path,
        "rna_matrix": rna_npz_path,
    }
    for name, path in bindings.items():
        if path is None or sha(path) != lock["sources"][name]["sha256"]:
            raise ValueError(f"Locked image experiment source mismatch: {name}")
    contract_path = Path(lock["sources"]["cohort_contract"]["path"])
    audit_path = Path(lock["sources"]["independent_input_audit"]["path"])
    if (
        sha(contract_path) != lock["sources"]["cohort_contract"]["sha256"]
        or sha(audit_path) != lock["sources"]["independent_input_audit"]["sha256"]
    ):
        raise ValueError("Image contract/independent input audit changed")
    contract, audit = (
        json.loads(contract_path.read_text()),
        json.loads(audit_path.read_text()),
    )
    if audit.get("status") != "passed" or audit.get("cohort_contract_sha256") != sha(
        contract_path
    ):
        raise ValueError(
            "Independent source-to-pooled-input reconstruction has not passed"
        )
    for name, path in (
        ("patient_features.csv", input_path),
        ("modalities.json", modalities_path),
    ):
        if sha(path) != contract["artifacts_sha256"][name]:
            raise ValueError(
                "Locked input differs from the independently audited image contract"
            )
    if sha(rna_npz_path) != contract["rna"]["sha256"]:
        raise ValueError("Locked RNA matrix differs from the image cohort contract")
    features = pd.read_csv(input_path)
    modalities = json.loads(Path(modalities_path).read_text())
    if set(modalities) != {"cli", "wsi", "rad"}:
        raise ValueError(
            "Expected explicit CLI/WSI/CT vectors; RNA is joined separately by patient identity"
        )
    if features.Patient_ID.duplicated().any():
        raise ValueError("Duplicate image experiment patient")
    with np.load(rna_npz_path, allow_pickle=False) as matrix:
        ids = matrix["patient_ids"].astype(str)
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate RNA patient identity")
        rna_ids = set(ids[np.isfinite(matrix["values"]).any(1)])
    present = np.stack(
        [
            np.isfinite(features[cols].to_numpy(float)).any(1)
            for cols in modalities.values()
        ]
    ).all(0)
    present &= features.Patient_ID.isin(rna_ids).to_numpy()
    admitted = sorted(features.loc[present, "Patient_ID"])
    if (
        admitted != sorted(lock.get("admitted_patient_ids", []))
        or admitted != audit["complete_input_patient_ids"]
    ):
        raise ValueError(
            "Root-locked complete-input identities differ from independently verified availability"
        )
    if (
        len(admitted) != lock["admitted_patients"]
        or int(features.loc[present, "event"].sum()) != lock["admitted_os_deaths"]
    ):
        raise ValueError(
            "Root-locked patient/death counts differ from exact source outcomes"
        )
    if contract["clinical_source"]["sha256"] != lock["candidate_clinical_sha256"]:
        raise ValueError("Image cohort changed its candidate clinical source")
    if (
        lock.get("patient_fold_policy")
        != "retain_original_locked_assignments_no_resplitting"
    ):
        raise ValueError(
            "Original patient assignments must survive image QC exclusions"
        )
    return {
        "path": str(lock_path),
        "sha256": sha(lock_path),
        "admitted_patient_ids": admitted,
        "admitted_patients": len(admitted),
        "admitted_os_deaths": lock["admitted_os_deaths"],
        "scope": "Root-locked retrospective complete-input comparison; clinical timing/validity not established",
        "prospective_prediction_eligible": False,
    }
