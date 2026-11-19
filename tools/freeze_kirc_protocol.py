"""Freeze candidate KIRC patient assignments and a retrospective analysis plan.

No feature fitting, model training, prediction or performance selection occurs.
Image QC may remove candidates only through a separate, explicit exclusion ledger;
retained patients keep these original assignments. Existing locks are immutable.
"""

from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_assignments(frame):
    frame = frame.sort_values("Patient_ID").reset_index(drop=True)
    if frame.Patient_ID.duplicated().any() or not frame.event.isin([0, 1]).all():
        raise ValueError("Unique patients and binary recorded events required")
    if not np.isfinite(frame.time).all() or not (frame.time > 0).all():
        raise ValueError("Positive recorded OS times required")
    assignments, counts = {}, {}
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (development, test) in enumerate(outer.split(frame, frame.event), 1):
        train, validation = train_test_split(
            development,
            test_size=0.2,
            random_state=1042 + fold,
            stratify=frame.event.iloc[development],
        )
        ledger = frame[["Patient_ID"]].copy()
        ledger["split"] = ""
        counts[str(fold)] = {}
        for name, ix in {
            "train": train,
            "validation": validation,
            "test": test,
        }.items():
            ledger.loc[ix, "split"] = name
            counts[str(fold)][name] = {
                "patients": len(ix),
                "deaths": int(frame.event.iloc[ix].sum()),
            }
        if ledger.split.eq("").any():
            raise AssertionError("Unassigned patient")
        assignments[fold] = ledger
    return assignments, counts


def main():
    source = ROOT / "data/processed/tcga_kirc_clinical.csv"
    output = ROOT / "data/processed/kirc-fixed-folds"
    protocol_path = ROOT / "config/kirc_retrospective_os_protocol.json"
    if protocol_path.exists():
        recorded = json.loads(protocol_path.read_text())
        if recorded["clinical_source_sha256"] != sha256(source):
            raise ValueError(
                "Frozen clinical source changed; preserve this protocol and declare a new version"
            )
        for record in recorded["fold_ledgers"]:
            if sha256(ROOT / record["path"]) != record["sha256"]:
                raise ValueError("A frozen patient-fold ledger changed")
        print(
            json.dumps(
                {"status": "existing_lock_verified", "protocol": str(protocol_path)}
            )
        )
        return
    frame = pd.read_csv(source).sort_values("Patient_ID").reset_index(drop=True)
    assignments, counts = build_assignments(frame)
    output.mkdir(parents=True, exist_ok=True)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    ledgers = []
    for fold, ledger in assignments.items():
        path = output / f"fold_{fold:02d}.csv"
        if path.exists():
            raise ValueError("Unregistered existing ledger: refuse overwrite")
        ledger.to_csv(path, index=False, lineterminator="\n")
        ledgers.append(
            {
                "fold": fold,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
            }
        )
    modalities = ["cli", "rna", "wsi", "rad"]
    subsets = [
        list(x)
        for size in range(1, 5)
        for x in itertools.combinations(modalities, size)
    ]
    protocol = {
        "protocol_id": "KIRC-retrospective-OS-v1",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "Locked candidate analysis plan before KIRC performance evaluation; full image acquisition/QC pending",
        "study_scope": "Separate TCGA-KIRC retrospective OS association; no BRCA pooling, pan-cancer generalization or prospective diagnosis-time claim",
        "clinical_source": source.relative_to(ROOT).as_posix(),
        "clinical_source_sha256": sha256(source),
        "candidate_patients": len(frame),
        "candidate_os_deaths": int(frame.event.sum()),
        "endpoint": {
            "name": "OS",
            "source": "Original TCGA-CDR OS/OS.time",
            "unit": "days",
            "time_origin": "Recorded CDR diagnosis origin; not shifted by BCR receipt or DICOM calendar dates",
            "postcollection_landmark": False,
            "prospective_prediction_time_eligibility_verified": False,
        },
        "predictors": {
            "primary_clinical": ["age_years"],
            "retrospective_sensitivity_clinical": ["age_years", "pathologic_stage"],
            "forbidden": "Outcome, death, followup, last contact, receipt timing and identifiers as predictors; stages limited to declared sensitivity study",
            "rna": "Selected primary-tumour log1p TPM; variable-gene selection and all transforms fitted on training partition only",
            "images": "Actual source-verified primary WSI and CT bags; pooled mean of VALID items is shared-feature baseline, not MCAT or patch cross-attention",
        },
        "timing_definition": {
            "days_to_collection": "BCR/processing-centre receipt, not patient sampling",
            "days_to_sample_procurement": "Procedure/removal from patient",
            "rna_procurement": "Unknown; cannot assume baseline",
            "unknown_timing_policy": "Retain only for explicitly retrospective association with sensitivity/limitations; no clinical deployment eligibility inferred",
        },
        "partitions": {
            "unit": "patient",
            "outer_folds": 5,
            "outer_stratification": "recorded OS event",
            "outer_seed": 42,
            "ordering": "ascending Patient_ID before assignment",
            "inner_validation_fraction_of_outer_development": 0.2,
            "inner_seed_formula": "1042 + 1-based outer fold",
            "fold_counts": counts,
            "immutable_after_image_qc": True,
        },
        "fold_ledgers": ledgers,
        "qc_exclusions": {
            "rule": "Outcome-blind technical/identity/histology QC; record patient, source hash and reason before model evaluation",
            "retain_original_assignment": True,
            "no_event_based_replacement_or_resplitting": True,
            "compare_models_on_identical_admitted_patients": True,
            "if_fewer_than_two_events_in_any_partition": "Mark affected experiment data_constrained; no arbitrary resplit",
        },
        "four_input_admission": [
            "Each patient has exact case-linked CLI/RNA/primary diagnostic WSI/CT identity",
            "Raw source checksums or verified extraction provenance plus feature hashes",
            "Finite nonempty feature bags with binary valid-item masks and declared physical preprocessing",
            "CT patient/series/frame, physical ordering, geometry, HU, protocol/anatomy/phase QC recorded",
            "WSI primary sample, known physical MPP, deterministic tissue/patch QC and weights recorded",
            "No all-missing input; missing values and padded items never treated as observations",
        ],
        "hypotheses": {
            "primary_H1": "Attention with explicit masking and training modality dropout0.3 improves missing-input robustness versus identical masked attention with dropout0",
            "primary_subset_panel": subsets,
            "primary_summary": "Equal mean Harrell C across all15 nonempty subsets within each outer fold, then equal mean across5outer folds",
            "primary_paired_contrast": "dropout0.3 minus dropout0; identical patients/subsets/folds and seed ensemble",
            "complete_input_guardrail": "Report complete-input C, Brier and calibration separately; an aggregate cannot hide degradation",
            "secondary": "Clinical/unimodal/concatenation/mean comparisons, dropout0.1/0.5, stage sensitivity, uncertainty and phase/site analyses exploratory",
        },
        "training": {
            "seeds": [42, 123, 2026],
            "selection": "Inner-validation censored likelihood; no outer performance tuning",
            "transforms": "Training-only imputation/scaling/gene selection/dictionaries; frozen externally pretrained encoders permitted with provenance",
            "all_missing_dropout": "Restore one uniformly selected actually available input; do not always retain clinical input",
        },
        "evaluation": {
            "primary_metric": "Harrell C, higher risk means earlier death",
            "horizons_days": [365, 1095, 1825],
            "horizon_policy": "Descriptive IPCW AUC/Brier/Uno and calibration only with positive censoring support, observed cases and controls; unsupported->not_evaluated",
            "report_counts": "Per-fold patients/deaths, cases/controls/early censoring, G(t), weight extremes and valid bootstrap replicates",
            "sparse_warning_convention": "Fewer than10 cases or10 controls flagged descriptive instability; not a universal statistical adequacy rule",
            "fold_summary": "Fold-specific risks/metrics; no arbitrary pooling of unscaled Cox risk across fitted folds",
            "bootstrap": {
                "replicates": 1000,
                "seed": 20261118,
                "unit": "patient, paired across models/subsets within outer fold",
                "scope": "Conditional on fitted models/fixed folds; does not cover training-selection uncertainty or justify independent-fold significance claims",
            },
            "censoring": "Training-reference reverse KM per internal fold; independent censoring assumption and weight diagnostics disclosed",
            "multiplicity": "One primary H1 contrast; other comparisons exploratory, no uncorrected multi-contrast superiority claim",
            "external": "No independent paired four-input KIRC cohort currently verified",
        },
        "uncertainty": {
            "baseline": "Censored survival likelihood and3independently trained seeds; survival-probability SD is empirical disagreement",
            "validation": "Held-out IPCW error association, calibration, selective risk and missing-input stress with intervals",
            "edl": "Not validated; no Dirichlet classification loss on continuous risk/followup, no confidence claim from evidence magnitude",
        },
        "published_baselines": {
            "MultiSurv": "Pinned author mRNA-only model/loss feasible; shared-cohort preprocessing/training adaptations recorded separately. Official model has no CT branch and current pooled ViT/RAD vectors do not reproduce its WSI/clinical pathways",
            "MCAT": "Requires actual patch bags and six curated genomic functional groups with author source/config; pooled four tokens are incompatible",
            "PathomicFusion": "Requires original pathology/cell-graph/genomic representations and tensor gating; ordinary concatenation is not a reproduction",
        },
    }
    protocol_path.write_text(
        json.dumps(protocol, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "locked",
                "protocol": str(protocol_path),
                "patients": len(frame),
                "deaths": int(frame.event.sum()),
                "fold_counts": counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
