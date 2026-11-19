"""Reconstruct clinical/RNA-run predictions from raw rows and saved bundles.

Read-only scientific artifact validation; writes a separate report only.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.survival_metrics import bootstrap_concordance, evaluate_survival, harrell_c
from src.survival_models import HazardFusion


def reconstruct_neural(run, manifest, cohort, predictions):
    """Rebuild all neural inputs independently, including training gene selection."""
    prep = json.loads((run / "neural_preprocessing.json").read_text())
    raw = {}
    rna_path = manifest["arguments"].get("rna_npz")
    for mode, columns in manifest["features"].items():
        if mode == "rna" and rna_path:
            source = Path(rna_path)
            if (
                hashlib.sha256(source.read_bytes()).hexdigest()
                != manifest["rna_provenance"]["sha256"]
            ):
                raise ValueError("RNA source provenance mismatch")
            with np.load(source, allow_pickle=False) as matrix:
                ids = matrix["patient_ids"].astype(str)
                genes = matrix["gene_ids"].astype(str)
                values = matrix["values"]
                if len(set(ids)) != len(ids) or len(set(genes)) != len(genes):
                    raise ValueError("Nonunique RNA identity/feature order")
                lookup = {patient: i for i, patient in enumerate(ids)}
                joined = np.full((len(cohort), len(genes)), np.nan, dtype=np.float32)
                for i, patient in enumerate(cohort.Patient_ID):
                    if patient in lookup:
                        joined[i] = values[lookup[patient]]
            variance = np.nanvar(joined[cohort.split.eq("train")], axis=0)
            eligible = np.flatnonzero(np.isfinite(variance) & (variance > 0))
            selected = eligible[
                np.argsort(variance[eligible], kind="stable")[-len(columns) :]
            ]
            if genes[selected].tolist() != columns:
                raise ValueError(
                    "RNA selected genes/order differ from training-only variance reconstruction"
                )
            if manifest["rna_provenance"]["selected_gene_ids"] != columns:
                raise ValueError("RNA provenance and model feature contract differ")
            raw[mode] = joined[:, selected]
        else:
            raw[mode] = cohort[columns].to_numpy(float)
    available = {name: np.isfinite(values).any(1) for name, values in raw.items()}
    features = {}
    for name, values in raw.items():
        observed = values[cohort.split.eq("train").to_numpy() & available[name]].astype(
            float
        )
        median = np.nanmedian(observed, axis=0)
        filled = np.where(np.isfinite(observed), observed, median)
        mean, scale = filled.mean(0), filled.std(0)
        scale[scale < 1e-8] = 1
        for actual, expected in (
            (median, prep[name]["median"]),
            (mean, prep[name]["mean"]),
            (scale, prep[name]["scale"]),
        ):
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        features[name] = (
            (np.where(np.isfinite(values), values, median) - mean) / scale
        ).astype("float32")
    position = {patient: i for i, patient in enumerate(cohort.Patient_ID)}
    withheld_path = run / "inference_withholding_predictions.csv"
    withheld = pd.read_csv(withheld_path) if withheld_path.exists() else pd.DataFrame()
    results = {}
    torch.set_num_threads(2)
    for label in predictions.model.unique():
        first = run / f"{label}_seed{manifest['seeds'][0]}.pth"
        if not first.exists():
            continue
        saved = predictions.loc[predictions.model.eq(label)]
        ix = np.array([position[p] for p in saved.Patient_ID])
        if not cohort.iloc[ix].split.eq("test").all():
            raise ValueError("A prediction was not from the held-out partition")
        members, withholding_members = [], {}
        for seed in manifest["seeds"]:
            bundle = torch.load(
                run / f"{label}_seed{seed}.pth", map_location="cpu", weights_only=True
            )
            names = bundle["modalities"]
            expected = cohort.split.eq("test").to_numpy() & np.stack(
                [available[n] for n in names], 1
            ).any(1)
            if set(saved.Patient_ID) != set(cohort.loc[expected, "Patient_ID"]):
                raise ValueError(f"Unexpected eligible heldout population: {label}")
            tensors = {n: torch.from_numpy(features[n][ix]) for n in names}
            mask = torch.from_numpy(np.stack([available[n][ix] for n in names], 1))
            model = HazardFusion(
                bundle["dimensions"],
                bundle["boundaries"],
                fusion=bundle["fusion"],
                use_mask=bundle["use_mask"],
            )
            model.load_state_dict(bundle["state_dict"])
            members.append(
                model.predict_survival(tensors, mask, manifest["horizons_days"])
            )
            if len(withheld) and label in set(withheld.model):
                for kept, rows in withheld.loc[withheld.model.eq(label)].groupby(
                    "kept_modalities", sort=False
                ):
                    wi = np.array([position[p] for p in rows.Patient_ID])
                    wm = np.stack(
                        [available[n][wi] & (n in kept.split("+")) for n in names], 1
                    )
                    if not np.stack([available[n][wi] for n in names], 1).all():
                        raise ValueError(
                            "Withholding audit contains an incomplete baseline patient"
                        )
                    wp = model.predict_survival(
                        {n: torch.from_numpy(features[n][wi]) for n in names},
                        torch.from_numpy(wm),
                        manifest["horizons_days"],
                    )
                    withholding_members.setdefault(kept, []).append(wp)
        probabilities, spread = np.mean(members, axis=0), np.std(members, axis=0)
        horizons = manifest["horizons_days"]
        np.testing.assert_allclose(
            probabilities,
            saved[[f"survival_{int(h)}" for h in horizons]],
            atol=1e-6,
            rtol=0,
        )
        np.testing.assert_allclose(
            spread,
            saved[[f"ensemble_sd_{int(h)}" for h in horizons]],
            atol=1e-6,
            rtol=0,
        )
        np.testing.assert_allclose(
            -probabilities.mean(1), saved.risk, atol=1e-6, rtol=0
        )
        for kept, values in withholding_members.items():
            rows = withheld.loc[
                withheld.model.eq(label) & withheld.kept_modalities.eq(kept)
            ]
            wp = np.mean(values, axis=0)
            np.testing.assert_allclose(
                wp, rows[[f"survival_{int(h)}" for h in horizons]], atol=1e-6, rtol=0
            )
            np.testing.assert_allclose(-wp.mean(1), rows.risk, atol=1e-6, rtol=0)
        results[label] = {
            "patients": len(saved),
            "seed_checkpoints": manifest["seeds"],
            "training_only_preprocessing_and_gene_selection_match": True,
            "risk_survival_and_disagreement_reconstruction_matches": True,
            "withheld_input_subsets_reconstructed": list(withholding_members),
            "recomputed_harrell_c": harrell_c(saved.time, saved.event, saved.risk),
        }
    return results


def validate_recorded_metrics(run, manifest, cohort, predictions):
    """Recompute complete patient-bootstrap and supported horizon metrics."""
    recorded = json.loads((run / "metrics.json").read_text())
    reference = predictions.loc[
        predictions.model.eq("clinical_cox"), ["Patient_ID", "risk"]
    ].rename(columns={"risk": "reference_risk"})
    report = {}
    for label, saved in predictions.groupby("model", sort=False):
        expected = recorded[label]["test"]
        # These runs use complete CLI/RNA cohorts. More general missing-input
        # runs must reconstruct each configuration's eligible training cohort.
        if manifest["arguments"].get("rna_npz") and not manifest["arguments"].get(
            "complete_cohort"
        ):
            raise ValueError(
                "Metric reference for partially observed RNA cohorts requires explicit availability reconstruction"
            )
        train = cohort.loc[cohort.split.eq("train")]
        result = evaluate_survival(
            saved.time,
            saved.event,
            saved.risk,
            train_time=train.time,
            train_event=train.event,
            horizons=manifest["horizons_days"],
            survival_probabilities=saved[
                [f"survival_{int(h)}" for h in manifest["horizons_days"]]
            ].to_numpy(),
            n_bootstrap=manifest["arguments"]["bootstrap"],
            seed=manifest["arguments"].get("bootstrap_seed", 42),
        )
        np.testing.assert_allclose(
            result["harrell_c"], expected["harrell_c"], atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            result["bootstrap_95_ci"], expected["bootstrap_95_ci"], atol=1e-12, rtol=0
        )
        horizon_report = []
        for actual, prior in zip(result["horizons"], expected["horizons"]):
            if actual["status"] != prior["status"]:
                raise ValueError("Horizon support status changed during reconstruction")
            if actual["status"] == "evaluated":
                for field in ("ipcw_auc", "ipcw_brier", "uno_c_tau"):
                    np.testing.assert_allclose(
                        actual[field], prior[field], atol=1e-9, rtol=0
                    )
            horizon_report.append(
                {
                    "horizon": actual["horizon"],
                    "status": actual["status"],
                    "reason": actual.get("reason"),
                }
            )
        paired = saved.merge(reference, on="Patient_ID", validate="one_to_one")
        if "paired_vs_clinical_cox" in recorded[label]:
            delta = bootstrap_concordance(
                paired.time,
                paired.event,
                paired.risk,
                reference_risk=paired.reference_risk,
                n_bootstrap=manifest["arguments"]["bootstrap"],
                seed=manifest["arguments"].get("bootstrap_seed", 42),
            )
            old = recorded[label]["paired_vs_clinical_cox"]
            for field in (
                "model_minus_reference_c",
                "paired_difference_bootstrap_95_ci",
            ):
                np.testing.assert_allclose(delta[field], old[field], atol=1e-12, rtol=0)
        report[label] = {
            "complete_risk_ci_horizon_and_paired_delta_match": True,
            "horizon_support": horizon_report,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--legacy-external-float64",
        action="store_true",
        help="Explicitly reconstruct archived external exports made before the float32 input contract; never alters saved predictions",
    )
    args = parser.parse_args()
    run = args.run
    manifest = json.loads((run / "run_manifest.json").read_text())
    rawpath = Path(manifest["arguments"]["input"])
    raw = pd.read_csv(rawpath)
    splits = pd.read_csv(run / "patient_splits.csv")
    predictions = pd.read_csv(run / "test_predictions.csv")
    if hashlib.sha256(rawpath.read_bytes()).hexdigest() != manifest["input_sha256"]:
        raise ValueError("Input provenance mismatch")
    if (
        hashlib.sha256((run / "patient_splits.csv").read_bytes()).hexdigest()
        != manifest["split_sha256"]
    ):
        raise ValueError("Split provenance mismatch")
    if raw.Patient_ID.duplicated().any() or splits.Patient_ID.duplicated().any():
        raise ValueError("Duplicate patient identity")
    locked_protocol_checks = {}
    fixed_path = manifest["arguments"].get("split_csv")
    if fixed_path:
        fixed_path = Path(fixed_path)
        fixed_hash = hashlib.sha256(fixed_path.read_bytes()).hexdigest()
        if fixed_hash != manifest["fixed_split_source"]["sha256"]:
            raise ValueError("Locked source patient assignments changed")
        fixed = pd.read_csv(fixed_path)
        if (
            set(fixed.columns) != {"Patient_ID", "split"}
            or fixed.Patient_ID.duplicated().any()
        ):
            raise ValueError("Invalid source patient assignment ledger")
        if set(fixed.Patient_ID) != set(raw.Patient_ID):
            raise ValueError(
                "Original model input IDs differ from frozen candidate assignments"
            )
        assigned = splits.merge(
            fixed, on="Patient_ID", suffixes=("", "_locked"), validate="one_to_one"
        )
        if (
            len(assigned) != len(splits)
            or not assigned.split.eq(assigned.split_locked).all()
        ):
            raise ValueError("Retained patient assignment differs from locked split")
        excluded_record = manifest.get("qc_exclusions", [])
        if isinstance(excluded_record, int):
            exclusions = pd.read_csv(run / "excluded_patients.csv")
            if (
                len(exclusions) != excluded_record
                or exclusions.Patient_ID.duplicated().any()
            ):
                raise ValueError("QC exclusion count/identity mismatch")
            excluded = set(exclusions.Patient_ID)
        else:
            excluded = {row["Patient_ID"] for row in excluded_record}
        if set(raw.Patient_ID) - set(splits.Patient_ID) != excluded:
            raise ValueError(
                "Excluded source patients lack matching explicit QC reasons"
            )
        locked_protocol_checks["fixed_patient_assignments_and_exclusions_match"] = True
        protocol_path = manifest["arguments"].get("study_protocol")
        if protocol_path:
            protocol_path = Path(protocol_path)
            protocol = json.loads(protocol_path.read_text())
            if (
                hashlib.sha256(protocol_path.read_bytes()).hexdigest()
                != manifest["study_protocol"]["sha256"]
            ):
                raise ValueError("Analysis protocol changed after model training")
            clinical_source = Path(protocol["clinical_source"])
            if (
                hashlib.sha256(clinical_source.read_bytes()).hexdigest()
                != protocol["clinical_source_sha256"]
            ):
                raise ValueError("Protocol original clinical source changed")
            if fixed_hash not in {row["sha256"] for row in protocol["fold_ledgers"]}:
                raise ValueError(
                    "Assignment ledger is not one of the protocol-locked folds"
                )
            clinical = pd.read_csv(clinical_source)
            observed = raw[["Patient_ID", "time", "event"]].merge(
                clinical[["Patient_ID", "time", "event"]],
                on="Patient_ID",
                suffixes=("", "_locked"),
                validate="one_to_one",
            )
            if len(observed) != len(raw):
                raise ValueError(
                    "Model input contains patients absent from protocol source"
                )
            np.testing.assert_array_equal(
                observed[["time", "event"]], observed[["time_locked", "event_locked"]]
            )
            if (
                protocol["endpoint"]["prospective_prediction_time_eligibility_verified"]
                is not False
            ):
                raise ValueError(
                    "This KIRC protocol must remain explicitly retrospective"
                )
            # Read original CDR directly; do not validate one derived table only
            # against another derived copy of the same potentially wrong labels.
            cdr = pd.read_excel(
                "data/tcga-brca/TCGA-CDR-SupplementalTableS1.xlsx",
                sheet_name="TCGA-CDR",
            )
            original = cdr.loc[cdr.type.eq("KIRC")].set_index(
                "bcr_patient_barcode", verify_integrity=True
            )
            np.testing.assert_array_equal(
                original.loc[raw.Patient_ID, "OS.time"], raw.time
            )
            np.testing.assert_array_equal(original.loc[raw.Patient_ID, "OS"], raw.event)
            locked_protocol_checks.update(
                {
                    "protocol_hash_and_clinical_source_match": True,
                    "outcomes_equal_original_cdr": True,
                    "prospective_prediction_eligible": False,
                    "protocol_id": protocol["protocol_id"],
                }
            )
    cohort = splits.merge(
        raw, on=["Patient_ID", "time", "event"], validate="one_to_one"
    )
    if len(cohort) != len(splits):
        raise ValueError("Some split outcomes do not match raw data")
    train = cohort[cohort.split.eq("train")]
    results = {}
    horizons = manifest["horizons_days"]
    for artifact, expected_hash in manifest.get("artifacts_sha256", {}).items():
        if hashlib.sha256((run / artifact).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Run artifact hash mismatch: {artifact}")
    for name in ["clinical_cox", "age_only_cox"]:
        if name not in predictions.model.unique():
            continue
        bundle = json.loads((run / f"{name}_bundle.json").read_text())
        columns = bundle["columns"]
        pre = bundle["preprocessing"]
        observed = train[columns].to_numpy(float)
        median = np.nanmedian(observed, axis=0)
        filled = np.where(np.isfinite(observed), observed, median)
        mean = filled.mean(0)
        scale = filled.std(0)
        scale[scale < 1e-8] = 1
        for actual, expected in [
            (median, pre["median"]),
            (mean, pre["mean"]),
            (scale, pre["scale"]),
        ]:
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        saved = predictions[predictions.model.eq(name)]
        data = saved.merge(
            raw, on=["Patient_ID", "time", "event"], validate="one_to_one"
        )
        x = data[columns].to_numpy(float)
        x = (
            ((np.where(np.isfinite(x), x, median) - mean) / scale)
            .astype(pre.get("output_dtype", "float32"))
            .astype(float)
        )
        risk = x @ np.array(bundle["coefficients"])
        np.testing.assert_allclose(risk, data.risk, atol=1e-10, rtol=0)
        times = np.array(bundle["event_times"])
        hazard = np.array(bundle["baseline_cumulative_hazard"])
        indices = np.searchsorted(times, horizons, side="right") - 1
        cumulative = np.where(indices >= 0, hazard[np.maximum(indices, 0)], 0)
        survival = np.exp(-np.exp(risk)[:, None] * cumulative[None, :])
        np.testing.assert_allclose(
            survival, data[[f"survival_{int(h)}" for h in horizons]], atol=1e-10, rtol=0
        )
        results[name] = {
            "patients": len(data),
            "training_only_preprocessing_matches": True,
            "risk_and_survival_reconstruction_matches": True,
            "recomputed_harrell_c": harrell_c(data.time, data.event, risk),
        }
    if (run / "neural_preprocessing.json").exists():
        results.update(reconstruct_neural(run, manifest, cohort, predictions))
    metric_checks = validate_recorded_metrics(run, manifest, cohort, predictions)
    external_path = manifest["arguments"].get("external")
    if external_path and (run / "external_age_only_predictions.csv").exists():
        external = pd.read_csv(external_path)
        external_predictions = pd.read_csv(run / "external_age_only_predictions.csv")
        candidates = [
            name
            for name in ["age_only_cox", "clinical_cox"]
            if (run / f"{name}_bundle.json").exists()
            and json.loads((run / f"{name}_bundle.json").read_text())["columns"]
            == ["age_years"]
        ]
        if not candidates:
            raise ValueError("No portable age-only bundle for external predictions")
        name = candidates[0]
        bundle = json.loads((run / f"{name}_bundle.json").read_text())
        recorded = json.loads((run / "metrics.json").read_text())[name]
        if (
            hashlib.sha256(Path(external_path).read_bytes()).hexdigest()
            != recorded["external_input_sha256"]
        ):
            raise ValueError("External source hash mismatch")
        data = external_predictions.merge(
            external, on=["Patient_ID", "time", "event"], validate="one_to_one"
        )
        if len(data) != len(external_predictions):
            raise ValueError("External outcomes fail raw-source join")
        prep = bundle["preprocessing"]
        values = data[bundle["columns"]].to_numpy(float)
        external_dtype = (
            "float64"
            if args.legacy_external_float64
            else prep.get("output_dtype", "float32")
        )
        x = (
            (
                (np.where(np.isfinite(values), values, prep["median"]) - prep["mean"])
                / prep["scale"]
            )
            .astype(external_dtype)
            .astype(float)
        )
        risk = x @ np.array(bundle["coefficients"])
        np.testing.assert_allclose(risk, data.risk, atol=1e-10, rtol=0)
        indices = np.searchsorted(bundle["event_times"], horizons, side="right") - 1
        hazard = np.array(bundle["baseline_cumulative_hazard"])
        cumulative = np.where(indices >= 0, hazard[np.maximum(indices, 0)], 0)
        survival = np.exp(-np.exp(risk)[:, None] * cumulative[None, :])
        np.testing.assert_allclose(
            survival, data[[f"survival_{int(h)}" for h in horizons]], atol=1e-10, rtol=0
        )
        results["external_frozen_age_only_cox"] = {
            "patients": len(data),
            "events": int(data.event.sum()),
            "source_hash_and_outcomes_match": True,
            "frozen_bundle_risk_and_survival_match": True,
            "external_feature_dtype": external_dtype,
            "legacy_external_precision_override": args.legacy_external_float64,
            "recomputed_harrell_c": harrell_c(data.time, data.event, risk),
        }
    report = {
        "status": "passed",
        "scope": "Independent row, preprocessing and bundle reconstruction; not clinical efficacy validation",
        "models": results,
        "metric_reconstruction": metric_checks,
        "locked_protocol_checks": locked_protocol_checks,
        "recorded_artifact_hashes_verified": len(manifest.get("artifacts_sha256", {})),
        "source_run_manifest_sha256": hashlib.sha256(
            (run / "run_manifest.json").read_bytes()
        ).hexdigest(),
        "current_code_hash_matches_recorded": {
            name: Path(name).exists()
            and hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
            for name, expected in manifest.get("code_sha256", {}).items()
        },
        "code_hash_scope": "Records current source agreement separately; a changed runner is not an exact source rerun. Prediction reconstruction uses saved states and independently rebuilt preprocessing.",
    }
    (run / "independent_prediction_reconstruction.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
