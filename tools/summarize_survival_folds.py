"""Separate locked within-fold survival summaries from secondary pooled OOF scores.

Bootstrap intervals condition on the saved fitted models. They do not include
retraining variability from overlapping development partitions.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.survival_metrics import bootstrap_concordance, harrell_c
from tools.run_survival_experiments import dump


def pairs(time, event):
    return np.where(
        event[:, None]
        & (
            (time[:, None] < time[None, :])
            | ((time[:, None] == time[None, :]) & ~event[None, :])
        )
    )


def credits(risk, earlier, later):
    difference = np.asarray(risk)[earlier] - np.asarray(risk)[later]
    return (difference > 1e-8).astype(float) + 0.5 * (np.abs(difference) <= 1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("Positive bootstrap count required")
    folders = sorted(p for p in args.runs_root.glob("fold_*") if p.is_dir())
    if len(folders) != 5:
        raise ValueError(
            "This locked study requires exactly five completed outer folds"
        )
    oof, fold_rows, fold_payloads, uncertainty, support = [], [], [], {}, []
    manifests, source_hashes = [], {}
    models = None
    panel_models = ["fusion_attention", "attention_dropout_03"]
    panel_names = None
    for folder in folders:
        manifest = json.loads((folder / "run_manifest.json").read_text())
        if manifest["status"] != "completed":
            raise ValueError(f"Incomplete fold: {folder.name}")
        for name, expected in manifest["artifacts_sha256"].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Changed source artifact: {folder.name}/{name}")
        manifests.append(manifest)
        source_hashes[folder.name] = {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in [
                "run_manifest.json",
                "metrics.json",
                "test_predictions.csv",
                "patient_splits.csv",
                "inference_withholding_predictions.csv",
            ]
        }
        metrics = json.loads((folder / "metrics.json").read_text())
        predictions = pd.read_csv(folder / "test_predictions.csv")
        withheld = pd.read_csv(folder / "inference_withholding_predictions.csv")
        if predictions.duplicated(["model", "Patient_ID"]).any():
            raise ValueError("Duplicate model/patient test predictions")
        current_models = sorted(predictions.model.unique())
        if models is None:
            models = current_models
        if current_models != models:
            raise ValueError(
                "Different model panels across folds require a separate constrained study"
            )
        base = (
            predictions[predictions.model.eq("clinical_cox")]
            .sort_values("Patient_ID")
            .reset_index(drop=True)
        )
        time, event = base.time.to_numpy(), base.event.to_numpy(bool)
        earlier, later = pairs(time, event)
        model_credits = []
        for model in models:
            rows = predictions[predictions.model.eq(model)].set_index("Patient_ID")
            if set(rows.index) != set(base.Patient_ID):
                raise ValueError(
                    "Primary model comparisons must use identical admitted test patients"
                )
            rows = rows.loc[base.Patient_ID].reset_index()
            np.testing.assert_array_equal(
                rows[["time", "event"]], base[["time", "event"]]
            )
            c = harrell_c(rows.time, rows.event, rows.risk)
            np.testing.assert_allclose(
                c, metrics[model]["test"]["harrell_c"], rtol=0, atol=1e-12
            )
            model_credits.append(credits(rows.risk.to_numpy(), earlier, later))
            fold_rows.append(
                {
                    "fold": folder.name,
                    "model": model,
                    "test_patients": len(rows),
                    "test_deaths": int(event.sum()),
                    "within_fold_c": c,
                    "bootstrap_lower": metrics[model]["test"]["bootstrap_95_ci"][0],
                    "bootstrap_upper": metrics[model]["test"]["bootstrap_95_ci"][1],
                }
            )
            rows["fold"] = folder.name
            rows["oof_common_probability_risk"] = -rows[
                ["survival_365", "survival_1095", "survival_1825"]
            ].mean(axis=1)
            oof.append(rows)
            if "uncertainty_analysis" in metrics[model]:
                uncertainty[f"{folder.name}/{model}"] = metrics[model][
                    "uncertainty_analysis"
                ]
            for horizon in metrics[model]["test"].get("horizons", []):
                support.append(
                    {
                        "fold": folder.name,
                        "model": model,
                        "horizon": horizon["horizon"],
                        "status": horizon["status"],
                        "cases": horizon.get("cases"),
                        "controls": horizon.get("controls"),
                        "sparse_descriptive_warning": horizon.get("cases", 0) < 10
                        or horizon.get("controls", 0) < 10,
                        "reason": horizon.get("reason"),
                    }
                )
        panel_credits = []
        for model in panel_models:
            available_subsets = sorted(
                withheld.loc[withheld.model.eq(model), "kept_modalities"].unique()
            )
            current_names = ["all"] + available_subsets
            if panel_names is None:
                panel_names = current_names
            if panel_names != current_names:
                raise ValueError(
                    "Missingness comparison panels differ across models/folds"
                )
            full = (
                predictions[predictions.model.eq(model)]
                .set_index("Patient_ID")
                .loc[base.Patient_ID]
            )
            member = [credits(full.risk.to_numpy(), earlier, later)]
            for subset in available_subsets:
                rows = withheld[
                    withheld.model.eq(model) & withheld.kept_modalities.eq(subset)
                ].set_index("Patient_ID")
                if set(rows.index) != set(base.Patient_ID):
                    raise ValueError(
                        "Missing-input comparisons must retain identical complete-input patients"
                    )
                rows = rows.loc[base.Patient_ID]
                member.append(credits(rows.risk.to_numpy(), earlier, later))
            panel_credits.append(member)
        fold_payloads.append(
            {
                "patients": len(base),
                "earlier": earlier,
                "later": later,
                "model_credits": np.asarray(model_credits),
                "panel_credits": np.asarray(panel_credits),
            }
        )
    for key in ["input_sha256", "code_sha256", "study_protocol"]:
        if any(manifest[key] != manifests[0][key] for manifest in manifests):
            raise ValueError(f"Fold source/protocol implementations differ: {key}")
    table = pd.concat(oof, ignore_index=True)
    if table.duplicated(["model", "Patient_ID"]).any():
        raise ValueError("A patient appeared in more than one outer test fold")
    source = pd.read_csv(manifests[0]["arguments"]["input"])
    excluded = pd.read_csv(folders[0] / "excluded_patients.csv")
    admitted_ids = set(source.Patient_ID) - set(excluded.Patient_ID)
    for model in models:
        if set(table.loc[table.model.eq(model), "Patient_ID"]) != admitted_ids:
            raise ValueError(
                "OOF panel is not exactly one prediction per admitted patient/model"
            )
    rng = np.random.default_rng(args.seed)
    mean_samples, panel_samples = [], []
    for _ in range(args.bootstrap):
        fold_scores, fold_panels = [], []
        for payload in fold_payloads:
            count = np.bincount(
                rng.integers(payload["patients"], size=payload["patients"]),
                minlength=payload["patients"],
            )
            weight = count[payload["earlier"]] * count[payload["later"]]
            denominator = weight.sum()
            if denominator == 0:
                break
            fold_scores.append(payload["model_credits"] @ weight / denominator)
            fold_panels.append(
                (payload["panel_credits"] @ weight / denominator).mean(axis=1)
            )
        if len(fold_scores) == len(folders):
            mean_samples.append(np.mean(fold_scores, axis=0))
            panel_samples.append(np.mean(fold_panels, axis=0))
    mean_samples, panel_samples = np.asarray(mean_samples), np.asarray(panel_samples)
    reference_index = models.index("clinical_cox")
    per_fold = pd.DataFrame(fold_rows)
    primary, pooled_rows = {}, []
    reference = table[table.model.eq("clinical_cox")].set_index("Patient_ID")
    pooled = {}
    for index, model in enumerate(models):
        point = float(per_fold.loc[per_fold.model.eq(model), "within_fold_c"].mean())
        delta = point - float(
            per_fold.loc[per_fold.model.eq("clinical_cox"), "within_fold_c"].mean()
        )
        primary[model] = {
            "mean_within_fold_c": point,
            "conditional_fold_stratified_patient_bootstrap_95_ci": np.quantile(
                mean_samples[:, index], [0.025, 0.975]
            ).tolist(),
            "model_minus_clinical_cox": delta,
            "paired_delta_95_ci": np.quantile(
                mean_samples[:, index] - mean_samples[:, reference_index],
                [0.025, 0.975],
            ).tolist(),
        }
        rows = table[table.model.eq(model)].sort_values("Patient_ID")
        paired_reference = reference.loc[
            rows.Patient_ID, "oof_common_probability_risk"
        ].to_numpy()
        pooled[model] = bootstrap_concordance(
            rows.time,
            rows.event,
            rows.oof_common_probability_risk,
            reference_risk=paired_reference,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
        ci = pooled[model]["bootstrap_95_ci"]
        pooled_rows.append(
            {
                "model": model,
                "patients": len(rows),
                "deaths": int(rows.event.sum()),
                "pooled_oof_common_probability_c": pooled[model]["harrell_c"],
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "delta_vs_common_probability_cox": pooled[model][
                    "model_minus_reference_c"
                ],
                "paired_delta_lower": pooled[model][
                    "paired_difference_bootstrap_95_ci"
                ][0],
                "paired_delta_upper": pooled[model][
                    "paired_difference_bootstrap_95_ci"
                ][1],
            }
        )
    panel_point = np.mean(
        [p["panel_credits"].mean(axis=2).mean(axis=1) for p in fold_payloads], axis=0
    )
    panel_scope = (
        "primary_four_input_panel"
        if len(panel_names) == 15
        and set(manifests[0]["features"]) == {"cli", "rna", "wsi", "rad"}
        else "exploratory_two_input_panel; four-input H1 remains untested"
    )
    summary = {
        "status": "completed",
        "study_protocol": manifests[0]["study_protocol"],
        "scope": "Separate retrospective KIRC clinical/RNA association; procurement timing unknown, no prospective or four-input claim",
        "oof_patients": len(admitted_ids),
        "oof_observed_deaths": int(reference.event.sum()),
        "primary_locked_within_fold_summary": primary,
        "secondary_pooled_oof_common_probability_summary": pooled,
        "pooled_risk_definition": "negative mean of the fold-specific saved survival probabilities at365,1095,1825days, including Cox; raw Cox linear predictors are not pooled",
        "pooled_analysis_status": "Secondary descriptive aggregation requested after fitting; does not replace the locked within-fold primary estimator",
        "missing_input_panel": {
            "scope": panel_scope,
            "subsets": panel_names,
            "attention_dropout0_mean_fold_subset_c": float(panel_point[0]),
            "attention_dropout03_mean_fold_subset_c": float(panel_point[1]),
            "dropout03_minus_dropout0": float(panel_point[1] - panel_point[0]),
            "paired_conditional_fold_stratified_patient_bootstrap_95_ci": np.quantile(
                panel_samples[:, 1] - panel_samples[:, 0], [0.025, 0.975]
            ).tolist(),
        },
        "bootstrap": {
            "requested": args.bootstrap,
            "seed": args.seed,
            "valid_stratified_replicates": len(mean_samples),
            "ordering": "replicate outer loop; folds01..05; within each fold ascending Patient_ID; one resample shared across models/subsets",
            "primary_unit": "Patients sampled with replacement within fixed outer folds, then equal mean of5foldC; folds are not bootstrapped as independent observations",
            "secondary_unit": "Patients sampled from the complete aligned OOF prediction table in ascendingPatient_ID order",
            "limitation": "Conditional on saved fitted models; does not include retraining/selection uncertainty and overlapping development-fold dependence",
        },
        "uncertainty_by_fold": uncertainty,
        "source_artifact_hashes": source_hashes,
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "training_code_sha256": manifests[0]["code_sha256"],
    }
    table.to_csv(args.runs_root / "oof_predictions.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(
        args.runs_root / "oof_model_metrics.csv", index=False
    )
    per_fold.to_csv(args.runs_root / "fold_metrics.csv", index=False)
    pd.DataFrame(support).to_csv(
        args.runs_root / "fold_horizon_support.csv", index=False
    )
    dump(args.runs_root / "five_fold_summary.json", summary)
    dump(
        args.runs_root / "pooled_oof_summary.json",
        {
            "status": "completed",
            "scope": "Secondary descriptive pooled OOF analysis of217unique retrospective KIRC patients; not the locked within-fold primary estimator",
            "patients": summary["oof_patients"],
            "observed_deaths": summary["oof_observed_deaths"],
            "risk_definition": summary["pooled_risk_definition"],
            "models": pooled,
            "bootstrap": summary["bootstrap"],
            "training_code_sha256": summary["training_code_sha256"],
            "analysis_code_sha256": summary["implementation_sha256"],
            "source_artifact_hashes": source_hashes,
            "primary_summary_file": "five_fold_summary.json",
            "uncertainty_interpretation": "Fold-specific between-seed disagreement/error relations are inconsistent; calibrated confidence is not established. See five_fold_summary uncertainty_by_fold.",
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.runs_root / "five_fold_summary.json"),
                "patients": summary["oof_patients"],
                "deaths": summary["oof_observed_deaths"],
                "primary_mean_fold_c": {
                    k: v["mean_within_fold_c"] for k, v in primary.items()
                },
                "secondary_pooled_oof_c": {
                    k: v["harrell_c"] for k, v in pooled.items()
                },
                "missing_input_panel": summary["missing_input_panel"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
