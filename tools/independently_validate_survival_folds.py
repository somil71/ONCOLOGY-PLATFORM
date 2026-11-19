"""Independent fixed-fold survival summary and patient-paired bootstrap audit.

Primary summaries preserve the locked equal-fold estimand. Pooled predictions
are secondary and use the same fixed-horizon survival-probability risk for Cox
and neural models; arbitrary fitted-fold Cox linear predictors are not pooled.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.survival_metrics import bootstrap_concordance


def pair_credits(time, event, risk):
    time, event, risk = np.asarray(time), np.asarray(event, bool), np.asarray(risk)
    before = (time[:, None] < time[None, :]) | (
        (time[:, None] == time[None, :]) & ~event[None, :]
    )
    i, j = np.where(event[:, None] & before)
    if not len(i):
        raise ValueError("No comparable survival pairs in an outer fold")
    diff = risk[..., i] - risk[..., j]
    return i, j, (diff > 1e-8) + 0.5 * (np.abs(diff) <= 1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    folder = args.root
    paths = sorted(folder.glob("fold_*/run_manifest.json"))
    if len(paths) != 5:
        raise ValueError("Expected exactly the five protocol outer folds")
    cache, oof, records = [], [], []
    panel_models = ["fusion_attention", "attention_dropout_03"]
    protocol_hash = None
    names = None
    hashes = {}
    for path in paths:
        run = path.parent
        manifest = json.loads(path.read_text())
        audit = json.loads(
            (run / "independent_prediction_reconstruction.json").read_text()
        )
        if (
            audit["status"] != "passed"
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != audit["source_run_manifest_sha256"]
        ):
            raise ValueError("Missing or stale independent per-fold reconstruction")
        if not audit["locked_protocol_checks"].get("outcomes_equal_original_cdr"):
            raise ValueError("Original CDR outcome audit missing")
        this_protocol = manifest["study_protocol"]["sha256"]
        if protocol_hash is not None and this_protocol != protocol_hash:
            raise ValueError("Outer folds used different analysis protocols")
        protocol_hash = this_protocol
        predictions = pd.read_csv(run / "test_predictions.csv")
        if names is None:
            names = list(predictions.model.unique())
        if set(names) != set(predictions.model):
            raise ValueError("Different model panel across outer folds")
        reference = predictions.loc[predictions.model.eq("clinical_cox")].sort_values(
            "Patient_ID"
        )
        ids = reference.Patient_ID.tolist()
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate held-out clinical patient")
        risks = []
        for label in names:
            rows = (
                predictions.loc[predictions.model.eq(label)]
                .sort_values("Patient_ID")
                .copy()
            )
            if rows.Patient_ID.tolist() != ids:
                raise ValueError(
                    "Primary paired fold summary requires identical patients for every model"
                )
            np.testing.assert_array_equal(
                rows[["time", "event"]], reference[["time", "event"]]
            )
            risks.append(rows.risk.to_numpy())
            rows["outer_fold"] = run.name
            horizons = manifest["horizons_days"]
            rows["common_probability_risk"] = -rows[
                [f"survival_{int(h)}" for h in horizons]
            ].mean(1)
            oof.append(rows)
        time, event = reference.time.to_numpy(), reference.event.to_numpy()
        i, j, credit = pair_credits(time, event, np.stack(risks))
        withheld = pd.read_csv(run / "inference_withholding_predictions.csv")
        panel = []
        for label in panel_models:
            full = predictions.loc[predictions.model.eq(label)].sort_values(
                "Patient_ID"
            )
            subset_risks = [full.risk.to_numpy()]
            for subset in ("cli", "rna"):
                rows = withheld.loc[
                    withheld.model.eq(label) & withheld.kept_modalities.eq(subset)
                ].sort_values("Patient_ID")
                if rows.Patient_ID.tolist() != ids:
                    raise ValueError(
                        "The exploratory paired missing-input panel is incomplete"
                    )
                np.testing.assert_array_equal(
                    rows[["time", "event"]], reference[["time", "event"]]
                )
                subset_risks.append(rows.risk.to_numpy())
            _, _, pc = pair_credits(time, event, np.stack(subset_risks))
            panel.append(pc)
        cache.append(
            {
                "patients": len(ids),
                "i": i,
                "j": j,
                "credit": credit,
                "panel_credit": np.stack(panel),
            }
        )
        records.append(
            {
                "fold": run.name,
                "patients": len(ids),
                "deaths": int(event.sum()),
                "model_c": dict(zip(names, credit.mean(1).tolist())),
            }
        )
        hashes[str(run / "test_predictions.csv")] = hashlib.sha256(
            (run / "test_predictions.csv").read_bytes()
        ).hexdigest()
    table = pd.concat(oof, ignore_index=True)
    for label, frame in table.groupby("model"):
        if frame.Patient_ID.duplicated().any():
            raise ValueError("A patient was evaluated in more than one outer fold")
    mean_c = np.mean([x["credit"].mean(1) for x in cache], axis=0)
    fold_sd = np.std([x["credit"].mean(1) for x in cache], axis=0, ddof=1)
    rng = np.random.default_rng(20261118)
    boot, panel_boot = [], []
    for _ in range(1000):
        fold_scores, fold_panels = [], []
        for item in cache:
            counts = np.bincount(
                rng.integers(item["patients"], size=item["patients"]),
                minlength=item["patients"],
            )
            weights = counts[item["i"]] * counts[item["j"]]
            if not weights.sum():
                break
            fold_scores.append((item["credit"] * weights).sum(1) / weights.sum())
            fold_panels.append(
                ((item["panel_credit"] * weights).sum(2) / weights.sum()).mean(1)
            )
        if len(fold_scores) == 5:
            boot.append(np.mean(fold_scores, axis=0))
            panel_boot.append(np.mean(fold_panels, axis=0))
    boot = np.stack(boot)
    reference_index = names.index("clinical_cox")
    primary, secondary = {}, {}
    for k, label in enumerate(names):
        primary[label] = {
            "mean_outer_fold_c": float(mean_c[k]),
            "descriptive_outer_fold_sd": float(fold_sd[k]),
            "conditional_patient_bootstrap_mean_c_95_ci": np.quantile(
                boot[:, k], [0.025, 0.975]
            ).tolist(),
            "mean_fold_delta_vs_clinical_cox": float(
                mean_c[k] - mean_c[reference_index]
            ),
            "conditional_paired_delta_95_ci": np.quantile(
                boot[:, k] - boot[:, reference_index], [0.025, 0.975]
            ).tolist(),
        }
        rows = table.loc[table.model.eq(label)].sort_values("Patient_ID")
        reference = table.loc[table.model.eq("clinical_cox")].sort_values("Patient_ID")
        secondary[label] = bootstrap_concordance(
            rows.time,
            rows.event,
            rows.common_probability_risk,
            reference_risk=reference.common_probability_risk,
            n_bootstrap=1000,
            seed=20261118,
        )
    panel_boot = np.stack(panel_boot)
    panel_point = np.mean(
        [item["panel_credit"].mean(2).mean(1) for item in cache], axis=0
    )
    panel_delta_ci = np.quantile(
        panel_boot[:, 1] - panel_boot[:, 0], [0.025, 0.975]
    ).tolist()
    summary_checks = {"status": "summary_not_available_for_comparison"}
    summary_path = folder / "five_fold_summary.json"
    if summary_path.exists():
        prior = json.loads(summary_path.read_text())
        for label in names:
            a, b = primary[label], prior["primary_locked_within_fold_summary"][label]
            for independent_field, recorded_field in (
                ("mean_outer_fold_c", "mean_within_fold_c"),
                (
                    "conditional_patient_bootstrap_mean_c_95_ci",
                    "conditional_fold_stratified_patient_bootstrap_95_ci",
                ),
                ("mean_fold_delta_vs_clinical_cox", "model_minus_clinical_cox"),
                ("conditional_paired_delta_95_ci", "paired_delta_95_ci"),
            ):
                np.testing.assert_allclose(
                    a[independent_field], b[recorded_field], atol=1e-12, rtol=0
                )
            a, b = (
                secondary[label],
                prior["secondary_pooled_oof_common_probability_summary"][label],
            )
            for field in (
                "harrell_c",
                "bootstrap_95_ci",
                "model_minus_reference_c",
                "paired_difference_bootstrap_95_ci",
            ):
                np.testing.assert_allclose(a[field], b[field], atol=1e-12, rtol=0)
        old_panel = prior["missing_input_panel"]
        np.testing.assert_allclose(
            panel_point,
            [
                old_panel["attention_dropout0_mean_fold_subset_c"],
                old_panel["attention_dropout03_mean_fold_subset_c"],
            ],
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            float(panel_point[1] - panel_point[0]),
            old_panel["dropout03_minus_dropout0"],
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            panel_delta_ci,
            old_panel["paired_conditional_fold_stratified_patient_bootstrap_95_ci"],
            atol=1e-12,
            rtol=0,
        )
        for fold, sources in prior["source_artifact_hashes"].items():
            for name, expected in sources.items():
                if (
                    hashlib.sha256((folder / fold / name).read_bytes()).hexdigest()
                    != expected
                ):
                    raise ValueError("Summary source provenance mismatch")
        exported = (
            pd.read_csv(folder / "oof_predictions.csv")
            .sort_values(["model", "Patient_ID"])
            .reset_index(drop=True)
        )
        expected = table.sort_values(["model", "Patient_ID"]).reset_index(drop=True)
        if (
            exported[["Patient_ID", "model", "fold"]].to_numpy().tolist()
            != expected[["Patient_ID", "model", "outer_fold"]].to_numpy().tolist()
        ):
            raise ValueError(
                "Exported OOF identity/fold rows do not match per-fold sources"
            )
        for field in (
            "time",
            "event",
            "risk",
            "survival_365",
            "survival_1095",
            "survival_1825",
        ):
            np.testing.assert_allclose(
                exported[field], expected[field], atol=1e-12, rtol=0
            )
        np.testing.assert_allclose(
            exported.oof_common_probability_risk,
            expected.common_probability_risk,
            atol=1e-12,
            rtol=0,
        )
        summary_checks = {
            "status": "passed",
            "all_primary_and_secondary_metrics_and_cis_match": True,
            "missing_input_patient_paired_panel_matches": True,
            "exported_oof_rows_and_common_probability_risk_match": True,
            "summary_source_sha256": hashlib.sha256(
                summary_path.read_bytes()
            ).hexdigest(),
        }
    report = {
        "status": "independently_reconstructed",
        "scope": "Retrospective clinical/RNA fixed-fold experiment only; no image training, four-input efficacy or prospective validity",
        "protocol_sha256": protocol_hash,
        "source_prediction_sha256": hashes,
        "patient_tested_once_in_outer_folds": True,
        "patients": int(table.loc[table.model.eq("clinical_cox")].Patient_ID.nunique()),
        "deaths": int(table.loc[table.model.eq("clinical_cox")].event.sum()),
        "per_fold": records,
        "primary_equal_fold_summary": primary,
        "secondary_pooled_common_probability_risk": secondary,
        "independent_summary_comparison": summary_checks,
        "exploratory_two_input_panel": {
            "models": panel_models,
            "subsets": ["all", "cli", "rna"],
            "point_mean_subset_c": panel_point.tolist(),
            "dropout03_minus_dropout0": float(panel_point[1] - panel_point[0]),
            "conditional_patient_paired_delta_95_ci": panel_delta_ci,
            "four_input_H1_tested": False,
        },
        "bootstrap": {
            "replicates_requested": 1000,
            "replicates_valid": len(boot),
            "seed": 20261118,
            "ordering": "Ascending Patient_ID within outer folds1..5; sample once per fold per replicate, shared by all models",
            "scope": "Conditional fixed-prediction intervals; overlapping training folds and model selection uncertainty not included",
        },
    }
    output = folder / "independent_five_fold_summary.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "patients": report["patients"],
                "deaths": report["deaths"],
                "primary": primary,
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
