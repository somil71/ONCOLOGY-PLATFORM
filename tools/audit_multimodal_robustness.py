"""Exploratory paired missing-input audit from actual held-out predictions.

Resample patients once across every input subset; never treat multiple predictions
of one patient as independent subjects. This does not retrain/bootstrap models.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.survival_metrics import bootstrap_concordance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    manifest = json.loads((run / "run_manifest.json").read_text())
    predictions = pd.read_csv(run / "test_predictions.csv")
    withheld = pd.read_csv(run / "inference_withholding_predictions.csv")
    baseline = "fusion_attention"
    reference = predictions.loc[predictions.model.eq(baseline)].sort_values(
        "Patient_ID"
    )
    ids = reference.Patient_ID.tolist()
    times, events = reference.time.to_numpy(), reference.event.to_numpy(bool)
    earlier = (times[:, None] < times[None, :]) | (
        (times[:, None] == times[None, :]) & ~events[None, :]
    )
    i, j = np.where(events[:, None] & earlier)
    if not len(i):
        raise ValueError("No survival-comparable held-out patients")
    rows = {}
    models = [
        baseline,
        "attention_dropout_01",
        "attention_dropout_03",
        "attention_dropout_05",
    ]
    for model in models:
        rows[model] = {"complete": predictions.loc[predictions.model.eq(model)]}
        for subset, frame in withheld.loc[withheld.model.eq(model)].groupby(
            "kept_modalities"
        ):
            rows[model][subset] = frame
    subsets = sorted(rows[baseline])
    credits = {}
    for model, panel in rows.items():
        if sorted(panel) != subsets:
            raise ValueError("Input subset panel differs across models")
        model_credits = []
        for subset in subsets:
            frame = panel[subset].sort_values("Patient_ID")
            if frame.Patient_ID.duplicated().any() or frame.Patient_ID.tolist() != ids:
                raise ValueError("Incomplete or unmatched paired patient panel")
            np.testing.assert_array_equal(frame.time, times)
            np.testing.assert_array_equal(frame.event.to_numpy(bool), events)
            difference = frame.risk.to_numpy()[i] - frame.risk.to_numpy()[j]
            model_credits.append(
                (difference > 1e-8) + 0.5 * (np.abs(difference) <= 1e-8)
            )
        credits[model] = np.stack(model_credits)
    nboot = manifest["arguments"]["bootstrap"]
    rng = np.random.default_rng(42)
    bootstrap = {model: [] for model in models}
    for _ in range(nboot):
        counts = np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))
        weights = counts[i] * counts[j]
        if weights.sum() == 0:
            continue
        reference_mean = np.mean((credits[baseline] * weights).sum(1) / weights.sum())
        for model in models:
            bootstrap[model].append(
                float(
                    np.mean((credits[model] * weights).sum(1) / weights.sum())
                    - reference_mean
                )
            )
    comparisons = {}
    for model in models:
        point = credits[model].mean(1)
        delta = float(point.mean() - credits[baseline].mean())
        comparisons[model] = {
            "subset_harrell_c": dict(zip(subsets, point.tolist())),
            "equal_weight_mean_subset_c": float(point.mean()),
            "aggregate_delta_vs_no_dropout": delta,
            "patient_paired_bootstrap_delta_95_ci": np.quantile(
                bootstrap[model], [0.025, 0.975]
            ).tolist(),
            "valid_bootstrap_replicates": len(bootstrap[model]),
        }
    full_contrasts = {}
    for model, ref in (
        ("fusion_attention", "fusion_concat"),
        ("fusion_concat", "unimodal_rna"),
    ):
        a = predictions.loc[predictions.model.eq(model)].sort_values("Patient_ID")
        b = predictions.loc[predictions.model.eq(ref)].sort_values("Patient_ID")
        if a.Patient_ID.tolist() != b.Patient_ID.tolist():
            raise ValueError("Unmatched full-input comparison")
        full_contrasts[f"{model}_minus_{ref}"] = bootstrap_concordance(
            a.time, a.event, a.risk, reference_risk=b.risk, n_bootstrap=nboot, seed=42
        )
    report = {
        "scope": "Exploratory patient-paired aggregate robustness audit; no training-selection uncertainty or multiplicity correction",
        "patients": len(ids),
        "deaths": int(events.sum()),
        "fixed_subset_panel": subsets,
        "bootstrap_unit": "patient resampled once across all input subsets and model predictions",
        "source_sha256": {
            name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in (
                "run_manifest.json",
                "test_predictions.csv",
                "inference_withholding_predictions.csv",
            )
        },
        "comparisons": comparisons,
        "full_input_secondary_contrasts": full_contrasts,
    }
    output = run / "independent_robustness_audit.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
