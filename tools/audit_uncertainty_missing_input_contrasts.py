"""Paired patient-bootstrap contrasts for actual-patient input withholding."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "RUN REPORT/uncertainty-validation-20261118-v1"


def main():
    path = FOLDER / "patient_diagnostics.csv"
    source = pd.read_csv(path)
    summary = json.loads((FOLDER / "uncertainty_report.json").read_text())
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != summary["patient_diagnostics_sha256"]
    ):
        raise ValueError("Patient diagnostic source changed")
    results = []
    for (cohort, model, horizon), rows in source.groupby(
        ["cohort", "model", "horizon"], sort=True
    ):
        full = rows[rows.condition.eq("all_inputs")]
        for condition, missing in rows[~rows.condition.eq("all_inputs")].groupby(
            "condition", sort=True
        ):
            data = (
                full.merge(
                    missing,
                    on=["Patient_ID", "fold"],
                    validate="one_to_one",
                    suffixes=("_full", "_missing"),
                )
                .sort_values(["fold", "Patient_ID"])
                .reset_index(drop=True)
            )
            if len(data) != len(full):
                raise ValueError("Withholding changed the patient comparison set")
            for name in ["case", "control", "weight", "time", "event"]:
                np.testing.assert_array_equal(
                    data[name + "_full"], data[name + "_missing"]
                )
            p, q = data.probability_full.to_numpy(), data.probability_missing.to_numpy()
            truth = data.control_full.to_numpy(float)
            w = data.weight_full.to_numpy()
            sd = (
                data.disagreement_missing.to_numpy() - data.disagreement_full.to_numpy()
            )
            brier = w * ((q - truth) ** 2 - (p - truth) ** 2)
            groups = [
                np.flatnonzero(data.fold.to_numpy() == fold)
                for fold in sorted(data.fold.unique())
            ]
            rng = np.random.default_rng(20261118)
            boot = []
            for _ in range(1000):
                ix = np.concatenate(
                    [rng.choice(g, size=len(g), replace=True) for g in groups]
                )
                if (
                    not data.case_full.to_numpy()[ix].any()
                    or not data.control_full.to_numpy()[ix].any()
                ):
                    continue
                boot.append([float(sd[ix].mean()), float(brier[ix].mean())])
            boot = np.asarray(boot)
            results.append(
                {
                    "cohort": cohort,
                    "model": model,
                    "horizon": int(horizon),
                    "withheld_condition": condition,
                    "patients": len(data),
                    "mean_disagreement_missing_minus_full": float(sd.mean()),
                    "disagreement_delta_conditional95_ci": np.quantile(
                        boot[:, 0], [0.025, 0.975]
                    ).tolist(),
                    "ipcw_brier_missing_minus_full": float(brier.mean()),
                    "brier_delta_conditional95_ci": np.quantile(
                        boot[:, 1], [0.025, 0.975]
                    ).tolist(),
                    "valid_bootstrap_replicates": len(boot),
                }
            )
    report = {
        "scope": "Exploratory paired inference-withholding contrasts; fixed training censor weights/models; no multiplicity correction or natural-missingness validation",
        "patient_diagnostics_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "comparisons": results,
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    (FOLDER / "missing_input_paired_contrasts.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(results).to_csv(
        FOLDER / "missing_input_paired_contrasts.csv", index=False
    )
    print(
        json.dumps(
            [
                x
                for x in results
                if x["horizon"] == 1825
                and x["model"] == "fusion_concat"
                and x["withheld_condition"] == "cli_only"
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
