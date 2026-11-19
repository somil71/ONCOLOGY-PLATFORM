"""Censor-aware disagreement/seed/withheld-input diagnostics on fixed predictions.

No training or tuning. Sources are previously independently reconstructed runs.
"""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.survival_metrics import (
    CensoringDistribution,
    evaluate_uncertainty,
    horizon_metrics,
)
from src.survival_models import HazardFusion
from src.uncertainty_audit import bootstrap_diagnostics


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "RUN REPORT/uncertainty-validation-20261118-v1"
MODELS = ["fusion_concat", "fusion_attention", "attention_dropout_03", "unimodal_rna"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(run, label, manifest, saved):
    raw = pd.read_csv(manifest["arguments"]["input"]).set_index(
        "Patient_ID", verify_integrity=True
    )
    prep = json.loads((run / "neural_preprocessing.json").read_text())
    names = list(manifest["features"])
    features = {}
    with np.load(manifest["arguments"]["rna_npz"], allow_pickle=False) as matrix:
        ids = {p: n for n, p in enumerate(matrix["patient_ids"].astype(str))}
        genes = {g: n for n, g in enumerate(matrix["gene_ids"].astype(str))}
        for name in names:
            columns = manifest["features"][name]
            values = (
                matrix["values"][[ids[p] for p in saved.Patient_ID]][
                    :, [genes[g] for g in columns]
                ].astype(float)
                if name == "rna"
                else raw.loc[saved.Patient_ID, columns].to_numpy(float)
            )
            p = prep[name]
            features[name] = torch.tensor(
                (
                    (np.where(np.isfinite(values), values, p["median"]) - p["mean"])
                    / p["scale"]
                ).astype("float32")
            )
    predictions = {}
    for seed in manifest["seeds"]:
        bundle = torch.load(
            run / f"{label}_seed{seed}.pth", weights_only=True, map_location="cpu"
        )
        selected = bundle["modalities"]
        model = HazardFusion(
            bundle["dimensions"],
            bundle["boundaries"],
            fusion=bundle["fusion"],
            use_mask=bundle["use_mask"],
        )
        model.load_state_dict(bundle["state_dict"])
        for size in range(1, len(selected) + 1):
            for keep in itertools.combinations(selected, size):
                condition = (
                    "all_inputs"
                    if size == len(selected)
                    else "+".join(sorted(keep)) + "_only"
                )
                mask = torch.tensor(
                    [[n in keep for n in selected]] * len(saved), dtype=torch.bool
                )
                prediction = model.predict_survival(
                    {n: features[n] for n in selected}, mask, manifest["horizons_days"]
                )
                predictions.setdefault(condition, []).append(prediction)
    return predictions


def main():
    torch.set_num_threads(2)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    plan = {
        "models": MODELS,
        "horizons_days": [365, 1095, 1825],
        "bootstrap": 1000,
        "bootstrap_seed": 20261118,
        "comparison": "First prespecified seed42 versus mean of all3seeds; no best-seed selection or model tuning",
        "uncertainty": "Between-seed survival probability SD; not calibrated confidence",
        "missingness": "Inference-only withholding on identical real complete-input patients; not naturally missing cohorts",
        "bootstrap_scope": "Patients resampled within outer fold with training censor weights/fitted predictions fixed; no retraining uncertainty",
        "selective_ties": "Include every uncertainty tie at threshold; actual coverage reported",
        "censoring_warning": "Global fold-training reverse-KM assumes appropriate independent censoring; does not establish conditional independence inside selected strata",
    }
    (OUTPUT / "analysis_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    groups = {
        "KIRC_fixed5fold": [
            ROOT / "RUN REPORT/kirc-cli-rna-20261118-fixed5fold-v1" / f"fold_{n:02d}"
            for n in range(1, 6)
        ],
        "BRCA_heldout": [ROOT / "RUN REPORT/clean-cli-rna-1080-20261118-v3"],
    }
    rows = []
    sources = []
    unsupported = []
    checks = []
    for cohort, runs in groups.items():
        for run in runs:
            manifest = json.loads((run / "run_manifest.json").read_text())
            audit = json.loads(
                (run / "independent_prediction_reconstruction.json").read_text()
            )
            if audit["status"] != "passed" or audit[
                "source_run_manifest_sha256"
            ] != sha(run / "run_manifest.json"):
                raise ValueError("Missing/stale independent source reconstruction")
            splits = pd.read_csv(run / "patient_splits.csv")
            train = splits[splits.split.eq("train")]
            censor = CensoringDistribution(train.time, train.event)
            saved = pd.read_csv(run / "test_predictions.csv")
            withheld = pd.read_csv(run / "inference_withholding_predictions.csv")
            metrics = json.loads((run / "metrics.json").read_text())
            sources.append(
                {
                    "run": str(run),
                    "manifest_sha256": sha(run / "run_manifest.json"),
                    "predictions_sha256": sha(run / "test_predictions.csv"),
                    "withheld_sha256": sha(
                        run / "inference_withholding_predictions.csv"
                    ),
                    "metrics_sha256": sha(run / "metrics.json"),
                }
            )
            for label in MODELS:
                full = saved[saved.model.eq(label)].copy()
                members = reconstruct(run, label, manifest, full)
                full_members = members["all_inputs"]
                np.testing.assert_allclose(
                    np.mean(full_members, axis=0),
                    full[[f"survival_{h}" for h in plan["horizons_days"]]],
                    atol=1e-6,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    np.std(full_members, axis=0),
                    full[[f"ensemble_sd_{h}" for h in plan["horizons_days"]]],
                    atol=1e-6,
                    rtol=0,
                )
                for condition, values in members.items():
                    probability = np.mean(values, axis=0)
                    disagreement = np.std(values, axis=0)
                    if condition == "all_inputs":
                        probability = full[
                            [f"survival_{h}" for h in plan["horizons_days"]]
                        ].to_numpy()
                        disagreement = full[
                            [f"ensemble_sd_{h}" for h in plan["horizons_days"]]
                        ].to_numpy()
                    else:
                        keep = condition.removesuffix("_only")
                        subset = (
                            withheld[
                                withheld.model.eq(label)
                                & withheld.kept_modalities.eq(keep)
                            ]
                            .set_index("Patient_ID")
                            .loc[full.Patient_ID]
                        )
                        np.testing.assert_allclose(
                            probability,
                            subset[[f"survival_{h}" for h in plan["horizons_days"]]],
                            atol=1e-6,
                            rtol=0,
                        )
                        probability = subset[
                            [f"survival_{h}" for h in plan["horizons_days"]]
                        ].to_numpy()
                    t, e = full.time.to_numpy(), full.event.to_numpy(bool)
                    for column, horizon in enumerate(plan["horizons_days"]):
                        try:
                            support = horizon_metrics(
                                t,
                                e,
                                -probability[:, column],
                                train.time,
                                train.event,
                                horizon,
                                probability[:, column],
                            )
                        except ValueError as exc:
                            unsupported.append(
                                {
                                    "cohort": cohort,
                                    "fold": run.name,
                                    "model": label,
                                    "condition": condition,
                                    "horizon": horizon,
                                    "status": "not_evaluated",
                                    "reason": str(exc),
                                }
                            )
                            continue
                        if condition == "all_inputs":
                            source = evaluate_uncertainty(
                                t,
                                e,
                                probability[:, column],
                                disagreement[:, column],
                                train.time,
                                train.event,
                                horizon,
                            )
                            old = metrics[label]["uncertainty_analysis"][str(horizon)]
                            for name in ["ipcw_error_uncertainty_correlation"]:
                                if source[name] is not None:
                                    np.testing.assert_allclose(
                                        source[name], old[name], atol=1e-10, rtol=0
                                    )
                            for a, b in zip(
                                source["selective_prediction"],
                                old["selective_prediction"],
                            ):
                                if a["ipcw_brier"] is not None:
                                    np.testing.assert_allclose(
                                        a["ipcw_brier"],
                                        b["ipcw_brier"],
                                        atol=1e-10,
                                        rtol=0,
                                    )
                            checks.append(
                                {
                                    "run": run.name,
                                    "cohort": cohort,
                                    "model": label,
                                    "horizon": horizon,
                                    "saved_uncertainty_correlation_and_selective_brier_reconstructed": True,
                                }
                            )
                        cases = e & (t <= horizon)
                        controls = t > horizon
                        weight = np.zeros(len(t))
                        weight[cases] = 1 / censor.at(t[cases])
                        weight[controls] = 1 / float(censor.at(horizon))
                        for n, patient in enumerate(full.Patient_ID):
                            rows.append(
                                {
                                    "cohort": cohort,
                                    "fold": run.name,
                                    "model": label,
                                    "condition": condition,
                                    "horizon": horizon,
                                    "Patient_ID": patient,
                                    "time": float(t[n]),
                                    "event": bool(e[n]),
                                    "case": bool(cases[n]),
                                    "control": bool(controls[n]),
                                    "weight": float(weight[n]),
                                    "probability": float(probability[n, column]),
                                    "seed42_probability": float(values[0][n, column]),
                                    "disagreement": float(disagreement[n, column]),
                                    "training_censor_survival_at_horizon": support[
                                        "censoring_survival"
                                    ],
                                }
                            )
            print("Reconstructed", cohort, run.name, flush=True)
    table = (
        pd.DataFrame(rows)
        .sort_values(["cohort", "model", "condition", "horizon", "fold", "Patient_ID"])
        .reset_index(drop=True)
    )
    table.to_csv(OUTPUT / "patient_diagnostics.csv", index=False)
    reports = []
    for (cohort, model, condition, horizon), frame in table.groupby(
        ["cohort", "model", "condition", "horizon"], sort=True
    ):
        result = bootstrap_diagnostics(frame, n_bootstrap=1000, seed=20261118)
        result.update(
            cohort=cohort,
            model=model,
            condition=condition,
            horizon=int(horizon),
            sparse_descriptive_warning=bool(
                result["cases"] < 10 or result["controls"] < 10
            ),
        )
        reports.append(result)
        print("Analyzed", cohort, model, condition, horizon, flush=True)
    report = {
        "status": "completed",
        "scope": plan,
        "source_artifacts": sources,
        "source_metric_reconstruction_checks": checks,
        "unsupported_horizon_records": unsupported,
        "groups": reports,
        "patient_diagnostics_sha256": sha(OUTPUT / "patient_diagnostics.csv"),
        "implementation_sha256": {
            name: sha(ROOT / name)
            for name in [
                "src/uncertainty_audit.py",
                "tools/audit_survival_uncertainty.py",
            ]
        },
    }
    (OUTPUT / "uncertainty_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    flat = [
        {k: v for k, v in record.items() if not isinstance(v, (list, dict))}
        for record in reports
    ]
    pd.DataFrame(flat).to_csv(OUTPUT / "uncertainty_summary.csv", index=False)
    print("Completed", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
