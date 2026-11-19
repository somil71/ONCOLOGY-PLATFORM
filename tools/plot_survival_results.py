"""Generate actual-run scientific figures with a training-chosen KM threshold."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.survival_metrics import calibration_table, kaplan_meier
from tools.run_survival_experiments import dump, uncertainty_if_supported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--horizon", type=float, default=1825)
    args = parser.parse_args()
    run = args.run
    metrics = json.loads((run / "metrics.json").read_text())
    manifest = json.loads((run / "run_manifest.json").read_text())
    clinical_columns = manifest["features"].get("cli", [])
    predictor_label = "Age-only" if clinical_columns == ["age_years"] else "Clinical"
    model_labels = {
        "clinical_cox": f"{predictor_label} Cox",
        "age_only_cox": "Age-only Cox",
        "unimodal_cli": f"{predictor_label} neural ensemble",
        "unimodal_rna": "RNA neural ensemble",
        "fusion_concat": "Concatenation fusion",
        "fusion_mean": "Mean fusion",
        "fusion_attention": "Attention fusion",
        "attention_no_mask": "Attention without masking",
        "attention_dropout_01": "Attention: modality dropout 0.1",
        "attention_dropout_03": "Attention: modality dropout 0.3",
        "attention_dropout_05": "Attention: modality dropout 0.5",
        "without_cli": "Retrained without clinical input",
        "without_rna": "Retrained without RNA input",
    }
    predictions = pd.read_csv(run / "test_predictions.csv")
    splits = pd.read_csv(run / "patient_splits.csv")
    frame = pd.read_csv(manifest["arguments"]["input"])
    if (
        hashlib.sha256(Path(manifest["arguments"]["input"]).read_bytes()).hexdigest()
        != manifest["input_sha256"]
    ):
        raise ValueError("Input file changed since training")
    if (
        hashlib.sha256((run / "patient_splits.csv").read_bytes()).hexdigest()
        != manifest["split_sha256"]
    ):
        raise ValueError("Split file changed since training")
    train = splits[splits.split.eq("train")].merge(
        frame, on=["Patient_ID", "time", "event"], validate="one_to_one"
    )
    out = run / "paper_figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )
    saved = []

    def save(fig, name):
        fig.savefig(out / (name + ".png"), bbox_inches="tight", dpi=200)
        fig.savefig(out / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)
        saved.append(name)

    labels = []
    points = []
    lower = []
    upper = []
    for label, value in metrics.items():
        test = value.get("test", {})
        ci = test.get("bootstrap_95_ci")
        if ci and test.get("harrell_c") is not None:
            labels.append(model_labels.get(label, label.replace("_", " ")))
            points.append(test["harrell_c"])
            lower.append(ci[0])
            upper.append(ci[1])
    if labels:
        fig, ax = plt.subplots(figsize=(8, max(3.5, 1.6 + 0.36 * len(labels))))
        y = np.arange(len(labels))
        points = np.array(points)
        ax.errorbar(
            points,
            y,
            xerr=[points - lower, np.array(upper) - points],
            fmt="o",
            color="#156b65",
            capsize=4,
        )
        ax.axvline(0.5, color=".6", ls="--", lw=1)
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Harrell C-index (patient bootstrap 95% interval)")
        ax.set_title("Exploratory held-out survival ranking")
        fig.tight_layout()
        save(fig, "01_heldout_cindex_intervals")
    bundle = json.loads((run / "clinical_cox_bundle.json").read_text())
    columns = bundle["columns"]
    prep = bundle["preprocessing"]
    values = train[columns].to_numpy(float)
    values = np.where(np.isfinite(values), values, np.array(prep["median"]))
    standardized = (
        ((values - prep["mean"]) / prep["scale"])
        .astype(prep.get("output_dtype", "float32"))
        .astype(float)
    )
    train_risk = standardized @ np.array(bundle["coefficients"])
    threshold = float(np.median(train_risk))
    test = predictions[predictions.model.eq("clinical_cox")]
    fig, (ax, counts_ax) = plt.subplots(
        2, 1, figsize=(7, 5.2), gridspec_kw={"height_ratios": [4, 1.2]}
    )
    ticks = np.array([0, 365, 1095, 1825, 3650])
    risk_table = []
    for name, mask, color in [
        ("Lower predicted risk", test.risk < threshold, "#156b65"),
        ("Higher predicted risk", test.risk >= threshold, "#b05a36"),
    ]:
        rows = test.loc[mask]
        steps = kaplan_meier(rows.time, rows.event)
        times = np.r_[0, [r["time"] for r in steps]] / 365.25
        surv = np.r_[1, [r["survival"] for r in steps]]
        ax.step(
            times,
            surv,
            where="post",
            color=color,
            label=f"{name} (n={len(rows)}, deaths={int(rows.event.sum())})",
        )
        ax.fill_between(
            times,
            np.r_[1, [r["lower_95"] for r in steps]],
            np.r_[1, [r["upper_95"] for r in steps]],
            step="post",
            color=color,
            alpha=0.12,
        )
        risk_table.append([name] + [int((rows.time >= t).sum()) for t in ticks])
    ax.set(
        xlabel="Years from endpoint origin",
        ylabel="Observed overall survival",
        ylim=(0, 1.02),
        xlim=(0, 10),
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(
        f"{predictor_label} Cox: observed-outcome Kaplan–Meier\nRisk cutoff fixed at the training median"
    )
    counts_ax.axis("off")
    counts_ax.table(
        cellText=risk_table,
        colLabels=["Number at risk"] + [f"{t / 365.25:.1f} y" for t in ticks],
        loc="center",
        cellLoc="center",
        colWidths=[0.38] + [0.12] * 5,
    ).auto_set_font_size(False)
    fig.tight_layout()
    save(fig, "02_clinical_cox_observed_km_training_cutoff")
    horizon = args.horizon
    probcol = f"survival_{int(horizon)}"
    uncertainty_records = {}
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", color=".6")
    for label in predictions.model.unique():
        rows = predictions[predictions.model.eq(label)]
        if probcol not in rows or rows[probcol].isna().all():
            continue
        bins = calibration_table(rows.time, rows.event, rows[probcol], horizon)
        valid = [b for b in bins if b["supported"]]
        ax.plot(
            [b["mean_predicted_survival"] for b in valid],
            [b["km_survival"] for b in valid],
            marker="o",
            label=model_labels.get(label, label.replace("_", " ")),
        )
    ax.set(
        xlabel="Mean predicted survival",
        ylabel="Within-bin observed KM survival",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.set_title(
        f"{horizon / 365.25:.1f}-year descriptive calibration\nSmall-bin estimates; no fitted recalibration"
    )
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save(fig, "03_horizon_calibration")
    for label in predictions.model.unique():
        rows = predictions[predictions.model.eq(label)]
        sdcol = f"ensemble_sd_{int(horizon)}"
        if sdcol not in rows or rows[sdcol].isna().all():
            continue
        analysis = uncertainty_if_supported(
            rows.time,
            rows.event,
            rows[probcol],
            rows[sdcol],
            train.time,
            train.event,
            horizon,
        )
        uncertainty_records[label] = analysis
        if analysis["status"] != "evaluated":
            continue
        known = ((rows.event == 1) & (rows.time <= horizon)) | (rows.time > horizon)
        residual = (
            rows.loc[known, probcol] - (rows.loc[known, "time"] > horizon).astype(float)
        ) ** 2
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(
            rows.loc[known, sdcol], residual, s=16, color="#156b65", alpha=0.65
        )
        correlation = analysis["ipcw_error_uncertainty_correlation"]
        axes[0].set(
            xlabel="Between-seed survival SD",
            ylabel="Squared survival residual (known status)",
            title=f"Descriptive IPCW correlation: {correlation:.3f}"
            if correlation is not None
            else "Uncertainty has no measurable variation",
        )
        selective = analysis["selective_prediction"]
        axes[1].plot(
            [r["coverage_actual"] for r in selective],
            [r["ipcw_brier"] for r in selective],
            marker="o",
            color="#156b65",
        )
        axes[1].set(
            xlabel="Fraction retained (least uncertain first)",
            ylabel="IPCW Brier score",
            title="Selective prediction; lower error is better",
            xlim=(0, 1.02),
        )
        fig.suptitle(
            f"{model_labels.get(label, label.replace('_', ' '))}: {horizon / 365.25:.1f}-year uncertainty diagnostics",
            fontsize=12,
        )
        fig.tight_layout()
        save(fig, "04_" + label + "_uncertainty_diagnostics")
    histories = list(run.glob("*_seed*_history.csv"))
    if histories:
        fig, ax = plt.subplots(figsize=(8, 5))
        configurations = sorted({p.stem.split("_seed")[0] for p in histories})
        colors = plt.get_cmap("tab20")
        labelled = set()
        for path in histories:
            history = pd.read_csv(path)
            name = path.stem.split("_seed")[0]
            label = (
                model_labels.get(name, name.replace("_", " "))
                if name not in labelled
                else "_nolegend_"
            )
            ax.plot(
                history.epoch,
                history.validation_nll,
                label=label,
                color=colors(configurations.index(name) % 20),
                alpha=0.65,
                lw=1,
            )
            labelled.add(name)
        ax.set(
            xlabel="Epoch",
            ylabel="Validation negative log-likelihood",
            title="Censoring-aware model selection history\nOne line per training seed; color identifies configuration",
        )
        ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        save(fig, "05_validation_likelihood_history")
    provenance = {
        "run_id": manifest["run_id"],
        "scope": manifest["scope"],
        "figures": saved,
        "sources": {
            name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in [
                "test_predictions.csv",
                "metrics.json",
                "patient_splits.csv",
                "run_manifest.json",
                "clinical_cox_bundle.json",
            ]
        },
        "clinical_cox_training_median_risk_cutoff": threshold,
        "training_patients_for_cutoff": len(train),
        "predictors": clinical_columns,
        "preprocessing_output_dtype": prep.get("output_dtype", "float32"),
        "horizon_days": horizon,
        "uncertainty_diagnostics": uncertainty_records,
        "limitations": [
            "Exploratory study; no prospective clinical validation.",
            "KM uses observed events and training-chosen threshold.",
            "No multimodal superiority implied by clinical-only figures.",
            "Calibration bins and uncertainty relations are descriptive and event-limited.",
        ],
    }
    dump(out / "figure_provenance.json", provenance)
    print(
        json.dumps(
            {"output": str(out), "figures": saved, "training_threshold": threshold}
        )
    )


if __name__ == "__main__":
    main()
