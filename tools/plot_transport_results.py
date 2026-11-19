"""Figures from frozen internal/external transport predictions and actual outcomes."""

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


LABELS = {
    "age_only_cox": "Age-only Cox",
    "unimodal_rna": "RNA ensemble",
    "fusion_concat": "Concatenation",
    "fusion_mean": "Mean fusion",
    "fusion_attention": "Attention fusion",
}
COLORS = ["#536471", "#866b9b", "#146c65", "#b36d3d", "#467eac"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--horizon", type=float, default=1825)
    args = parser.parse_args()
    run = args.run
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Final figures require a completed run")
    for name, expected in manifest["artifacts_sha256"].items():
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source artifact changed: {name}")
    metrics = json.loads((run / "metrics.json").read_text())
    predictions = pd.read_csv(run / "predictions.csv")
    splits = pd.read_csv(run / "patient_splits.csv")
    train = splits[splits.split.eq("train")]
    output = run / "paper_figures"
    output.mkdir(exist_ok=True)
    saved = []
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=200, bbox_inches="tight")
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        saved.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, cohort, title in zip(
        axes, ["internal", "external"], ["TCGA-BRCA held-out", "METABRIC external"]
    ):
        values = [metrics[name][cohort] for name in LABELS]
        point = np.array([v["harrell_c"] for v in values])
        ci = np.array([v["bootstrap_95_ci"] for v in values])
        ax.errorbar(
            point,
            np.arange(len(point)),
            xerr=[point - ci[:, 0], ci[:, 1] - point],
            fmt="o",
            color="#146c65",
            capsize=4,
        )
        ax.axvline(0.5, ls="--", color=".6", lw=1)
        ax.set(
            xlim=(0, 1),
            xlabel="Harrell C-index (95% patient bootstrap CI)",
            title=title,
            yticks=np.arange(len(LABELS)),
            yticklabels=list(LABELS.values()),
        )
    fig.suptitle("Exploratory shared-rank clinical + RNA transport", fontsize=12)
    fig.tight_layout()
    save(fig, "01_internal_external_cindex_intervals")

    names = [n for n in LABELS if n != "age_only_cox"]
    point = np.array(
        [
            metrics[n]["external"]["paired_vs_age_only"]["model_minus_reference_c"]
            for n in names
        ]
    )
    ci = np.array(
        [
            metrics[n]["external"]["paired_vs_age_only"][
                "paired_difference_bootstrap_95_ci"
            ]
            for n in names
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.errorbar(
        point,
        np.arange(len(names)),
        xerr=[point - ci[:, 0], ci[:, 1] - point],
        fmt="o",
        color="#146c65",
        capsize=4,
    )
    ax.axvline(0, ls="--", color=".6", lw=1)
    ax.set(
        yticks=np.arange(len(names)),
        yticklabels=[LABELS[n] for n in names],
        xlabel="C-index difference versus frozen age-only Cox (95% paired CI)",
        title="METABRIC external: exploratory model comparisons\nIntervals are unadjusted for multiple comparisons",
    )
    fig.tight_layout()
    save(fig, "02_external_paired_differences")

    horizon = args.horizon
    probability = f"survival_{int(horizon)}"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    uncertainty = {}
    for ax, cohort in zip(axes, ["internal", "external"]):
        ax.plot([0, 1], [0, 1], color=".6", ls="--", lw=1)
        for (name, label), color in zip(LABELS.items(), COLORS):
            rows = predictions[
                predictions.cohort.eq(cohort) & predictions.model.eq(name)
            ]
            bins = calibration_table(rows.time, rows.event, rows[probability], horizon)
            supported = [b for b in bins if b["supported"]]
            ax.plot(
                [b["mean_predicted_survival"] for b in supported],
                [b["km_survival"] for b in supported],
                marker="o",
                color=color,
                label=label,
            )
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Mean predicted survival",
            ylabel="Within-bin observed KM survival",
            title=f"{cohort.title()}: {horizon / 365.25:.1f}-year calibration",
        )
    axes[-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Descriptive calibration; no target-fitted recalibration", fontsize=12)
    fig.tight_layout()
    save(fig, "03_internal_external_descriptive_calibration")

    for cohort in ["internal", "external"]:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        for (name, label), color in zip(LABELS.items(), COLORS):
            if name == "age_only_cox":
                continue
            rows = predictions[
                predictions.cohort.eq(cohort) & predictions.model.eq(name)
            ]
            reference = train if cohort == "internal" else rows
            analysis = uncertainty_if_supported(
                rows.time,
                rows.event,
                rows[probability],
                rows[f"ensemble_sd_{int(horizon)}"],
                reference.time,
                reference.event,
                horizon,
            )
            uncertainty[f"{cohort}/{name}"] = analysis
            if analysis["status"] == "evaluated":
                selective = analysis["selective_prediction"]
                ax.plot(
                    [r["coverage_actual"] for r in selective],
                    [r["ipcw_brier"] for r in selective],
                    marker="o",
                    color=color,
                    label=label,
                )
        ax.set(
            xlim=(0, 1.02),
            xlabel="Fraction retained (least uncertain first)",
            ylabel="IPCW Brier score; lower is better",
            title=f"{cohort.title()}: {horizon / 365.25:.1f}-year selective prediction\nBetween-seed disagreement; descriptive evidence only",
        )
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
        fig.tight_layout()
        save(fig, f"04_{cohort}_uncertainty_risk_coverage")

    fig, (ax, counts) = plt.subplots(
        2, 1, figsize=(7, 5.3), gridspec_kw={"height_ratios": [4, 1.3]}
    )
    tick_days = np.array([0, 365, 1095, 1825, 3650])
    count_rows = []
    for cohort, label, color in [
        ("internal", "TCGA-BRCA held-out", COLORS[2]),
        ("external", "METABRIC external", COLORS[3]),
    ]:
        rows = predictions[
            predictions.cohort.eq(cohort) & predictions.model.eq("age_only_cox")
        ]
        steps = kaplan_meier(rows.time, rows.event)
        times = np.r_[0, [r["time"] for r in steps]] / 365.25
        ax.step(
            times,
            np.r_[1, [r["survival"] for r in steps]],
            where="post",
            color=color,
            label=f"{label}: n={len(rows)}, deaths={int(rows.event.sum())}",
        )
        ax.fill_between(
            times,
            np.r_[1, [r["lower_95"] for r in steps]],
            np.r_[1, [r["upper_95"] for r in steps]],
            step="post",
            color=color,
            alpha=0.12,
        )
        count_rows.append([label] + [int((rows.time >= t).sum()) for t in tick_days])
    ax.set(
        xlim=(0, 10),
        ylim=(0, 1.02),
        xlabel="Years from recorded endpoint origin",
        ylabel="Observed overall survival",
        title="Observed survival in the analyzed cohorts\nCohorts differ; these curves do not measure model performance",
    )
    ax.legend(fontsize=8, loc="lower left")
    counts.axis("off")
    counts.table(
        cellText=count_rows,
        colLabels=["Number at risk"] + [f"{t / 365.25:.1f}y" for t in tick_days],
        cellLoc="center",
        colWidths=[0.4] + [0.12] * 5,
        loc="center",
    ).auto_set_font_size(False)
    fig.tight_layout()
    save(fig, "05_observed_cohort_survival")
    dump(
        output / "figure_provenance.json",
        {
            "scope": manifest["scope"],
            "representation": manifest["representation"],
            "figures": saved,
            "horizon_days": horizon,
            "uncertainty_diagnostics": uncertainty,
            "sources_sha256": {
                name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                for name in [
                    "run_manifest.json",
                    "metrics.json",
                    "predictions.csv",
                    "patient_splits.csv",
                ]
            },
            "censoring_reference": {
                "internal": "TCGA training outcomes",
                "external": "METABRIC evaluation outcomes; model and transforms frozen",
            },
            "limitations": [
                "Exploratory cross-assay representation; clinical validation is not established.",
                "Shared-rank transport does not prove platform invariance.",
                "Paired intervals are not multiplicity-adjusted.",
                "Calibration and uncertainty diagnostics are descriptive under independent censoring.",
            ],
        },
    )
    print(json.dumps({"output": str(output), "figures": saved}))


if __name__ == "__main__":
    main()
