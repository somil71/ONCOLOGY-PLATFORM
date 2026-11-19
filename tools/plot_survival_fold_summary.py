"""Scientific figures for locked fold estimates and separate secondary OOF scores."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.survival_metrics import kaplan_meier
from tools.run_survival_experiments import dump


LABELS = {
    "clinical_cox": "Age-only Cox",
    "unimodal_cli": "Age-only neural",
    "unimodal_rna": "RNA ensemble",
    "fusion_concat": "Concatenation",
    "fusion_mean": "Mean fusion",
    "fusion_attention": "Attention fusion",
    "attention_no_mask": "Attention without masking",
    "attention_dropout_01": "Attention: dropout 0.1",
    "attention_dropout_03": "Attention: dropout 0.3",
    "attention_dropout_05": "Attention: dropout 0.5",
    "without_cli": "Retrained without clinical",
    "without_rna": "Retrained without RNA",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.runs_root
    summary = json.loads((root / "five_fold_summary.json").read_text())
    folds = pd.read_csv(root / "fold_metrics.csv")
    oof = pd.read_csv(root / "oof_predictions.csv")
    output = root / "paper_figures"
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

    names = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.7), sharey=True)
    for ax, key, point_name, ci_name, title in [
        (
            axes[0],
            "primary_locked_within_fold_summary",
            "mean_within_fold_c",
            "conditional_fold_stratified_patient_bootstrap_95_ci",
            "Primary: equal mean of within-fold C",
        ),
        (
            axes[1],
            "secondary_pooled_oof_common_probability_summary",
            "harrell_c",
            "bootstrap_95_ci",
            "Secondary: pooled OOF probability score",
        ),
    ]:
        values = [summary[key][name] for name in names]
        points = np.array([v[point_name] for v in values])
        intervals = np.array([v[ci_name] for v in values])
        ax.errorbar(
            points,
            np.arange(len(names)),
            xerr=[points - intervals[:, 0], intervals[:, 1] - points],
            fmt="o",
            color="#146c65",
            capsize=3,
        )
        ax.axvline(0.5, color=".6", ls="--", lw=1)
        ax.set(
            xlim=(0, 1),
            xlabel="C-index and conditional 95% patient-bootstrap CI",
            title=title,
            yticks=np.arange(len(names)),
            yticklabels=[LABELS[n] for n in names],
        )
    fig.suptitle(
        "Retrospective KIRC clinical + RNA: 217 patients, 66 OS deaths\nDifferent estimators; neither interval includes retraining uncertainty",
        fontsize=12,
    )
    fig.tight_layout()
    save(fig, "01_primary_fold_mean_and_secondary_oof")

    selected = [
        "clinical_cox",
        "unimodal_rna",
        "fusion_concat",
        "fusion_attention",
        "attention_dropout_03",
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in selected:
        values = folds[folds.model.eq(name)].sort_values("fold")
        ax.plot(np.arange(1, 6), values.within_fold_c, marker="o", label=LABELS[name])
    ax.set(
        xticks=np.arange(1, 6),
        xlabel="Locked outer test fold",
        ylabel="Held-out Harrell C-index",
        ylim=(0, 1),
        title="Fold-to-fold variation\nTest deaths: 14, 13, 13, 13, 13; no test-based configuration selection",
    )
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    save(fig, "02_locked_fold_variation")

    compare = [n for n in names if n != "clinical_cox"]
    point = np.array(
        [
            summary["primary_locked_within_fold_summary"][n]["model_minus_clinical_cox"]
            for n in compare
        ]
    )
    intervals = np.array(
        [
            summary["primary_locked_within_fold_summary"][n]["paired_delta_95_ci"]
            for n in compare
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 5.7))
    ax.errorbar(
        point,
        np.arange(len(compare)),
        xerr=[point - intervals[:, 0], intervals[:, 1] - point],
        fmt="o",
        color="#146c65",
        capsize=3,
    )
    ax.axvline(0, color=".6", ls="--", lw=1)
    ax.set(
        yticks=np.arange(len(compare)),
        yticklabels=[LABELS[n] for n in compare],
        xlabel="Mean within-fold C difference versus matched age-only Cox",
        title="Exploratory paired comparisons\nPatient resampling within fixed folds; intervals are not multiplicity-adjusted",
    )
    fig.tight_layout()
    save(fig, "03_primary_paired_differences")

    panel = summary["missing_input_panel"]
    values = {"fusion_attention": [], "attention_dropout_03": []}
    for name in values:
        per_fold = []
        for folder in sorted(p for p in root.glob("fold_*") if p.is_dir()):
            measured = json.loads((folder / "metrics.json").read_text())[name]
            per_fold.append(
                [
                    measured["test"]["harrell_c"],
                    measured["inference_withholding"]["cli"]["harrell_c"],
                    measured["inference_withholding"]["rna"]["harrell_c"],
                ]
            )
        values[name] = np.mean(per_fold, axis=0)
    fig, (ax, delta_ax) = plt.subplots(
        2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1.5]}
    )
    x = np.arange(3)
    for offset, name, color, label in [
        (-0.18, "fusion_attention", "#536471", "Masked attention, dropout 0"),
        (0.18, "attention_dropout_03", "#146c65", "Masked attention, dropout 0.3"),
    ]:
        ax.bar(x + offset, values[name], width=0.36, color=color, label=label)
    ax.set(
        xticks=x,
        xticklabels=["Both inputs", "Clinical only", "RNA only"],
        ylim=(0, 1),
        ylabel="Mean within-fold C",
        title="Exploratory two-input missingness comparison\nThree observed-input subsets; the four-input 15-subset hypothesis is untested",
    )
    ax.legend(fontsize=8)
    point = panel["dropout03_minus_dropout0"]
    low, high = panel["paired_conditional_fold_stratified_patient_bootstrap_95_ci"]
    delta_ax.errorbar(
        [point],
        [0],
        xerr=[[point - low], [high - point]],
        fmt="o",
        color="#146c65",
        capsize=4,
    )
    delta_ax.axvline(0, color=".6", ls="--", lw=1)
    delta_ax.set(
        yticks=[],
        xlabel="Dropout 0.3 minus 0: average subset C difference (paired conditional 95% CI)",
    )
    fig.tight_layout()
    save(fig, "04_exploratory_two_input_missingness")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for i, ax in enumerate(axes.flat):
        if i == 5:
            ax.axis("off")
            continue
        for name, color, label in [
            ("fusion_attention", "#536471", "Attention, dropout 0"),
            ("attention_dropout_03", "#146c65", "Attention, dropout 0.3"),
        ]:
            analysis = summary["uncertainty_by_fold"][f"fold_{i + 1:02d}/{name}"][
                "1825"
            ]
            if analysis["status"] == "evaluated":
                curve = analysis["selective_prediction"]
                ax.plot(
                    [r["coverage_actual"] for r in curve],
                    [r["ipcw_brier"] for r in curve],
                    color=color,
                    marker="o",
                    label=label,
                )
        ax.set(
            title=f"Outer fold {i + 1}",
            xlabel="Fraction retained",
            ylabel="IPCW Brier; lower is better",
            xlim=(0, 1.02),
        )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "5-year uncertainty diagnostics: retain least-uncertain cases first\nInconsistent risk-coverage curves do not establish calibrated confidence",
        fontsize=12,
    )
    fig.tight_layout()
    save(fig, "05_fold_specific_uncertainty_risk_coverage")

    rows = oof[oof.model.eq("clinical_cox")]
    if rows.Patient_ID.duplicated().any():
        raise ValueError("Observed cohort KM requires exactly one row per patient")
    steps = kaplan_meier(rows.time, rows.event)
    times = np.r_[0, [s["time"] for s in steps]] / 365.25
    fig, (ax, counts) = plt.subplots(
        2, 1, figsize=(7, 5.2), gridspec_kw={"height_ratios": [4, 1]}
    )
    ax.step(
        times, np.r_[1, [s["survival"] for s in steps]], where="post", color="#146c65"
    )
    ax.fill_between(
        times,
        np.r_[1, [s["lower_95"] for s in steps]],
        np.r_[1, [s["upper_95"] for s in steps]],
        step="post",
        color="#146c65",
        alpha=0.13,
    )
    ax.set(
        xlim=(0, 10),
        ylim=(0, 1.02),
        xlabel="Years from recorded CDR diagnosis origin",
        ylabel="Observed overall survival",
        title=f"Analyzed KIRC cohort: n={len(rows)}, OS deaths={int(rows.event.sum())}\nObserved outcome distribution; this is not a model performance curve",
    )
    ticks = [0, 365, 1095, 1825, 3650]
    counts.axis("off")
    counts.table(
        cellText=[[int((rows.time >= t).sum()) for t in ticks]],
        colLabels=[f"{t / 365.25:.1f}y" for t in ticks],
        rowLabels=["At risk"],
        loc="center",
        cellLoc="center",
    )
    fig.tight_layout()
    save(fig, "06_observed_cohort_kaplan_meier")
    dump(
        output / "figure_provenance.json",
        {
            "scope": summary["scope"],
            "figures": saved,
            "source_sha256": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in [
                    "five_fold_summary.json",
                    "pooled_oof_summary.json",
                    "oof_predictions.csv",
                    "fold_metrics.csv",
                ]
            },
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "bootstrap_limitation": summary["bootstrap"]["limitation"],
            "interpretation": "All multimodal paired improvements cross zero; four-input H1 remains untested and calibrated uncertainty is not established.",
        },
    )
    print(json.dumps({"output": str(output), "figures": saved}))


if __name__ == "__main__":
    main()
