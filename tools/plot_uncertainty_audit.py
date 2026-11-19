"""Plot measured conditional uncertainty diagnostics without confidence claims."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "RUN REPORT/uncertainty-validation-20261118-v1"


def main():
    source = FOLDER / "uncertainty_report.json"
    report = json.loads(source.read_text())
    groups = report["groups"]
    out = FOLDER / "paper_figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    cohorts = ["KIRC_fixed5fold", "BRCA_heldout"]
    labels = {
        "KIRC_fixed5fold": "KIRC: fixed-fold OOF",
        "BRCA_heldout": "BRCA: held-out",
    }
    names = {
        "fusion_attention": "Attention",
        "fusion_concat": "Concatenation",
        "attention_dropout_03": "Attention + dropout0.3",
        "unimodal_rna": "RNA only",
    }
    colors = {
        "fusion_attention": "#176B87",
        "fusion_concat": "#9D5C0D",
        "attention_dropout_03": "#7A4D9B",
        "unimodal_rna": "#4D6F32",
    }

    def get(cohort, model, condition="all_inputs", horizon=1825):
        return next(
            x
            for x in groups
            if x["cohort"] == cohort
            and x["model"] == model
            and x["condition"] == condition
            and x["horizon"] == horizon
        )

    def save(fig, name):
        fig.savefig(out / (name + ".png"), dpi=220, bbox_inches="tight")
        fig.savefig(out / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharex=True)
    for ax, cohort in zip(axes, cohorts):
        for y, model in enumerate(names):
            g = get(cohort, model)
            v = g["ipcw_error_disagreement_correlation"]
            ci = g["ipcw_error_disagreement_correlation_conditional95_ci"]
            ax.errorbar(
                v,
                y,
                xerr=[[v - ci[0]], [ci[1] - v]],
                fmt="o",
                color=colors[model],
                capsize=3,
            )
        ax.axvline(0, color=".5", lw=0.8)
        ax.set_yticks(range(4), names.values())
        ax.set_title(labels[cohort])
        ax.set_xlabel("IPCW error–disagreement correlation")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Five-year association: conditional patient-bootstrap95% intervals")
    fig.tight_layout()
    save(fig, "five_year_error_disagreement_correlation")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for row, cohort in enumerate(cohorts):
        for col, model in enumerate(["fusion_concat", "fusion_attention"]):
            ax = axes[row, col]
            g = get(cohort, model)
            curve = g["selective_curve"]
            x = [p["coverage_actual"] for p in curve]
            y = [p["ipcw_brier"] for p in curve]
            lo = [p["conditional95_ci"][0] for p in curve]
            hi = [p["conditional95_ci"][1] for p in curve]
            ax.plot(x, y, "o-", color=colors[model], label="Three-seed ensemble")
            ax.fill_between(x, lo, hi, color=colors[model], alpha=0.12)
            ax.axhline(
                g["seed42_ipcw_brier"],
                ls="--",
                color=".4",
                label="Seed42: all patients",
            )
            ax.set_title(labels[cohort] + " / " + names[model])
            ax.set_xlabel("Coverage after retaining lower-disagreement cases")
            ax.set_ylabel("Five-year IPCW Brier (lower is better)")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
    fig.suptitle(
        "Selective prediction: ties retained; intervals conditional on fitted models"
    )
    fig.tight_layout()
    save(fig, "five_year_selective_brier")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    conditions = ["all_inputs", "cli_only", "rna_only"]
    ticks = ["Both inputs", "Clinical only", "RNA only"]
    for col, cohort in enumerate(cohorts):
        for model, offset in [("fusion_concat", -0.12), ("fusion_attention", 0.12)]:
            records = [get(cohort, model, c) for c in conditions]
            for row, field in enumerate(["mean_disagreement", "ensemble_ipcw_brier"]):
                point = np.array([g[field] for g in records])
                ci = np.array([g[field + "_conditional95_ci"] for g in records])
                axes[row, col].errorbar(
                    np.arange(3) + offset,
                    point,
                    yerr=np.stack([point - ci[:, 0], ci[:, 1] - point]),
                    fmt="o",
                    capsize=3,
                    label=names[model],
                    color=colors[model],
                )
                axes[row, col].set_xticks(range(3), ticks)
                axes[row, col].grid(axis="y", alpha=0.2)
                axes[row, col].set_ylabel(
                    "Mean between-seed survival SD"
                    if row == 0
                    else "Five-year IPCW Brier"
                )
        axes[0, col].set_title(labels[cohort])
        axes[0, col].legend(fontsize=7)
    fig.suptitle(
        "Actual-patient inference withholding; no naturally-missing cohort claim"
    )
    fig.tight_layout()
    save(fig, "missing_input_disagreement_and_brier")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
    for ax, cohort in zip(axes, cohorts):
        for y, model in enumerate(names):
            g = get(cohort, model)
            v = g["ensemble_minus_seed42_brier"]
            ci = g["ensemble_minus_seed42_brier_conditional95_ci"]
            ax.errorbar(
                v,
                y,
                xerr=[[v - ci[0]], [ci[1] - v]],
                fmt="o",
                color=colors[model],
                capsize=3,
            )
        ax.axvline(0, color=".5", lw=0.8)
        ax.set_yticks(range(4), names.values())
        ax.set_title(labels[cohort])
        ax.set_xlabel("Brier difference: three-seed ensemble minus seed42")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Five-year fixed-seed comparison: negative favours ensemble; conditional95% intervals"
    )
    fig.tight_layout()
    save(fig, "ensemble_vs_seed42_brier")
    provenance = {
        "source_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "Censored survival disagreement diagnostics; not calibrated confidence",
        "plots": [p.name for p in out.glob("*.png")],
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    (out / "figure_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print("Wrote4PNG+4PDF figures:", out)


if __name__ == "__main__":
    main()
