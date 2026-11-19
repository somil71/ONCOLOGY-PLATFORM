"""Exploratory frozen CLI(age)+RNA transfer using shared within-patient ranks.

RNA-seq logTPM and microarray values are never passed through a common raw-scale
normalizer. The shared-rank representation defines a separate model and has no
claim of established platform invariance. No target outcomes tune any model.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from src.survival_models import RegularizedCox, HazardFusion, train_hazard
from src.survival_transport import shared_gene_symbols, rank_expression
from src.survival_metrics import harrell_c, bootstrap_concordance
from tools.run_survival_experiments import dump, fit_transform, metric


def align(frame, assay, symbols):
    patient_ids = assay["patient_ids"].astype(str)
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Duplicate expression patients")
    lookup = {p: i for i, p in enumerate(patient_ids)}
    frame = frame.loc[frame.Patient_ID.isin(lookup)].reset_index(drop=True)
    expression = assay["values"][[lookup[p] for p in frame.Patient_ID]]
    return frame, rank_expression(
        expression, assay["gene_symbols"].astype(str), symbols
    )


def frozen_transform(values, parameters):
    return (
        (
            np.where(np.isfinite(values), values, np.array(parameters["median"]))
            - parameters["mean"]
        )
        / parameters["scale"]
    ).astype("float32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-clinical",
        type=Path,
        default=Path("data/processed/tcga_brca_clinical.csv"),
    )
    parser.add_argument(
        "--source-rna",
        type=Path,
        default=Path("data/processed/tcga_brca_rna_logtpm.npz"),
    )
    parser.add_argument(
        "--target-clinical",
        type=Path,
        default=Path("data/processed/metabric_confirmed_carcinoma_clinical.csv"),
    )
    parser.add_argument(
        "--target-rna",
        type=Path,
        default=Path("data/processed/metabric_microarray.npz"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--genes", type=int, default=256)
    args = parser.parse_args()
    if args.epochs < 1 or args.genes < 1:
        parser.error("--epochs and --genes must be positive integers")
    start = time.perf_counter()
    cpu_start = time.process_time()
    torch.set_num_threads(4)
    source_assay = np.load(args.source_rna, allow_pickle=False)
    target_assay = np.load(args.target_rna, allow_pickle=False)
    symbols = shared_gene_symbols(
        source_assay["gene_symbols"], target_assay["gene_symbols"]
    )
    if len(symbols) < args.genes:
        raise ValueError("Requested gene count exceeds the shared assay gene contract")
    source, source_rank = align(
        pd.read_csv(args.source_clinical), source_assay, symbols
    )
    target, target_rank = align(
        pd.read_csv(args.target_clinical), target_assay, symbols
    )
    if source.Patient_ID.duplicated().any() or target.Patient_ID.duplicated().any():
        raise ValueError("Duplicate cohort patients")
    train, holdout = train_test_split(
        np.arange(len(source)), test_size=0.4, random_state=42, stratify=source.event
    )
    validation, test = train_test_split(
        holdout, test_size=0.5, random_state=42, stratify=source.event.iloc[holdout]
    )
    split = {"train": train, "validation": validation, "test": test}
    variance = source_rank[train].var(axis=0)
    selected = np.argsort(variance, kind="stable")[-args.genes :]
    source_features = {}
    target_features = {}
    preprocessing = {}
    for mode, raw, external in [
        ("cli", source[["age_years"]].to_numpy(), target[["age_years"]].to_numpy()),
        ("rna", source_rank[:, selected], target_rank[:, selected]),
    ]:
        source_features[mode], preprocessing[mode] = fit_transform(raw[train], raw)
        target_features[mode] = frozen_transform(external, preprocessing[mode])
    args.output.mkdir(parents=True, exist_ok=False)
    source[["Patient_ID", "time", "event"]].assign(
        split=[
            "train"
            if i in set(train)
            else "validation"
            if i in set(validation)
            else "test"
            for i in range(len(source))
        ]
    ).to_csv(args.output / "patient_splits.csv", index=False)
    manifest = {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "horizons_days": [365.0, 1095.0, 1825.0],
        "code_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                Path("src/survival_models.py"),
                Path("src/survival_metrics.py"),
                Path("src/survival_transport.py"),
                Path("tools/run_survival_experiments.py"),
            ]
        },
        "scope": "Exploratory cross-assay rank transport; legacy patient reuse, not untouched prospective validation",
        "features": {
            "cli": ["age_years"],
            "rna_shared_gene_symbols": symbols,
            "selected_gene_symbols": [symbols[i] for i in selected],
        },
        "preprocessing": preprocessing,
        "sources": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in vars(args).items()
            if name
            in ["source_clinical", "source_rna", "target_clinical", "target_rna"]
        },
        "partitions": {
            n: {"patients": len(i), "events": int(source.event.iloc[i].sum())}
            for n, i in split.items()
        },
        "target": {"patients": len(target), "events": int(target.event.sum())},
        "seeds": [42, 123, 2026],
        "training_parameters": {
            "epochs": args.epochs,
            "gene_count": args.genes,
            "hidden_dimension": 32,
            "learning_rate": 0.003,
            "weight_decay": 0.01,
            "early_stopping_patience": 35,
            "selection": "validation censored likelihood; no external outcome selection",
        },
        "representation": "Per-patient shared-gene average ranks; duplicate gene symbols averaged; patient-median fill; top variance selection and scaling source-training-only",
    }
    dump(args.output / "run_manifest.json", manifest)
    horizons = [365.0, 1095.0, 1825.0]
    boundaries = [365.0, 1095.0, 1825.0, 3650.0]
    summary = {}
    records = []
    candidates = []
    for penalty in [0.0, 0.01, 0.1, 1.0]:
        model = RegularizedCox(penalty).fit(
            source_features["cli"][train],
            source.time.iloc[train],
            source.event.iloc[train],
        )
        candidates.append(
            (
                harrell_c(
                    source.time.iloc[validation],
                    source.event.iloc[validation],
                    model.predict_risk(source_features["cli"][validation]),
                ),
                penalty,
                model,
            )
        )
    _, penalty, model = max(candidates, key=lambda item: item[0])
    dump(
        args.output / "age_only_cox_bundle.json",
        {
            "columns": ["age_years"],
            "preprocessing": preprocessing["cli"],
            "penalty": penalty,
            "coefficients": model.coef_.tolist(),
            "event_times": model.event_times_.tolist(),
            "baseline_cumulative_hazard": model.baseline_cumulative_hazard_.tolist(),
            "optimizer": model.optimizer_,
        },
    )
    summary["age_only_cox"] = {"penalty": penalty}
    for cohort, frame, features, index in [
        ("internal", source, source_features, test),
        ("external", target, target_features, np.arange(len(target))),
    ]:
        risk = model.predict_risk(features["cli"][index])
        probability = model.predict_survival(features["cli"][index], horizons)
        censor_t, censor_e = (
            (source.time.iloc[train], source.event.iloc[train])
            if cohort == "internal"
            else (target.time, target.event)
        )
        summary["age_only_cox"][cohort] = metric(
            frame.time.iloc[index].to_numpy(),
            frame.event.iloc[index].to_numpy(),
            risk,
            np.asarray(censor_t),
            np.asarray(censor_e),
            probability,
            horizons,
            1000,
        )
        for j, i in enumerate(index):
            records.append(
                {
                    "cohort": cohort,
                    "model": "age_only_cox",
                    **frame.iloc[i][["Patient_ID", "time", "event"]].to_dict(),
                    "risk": float(risk[j]),
                    **{
                        f"survival_{int(h)}": float(probability[j, k])
                        for k, h in enumerate(horizons)
                    },
                }
            )
    for label, modes, fusion in [
        ("unimodal_rna", ["rna"], "mean"),
        ("fusion_concat", ["cli", "rna"], "concat"),
        ("fusion_mean", ["cli", "rna"], "mean"),
        ("fusion_attention", ["cli", "rna"], "attention"),
    ]:
        features = {
            n: {m: torch.tensor(source_features[m][ix]) for m in modes}
            for n, ix in split.items()
        }
        masks = {
            n: torch.ones((len(ix), len(modes)), dtype=torch.bool)
            for n, ix in split.items()
        }
        targets = {
            n: (
                torch.tensor(source.time.iloc[ix].to_numpy(), dtype=torch.float32),
                torch.tensor(source.event.iloc[ix].to_numpy(), dtype=torch.float32),
            )
            for n, ix in split.items()
        }
        ext_features = {m: torch.tensor(target_features[m]) for m in modes}
        ext_mask = torch.ones((len(target), len(modes)), dtype=torch.bool)
        members = {"internal": [], "external": []}
        histories = []
        for seed in [42, 123, 2026]:
            torch.manual_seed(seed)
            model = HazardFusion(
                {m: source_features[m].shape[1] for m in modes},
                boundaries,
                fusion=fusion,
            )
            history = train_hazard(
                model,
                features["train"],
                masks["train"],
                *targets["train"],
                features["validation"],
                masks["validation"],
                *targets["validation"],
                epochs=args.epochs,
                seed=seed,
            )
            pd.DataFrame(history).to_csv(
                args.output / f"{label}_seed{seed}_history.csv", index=False
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "dimensions": {m: source_features[m].shape[1] for m in modes},
                    "modalities": modes,
                    "boundaries": boundaries,
                    "fusion": fusion,
                    "seed": seed,
                },
                args.output / f"{label}_seed{seed}.pth",
            )
            histories.append(
                {
                    "seed": seed,
                    "epochs": len(history),
                    "best_validation_nll": min(r["validation_nll"] for r in history),
                }
            )
            members["internal"].append(
                model.predict_survival(features["test"], masks["test"], horizons)
            )
            members["external"].append(
                model.predict_survival(ext_features, ext_mask, horizons)
            )
        summary[label] = {"members": histories}
        for cohort, frame, index in [
            ("internal", source, test),
            ("external", target, np.arange(len(target))),
        ]:
            probability = np.mean(members[cohort], axis=0)
            spread = np.std(members[cohort], axis=0)
            risk = -probability.mean(1)
            censor_t, censor_e = (
                (source.time.iloc[train], source.event.iloc[train])
                if cohort == "internal"
                else (target.time, target.event)
            )
            summary[label][cohort] = metric(
                frame.time.iloc[index].to_numpy(),
                frame.event.iloc[index].to_numpy(),
                risk,
                np.asarray(censor_t),
                np.asarray(censor_e),
                probability,
                horizons,
                1000,
            )
            for j, i in enumerate(index):
                records.append(
                    {
                        "cohort": cohort,
                        "model": label,
                        **frame.iloc[i][["Patient_ID", "time", "event"]].to_dict(),
                        "risk": float(risk[j]),
                        **{
                            f"survival_{int(h)}": float(probability[j, k])
                            for k, h in enumerate(horizons)
                        },
                        **{
                            f"ensemble_sd_{int(h)}": float(spread[j, k])
                            for k, h in enumerate(horizons)
                        },
                    }
                )
    result = pd.DataFrame(records)
    result.to_csv(args.output / "predictions.csv", index=False)
    for cohort in ["internal", "external"]:
        ref = result.loc[
            result.cohort.eq(cohort) & result.model.eq("age_only_cox"),
            ["Patient_ID", "risk"],
        ].rename(columns={"risk": "reference_risk"})
        for label in summary:
            if label == "age_only_cox":
                continue
            paired = result.loc[
                result.cohort.eq(cohort) & result.model.eq(label)
            ].merge(ref, on="Patient_ID", validate="one_to_one")
            summary[label][cohort]["paired_vs_age_only"] = bootstrap_concordance(
                paired.time,
                paired.event,
                paired.risk,
                reference_risk=paired.reference_risk,
            )
    manifest["status"] = "completed"
    manifest["elapsed_seconds"] = time.perf_counter() - start
    manifest["cpu_process_seconds"] = time.process_time() - cpu_start
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    dump(args.output / "metrics.json", summary)
    manifest["artifacts_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.iterdir()
        if p.is_file() and p.name != "run_manifest.json"
    }
    dump(args.output / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "partitions": manifest["partitions"],
                "external": manifest["target"],
            }
        )
    )


if __name__ == "__main__":
    main()
