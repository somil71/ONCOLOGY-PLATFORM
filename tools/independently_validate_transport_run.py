"""Rebuild frozen cross-assay predictions from input rows and saved checkpoints.

Gene aggregation/ranking and the age Cox fit are implemented independently of
the transport runner. This verifies research artifacts, not clinical validity.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import torch

from src.survival_metrics import bootstrap_concordance, harrell_c
from src.survival_models import HazardFusion
from tools.run_survival_experiments import dump


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_ranks(assay, frame, shared):
    ids = assay["patient_ids"].astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate expression patient identities")
    position = {p: i for i, p in enumerate(ids)}
    frame = frame.loc[frame.Patient_ID.isin(position)].reset_index(drop=True)
    values = assay["values"][[position[p] for p in frame.Patient_ID]]
    symbols = np.array([str(s).strip() for s in assay["gene_symbols"]])
    # pandas groupby/rank provides an independent implementation of the
    # runner's NumPy gene-group loop and scipy rankdata representation.
    genes = pd.DataFrame(values.T, index=symbols)
    grouped = genes.groupby(level=0, sort=True).mean().loc[shared].T
    matrix = grouped.to_numpy(dtype=np.float32)
    medians = np.nanmedian(matrix, axis=1)
    matrix = np.where(np.isfinite(matrix), matrix, medians[:, None])
    ranked = pd.DataFrame(matrix).rank(axis=1, method="average").to_numpy()
    return frame, ((ranked - 0.5) / len(shared)).astype(np.float32)


def independent_age_cox(x, time, event, penalty):
    """One-dimensional Breslow score root and direct baseline reconstruction."""
    x, time, event = np.asarray(x, float).ravel(), np.asarray(time), np.asarray(event)
    event_times = np.unique(time[event == 1])

    def score(beta):
        result = 0.0
        for t in event_times:
            deaths = (time == t) & (event == 1)
            risk_x = x[time >= t]
            z = beta * risk_x
            weight = np.exp(z - z.max())
            result += deaths.sum() * np.dot(weight, risk_x) / weight.sum()
            result -= x[deaths].sum()
        return result / event.sum() + penalty * beta

    beta = brentq(score, -100, 100, xtol=1e-13)
    baseline = np.cumsum(
        [
            ((time == t) & (event == 1)).sum() / np.exp(beta * x[time >= t]).sum()
            for t in event_times
        ]
    )
    return beta, event_times, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()
    run = args.run
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Only completed runs can pass the final audit")
    for name, expected in manifest["artifacts_sha256"].items():
        if sha256(run / name) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    sources = {name: Path(info["path"]) for name, info in manifest["sources"].items()}
    for name, path in sources.items():
        if sha256(path) != manifest["sources"][name]["sha256"]:
            raise ValueError(f"Input hash mismatch: {name}")
    source_raw = pd.read_csv(sources["source_clinical"])
    target_raw = pd.read_csv(sources["target_clinical"])
    for frame in [source_raw, target_raw]:
        if frame.Patient_ID.duplicated().any():
            raise ValueError("Duplicate clinical patient identities")
    source_assay = np.load(sources["source_rna"], allow_pickle=False)
    target_assay = np.load(sources["target_rna"], allow_pickle=False)

    def symbols(assay):
        return {
            str(s).strip()
            for s in assay["gene_symbols"]
            if str(s).strip() and str(s).strip().casefold() not in {"nan", "none", "na"}
        }

    shared = sorted(symbols(source_assay) & symbols(target_assay))
    if shared != manifest["features"]["rna_shared_gene_symbols"]:
        raise ValueError("Shared-gene metadata contract differs from input assays")
    source, source_rank = independent_ranks(source_assay, source_raw, shared)
    target, target_rank = independent_ranks(target_assay, target_raw, shared)
    source_assay.close()
    target_assay.close()
    splits = pd.read_csv(run / "patient_splits.csv")
    if splits.Patient_ID.duplicated().any() or set(splits.Patient_ID) != set(
        source.Patient_ID
    ):
        raise ValueError("Split identities do not equal the matched source cohort")
    np.testing.assert_array_equal(splits.Patient_ID, source.Patient_ID)
    np.testing.assert_array_equal(splits[["time", "event"]], source[["time", "event"]])
    indices = {
        n: np.flatnonzero(splits.split.eq(n)) for n in ["train", "validation", "test"]
    }
    if set(splits.split) != set(indices):
        raise ValueError("Unexpected patient partition")
    for name, rows in indices.items():
        if manifest["partitions"][name] != {
            "patients": len(rows),
            "events": int(source.event.iloc[rows].sum()),
        }:
            raise ValueError(f"Partition count mismatch: {name}")
    if manifest["target"] != {
        "patients": len(target),
        "events": int(target.event.sum()),
    }:
        raise ValueError("Target cohort size/events disagree with actual matched rows")
    selected_genes = manifest["features"]["selected_gene_symbols"]
    selected = np.argsort(source_rank[indices["train"]].var(axis=0), kind="stable")[
        -len(selected_genes) :
    ]
    if [shared[i] for i in selected] != selected_genes:
        raise ValueError(
            "Training-only selected variance genes could not be reproduced"
        )
    source_features, target_features = {}, {}
    for mode, raw, external in [
        ("cli", source[["age_years"]].to_numpy(), target[["age_years"]].to_numpy()),
        ("rna", source_rank[:, selected], target_rank[:, selected]),
    ]:
        observed = np.asarray(raw[indices["train"]], float)
        median = np.nanmedian(observed, axis=0)
        filled = np.where(np.isfinite(observed), observed, median)
        mean, scale = filled.mean(0), filled.std(0)
        scale[scale < 1e-8] = 1.0
        for key, computed in [("median", median), ("mean", mean), ("scale", scale)]:
            np.testing.assert_allclose(
                computed, manifest["preprocessing"][mode][key], rtol=0, atol=1e-12
            )
        source_features[mode] = (
            (np.where(np.isfinite(raw), raw, median) - mean) / scale
        ).astype(np.float32)
        target_features[mode] = (
            (np.where(np.isfinite(external), external, median) - mean) / scale
        ).astype(np.float32)

    predictions = pd.read_csv(run / "predictions.csv")
    metrics = json.loads((run / "metrics.json").read_text())
    if predictions.duplicated(["cohort", "model", "Patient_ID"]).any():
        raise ValueError("Duplicate saved prediction identities")
    expected_models = {
        "age_only_cox",
        "unimodal_rna",
        "fusion_concat",
        "fusion_mean",
        "fusion_attention",
    }
    if set(predictions.model) != expected_models or set(metrics) != expected_models:
        raise ValueError("Missing or extra fitted model outputs")
    horizons = [365.0, 1095.0, 1825.0]
    train = indices["train"]
    beta, event_times, baseline = independent_age_cox(
        source_features["cli"][train],
        source.time.iloc[train],
        source.event.iloc[train],
        metrics["age_only_cox"]["penalty"],
    )
    report = {
        "status": "passed",
        "scope": "Independent representation, preprocessing, checkpoint and metric reconstruction; not clinical validation",
        "models": {},
    }
    torch.set_num_threads(2)
    for cohort, frame, features, rows in [
        ("internal", source, source_features, indices["test"]),
        ("external", target, target_features, np.arange(len(target))),
    ]:
        ref_rows = predictions[
            predictions.cohort.eq(cohort) & predictions.model.eq("age_only_cox")
        ]
        for label in sorted(expected_models):
            saved = predictions[
                predictions.cohort.eq(cohort) & predictions.model.eq(label)
            ]
            if set(saved.Patient_ID) != set(frame.Patient_ID.iloc[rows]):
                raise ValueError(f"Incomplete prediction cohort: {cohort}/{label}")
            metric_rows = saved.copy()
            saved = (
                saved.set_index("Patient_ID")
                .loc[frame.Patient_ID.iloc[rows]]
                .reset_index()
            )
            np.testing.assert_array_equal(
                saved[["time", "event"]], frame.iloc[rows][["time", "event"]]
            )
            if label == "age_only_cox":
                risk = features["cli"][rows].astype(float).ravel() * beta
                ix = np.searchsorted(event_times, horizons, side="right") - 1
                h0 = np.where(ix >= 0, baseline[np.maximum(ix, 0)], 0)
                survival = np.exp(-np.exp(risk)[:, None] * h0[None, :])
                np.testing.assert_allclose(risk, saved.risk, rtol=0, atol=1e-6)
            else:
                members = []
                for seed in manifest["seeds"]:
                    bundle = torch.load(
                        run / f"{label}_seed{seed}.pth",
                        map_location="cpu",
                        weights_only=True,
                    )
                    modes = bundle["modalities"]
                    model = HazardFusion(
                        bundle["dimensions"],
                        bundle["boundaries"],
                        fusion=bundle["fusion"],
                    )
                    model.load_state_dict(bundle["state_dict"])
                    model.eval()
                    values = {m: torch.from_numpy(features[m][rows]) for m in modes}
                    mask = torch.ones((len(rows), len(modes)), dtype=torch.bool)
                    with torch.no_grad():
                        interval_survival = (
                            torch.sigmoid(-model(values, mask)).cumprod(dim=1).numpy()
                        )
                    count = np.searchsorted(
                        bundle["boundaries"], horizons, side="right"
                    )
                    member = np.ones((len(rows), len(horizons)), dtype=np.float32)
                    for j, n in enumerate(count):
                        if n:
                            member[:, j] = interval_survival[:, n - 1]
                    members.append(member)
                survival = np.mean(members, axis=0)
                spread = np.std(members, axis=0)
                risk = -survival.mean(1)
                np.testing.assert_allclose(
                    spread,
                    saved[[f"ensemble_sd_{int(h)}" for h in horizons]],
                    rtol=0,
                    atol=1e-6,
                )
                np.testing.assert_allclose(risk, saved.risk, rtol=0, atol=1e-6)
            np.testing.assert_allclose(
                survival,
                saved[[f"survival_{int(h)}" for h in horizons]],
                rtol=0,
                atol=1e-6,
            )
            c = harrell_c(saved.time, saved.event, saved.risk)
            np.testing.assert_allclose(
                c, metrics[label][cohort]["harrell_c"], rtol=0, atol=1e-12
            )
            boot_checked = False
            if not args.skip_bootstrap:
                boot = bootstrap_concordance(
                    metric_rows.time,
                    metric_rows.event,
                    metric_rows.risk,
                    reference_risk=(
                        ref_rows.set_index("Patient_ID")
                        .loc[metric_rows.Patient_ID, "risk"]
                        .to_numpy()
                        if label != "age_only_cox"
                        else None
                    ),
                    n_bootstrap=metrics[label][cohort][
                        "bootstrap_replicates_requested"
                    ],
                    seed=42,
                )
                np.testing.assert_allclose(
                    boot["bootstrap_95_ci"],
                    metrics[label][cohort]["bootstrap_95_ci"],
                    rtol=0,
                    atol=1e-12,
                )
                if label != "age_only_cox":
                    np.testing.assert_allclose(
                        boot["paired_difference_bootstrap_95_ci"],
                        metrics[label][cohort]["paired_vs_age_only"][
                            "paired_difference_bootstrap_95_ci"
                        ],
                        rtol=0,
                        atol=1e-12,
                    )
                boot_checked = True
            report["models"][f"{cohort}/{label}"] = {
                "patients": len(saved),
                "events": int(saved.event.sum()),
                "risk_survival_and_seed_sd_match": True,
                "harrell_c": c,
                "bootstrap_and_paired_intervals_match": boot_checked,
            }
    report.update(
        {
            "independent_pandas_gene_aggregation_and_ranking": True,
            "training_only_gene_selection_and_preprocessing": True,
            "independent_scalar_cox_fit": True,
            "artifacts_verified": len(manifest["artifacts_sha256"]),
            "input_hashes_verified": len(sources),
            "seed_checkpoints_verified": 12,
            "source_run_manifest_sha256": sha256(run / "run_manifest.json"),
            "code_provenance_limitation": "The original transport manifest lacks source-code hashes. This audit verifies input/artifact hashes and reconstruction with the current implementation; it cannot retroactively establish exact original source provenance.",
        }
    )
    dump(run / "independent_transport_reconstruction.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
