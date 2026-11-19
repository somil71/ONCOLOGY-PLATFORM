"""Independently reconstruct the adapted author MultiSurv RNA fixed-fold run."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.published_baselines import AuthorMultiSurvRNA, multisurv_source_provenance
from src.survival_metrics import bootstrap_concordance, evaluate_survival


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "RUN REPORT/kirc-multisurv-author-rna-20261118-v1"
REFERENCE = ROOT / "RUN REPORT/kirc-cli-rna-20261118-fixed5fold-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    root_manifest = json.loads((RUN / "run_manifest.json").read_text())
    config = root_manifest["config"]
    protocol_path = ROOT / "config/kirc_retrospective_os_protocol.json"
    if sha(protocol_path) != root_manifest["protocol_sha256"]:
        raise ValueError("Protocol changed")
    protocol = json.loads(protocol_path.read_text())
    clinical_path = ROOT / protocol["clinical_source"]
    rna_path = ROOT / "data/processed/tcga_kirc_rna_logtpm.npz"
    if (
        sha(clinical_path) != root_manifest["clinical_sha256"]
        or sha(rna_path) != root_manifest["rna_sha256"]
    ):
        raise ValueError("Source input hash mismatch")
    if multisurv_source_provenance() != root_manifest["source_provenance"]:
        raise ValueError("Author source provenance changed")
    for relative, expected in root_manifest["code_sha256"].items():
        if (
            sha(ROOT / relative) != expected
            or sha(RUN / "source_snapshot" / relative) != expected
        ):
            raise ValueError("Executed/source-snapshot implementation mismatch")
    clinical = (
        pd.read_csv(clinical_path).sort_values("Patient_ID").reset_index(drop=True)
    )
    original = pd.read_excel(
        ROOT / "data/tcga-brca/TCGA-CDR-SupplementalTableS1.xlsx", sheet_name="TCGA-CDR"
    )
    original = original.loc[original.type.eq("KIRC")].set_index(
        "bcr_patient_barcode", verify_integrity=True
    )
    np.testing.assert_array_equal(
        original.loc[clinical.Patient_ID, "OS.time"], clinical.time
    )
    np.testing.assert_array_equal(
        original.loc[clinical.Patient_ID, "OS"], clinical.event
    )
    with np.load(rna_path, allow_pickle=False) as source:
        ids, genes = source["patient_ids"].astype(str), source["gene_ids"].astype(str)
        lookup = {patient: n for n, patient in enumerate(ids)}
        matrix = np.stack([source["values"][lookup[p]] for p in clinical.Patient_ID])
    reports, fold_pairs, all_rows = [], [], []
    for locked in protocol["fold_ledgers"]:
        folder = RUN / f"fold_{locked['fold']:02d}"
        if not (folder / "run_manifest.json").exists():
            continue
        manifest = json.loads((folder / "run_manifest.json").read_text())
        for name, expected in manifest["artifacts_sha256"].items():
            if sha(folder / name) != expected:
                raise ValueError("Saved fold artifact changed")
        ledger_path = ROOT / locked["path"]
        if (
            sha(ledger_path) != locked["sha256"]
            or manifest["fixed_split_sha256"] != locked["sha256"]
        ):
            raise ValueError("Frozen assignment hash differs")
        frame = clinical.merge(
            pd.read_csv(ledger_path), on="Patient_ID", validate="one_to_one"
        )
        split = pd.read_csv(folder / "patient_splits.csv")
        if (
            frame[["Patient_ID", "time", "event", "split"]].to_numpy().tolist()
            != split.to_numpy().tolist()
        ):
            raise ValueError(
                "Saved partition/outcomes differ from original frozen source"
            )
        train_ix = np.flatnonzero(frame.split.eq("train"))
        val_ix = np.flatnonzero(frame.split.eq("validation"))
        test_ix = np.flatnonzero(frame.split.eq("test"))
        variance = np.var(matrix[train_ix], axis=0)
        eligible = np.flatnonzero(np.isfinite(variance) & (variance > 0))
        selected = eligible[np.argsort(variance[eligible], kind="stable")[-1000:]]
        preprocessing = json.loads((folder / "preprocessing.json").read_text())
        if (
            genes[selected].tolist() != preprocessing["selected_gene_ids"]
            or frame.Patient_ID.iloc[train_ix].tolist() != preprocessing["fit_patients"]
        ):
            raise ValueError("Gene selection order/training identities mismatch")
        selected_values = matrix[:, selected].astype(float)
        mean, scale = (
            selected_values[train_ix].mean(0),
            selected_values[train_ix].std(0),
        )
        scale[scale < 1e-8] = 1
        np.testing.assert_array_equal(mean, preprocessing["mean"])
        np.testing.assert_array_equal(scale, preprocessing["scale"])
        features = ((selected_values - mean) / scale).astype("float32")
        members = []
        for record in manifest["members"]:
            seed = record["seed"]
            bundle = torch.load(
                folder / f"seed_{seed}.pth", weights_only=True, map_location="cpu"
            )
            if bundle["source_commit"] != root_manifest["source_provenance"]["commit"]:
                raise ValueError(
                    "Checkpoint does not identify the pinned source commit"
                )
            model = AuthorMultiSurvRNA(bundle["breaks_days"])
            model.load_state_dict(bundle["state_dict"])
            model.eval()
            with torch.no_grad():
                val = torch.from_numpy(features[val_ix])
                actual_nll = float(
                    model.loss_from_days(
                        model(val),
                        frame.time.iloc[val_ix].to_numpy(),
                        frame.event.iloc[val_ix].to_numpy(),
                    )
                )
            np.testing.assert_allclose(
                actual_nll,
                record["best_inner_validation_author_nll"],
                atol=1e-8,
                rtol=0,
            )
            members.append(
                model.predict_survival(
                    torch.from_numpy(features[test_ix]), config["horizons_days"]
                )
            )
        probabilities, spread = np.mean(members, axis=0), np.std(members, axis=0)
        predictions = pd.read_csv(folder / "test_predictions.csv")
        test = frame.iloc[test_ix]
        if predictions.Patient_ID.tolist() != test.Patient_ID.tolist():
            raise ValueError("Outer prediction population/order mismatch")
        np.testing.assert_array_equal(
            predictions[["time", "event"]], test[["time", "event"]]
        )
        np.testing.assert_allclose(
            probabilities,
            predictions[[f"survival_{h}" for h in config["horizons_days"]]],
            atol=1e-6,
            rtol=0,
        )
        np.testing.assert_allclose(
            spread,
            predictions[[f"ensemble_sd_{h}" for h in config["horizons_days"]]],
            atol=1e-6,
            rtol=0,
        )
        np.testing.assert_allclose(
            -probabilities.mean(1), predictions.risk, atol=1e-6, rtol=0
        )
        # The source runner assigned float32 arrays to pandas columns, whose
        # default CSV rendering rounds their printed decimals. Reconstruct the
        # exact saved-model probabilities for arithmetic validation and export a
        # separate lossless table; preserve the original export and its hash.
        roundtrip = {
            "risk": float(
                np.max(np.abs(-probabilities.mean(1) - predictions.risk.to_numpy()))
            )
        }
        reconstructed = predictions.copy()
        reconstructed["risk"] = (-probabilities.mean(1)).astype(float)
        for column, horizon in enumerate(config["horizons_days"]):
            for prefix, values in (
                ("survival", probabilities),
                ("ensemble_sd", spread),
            ):
                name = f"{prefix}_{horizon}"
                roundtrip[name] = float(
                    np.max(np.abs(values[:, column] - predictions[name].to_numpy()))
                )
                reconstructed[name] = values[:, column].astype(float)
        if max(roundtrip.values()) > 1e-7:
            raise ValueError(
                "Prediction export differs beyond float32 decimal rendering precision"
            )
        reconstructed.to_csv(
            folder / "reconstructed_full_precision_predictions.csv", index=False
        )
        predictions = reconstructed
        recorded = json.loads((folder / "metrics.json").read_text())
        metric = evaluate_survival(
            test.time,
            test.event,
            predictions.risk,
            train_time=frame.time.iloc[train_ix],
            train_event=frame.event.iloc[train_ix],
            survival_probabilities=predictions[
                [f"survival_{h}" for h in config["horizons_days"]]
            ].to_numpy(),
            horizons=config["horizons_days"],
            n_bootstrap=1000,
            seed=20261118,
        )
        for name in ("harrell_c", "bootstrap_95_ci"):
            np.testing.assert_allclose(
                metric[name], recorded["test"][name], atol=1e-12, rtol=0
            )
        for actual, old in zip(metric["horizons"], recorded["test"]["horizons"]):
            if actual["status"] != old["status"]:
                raise ValueError("Horizon support changed")
            if actual["status"] == "evaluated":
                for name in ("ipcw_auc", "ipcw_brier", "uno_c_tau"):
                    np.testing.assert_allclose(
                        actual[name], old[name], atol=1e-9, rtol=0
                    )
        reference_path = REFERENCE / folder.name / "test_predictions.csv"
        if sha(reference_path) != manifest["reference_predictions_sha256"]:
            raise ValueError("Matched reference prediction source changed")
        refs = pd.read_csv(reference_path)
        paired_risks = [predictions.risk.to_numpy()]
        for label in ("clinical_cox", "unimodal_rna"):
            rows = (
                refs.loc[refs.model.eq(label)]
                .set_index("Patient_ID")
                .loc[test.Patient_ID]
            )
            np.testing.assert_array_equal(
                rows[["time", "event"]], test[["time", "event"]]
            )
            contrast = bootstrap_concordance(
                test.time,
                test.event,
                predictions.risk,
                reference_risk=rows.risk,
                n_bootstrap=1000,
                seed=20261118,
            )
            for field in (
                "model_minus_reference_c",
                "paired_difference_bootstrap_95_ci",
            ):
                np.testing.assert_allclose(
                    contrast[field],
                    recorded["paired_references"][label][field],
                    atol=1e-12,
                    rtol=0,
                )
            paired_risks.append(rows.risk.to_numpy())
        time, event = test.time.to_numpy(), test.event.to_numpy(bool)
        before = (time[:, None] < time[None, :]) | (
            (time[:, None] == time[None, :]) & ~event[None, :]
        )
        i, j = np.where(event[:, None] & before)
        risk = np.stack(paired_risks)
        delta = risk[:, i] - risk[:, j]
        credits = (delta > 1e-8) + 0.5 * (np.abs(delta) <= 1e-8)
        fold_pairs.append((len(test), i, j, credits))
        all_rows.append(predictions)
        result = {
            "fold": locked["fold"],
            "status": "passed",
            "artifact_hashes": len(manifest["artifacts_sha256"]),
            "source_outcomes_folds_training_gene_selection_and_scaling_match": True,
            "all_seed_checkpoints_validation_selection_predictions_and_sd_match": True,
            "metrics_cis_horizon_support_and_paired_references_match": True,
            "harrell_c": metric["harrell_c"],
            "original_float32_csv_roundtrip_max_abs_errors": roundtrip,
            "reconstructed_full_precision_export_sha256": sha(
                folder / "reconstructed_full_precision_predictions.csv"
            ),
            "export_precision_note": "Original CSV preserved; metrics validated against exact checkpoint-reconstructed float32 probabilities and separately lossless float64-rendered export",
        }
        (folder / "independent_validation.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        reports.append(result)
    aggregate = None
    if len(reports) == 5:
        rng, samples = np.random.default_rng(20261118), []
        for _ in range(1000):
            current = []
            for n, i, j, credit in fold_pairs:
                count = np.bincount(rng.integers(n, size=n), minlength=n)
                weights = count[i] * count[j]
                if weights.sum() == 0:
                    break
                current.append((credit * weights).sum(1) / weights.sum())
            if len(current) == 5:
                samples.append(np.mean(current, axis=0))
        samples = np.stack(samples)
        point = np.mean([item[3].mean(1) for item in fold_pairs], axis=0)
        oof = pd.concat(all_rows, ignore_index=True)
        if oof.Patient_ID.duplicated().any() or set(oof.Patient_ID) != set(
            clinical.Patient_ID
        ):
            raise ValueError("Each candidate must be predicted once in the outer folds")
        aggregate = {
            "mean_outer_fold_c": float(point[0]),
            "conditional_patient_bootstrap_95_ci": np.quantile(
                samples[:, 0], [0.025, 0.975]
            ).tolist(),
            "paired_mean_fold_deltas": {
                label: {
                    "delta": float(point[0] - point[k]),
                    "conditional95_ci": np.quantile(
                        samples[:, 0] - samples[:, k], [0.025, 0.975]
                    ).tolist(),
                }
                for k, label in [
                    (1, "matched_age_cox"),
                    (2, "matched_256gene_rna_reference"),
                ]
            },
            "patients": len(oof),
            "deaths": int(oof.event.sum()),
            "valid_bootstrap_replicates": len(samples),
            "warning": "Post hoc secondary adapted benchmark;1000gene author model versus256gene local RNA reference does not isolate architecture. Conditional fixed-prediction bootstrap excludes retraining selection uncertainty.",
        }
    report = {
        "status": "passed" if len(reports) == 5 else "partial_completed_fold_audit",
        "completed_folds": len(reports),
        "scope": root_manifest["scope"],
        "fold_reports": reports,
        "aggregate": aggregate,
        "root_manifest_sha256": sha(RUN / "run_manifest.json"),
    }
    (RUN / "independent_validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
