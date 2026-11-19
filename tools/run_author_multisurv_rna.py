"""Fit an explicitly adapted, pinned author MultiSurv mRNA model on locked KIRC folds.

Secondary benchmark added after the initial CLI/RNA analysis. Architecture and
loss are original; shared-cohort logTPM, folds and optimization are adaptations.
"""

import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from src.published_baselines import (
    AuthorMultiSurvRNA,
    MULTISURV_ADAPTATION_NOTICE,
    multisurv_source_provenance,
)
from src.survival_metrics import bootstrap_concordance, evaluate_survival


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "RUN REPORT/kirc-multisurv-author-rna-20261118-v1"
REFERENCE = ROOT / "RUN REPORT/kirc-cli-rna-20261118-fixed5fold-v1"
CONFIG = {
    "model": "MultiSurv-author-mRNA shared-cohort adaptation",
    "gene_count": 1000,
    "seeds": [42, 123, 2026],
    "optimizer": "Adam",
    "learning_rate": 0.0001,
    "weight_decay": 0.0,
    "max_epochs": 300,
    "patience": 35,
    "minimum_validation_improvement": 1e-7,
    "gradient_clip_norm": 5.0,
    "validation_selection": "Original author likelihood, inner validation only",
    "breaks_days": (np.arange(31) * 365).tolist(),
    "horizons_days": [365, 1095, 1825],
    "cpu_threads": 2,
    "bootstrap": 1000,
    "bootstrap_seed": 20261118,
    "analysis_status": "Secondary benchmark specified after initial CLI/RNA results; not a prespecified new primary contrast",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    path = Path(path)
    data = (
        value
        if isinstance(value, bytes)
        else json.dumps(value, indent=2, allow_nan=False).encode()
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    start = time.perf_counter()
    torch.set_num_threads(CONFIG["cpu_threads"])
    protocol_path = ROOT / "config/kirc_retrospective_os_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    clinical_path = ROOT / protocol["clinical_source"]
    if sha(clinical_path) != protocol["clinical_source_sha256"]:
        raise ValueError("Locked source clinical cohort changed")
    clinical = (
        pd.read_csv(clinical_path).sort_values("Patient_ID").reset_index(drop=True)
    )
    rna_path = ROOT / "data/processed/tcga_kirc_rna_logtpm.npz"
    with np.load(rna_path, allow_pickle=False) as source:
        ids = source["patient_ids"].astype(str)
        genes = source["gene_ids"].astype(str)
        values = source["values"]
        if len(ids) != len(set(ids)) or len(genes) != len(set(genes)):
            raise ValueError("Duplicate RNA identities/features")
        lookup = {p: n for n, p in enumerate(ids)}
        matrix = np.stack([values[lookup[p]] for p in clinical.Patient_ID])
    if not np.isfinite(matrix).all():
        raise ValueError(
            "This author mRNA-only benchmark requires observed finite RNA for every candidate"
        )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    code_paths = [
        Path(__file__),
        ROOT / "src/published_baselines.py",
        ROOT / "src/survival_metrics.py",
    ]
    run = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": CONFIG,
        "adaptation_notice": MULTISURV_ADAPTATION_NOTICE,
        "source_provenance": multisurv_source_provenance(),
        "protocol_sha256": sha(protocol_path),
        "clinical_sha256": sha(clinical_path),
        "rna_sha256": sha(rna_path),
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in code_paths},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "scope": "Separate retrospective KIRC mRNA-only adapted published baseline; no images, prospective claim or original pan-cancer reproduction",
    }
    for path in code_paths:
        snapshot = OUTPUT / "source_snapshot" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        publish(snapshot, path.read_bytes())
    publish(OUTPUT / "analysis_config.json", CONFIG)
    publish(OUTPUT / "run_manifest.json", run)
    for record in protocol["fold_ledgers"]:
        fold_start = time.perf_counter()
        ledger_path = ROOT / record["path"]
        if sha(ledger_path) != record["sha256"]:
            raise ValueError("Frozen patient assignment changed")
        ledger = pd.read_csv(ledger_path)
        if ledger.Patient_ID.duplicated().any() or set(ledger.Patient_ID) != set(
            clinical.Patient_ID
        ):
            raise ValueError("Invalid fixed patient-fold identities")
        frame = clinical.merge(ledger, on="Patient_ID", validate="one_to_one")
        indices = {
            n: np.flatnonzero(frame.split.eq(n))
            for n in ["train", "validation", "test"]
        }
        variance = np.var(matrix[indices["train"]], axis=0)
        eligible = np.flatnonzero(np.isfinite(variance) & (variance > 0))
        selected = eligible[
            np.argsort(variance[eligible], kind="stable")[-CONFIG["gene_count"] :]
        ]
        if len(selected) != CONFIG["gene_count"]:
            raise ValueError("Insufficient variable training genes")
        raw = matrix[:, selected].astype(float)
        train = raw[indices["train"]]
        mean, scale = train.mean(0), train.std(0)
        scale[scale < 1e-8] = 1
        x = ((raw - mean) / scale).astype("float32")
        folder = OUTPUT / f"fold_{record['fold']:02d}"
        folder.mkdir()
        frame[["Patient_ID", "time", "event", "split"]].to_csv(
            folder / "patient_splits.csv", index=False
        )
        preprocessing = {
            "selected_gene_ids": genes[selected].tolist(),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "fit_patients": frame.loc[indices["train"], "Patient_ID"].tolist(),
            "selection": "Training-only top1000variance genes; no imputation needed, all finite",
        }
        publish(folder / "preprocessing.json", preprocessing)
        tensors = {n: torch.from_numpy(x[ix]) for n, ix in indices.items()}
        times = {
            n: torch.tensor(frame.time.iloc[ix].to_numpy(), dtype=torch.float32)
            for n, ix in indices.items()
        }
        events = {
            n: torch.tensor(frame.event.iloc[ix].to_numpy(), dtype=torch.float32)
            for n, ix in indices.items()
        }
        members, member_records = [], []
        for seed in CONFIG["seeds"]:
            torch.manual_seed(seed)
            model = AuthorMultiSurvRNA(CONFIG["breaks_days"])
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=CONFIG["learning_rate"],
                weight_decay=CONFIG["weight_decay"],
            )
            best, best_state, stale, history = float("inf"), None, 0, []
            for epoch in range(CONFIG["max_epochs"]):
                epoch_start = time.perf_counter()
                model.train()
                optimizer.zero_grad()
                loss = model.loss_from_days(
                    model(tensors["train"]), times["train"], events["train"]
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite author survival training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG["gradient_clip_norm"]
                )
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    validation = float(
                        model.loss_from_days(
                            model(tensors["validation"]),
                            times["validation"],
                            events["validation"],
                        )
                    )
                history.append(
                    {
                        "epoch": epoch + 1,
                        "training_nll": float(loss.detach()),
                        "validation_nll": validation,
                        "epoch_wall_seconds": time.perf_counter() - epoch_start,
                    }
                )
                if validation < best - CONFIG["minimum_validation_improvement"]:
                    best, best_state, stale = (
                        validation,
                        copy.deepcopy(model.state_dict()),
                        0,
                    )
                else:
                    stale += 1
                if stale >= CONFIG["patience"]:
                    break
            model.load_state_dict(best_state)
            members.append(
                model.predict_survival(tensors["test"], CONFIG["horizons_days"])
            )
            pd.DataFrame(history).to_csv(
                folder / f"seed_{seed}_history.csv", index=False
            )
            payload = io.BytesIO()
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seed": seed,
                    "breaks_days": CONFIG["breaks_days"],
                    "source_commit": model.provenance["commit"],
                },
                payload,
            )
            publish(folder / f"seed_{seed}.pth", payload.getvalue())
            member_records.append(
                {
                    "seed": seed,
                    "epochs": len(history),
                    "best_inner_validation_author_nll": best,
                    "trainable_parameters": sum(p.numel() for p in model.parameters()),
                }
            )
            print(
                f"fold{record['fold']:02d} seed{seed} completed epochs={len(history)} innerNLL={best:.8f}",
                flush=True,
            )
        probability, spread = np.mean(members, axis=0), np.std(members, axis=0)
        test = frame.iloc[indices["test"]]
        predictions = test[["Patient_ID", "time", "event"]].copy()
        predictions["model"] = "multisurv_author_rna"
        predictions["risk"] = -probability.mean(1)
        for col, h in enumerate(CONFIG["horizons_days"]):
            predictions[f"survival_{h}"] = probability[:, col]
            predictions[f"ensemble_sd_{h}"] = spread[:, col]
        predictions.to_csv(folder / "test_predictions.csv", index=False)
        metrics = evaluate_survival(
            test.time,
            test.event,
            predictions.risk,
            train_time=frame.time.iloc[indices["train"]],
            train_event=frame.event.iloc[indices["train"]],
            survival_probabilities=probability,
            horizons=CONFIG["horizons_days"],
            n_bootstrap=CONFIG["bootstrap"],
            seed=CONFIG["bootstrap_seed"],
        )
        reference_path = REFERENCE / folder.name / "test_predictions.csv"
        reference = pd.read_csv(reference_path)
        contrasts = {}
        for name in ["clinical_cox", "unimodal_rna"]:
            ref = (
                reference.loc[reference.model.eq(name)]
                .set_index("Patient_ID")
                .loc[test.Patient_ID]
            )
            np.testing.assert_array_equal(
                ref[["time", "event"]], test[["time", "event"]]
            )
            contrasts[name] = bootstrap_concordance(
                test.time,
                test.event,
                predictions.risk,
                reference_risk=ref.risk,
                n_bootstrap=CONFIG["bootstrap"],
                seed=CONFIG["bootstrap_seed"],
            )
        publish(
            folder / "metrics.json",
            {
                "model": "multisurv_author_rna",
                "test": metrics,
                "paired_references": contrasts,
            },
        )
        manifest = {
            "status": "completed",
            "fold": record["fold"],
            "config": CONFIG,
            "members": member_records,
            "fixed_split_sha256": sha(ledger_path),
            "reference_predictions_sha256": sha(reference_path),
            "protocol_sha256": run["protocol_sha256"],
            "source_manifest_sha256": run["source_provenance"][
                "source_manifest_sha256"
            ],
            "wall_seconds": time.perf_counter() - fold_start,
            "artifacts_sha256": {
                p.name: sha(p) for p in folder.iterdir() if p.is_file()
            },
        }
        publish(folder / "run_manifest.json", manifest)
        print(
            f"fold{record['fold']:02d} complete C={metrics['harrell_c']:.6f}",
            flush=True,
        )
    run.update(
        status="completed",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter() - start,
    )
    publish(OUTPUT / "run_manifest.json", run)


if __name__ == "__main__":
    main()
