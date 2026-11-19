"""Run reproducible survival baselines/ablations on real patient-level CSV inputs.

Examples:
 python -m tools.run_survival_experiments --input data/processed/tcga_brca_clinical.csv
 python -m tools.run_survival_experiments --input ... --modalities-json config.json
Config JSON maps modality names to explicit numeric feature column lists. Inputs
are already patient-level extracted features; this runner does not manufacture
missing modalities, labels, images or biological annotations.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time as clock
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from src.clinical_contract import select_reviewed_clinical_features
from src.image_feature_cohort import safe_predictor_names
from src.survival_models import (
    RegularizedCox,
    HazardFusion,
    train_hazard,
    validate_outcomes,
)


def dump(path, value):
    """Publish a complete JSON artifact after flushing it to the same filesystem.

    Serializing first prevents an invalid metric from truncating an earlier file.
    The temporary file and atomic replace keep interruption during a write from
    exposing a partially written completed-run manifest to downstream consumers.
    """
    path = Path(path)
    payload = json.dumps(
        value,
        indent=2,
        allow_nan=False,
        default=lambda x: x.item() if hasattr(x, "item") else str(x),
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fit_transform(train, values):
    """Numeric median imputation/scaling fitted solely on training available rows."""
    train, values = np.asarray(train, float), np.asarray(values, float)
    if np.isinf(train).any() or np.isinf(values).any():
        raise ValueError("Infinite feature values")
    usable = np.isfinite(train).any(0)
    if not usable.all():
        raise ValueError(
            "An approved predictor has no observed training values; revise the contract."
        )
    median = np.nanmedian(train, axis=0)
    filled = np.where(np.isfinite(train), train, median)
    mean, scale = filled.mean(0), filled.std(0)
    scale[scale < 1e-8] = 1
    return ((np.where(np.isfinite(values), values, median) - mean) / scale).astype(
        "float32"
    ), {
        "median": median.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "output_dtype": "float32",
    }


def metric(
    time, event, risk, train_time, train_event, probabilities, horizons, nboot, seed=42
):
    from src.survival_metrics import evaluate_survival

    result = evaluate_survival(
        time,
        event,
        risk,
        train_time=train_time,
        train_event=train_event,
        survival_probabilities=probabilities,
        horizons=horizons,
        n_bootstrap=nboot,
        seed=seed,
    )
    return result


def locked_patient_assignments(path, patient_ids):
    """Read explicit patient partitions; never manufacture or replace assignments."""
    ledger = pd.read_csv(path)
    if set(ledger.columns) != {"Patient_ID", "split"}:
        raise ValueError("Fixed split CSV must contain only Patient_ID and split")
    if ledger.Patient_ID.duplicated().any() or set(ledger.Patient_ID) != set(
        patient_ids
    ):
        raise ValueError(
            "Fixed split IDs must exactly match the original input cohort, one row per patient"
        )
    if set(ledger.split) != {"train", "validation", "test"}:
        raise ValueError(
            "Fixed splits require nonempty train, validation and test partitions"
        )
    return ledger.set_index("Patient_ID").split.to_dict()


def verify_study_protocol(path, frame, clinical_columns, split_path):
    """Bind an explicit retrospective protocol to unchanged source outcomes/folds."""
    protocol = json.loads(path.read_text())
    source_path = Path(protocol["clinical_source"])
    if (
        hashlib.sha256(source_path.read_bytes()).hexdigest()
        != protocol["clinical_source_sha256"]
    ):
        raise ValueError("Clinical source changed after protocol lock")
    source = pd.read_csv(source_path)
    if frame.Patient_ID.duplicated().any() or source.Patient_ID.duplicated().any():
        raise ValueError("Duplicate clinical identity in protocol input")
    matched = frame[["Patient_ID", "time", "event"]].merge(
        source[["Patient_ID", "time", "event"]],
        on="Patient_ID",
        validate="one_to_one",
        suffixes=("", "_source"),
    )
    if len(matched) != len(frame) or not np.array_equal(
        matched[["time", "event"]].to_numpy(),
        matched[["time_source", "event_source"]].to_numpy(),
    ):
        raise ValueError(
            "Model cohort outcomes do not equal original protocol source outcomes"
        )
    if (
        "retrospective" not in protocol["study_scope"].lower()
        or protocol["endpoint"]["prospective_prediction_time_eligibility_verified"]
        is not False
    ):
        raise ValueError(
            "This admission requires an explicitly retrospective protocol without prospective eligibility claims"
        )
    allowed = [
        protocol["predictors"]["primary_clinical"],
        protocol["predictors"]["retrospective_sensitivity_clinical"],
    ]
    if clinical_columns not in allowed:
        raise ValueError(
            "Clinical predictors differ from the locked primary/sensitivity protocol"
        )
    if split_path is None or hashlib.sha256(
        split_path.read_bytes()
    ).hexdigest() not in {row["sha256"] for row in protocol["fold_ledgers"]}:
        raise ValueError("A protocol-locked patient split ledger is required")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "protocol_id": protocol["protocol_id"],
        "scope": protocol["study_scope"],
        "prospective_prediction_eligible": False,
    }


def uncertainty_if_supported(
    time, event, probabilities, uncertainty, train_time, train_event, horizon
):
    """Unsupported evaluation horizons are recorded, never invented or fatal."""
    from src.survival_metrics import evaluate_uncertainty

    try:
        result = evaluate_uncertainty(
            time, event, probabilities, uncertainty, train_time, train_event, horizon
        )
        result["status"] = "evaluated"
        return result
    except ValueError as exc:
        return {
            "horizon": float(horizon),
            "status": "not_evaluated",
            "reason": str(exc),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--modalities-json", type=Path)
    parser.add_argument(
        "--clinical-columns", nargs="+", default=["age_years", "pathologic_stage"]
    )
    parser.add_argument("--external", type=Path)
    parser.add_argument("--rna-npz", type=Path)
    parser.add_argument("--rna-genes", type=int, default=256)
    parser.add_argument("--complete-cohort", action="store_true")
    parser.add_argument(
        "--split-csv",
        type=Path,
        help="Locked Patient_ID,split ledger; no automatic resplitting after QC",
    )
    parser.add_argument(
        "--study-protocol",
        type=Path,
        help="Explicit retrospective protocol with source/fold hashes",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--cox-only", action="store_true")
    args = parser.parse_args()
    start = clock.perf_counter()
    cpu_start = clock.process_time()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    frame = pd.read_csv(args.input)
    if frame.Patient_ID.duplicated().any():
        raise ValueError("Patient IDs must be unique")
    validate_outcomes(frame.time, frame.event)
    modes = (
        json.loads(args.modalities_json.read_text())
        if args.modalities_json
        else {"cli": args.clinical_columns}
    )
    if "cli" not in modes:
        raise ValueError(
            "Explicit reviewed cli modality required for baseline comparison"
        )
    select_reviewed_clinical_features(frame, modes["cli"])
    assignments = (
        locked_patient_assignments(args.split_csv, frame.Patient_ID)
        if args.split_csv
        else None
    )
    study_protocol = (
        verify_study_protocol(args.study_protocol, frame, modes["cli"], args.split_csv)
        if args.study_protocol
        else None
    )
    feature_columns = [c for columns in modes.values() for c in columns]
    safe_predictor_names(feature_columns)
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("A feature cannot appear in multiple modalities")
    if any(
        c in ["Patient_ID", "time", "event", "OS", "OS.time"] for c in feature_columns
    ):
        raise ValueError("Labels/identifiers cannot be predictors")
    raw_modalities = {
        n: frame[columns].to_numpy(dtype=float) for n, columns in modes.items()
    }
    rna_provenance = None
    if args.rna_npz:
        if "rna" in modes:
            raise ValueError("RNA specified twice")
        rna = np.load(args.rna_npz, allow_pickle=False)
        rna_ids = rna["patient_ids"].astype(str)
        genes = rna["gene_ids"].astype(str)
        rna_values = rna["values"]
        if len(set(rna_ids)) != len(rna_ids):
            raise ValueError("Duplicated RNA patient identities")
        lookup = {patient: i for i, patient in enumerate(rna_ids)}
        matrix = np.full((len(frame), len(genes)), np.nan, dtype=np.float32)
        for i, patient in enumerate(frame.Patient_ID):
            if patient in lookup:
                matrix[i] = rna_values[lookup[patient]]
        raw_modalities["rna"] = matrix
        modes["rna"] = genes.tolist()
        rna_provenance = {
            "source": str(args.rna_npz),
            "sha256": hashlib.sha256(args.rna_npz.read_bytes()).hexdigest(),
            "transform": "provided log1p(TPM), training-only top-variance gene selection and median/scaling",
        }
    exclusions = []
    if args.complete_cohort:
        keep = np.stack(
            [np.isfinite(v).any(1) for v in raw_modalities.values()], axis=1
        ).all(1)
        for row in np.flatnonzero(~keep):
            exclusions.append(
                {
                    "Patient_ID": frame.Patient_ID.iloc[row],
                    "reason": "Missing requested input: "
                    + "+".join(
                        name
                        for name, values in raw_modalities.items()
                        if not np.isfinite(values[row]).any()
                    ),
                }
            )
        frame = frame.loc[keep].reset_index(drop=True)
        raw_modalities = {n: v[keep] for n, v in raw_modalities.items()}
    if assignments is None:
        train, holdout = train_test_split(
            np.arange(len(frame)), test_size=0.4, random_state=42, stratify=frame.event
        )
        validation, test = train_test_split(
            holdout, test_size=0.5, random_state=42, stratify=frame.event.iloc[holdout]
        )
        splits = {"train": train, "validation": validation, "test": test}
    else:
        retained_assignments = frame.Patient_ID.map(assignments)
        splits = {
            name: np.flatnonzero(retained_assignments.eq(name))
            for name in ["train", "validation", "test"]
        }
        if any(
            not len(rows) or frame.event.iloc[rows].sum() < 2
            for rows in splits.values()
        ):
            raise ValueError(
                "Data-constrained fixed split after QC: fewer than two events or no patients; resplitting is forbidden"
            )
        train, validation, test = (
            splits[name] for name in ["train", "validation", "test"]
        )
    if args.rna_npz:
        observed = raw_modalities["rna"][train]
        variance = np.nanvar(observed, axis=0)
        eligible = np.flatnonzero(np.isfinite(variance) & (variance > 0))
        if args.rna_genes < 1 or len(eligible) < args.rna_genes:
            raise ValueError("Insufficient variable training genes")
        selected = eligible[
            np.argsort(variance[eligible], kind="stable")[-args.rna_genes :]
        ]
        raw_modalities["rna"] = raw_modalities["rna"][:, selected]
        modes["rna"] = [modes["rna"][i] for i in selected]
        rna_provenance["selected_gene_ids"] = modes["rna"]
        rna_provenance["selection_fit_patient_count"] = len(train)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output or Path("RUN REPORT") / f"clean-survival-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    ledger = frame[["Patient_ID", "time", "event"]].copy()
    ledger["split"] = ""
    for name, rows in splits.items():
        ledger.loc[rows, "split"] = name
    ledger.to_csv(out / "patient_splits.csv", index=False)
    pd.DataFrame(exclusions, columns=["Patient_ID", "reason"]).to_csv(
        out / "excluded_patients.csv", index=False
    )
    manifest = {
        "run_id": out.name,
        "status": "running",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(
            (out / "patient_splits.csv").read_bytes()
        ).hexdigest(),
        "created_at_utc": stamp,
        "scope": "Exploratory clean-run heldout evaluation; previous legacy models used overlapping patients, so this is not a globally untouched cohort or prospective clinical validation",
        "study_protocol": study_protocol,
        "fixed_split_source": {
            "path": str(args.split_csv),
            "sha256": hashlib.sha256(args.split_csv.read_bytes()).hexdigest(),
        }
        if args.split_csv
        else None,
        "qc_exclusions": len(exclusions),
        "rna_provenance": rna_provenance,
        "features": modes,
        "seeds": args.seeds,
        "horizons_days": [365.0, 1095.0, 1825.0],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "code_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path("src/survival_models.py"),
                Path("src/survival_metrics.py"),
                Path("src/clinical_contract.py"),
                Path("src/image_feature_cohort.py"),
                Path(__file__),
            ]
        },
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True
            ).stdout
        ),
        "arguments": vars(args),
        "partitions": {
            n: {"patients": len(ix), "events": int(frame.event.iloc[ix].sum())}
            for n, ix in splits.items()
        },
    }
    dump(out / "run_manifest.json", manifest)
    horizons = manifest["horizons_days"]
    boundaries = [365.0, 1095.0, 1825.0, 3650.0]
    summary = {}
    prediction_records = []
    withholding_records = []
    # Independently compare restricted age-only and age+stage clinical baselines.
    clinical_sets = {"clinical_cox": modes["cli"]}
    if "age_years" in modes["cli"] and len(modes["cli"]) > 1:
        clinical_sets["age_only_cox"] = ["age_years"]
    for label, columns in clinical_sets.items():
        clinical_present = np.isfinite(frame[columns].to_numpy(float)).any(1)
        clinical_eligible = {
            name: rows[clinical_present[rows]] for name, rows in splits.items()
        }
        if any(
            not len(rows) or frame.event.iloc[rows].sum() < 2
            for rows in clinical_eligible.values()
        ):
            summary[label] = {
                "status": "data_constrained",
                "reason": "Clinical Cox requires actually available clinical inputs and at least two events in each partition",
            }
            continue
        clinical_outcomes = {
            name: (frame.time.iloc[rows].to_numpy(), frame.event.iloc[rows].to_numpy())
            for name, rows in clinical_eligible.items()
        }
        x, prep = fit_transform(
            frame.iloc[clinical_eligible["train"]][columns], frame[columns]
        )
        candidates = []
        from src.survival_metrics import harrell_c

        for penalty in [0.0, 0.01, 0.1, 1.0]:
            model = RegularizedCox(penalty).fit(
                x[clinical_eligible["train"]], *clinical_outcomes["train"]
            )
            c = harrell_c(
                *clinical_outcomes["validation"],
                model.predict_risk(x[clinical_eligible["validation"]]),
            )
            candidates.append((c if c is not None else -0.5, penalty, model))
        _, penalty, model = max(candidates, key=lambda v: v[0])
        risk = model.predict_risk(x[clinical_eligible["test"]])
        probs = model.predict_survival(x[clinical_eligible["test"]], horizons)
        summary[label] = {
            "penalty": penalty,
            "validation_candidates": [{"penalty": p, "c": c} for c, p, _ in candidates],
            "partition_counts": {
                name: {
                    "patients": len(rows),
                    "events": int(frame.event.iloc[rows].sum()),
                }
                for name, rows in clinical_eligible.items()
            },
            "test": metric(
                *clinical_outcomes["test"],
                risk,
                *clinical_outcomes["train"],
                probs,
                horizons,
                args.bootstrap,
                args.bootstrap_seed,
            ),
        }
        bundle = {
            "columns": columns,
            "preprocessing": prep,
            "coefficients": model.coef_.tolist(),
            "penalty": penalty,
            "event_times": model.event_times_.tolist(),
            "baseline_cumulative_hazard": model.baseline_cumulative_hazard_.tolist(),
            "optimizer": model.optimizer_,
        }
        dump(out / f"{label}_bundle.json", bundle)
        for i, row in enumerate(clinical_eligible["test"]):
            prediction_records.append(
                {
                    "model": label,
                    **frame.iloc[row][["Patient_ID", "time", "event"]].to_dict(),
                    "risk": float(risk[i]),
                    **{
                        f"survival_{int(h)}": float(probs[i, j])
                        for j, h in enumerate(horizons)
                    },
                }
            )
        if args.external and (
            label == "age_only_cox"
            or (label == "clinical_cox" and columns == ["age_years"])
        ):
            ext = pd.read_csv(args.external)
            validate_outcomes(ext.time, ext.event)
            if ext.Patient_ID.duplicated().any():
                raise ValueError("Duplicate external patient IDs")
            ep = np.array(prep["median"])
            ex = (
                ext[columns].fillna(dict(zip(columns, ep))).to_numpy() - prep["mean"]
            ) / prep["scale"]
            ex = ex.astype(np.float32)
            er = model.predict_risk(ex)
            es = model.predict_survival(ex, horizons)
            summary[label]["external"] = metric(
                ext.time.to_numpy(),
                ext.event.to_numpy(),
                er,
                ext.time.to_numpy(),
                ext.event.to_numpy(),
                es,
                horizons,
                args.bootstrap,
                args.bootstrap_seed,
            )
            summary[label]["external"]["censoring_reference"] = (
                "External evaluation outcomes for censoring nuisance only; prediction model and transforms frozen on TCGA"
            )
            pd.DataFrame(
                {
                    "Patient_ID": ext.Patient_ID,
                    "time": ext.time,
                    "event": ext.event,
                    "risk": er,
                    **{f"survival_{int(h)}": es[:, j] for j, h in enumerate(horizons)},
                }
            ).to_csv(out / "external_age_only_predictions.csv", index=False)
            summary[label]["external_input_sha256"] = hashlib.sha256(
                args.external.read_bytes()
            ).hexdigest()
    if not args.cox_only:
        names = list(modes)
        available = np.stack(
            [np.isfinite(v).any(axis=1) for v in raw_modalities.values()], axis=1
        )
        if not available.any(axis=1).all():
            raise ValueError(
                "Some patients have no input; revise cohort before training"
            )
        features = {}
        preprocessing = {}
        for i, (name, columns) in enumerate(modes.items()):
            train_present = train[available[train, i]]
            features[name], preprocessing[name] = fit_transform(
                raw_modalities[name][train_present], raw_modalities[name]
            )
        dump(out / "neural_preprocessing.json", preprocessing)
        configurations = [
            ("unimodal_" + name, [name], "mean", True, 0.0) for name in names
        ]
        if len(names) > 1:
            configurations += [
                ("fusion_concat", names, "concat", True, 0.0),
                ("fusion_mean", names, "mean", True, 0.0),
                ("fusion_attention", names, "attention", True, 0.0),
                ("attention_no_mask", names, "attention", False, 0.0),
                ("attention_dropout_01", names, "attention", True, 0.1),
                ("attention_dropout_03", names, "attention", True, 0.3),
                ("attention_dropout_05", names, "attention", True, 0.5),
            ]
            configurations += [
                (
                    "without_" + name,
                    [n for n in names if n != name],
                    "attention",
                    True,
                    0.0,
                )
                for name in names
            ]
        for label, selected, fusion, use_mask, dropout in configurations:
            si = [names.index(n) for n in selected]
            subavailable = available[:, si]
            eligible = {n: ix[subavailable[ix].any(1)] for n, ix in splits.items()}
            if any(
                not len(ix) or frame.event.iloc[ix].sum() < 2
                for ix in eligible.values()
            ):
                summary[label] = {
                    "status": "data_constrained",
                    "reason": "Fewer than two events or no patients in a partition",
                }
                continue
            tensors = {
                n: {mode: torch.tensor(features[mode][ix]) for mode in selected}
                for n, ix in eligible.items()
            }
            masks = {n: torch.tensor(subavailable[ix]) for n, ix in eligible.items()}
            targets = {
                n: (
                    torch.tensor(frame.time.iloc[ix].to_numpy(), dtype=torch.float32),
                    torch.tensor(frame.event.iloc[ix].to_numpy(), dtype=torch.float32),
                )
                for n, ix in eligible.items()
            }
            survival_members = []
            member_history = []
            withheld_members = {}
            for seed in args.seeds:
                torch.manual_seed(seed)
                model = HazardFusion(
                    {n: features[n].shape[1] for n in selected},
                    boundaries,
                    fusion=fusion,
                    use_mask=use_mask,
                )
                history = train_hazard(
                    model,
                    tensors["train"],
                    masks["train"],
                    *targets["train"],
                    tensors["validation"],
                    masks["validation"],
                    *targets["validation"],
                    epochs=args.epochs,
                    seed=seed,
                    modality_dropout=dropout,
                )
                survival_members.append(
                    model.predict_survival(tensors["test"], masks["test"], horizons)
                )
                if len(selected) > 1:
                    for size in range(1, len(selected)):
                        for kept in itertools.combinations(range(len(selected)), size):
                            kept_mask = torch.zeros_like(masks["test"])
                            kept_mask[:, list(kept)] = masks["test"][:, list(kept)]
                            # Fair paired withholding requires every selected real input.
                            complete = masks["test"].all(1)
                            if not complete.any():
                                continue
                            key = "+".join(selected[i] for i in kept)
                            subset_features = {
                                n: v[complete] for n, v in tensors["test"].items()
                            }
                            withheld_members.setdefault(key, []).append(
                                model.predict_survival(
                                    subset_features, kept_mask[complete], horizons
                                )
                            )
                member_history.append(
                    {
                        "seed": seed,
                        "epochs": len(history),
                        "training_epoch_wall_seconds": sum(
                            h["epoch_wall_seconds"] for h in history
                        ),
                        "training_epoch_cpu_process_seconds": sum(
                            h["epoch_cpu_process_seconds"] for h in history
                        ),
                        "best_validation_nll": min(
                            r["validation_nll"] for r in history
                        ),
                    }
                )
                pd.DataFrame(history).to_csv(
                    out / f"{label}_seed{seed}_history.csv", index=False
                )
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "modalities": selected,
                        "dimensions": {n: features[n].shape[1] for n in selected},
                        "boundaries": boundaries,
                        "fusion": fusion,
                        "use_mask": use_mask,
                        "seed": seed,
                    },
                    out / f"{label}_seed{seed}.pth",
                )
            probabilities = np.mean(survival_members, axis=0)
            spread = np.std(survival_members, axis=0)
            risk = -probabilities.mean(axis=1)
            ti = eligible["test"]
            tr = eligible["train"]
            summary[label] = {
                "members": member_history,
                "single_seed_c": {
                    str(seed): harrell_c(
                        frame.time.iloc[eligible["test"]],
                        frame.event.iloc[eligible["test"]],
                        -member.mean(axis=1),
                    )
                    for seed, member in zip(args.seeds, survival_members)
                },
                "modalities": selected,
                "test": metric(
                    frame.time.iloc[ti].to_numpy(),
                    frame.event.iloc[ti].to_numpy(),
                    risk,
                    frame.time.iloc[tr].to_numpy(),
                    frame.event.iloc[tr].to_numpy(),
                    probabilities,
                    horizons,
                    args.bootstrap,
                    args.bootstrap_seed,
                ),
                "uncertainty_scope": "Between-seed survival-probability standard deviation; not validated confidence",
                "partition_counts": {
                    n: {"patients": len(ix), "events": int(frame.event.iloc[ix].sum())}
                    for n, ix in eligible.items()
                },
            }
            summary[label]["uncertainty_analysis"] = {
                str(int(h)): uncertainty_if_supported(
                    frame.time.iloc[ti].to_numpy(),
                    frame.event.iloc[ti].to_numpy(),
                    probabilities[:, j],
                    spread[:, j],
                    frame.time.iloc[tr].to_numpy(),
                    frame.event.iloc[tr].to_numpy(),
                    h,
                )
                for j, h in enumerate(horizons)
            }
            summary[label]["inference_withholding"] = {}
            for kept, members in withheld_members.items():
                wi = ti[subavailable[ti].all(1)]
                wp = np.mean(members, axis=0)
                wr = -wp.mean(1)
                full_risk = risk[subavailable[ti].all(1)]
                wt, we = frame.time.iloc[wi].to_numpy(), frame.event.iloc[wi].to_numpy()
                summary[label]["inference_withholding"][kept] = metric(
                    wt,
                    we,
                    wr,
                    frame.time.iloc[tr].to_numpy(),
                    frame.event.iloc[tr].to_numpy(),
                    wp,
                    horizons,
                    args.bootstrap,
                    args.bootstrap_seed,
                )
                from src.survival_metrics import bootstrap_concordance

                summary[label]["inference_withholding"][kept][
                    "paired_vs_full_input"
                ] = bootstrap_concordance(
                    wt,
                    we,
                    wr,
                    reference_risk=full_risk,
                    n_bootstrap=args.bootstrap,
                    seed=args.bootstrap_seed,
                )
                for j, row in enumerate(wi):
                    withholding_records.append(
                        {
                            "model": label,
                            "kept_modalities": kept,
                            **frame.iloc[row][
                                ["Patient_ID", "time", "event"]
                            ].to_dict(),
                            "risk": float(wr[j]),
                            **{
                                f"survival_{int(h)}": float(wp[j, k])
                                for k, h in enumerate(horizons)
                            },
                        }
                    )
            for i, row in enumerate(ti):
                prediction_records.append(
                    {
                        "model": label,
                        **frame.iloc[row][["Patient_ID", "time", "event"]].to_dict(),
                        "risk": float(risk[i]),
                        **{
                            f"survival_{int(h)}": float(probabilities[i, j])
                            for j, h in enumerate(horizons)
                        },
                        **{
                            f"ensemble_sd_{int(h)}": float(spread[i, j])
                            for j, h in enumerate(horizons)
                        },
                    }
                )
    pd.DataFrame(prediction_records).to_csv(out / "test_predictions.csv", index=False)
    if withholding_records:
        pd.DataFrame(withholding_records).to_csv(
            out / "inference_withholding_predictions.csv", index=False
        )
    # Paired differences are restricted to the identical held-out patients.
    from src.survival_metrics import bootstrap_concordance

    table = pd.DataFrame(prediction_records)
    reference = table.loc[
        table.model.eq("clinical_cox"), ["Patient_ID", "risk"]
    ].rename(columns={"risk": "clinical_reference_risk"})
    for label in summary:
        if label == "clinical_cox":
            continue
        paired = table.loc[table.model.eq(label)].merge(
            reference, on="Patient_ID", validate="one_to_one"
        )
        if len(paired):
            summary[label]["paired_vs_clinical_cox"] = bootstrap_concordance(
                paired.time,
                paired.event,
                paired.risk,
                reference_risk=paired.clinical_reference_risk,
                n_bootstrap=args.bootstrap,
                seed=args.bootstrap_seed,
            )
    dump(out / "metrics.json", summary)
    dump(
        out / "runtime.json",
        {
            "elapsed_seconds": clock.perf_counter() - start,
            "cpu_process_seconds": clock.process_time() - cpu_start,
            "timing_note": "Wall duration may include scheduling/suspension; CPU process time sums process threads and is not GPU time.",
            "device": "cpu",
            "threads": torch.get_num_threads(),
        },
    )
    manifest["artifacts_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir()
        if p.is_file() and p.name != "run_manifest.json"
    }
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    dump(out / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(out),
                "partitions": manifest["partitions"],
                "models": list(summary),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
