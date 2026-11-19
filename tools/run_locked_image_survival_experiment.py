"""Gate a new retrospective image-feature experiment on a concrete QC cohort lock.

Delegates numeric survival fitting to the frozen shared engine. This separate
entry point preserves code hashes of completed clinical/RNA studies. Templates
cannot admit fitting, and validate-only never constructs or trains a model.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from src.image_cohort_lock import validate_image_cohort_lock
from tools.run_survival_experiments import (
    dump,
    locked_patient_assignments,
    verify_study_protocol,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    admission = validate_image_cohort_lock(
        args.image_cohort_lock,
        args.input,
        args.modalities_json,
        args.study_protocol,
        args.rna_npz,
        args.complete_cohort,
    )
    frame = pd.read_csv(args.input)
    modes = json.loads(args.modalities_json.read_text())
    assignments = locked_patient_assignments(args.split_csv, frame.Patient_ID)
    protocol = verify_study_protocol(
        args.study_protocol, frame, modes["cli"], args.split_csv
    )
    with np.load(args.rna_npz, allow_pickle=False) as rna:
        ids = rna["patient_ids"].astype(str)
        observed_rna = set(ids[np.isfinite(rna["values"]).any(1)])
    complete = np.stack(
        [np.isfinite(frame[cols].to_numpy(float)).any(1) for cols in modes.values()],
        axis=1,
    ).all(1)
    complete &= frame.Patient_ID.isin(observed_rna).to_numpy()
    retained = frame.loc[complete].copy()
    if sorted(retained.Patient_ID) != admission["admitted_patient_ids"]:
        raise ValueError(
            "Engine post-filter patient identities differ from the root-locked admitted cohort"
        )
    retained["split"] = retained.Patient_ID.map(assignments)
    counts = {
        name: {
            "patients": int(retained.split.eq(name).sum()),
            "observed_deaths": int(
                retained.loc[retained.split.eq(name), "event"].sum()
            ),
        }
        for name in ["train", "validation", "test"]
    }
    if any(
        value["patients"] == 0 or value["observed_deaths"] < 2
        for value in counts.values()
    ):
        raise ValueError(
            "Data-constrained locked image partition: fewer than two events or no patients; resplitting is forbidden"
        )
    if (
        args.epochs,
        args.rna_genes,
        args.bootstrap,
        args.bootstrap_seed,
        args.seeds,
    ) != (300, 256, 1000, 20261118, [42, 123, 2026]):
        raise ValueError(
            "This primary image runner uses the declared300epoch/256gene/three-seed/1000bootstrap protocol; deviations require a separately locked sensitivity runner"
        )
    command = [
        sys.executable,
        "-u",
        "-m",
        "tools.run_survival_experiments",
        "--input",
        str(args.input),
        "--modalities-json",
        str(args.modalities_json),
        "--rna-npz",
        str(args.rna_npz),
        "--rna-genes",
        str(args.rna_genes),
        "--complete-cohort",
        "--split-csv",
        str(args.split_csv),
        "--study-protocol",
        str(args.study_protocol),
        "--epochs",
        str(args.epochs),
        "--bootstrap",
        str(args.bootstrap),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
        "--seeds",
        *map(str, args.seeds),
        "--output",
        str(args.output),
    ]
    return {
        "schema": "cancerfusion.locked-image-execution.v1",
        "status": "validated_not_fitted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_cohort_lock": admission,
        "study_protocol": protocol,
        "retained_partition_counts": counts,
        "input_sha256": sha(args.input),
        "modalities_sha256": sha(args.modalities_json),
        "rna_sha256": sha(args.rna_npz),
        "fixed_split_sha256": sha(args.split_csv),
        "command": command,
        "representation": "Frozen valid-item mean image vectors; downstream training-only scaler. No patch-attention claim.",
        "prospective_prediction_eligible": False,
        "implementation_sha256": {
            name: sha(name)
            for name in [
                "tools/run_locked_image_survival_experiment.py",
                "src/image_cohort_lock.py",
                "src/image_feature_cohort.py",
                "tools/run_survival_experiments.py",
            ]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--modalities-json", type=Path, required=True)
    parser.add_argument("--rna-npz", type=Path, required=True)
    parser.add_argument("--study-protocol", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--image-cohort-lock", type=Path, required=True)
    parser.add_argument("--complete-cohort", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--rna-genes", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261118)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    record = prepare(args)
    if args.validate_only:
        print(json.dumps(record, indent=2))
        return
    if args.output.exists():
        raise ValueError(
            "Fresh output directory required; completed runs cannot be overwritten"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    admission_path = args.output.parent / (args.output.name + ".image-admission.json")
    if admission_path.exists():
        raise ValueError("An admission artifact already exists for this output name")
    dump(admission_path, record)
    subprocess.run(record["command"], check=True)
    after = prepare(args)
    for name in [
        "input_sha256",
        "modalities_sha256",
        "rna_sha256",
        "fixed_split_sha256",
        "image_cohort_lock",
    ]:
        if record[name] != after[name]:
            raise ValueError(
                "Locked inputs changed during fitting; do not promote this run"
            )
    saved = pd.read_csv(args.output / "patient_splits.csv")
    if sorted(saved.Patient_ID) != record["image_cohort_lock"]["admitted_patient_ids"]:
        raise ValueError(
            "Saved fitted population differs from root-locked admitted patients"
        )
    record.update(
        status="completed",
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        shared_run_manifest_sha256=sha(args.output / "run_manifest.json"),
    )
    dump(args.output / "locked_image_execution.json", record)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": record["status"],
                "admitted_patients": record["image_cohort_lock"]["admitted_patients"],
            }
        )
    )


if __name__ == "__main__":
    main()
