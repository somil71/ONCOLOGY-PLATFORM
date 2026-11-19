"""Reevaluate frozen external predictions using target-cohort censoring nuisance.

This command never fits the predictive model or alters saved probabilities.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.survival_metrics import evaluate_survival


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eligible-cohort", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output filename to preserve prior evaluations")
    frame = pd.read_csv(args.predictions)
    original_count = len(frame)
    if args.eligible_cohort:
        eligible = pd.read_csv(args.eligible_cohort)
        if eligible.Patient_ID.duplicated().any():
            raise ValueError("Duplicated eligible patient IDs")
        selected = eligible[["Patient_ID", "time", "event"]].merge(
            frame,
            on="Patient_ID",
            how="left",
            suffixes=("_eligible", ""),
            validate="one_to_one",
        )
        if (
            selected.risk.isna().any()
            or not ((selected.time_eligible - selected.time).abs() < 1e-7).all()
            or not selected.event_eligible.eq(selected.event).all()
        ):
            raise ValueError(
                "Eligibility subset does not match original recorded outcomes/predictions"
            )
        frame = selected.drop(columns=["time_eligible", "event_eligible"])
    if frame.Patient_ID.duplicated().any():
        raise ValueError("External predictions must have unique patients")
    horizons = sorted(
        float(c.removeprefix("survival_")) for c in frame if c.startswith("survival_")
    )
    probabilities = frame[[f"survival_{int(h)}" for h in horizons]].to_numpy()
    result = evaluate_survival(
        frame.time,
        frame.event,
        frame.risk,
        train_time=frame.time,
        train_event=frame.event,
        survival_probabilities=probabilities,
        horizons=horizons,
    )
    result["censoring_reference"] = (
        "External evaluation outcomes; evaluation nuisance only, frozen source model and predictions"
    )
    result["prediction_sha256"] = hashlib.sha256(
        args.predictions.read_bytes()
    ).hexdigest()
    result["independent_censoring_assumption"] = True
    if args.eligible_cohort:
        result["eligibility_cohort_sha256"] = hashlib.sha256(
            args.eligible_cohort.read_bytes()
        ).hexdigest()
        result["excluded_from_original_predictions"] = original_count - len(frame)
        frame.to_csv(args.output.with_suffix(".predictions.csv"), index=False)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "patients": result["patients"],
                "c": result["harrell_c"],
            }
        )
    )


if __name__ == "__main__":
    main()
