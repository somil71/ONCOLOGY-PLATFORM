"""Conditional patient-bootstrap diagnostics for censored survival disagreement.

IPCW references and fitted predictions stay fixed. These statistics do not prove
calibrated confidence, model-selection uncertainty or conditional censoring
independence inside selected uncertainty strata.
"""

import numpy as np


def diagnostics(
    probability,
    single_probability,
    uncertainty,
    case,
    control,
    weight,
    coverages=(0.25, 0.5, 0.75, 1.0),
):
    p, p1, u, w = [
        np.asarray(x, float)
        for x in (probability, single_probability, uncertainty, weight)
    ]
    case, control = np.asarray(case, bool), np.asarray(control, bool)
    if not case.any() or not control.any():
        raise ValueError("Observed cases and beyond-horizon controls required")
    known = case | control
    error, error1 = (p - control.astype(float)) ** 2, (p1 - control.astype(float)) ** 2
    normalized = w[known] / w[known].sum()
    a, b = u[known], error[known]
    a, b = a - np.dot(normalized, a), b - np.dot(normalized, b)
    denominator = np.sqrt(np.dot(normalized, a * a) * np.dot(normalized, b * b))
    correlation = (
        float(np.dot(normalized, a * b) / denominator) if denominator > 0 else None
    )
    curve = []
    for coverage in coverages:
        ordered = np.argsort(u, kind="stable")
        limit = u[ordered[max(1, int(np.ceil(coverage * len(u)))) - 1]]
        chosen = (
            u <= limit
        )  # Include every uncertainty tie, avoiding ID-based decisions.
        curve.append(
            {
                "coverage_requested": float(coverage),
                "coverage_actual": float(chosen.mean()),
                "patients": int(chosen.sum()),
                "known_status": int(known[chosen].sum()),
                "threshold_ties": int((u == limit).sum()),
                "ipcw_brier": float(np.mean((w * error)[chosen]))
                if known[chosen].any()
                else None,
            }
        )
    return {
        "patients": len(p),
        "cases": int(case.sum()),
        "controls": int(control.sum()),
        "known_status": int(known.sum()),
        "ipcw_error_disagreement_correlation": correlation,
        "ensemble_ipcw_brier": float(np.mean(w * error)),
        "seed42_ipcw_brier": float(np.mean(w * error1)),
        "ensemble_minus_seed42_brier": float(np.mean(w * (error - error1))),
        "mean_disagreement": float(u.mean()),
        "max_ipcw_weight": float(w.max()),
        "selective_curve": curve,
    }


def bootstrap_diagnostics(frame, n_bootstrap=1000, seed=20261118):
    """Resample each patient once within their fixed outer fold, sharing all statistics."""
    fields = [
        "probability",
        "seed42_probability",
        "disagreement",
        "case",
        "control",
        "weight",
    ]
    values = [frame[name].to_numpy() for name in fields]
    point = diagnostics(*values)
    groups = [
        np.flatnonzero(frame.fold.to_numpy() == fold)
        for fold in sorted(frame.fold.unique())
    ]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        selected = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        try:
            draws.append(diagnostics(*(value[selected] for value in values)))
        except ValueError:
            continue
    for name in [
        "ipcw_error_disagreement_correlation",
        "ensemble_ipcw_brier",
        "seed42_ipcw_brier",
        "ensemble_minus_seed42_brier",
        "mean_disagreement",
    ]:
        valid = [
            draw[name]
            for draw in draws
            if draw[name] is not None and np.isfinite(draw[name])
        ]
        point[name + "_conditional95_ci"] = (
            np.quantile(valid, [0.025, 0.975]).tolist() if valid else None
        )
        point[name + "_valid_bootstraps"] = len(valid)
    for column, row in enumerate(point["selective_curve"]):
        valid = [
            draw["selective_curve"][column]["ipcw_brier"]
            for draw in draws
            if draw["selective_curve"][column]["ipcw_brier"] is not None
        ]
        row["conditional95_ci"] = (
            np.quantile(valid, [0.025, 0.975]).tolist() if valid else None
        )
    point["bootstrap_requested"] = n_bootstrap
    point["bootstrap_valid_case_control_replicates"] = len(draws)
    return point
