"""Hand-computed IPCW diagnostics and conditional bootstrap tests; fixtures only."""

import numpy as np
import pandas as pd

from src.uncertainty_audit import bootstrap_diagnostics, diagnostics


def main():
    p = np.array([0.2, 0.1, 0.4, 0.7])
    p1 = np.full(4, 0.5)
    u = np.array([0.1, 0.5, 0.3, 0.2])
    case = np.array([1, 0, 1, 0], bool)
    control = np.array([0, 0, 0, 1], bool)
    w = np.array([1, 0, 2, 2])
    result = diagnostics(p, p1, u, case, control, w)
    np.testing.assert_allclose(result["ensemble_ipcw_brier"], 0.135, atol=1e-14)
    np.testing.assert_allclose(result["seed42_ipcw_brier"], 0.3125, atol=1e-14)
    np.testing.assert_allclose(
        result["ensemble_minus_seed42_brier"], -0.1775, atol=1e-14
    )
    expected = np.corrcoef(
        np.repeat(u, w), np.repeat((p - control.astype(float)) ** 2, w)
    )[0, 1]
    np.testing.assert_allclose(
        result["ipcw_error_disagreement_correlation"], expected, atol=1e-14
    )
    # Unknown censored observation has zero IPCW weight; changing its predicted
    # value cannot invent an observed event-free target or change full Brier.
    changed = p.copy()
    changed[1] = 0.99
    np.testing.assert_allclose(
        diagnostics(changed, p1, u, case, control, w)["ensemble_ipcw_brier"],
        0.135,
        atol=1e-14,
    )
    tied = diagnostics(p, p1, np.ones(4), case, control, w)
    assert tied["ipcw_error_disagreement_correlation"] is None
    assert tied["selective_curve"][0]["coverage_actual"] == 1
    frame = pd.DataFrame(
        {
            "probability": p,
            "seed42_probability": p1,
            "disagreement": u,
            "case": case,
            "control": control,
            "weight": w,
            "fold": "fixed",
        }
    )
    a = bootstrap_diagnostics(frame, 100, 42)
    b = bootstrap_diagnostics(frame, 100, 42)
    assert a == b and 0 < a["bootstrap_valid_case_control_replicates"] < 100
    try:
        diagnostics(p, p1, u, np.zeros(4, bool), control, w)
    except ValueError:
        pass
    else:
        raise AssertionError("No-case uncertainty analysis accepted")
    print(
        "PASS: manual censored IPCW Brier/delta, independent integer-weight Pearson reference, unknown-censor invariance, tie handling, deterministic patient bootstrap and unsupported-status rejection; synthetic fixtures only."
    )


if __name__ == "__main__":
    main()
