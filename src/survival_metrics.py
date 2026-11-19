"""Right-censored survival evaluation with explicit risk direction and support guards.

Censoring distributions use explicitly supplied reference outcomes. For external
evaluation, target-cohort evaluation outcomes may estimate this nuisance distribution
without refitting the predictive model; disclose the reference cohort.
These routines assume independent censoring; they do not correct predictor leakage.
"""
from __future__ import annotations

import numpy as np


def _outcomes(time, event):
    time, event = np.asarray(time, dtype=float), np.asarray(event)
    if time.ndim != 1 or event.shape != time.shape or time.size == 0:
        raise ValueError("Expected nonempty aligned one-dimensional outcomes.")
    if not np.isfinite(time).all() or np.any(time <= 0) or not np.isin(event, [0, 1]).all():
        raise ValueError("Require positive finite durations and binary event indicators.")
    return time, event.astype(bool)


def _risk(risk, size):
    risk = np.asarray(risk, dtype=float)
    if risk.shape != (size,) or not np.isfinite(risk).all():
        raise ValueError("Risk must be finite, aligned and one-dimensional.")
    return risk


def _pairs(time, event):
    # A failure precedes a later observation; event/censor ties are comparable.
    earlier = (time[:, None] < time[None, :]) | (
        (time[:, None] == time[None, :]) & ~event[None, :])
    return np.where(event[:, None] & earlier)


def _pair_credit(risk, i, j, tied_tol=1e-8):
    difference = risk[i] - risk[j]
    return (difference > tied_tol).astype(float) + 0.5 * (np.abs(difference) <= tied_tol)


def harrell_c(time, event, risk):
    """Higher risk predicts earlier failure. Undefined concordance returns None."""
    time, event = _outcomes(time, event)
    risk = _risk(risk, len(time))
    i, j = _pairs(time, event)
    return float(_pair_credit(risk, i, j).mean()) if len(i) else None


def bootstrap_concordance(time, event, risk, reference_risk=None, n_bootstrap=1000, seed=42):
    """Patient bootstrap; paired delta is model minus reference (positive is better)."""
    time, event = _outcomes(time, event)
    risk = _risk(risk, len(time))
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")
    i, j = _pairs(time, event)
    credit = _pair_credit(risk, i, j)
    reference = None if reference_risk is None else _risk(reference_risk, len(time))
    ref_credit = None if reference is None else _pair_credit(reference, i, j)
    rng, scores, differences = np.random.default_rng(seed), [], []
    for _ in range(n_bootstrap):
        counts = np.bincount(rng.integers(len(time), size=len(time)), minlength=len(time))
        pair_weights = counts[i] * counts[j]
        denominator = pair_weights.sum()
        if denominator:
            scores.append(float(np.sum(pair_weights * credit) / denominator))
            if reference is not None:
                differences.append(float(np.sum(pair_weights * (credit - ref_credit)) / denominator))
    result = {"harrell_c": harrell_c(time, event, risk),
              "bootstrap_95_ci": np.quantile(scores, [0.025, 0.975]).tolist() if scores else None,
              "bootstrap_replicates_requested": n_bootstrap,
              "bootstrap_replicates_valid": len(scores), "bootstrap_unit": "patient"}
    if reference is not None:
        point = result["harrell_c"]
        ref_point = harrell_c(time, event, reference)
        result.update({"reference_harrell_c": ref_point,
                       "model_minus_reference_c": None if point is None else point - ref_point,
                       "paired_difference_bootstrap_95_ci": np.quantile(differences, [0.025, 0.975]).tolist() if differences else None})
    return result


def kaplan_meier(time, event):
    """Observed survival steps, counts and Greenwood log-log 95% pointwise CIs."""
    time, event = _outcomes(time, event)
    survival, greenwood, rows = 1.0, 0.0, []
    for t in np.unique(time):
        at_risk = int((time >= t).sum())
        deaths = int((event & (time == t)).sum())
        censored = int((~event & (time == t)).sum())
        survival *= 1 - deaths / at_risk
        if 0 < deaths < at_risk:
            greenwood += deaths / (at_risk * (at_risk - deaths))
        if 0 < survival < 1:
            transform = np.log(-np.log(survival))
            se = np.sqrt(greenwood) / abs(np.log(survival))
            lower, upper = np.exp(-np.exp(transform + 1.95996398454 * se)), np.exp(-np.exp(transform - 1.95996398454 * se))
        else:
            lower = upper = survival
        rows.append({"time": float(t), "at_risk": at_risk, "events": deaths, "censored": censored,
                     "survival": float(survival), "lower_95": float(lower), "upper_95": float(upper)})
    return rows


class CensoringDistribution:
    """Reverse Kaplan-Meier: observed failures removed before tied censorings.

    Steps are right-continuous G(t). Event/censor tie handling follows the usual
    reverse KM convention; no arbitrary epsilon or clipping of zero G is used.
    """
    def __init__(self, time, event):
        time, event = _outcomes(time, event)
        self.maximum = float(time.max())
        self.times = np.unique(time)
        value, values = 1.0, []
        for t in self.times:
            deaths = int((event & (time == t)).sum())
            censored = int((~event & (time == t)).sum())
            denominator = int((time >= t).sum()) - deaths
            if censored:
                value *= 1 - censored / denominator
            values.append(value)
        self.values = np.asarray(values)

    def at(self, points):
        points = np.asarray(points, dtype=float)
        if not np.isfinite(points).all() or np.any(points > self.maximum):
            raise ValueError("Requested censoring weights outside development follow-up support.")
        index = np.searchsorted(self.times, points, side="right") - 1
        return np.where(index >= 0, self.values[np.maximum(index, 0)], 1.0)


def horizon_metrics(time, event, risk, train_time, train_event, horizon, survival_probability=None):
    """Cumulative/dynamic IPCW AUC and Graf IPCW Brier at a fixed horizon.

    Cases: observed failures by horizon. Controls: observed beyond horizon.
    Patients censored by horizon are not silently labelled event-free.
    """
    time, event = _outcomes(time, event)
    risk = _risk(risk, len(time))
    horizon = float(horizon)
    if not np.isfinite(horizon) or horizon < time.min() or horizon >= time.max():
        raise ValueError("Horizon must lie within evaluation follow-up [min, max).")
    censor = CensoringDistribution(train_time, train_event)
    g_horizon = float(censor.at(horizon))
    cases, controls = event & (time <= horizon), time > horizon
    if g_horizon <= 0 or not cases.any() or not controls.any():
        raise ValueError("Horizon lacks positive censoring support, observed cases or controls.")
    case_g = censor.at(time[cases])
    if np.any(case_g <= 0):
        raise ValueError("Zero censoring survival at an observed case time.")
    case_weights = 1.0 / case_g
    difference = risk[cases, None] - risk[None, controls]
    credit = (difference > 1e-8) + 0.5 * (np.abs(difference) <= 1e-8)
    auc = np.sum(case_weights[:, None] * credit) / (case_weights.sum() * controls.sum())
    result = {"horizon": horizon, "cases": int(cases.sum()), "controls": int(controls.sum()),
              "censored_by_horizon": int((~event & (time <= horizon)).sum()),
              "censoring_survival": g_horizon, "ipcw_auc": float(auc)}
    if survival_probability is not None:
        probability = _risk(survival_probability, len(time))
        if np.any((probability < 0) | (probability > 1)):
            raise ValueError("Survival probabilities must be in [0, 1].")
        contributions = np.zeros(len(time))
        contributions[cases] = probability[cases] ** 2 / case_g
        contributions[controls] = (1 - probability[controls]) ** 2 / g_horizon
        result["ipcw_brier"] = float(contributions.mean())
        result["mean_predicted_survival"] = float(probability.mean())
    return result


def uno_c(time, event, risk, train_time, train_event, tau):
    """IPCW concordance truncated at prespecified tau; higher risk is worse."""
    time, event = _outcomes(time, event)
    risk = _risk(risk, len(time))
    censor = CensoringDistribution(train_time, train_event)
    if not np.isfinite(tau) or tau <= 0 or float(censor.at(tau)) <= 0:
        raise ValueError("tau requires positive development censoring support.")
    i, j = _pairs(time, event)
    keep = time[i] < tau
    i, j = i[keep], j[keep]
    if not len(i):
        return None
    g = censor.at(time[i])
    if np.any(g <= 0):
        raise ValueError("Zero censoring survival at a comparable event.")
    weights = 1 / g ** 2
    return float(np.dot(weights, _pair_credit(risk, i, j)) / weights.sum())


def calibration_table(time, event, survival_probability, horizon, n_bins=4):
    """Descriptive equal-count calibration bins using observed within-bin KM.

    Bin KM assumes independent censoring within bins; small bins remain unstable.
    It does not fit a recalibration mapping or establish clinical calibration.
    """
    time, event = _outcomes(time, event)
    probability = _risk(survival_probability, len(time))
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("Calibration horizon must be finite and positive.")
    if np.any((probability < 0) | (probability > 1)) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("Invalid probabilities or calibration bin count.")
    rows = []
    for indices in np.array_split(np.argsort(probability, kind="stable"), min(n_bins, len(time))):
        steps = kaplan_meier(time[indices], event[indices])
        eligible_steps = [row for row in steps if row["time"] <= horizon]
        observed = eligible_steps[-1]["survival"] if eligible_steps else 1.0
        supported = bool((time[indices] > horizon).any()) or observed == 0
        rows.append({"patients": len(indices), "events_by_horizon": int((event[indices] & (time[indices] <= horizon)).sum()),
                     "at_risk_beyond_horizon": int((time[indices] > horizon).sum()),
                     "mean_predicted_survival": float(probability[indices].mean()),
                     "km_survival": float(observed) if supported else None,
                     "supported": supported})
    return rows


def evaluate_survival(time, event, risk, train_time=None, train_event=None,
                      survival_probabilities=None, horizons=None, n_bootstrap=1000, seed=42):
    """JSON-ready metrics; unsupported horizons have a reason, never fake scores.

    survival_probabilities must be [patients, horizons] for supplied horizons.
    train_time/train_event specify the censoring reference. For an external cohort
    use target evaluation outcomes when source censoring is not transportable; this
    estimates an evaluation nuisance and does not refit the predictive model.
    """
    time, event = _outcomes(time, event)
    risk = _risk(risk, len(time))
    result = {"patients": len(time), "observed_events": int(event.sum()),
              **bootstrap_concordance(time, event, risk, n_bootstrap=n_bootstrap, seed=seed),
              "risk_direction": "higher predicts earlier failure", "horizons": []}
    horizons = [] if horizons is None else list(horizons)
    if survival_probabilities is not None:
        probabilities = np.asarray(survival_probabilities, dtype=float)
        if probabilities.shape != (len(time), len(horizons)):
            raise ValueError("Expected survival probabilities with shape [patients, horizons].")
        if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Survival probabilities must be finite and in [0,1].")
        order = np.argsort(horizons)
        if np.any(np.diff(probabilities[:, order], axis=1) > 1e-8):
            raise ValueError("Survival probabilities must not increase with time.")
    else:
        probabilities = None
    if len(set(horizons)) != len(horizons):
        raise ValueError("Duplicate evaluation horizons.")
    for k, horizon in enumerate(horizons):
        if train_time is None or train_event is None:
            result["horizons"].append({"horizon": float(horizon), "status": "not_evaluated", "reason": "Development outcomes required for censoring weights."})
            continue
        probability = None if probabilities is None else probabilities[:, k]
        try:
            row = horizon_metrics(time, event, risk, train_time, train_event, horizon, probability)
            row["uno_c_tau"] = uno_c(time, event, risk, train_time, train_event, horizon)
            if probability is not None:
                row["calibration_bins"] = calibration_table(time, event, probability, horizon)
            row["status"] = "evaluated"
        except ValueError as exc:
            row = {"horizon": float(horizon), "status": "not_evaluated", "reason": str(exc)}
        result["horizons"].append(row)
    return result


def evaluate_uncertainty(time, event, survival_probability, uncertainty,
                         train_time, train_event, horizon, coverages=(.25, .5, .75, 1.0)):
    """Descriptive censoring-weighted error association and selective Brier.

    Larger uncertainty means less trusted. Error is squared survival-probability
    residual at a prespecified horizon, not an invented event classification.
    IPCW Pearson correlation is weighted over patients with known horizon status.
    This is not a calibrated error probability or a statistical significance test.
    """
    time, event = _outcomes(time, event)
    probability = _risk(survival_probability, len(time))
    uncertainty = _risk(uncertainty, len(time))
    horizon_metrics(time, event, -probability, train_time, train_event, horizon, probability)
    censor = CensoringDistribution(train_time, train_event)
    cases, controls = event & (time <= horizon), time > horizon
    known = cases | controls
    weights = np.zeros(len(time))
    weights[cases] = 1 / censor.at(time[cases])
    weights[controls] = 1 / float(censor.at(horizon))
    truth = controls.astype(float)
    error = (probability - truth) ** 2
    w = weights[known] / weights.sum()
    u, loss = uncertainty[known], error[known]
    uc, lc = u - np.dot(w, u), loss - np.dot(w, loss)
    denominator = np.sqrt(np.dot(w, uc**2) * np.dot(w, lc**2))
    correlation = float(np.dot(w, uc * lc) / denominator) if denominator > 0 else None
    ordered = np.argsort(uncertainty, kind='stable')
    selective = []
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError('Coverage must be in (0,1].')
        chosen = ordered[:max(1, int(np.ceil(coverage * len(time))))]
        selective.append({'coverage_requested': float(coverage), 'coverage_actual': len(chosen)/len(time),
            'patients': len(chosen), 'known_horizon_status': int(known[chosen].sum()),
            'ipcw_brier': float(np.sum(weights[chosen] * error[chosen]) / len(chosen)) if known[chosen].any() else None,
            'uncertainty_ties_at_threshold': int((uncertainty == uncertainty[chosen[-1]]).sum())})
    strata = []
    for name, chosen in zip(('lower_uncertainty','higher_uncertainty'), np.array_split(ordered, 2)):
        if len(chosen):
            strata.append({'stratum': name, 'patients': len(chosen),
                'known_horizon_status': int(known[chosen].sum()),
                'ipcw_brier': float(np.sum(weights[chosen] * error[chosen])/len(chosen)) if known[chosen].any() else None})
    return {'horizon': float(horizon), 'known_horizon_status': int(known.sum()),
            'ipcw_error_uncertainty_correlation': correlation,
            'selective_prediction': selective, 'uncertainty_strata': strata,
            'interpretation': 'Descriptive IPCW error association under independent censoring; not calibrated confidence or a significance test.'}
