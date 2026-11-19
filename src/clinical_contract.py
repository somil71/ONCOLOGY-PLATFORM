"""Prediction-time clinical input guard for research training."""

import pandas as pd


def select_reviewed_clinical_features(frame, approved_columns):
    """Require an explicit prediction-time feature contract; reject outcome proxies."""
    if not approved_columns or len(set(approved_columns)) != len(approved_columns):
        raise ValueError("Set clinical_feature_columns to a nonempty, unique list reviewed at the prediction time.")
    missing = set(approved_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Reviewed clinical columns missing: {sorted(missing)}")
    blocked = [name for name in approved_columns if any(token in name.casefold() for token in
        ('last_contact', 'follow_up', 'followup', 'death', 'vital_status', 'survival', 'os.time', 'pfi', 'dfi', 'dss'))
        or name.casefold() in ('os', 'event', 'time')]
    if blocked:
        raise ValueError(f"Outcome or follow-up fields cannot be predictors: {blocked}")
    selected = frame.loc[:, approved_columns]
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in selected.dtypes):
        raise ValueError("Reviewed clinical predictors must be numeric before model preprocessing.")
    return selected.copy()
