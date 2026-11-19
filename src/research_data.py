"""Leakage-restricted clinical cohorts from original, checksum-tracked sources.

Predictors are age at diagnosis and diagnostic pathologic stage only. Outcome
columns are retained in separate named fields for a survival runner; never pass
an entire exported frame to a model. Stage uses I-IV ordinal coding and NaN for
unknown, with an explicit missing indicator in train-fitted preprocessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLINICAL_FEATURES = ("age_years", "pathologic_stage")


def validate_survival_frame(frame):
    required = {"Patient_ID", "time", "event", *CLINICAL_FEATURES}
    if required - set(frame):
        raise ValueError(f"Missing cohort fields: {sorted(required - set(frame))}")
    if frame.Patient_ID.isna().any() or frame.Patient_ID.duplicated().any():
        raise ValueError("Missing or duplicated patient identifiers")
    if not frame.event.isin([0, 1]).all():
        raise ValueError("Events must be recorded binary observations")
    if not (np.isfinite(frame.time) & frame.time.gt(0)).all():
        raise ValueError("Survival durations must be positive and finite")
    for name, lower, upper in (("age_years", 0, 120), ("pathologic_stage", 0, 4)):
        values = frame[name].dropna()
        if not (np.isfinite(values) & values.between(lower, upper)).all():
            raise ValueError(f"Invalid baseline predictor: {name}")
    return frame


def tcga_clinical_cohort(source, cancer_type="BRCA", patient_ids=None):
    """Use CDR BRCA recorded OS; diagnosis-time age and pathologic stage.

    Landmark is after diagnostic pathologic staging, not before all diagnostic
    procedures. No treatment, follow-up, vital-status, or outcome proxy is used.
    """
    raw = pd.read_excel(source)
    raw = raw.loc[raw["type"].eq(cancer_type)].copy()
    if patient_ids is not None:
        raw = raw.loc[raw.bcr_patient_barcode.isin(set(patient_ids))].copy()
    if raw.bcr_patient_barcode.duplicated().any():
        raise ValueError("Duplicate CDR patient identifiers")
    stage = raw.ajcc_pathologic_tumor_stage.astype("string").str.strip().str.upper()
    stage = stage.str.extract(r"^STAGE\s+(IV|III|II|I)(?:[ABC])?$", expand=False).map(
        {"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}
    )
    frame = pd.DataFrame(
        {
            "Patient_ID": raw.bcr_patient_barcode,
            "time": pd.to_numeric(raw["OS.time"], errors="coerce"),
            "event": pd.to_numeric(raw.OS, errors="coerce"),
            "age_years": pd.to_numeric(
                raw.age_at_initial_pathologic_diagnosis, errors="coerce"
            ),
            "pathologic_stage": stage,
        }
    )
    keep = frame.event.isin([0, 1]) & np.isfinite(frame.time) & frame.time.gt(0)
    excluded = frame.loc[~keep, ["Patient_ID", "time", "event"]].assign(
        reason="Invalid/missing recorded OS or nonpositive duration"
    )
    return validate_survival_frame(frame.loc[keep].reset_index(drop=True)), excluded


def metabric_clinical_cohort(patient_source, sample_source):
    """METABRIC OS months converted using 365.25/12 days per month.

    Tumour stage is retained for exploratory use; do not assume equivalence to
    TCGA AJCC stage. Age-only is the strict shared baseline representation.
    """
    patient = pd.read_csv(patient_source, sep="\t", comment="#")
    samples = pd.read_csv(sample_source, sep="\t", comment="#")
    if patient.PATIENT_ID.duplicated().any() or samples.PATIENT_ID.duplicated().any():
        raise ValueError(
            "Resolve multiple samples per patient explicitly before merging"
        )
    raw = patient.merge(
        samples[["PATIENT_ID", "TUMOR_STAGE"]],
        on="PATIENT_ID",
        how="left",
        validate="one_to_one",
    )
    frame = pd.DataFrame(
        {
            "Patient_ID": raw.PATIENT_ID,
            "time": pd.to_numeric(raw.OS_MONTHS, errors="coerce") * (365.25 / 12),
            "event": raw.OS_STATUS.map({"0:LIVING": 0.0, "1:DECEASED": 1.0}),
            "age_years": pd.to_numeric(raw.AGE_AT_DIAGNOSIS, errors="coerce"),
            "pathologic_stage": pd.to_numeric(raw.TUMOR_STAGE, errors="coerce"),
        }
    )
    keep = frame.event.isin([0, 1]) & np.isfinite(frame.time) & frame.time.gt(0)
    excluded = frame.loc[~keep, ["Patient_ID", "time", "event"]].assign(
        reason="Invalid/missing recorded OS or nonpositive duration"
    )
    return validate_survival_frame(frame.loc[keep].reset_index(drop=True)), excluded


class TrainOnlyClinicalTransform:
    """Median imputation + standardization fit on explicit training rows only.

    Missing indicators are appended for every feature, even if training had no
    missing values, so the external feature contract does not change shape.
    """

    def __init__(self, columns=CLINICAL_FEATURES):
        if (
            not columns
            or len(set(columns)) != len(columns)
            or set(columns) - set(CLINICAL_FEATURES)
        ):
            raise ValueError(
                "Only explicitly supported diagnostic baseline predictors may be used"
            )
        self.columns = tuple(columns)
        self.parameters = None

    def fit(self, training):
        if training.empty:
            raise ValueError(
                "Training preprocessing requires at least one training patient"
            )
        values = (
            training.loc[:, self.columns]
            .apply(pd.to_numeric, errors="raise")
            .replace([np.inf, -np.inf], np.nan)
        )
        median = values.median().fillna(0.0)
        filled = values.fillna(median)
        mean = filled.mean()
        scale = filled.std(ddof=0).replace(0, 1).fillna(1.0)
        self.parameters = {
            "columns": list(self.columns),
            "median": median.to_dict(),
            "mean": mean.to_dict(),
            "scale": scale.to_dict(),
        }
        return self

    def transform(self, frame):
        if self.parameters is None:
            raise ValueError("Fit preprocessing on training patients before transform")
        values = (
            frame.loc[:, self.columns]
            .apply(pd.to_numeric, errors="raise")
            .replace([np.inf, -np.inf], np.nan)
        )
        missing = values.isna().to_numpy(dtype=np.float32)
        z = (
            values.fillna(pd.Series(self.parameters["median"]))
            - pd.Series(self.parameters["mean"])
        ) / pd.Series(self.parameters["scale"])
        result = np.concatenate([z.to_numpy(dtype=np.float32), missing], axis=1)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite preprocessed clinical features")
        return result
