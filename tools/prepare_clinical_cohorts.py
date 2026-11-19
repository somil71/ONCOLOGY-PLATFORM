"""Export recorded, leakage-restricted clinical cohorts and their provenance."""

import hashlib
import json
import pandas as pd
from pathlib import Path

from src.research_data import tcga_clinical_cohort, metabric_clinical_cohort


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "processed"
    out.mkdir(exist_ok=True)
    specs = {
        "tcga_brca": (
            [root / "data/tcga-brca/TCGA-CDR-SupplementalTableS1.xlsx"],
            tcga_clinical_cohort,
        ),
        "metabric": (
            [
                root / "data/metabric/data_clinical_patient.txt",
                root / "data/metabric/data_clinical_sample.txt",
            ],
            metabric_clinical_cohort,
        ),
    }
    summary = {
        "prediction_landmark": "After diagnostic pathologic staging; no follow-up or treatment predictors",
        "shared_external_primary_features": ["age_years"],
        "stage_external_warning": "TCGA AJCC pathologic stage and METABRIC TUMOR_STAGE equivalence has not been established; stage comparison exploratory only",
        "cohorts": {},
    }
    for name, (sources, prepare) in specs.items():
        frame, excluded = prepare(*sources)
        frame.to_csv(out / f"{name}_clinical.csv", index=False)
        excluded.to_csv(out / f"{name}_excluded.csv", index=False)
        summary["cohorts"][name] = {
            "cases": len(frame),
            "events": int(frame.event.sum()),
            "excluded": len(excluded),
            "missing_predictors": frame[["age_years", "pathologic_stage"]]
            .isna()
            .sum()
            .to_dict(),
            "age_years_quantiles": frame.age_years.quantile(
                [0, 0.25, 0.5, 0.75, 1]
            ).to_dict(),
            "stage_counts": frame.pathologic_stage.fillna("missing")
            .value_counts()
            .to_dict(),
            "followup_days_quantiles": frame.time.quantile(
                [0, 0.25, 0.5, 0.75, 1]
            ).to_dict(),
            "sources": [
                {
                    "path": str(p.relative_to(root)),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in sources
            ],
        }
    # Retain broad source cohort, and export a separately named histology-matched external subset.
    external = pd.read_csv(out / "metabric_clinical.csv")
    sample = pd.read_csv(
        root / "data/metabric/data_clinical_sample.txt", sep="\t", comment="#"
    )
    histology = external.Patient_ID.map(
        sample.set_index("PATIENT_ID").CANCER_TYPE_DETAILED
    )
    confirmed = histology.str.contains(
        "Carcinoma", case=False, na=False
    ) | histology.eq("Metaplastic Breast Cancer")
    for name, mask in (
        ("metabric_carcinoma_clinical", ~histology.eq("Breast Angiosarcoma")),
        ("metabric_confirmed_carcinoma_clinical", confirmed),
    ):
        external.loc[mask].to_csv(out / f"{name}.csv", index=False)
    histology_audit = {
        "all_valid_os_cases": len(external),
        "histology_counts": histology.value_counts().to_dict(),
        "confirmed_carcinoma": {
            "cases": int(confirmed.sum()),
            "events": int(external.loc[confirmed, "event"].sum()),
            "excludes": "Breast Angiosarcoma and unspecified Breast records; metaplastic carcinoma retained",
            "file": "metabric_confirmed_carcinoma_clinical.csv",
        },
    }
    (out / "metabric_histology_eligibility.json").write_text(
        json.dumps(histology_audit, indent=2)
    )
    (out / "clinical_cohort_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
