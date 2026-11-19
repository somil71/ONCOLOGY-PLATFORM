"""Read-only technical audit of strict CT bags; never admits a model cohort."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.independently_validate_image_cohort import validate_strict_ct_layout


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    folder = ROOT / "data/processed/kirc_ct_features_v2"
    qc = ROOT / "data/tcia-kirc/ct-cohort/download-qc-audit.json"
    qc_hash = sha(qc) if qc.exists() else None
    records = []
    for audit_path in sorted(folder.glob("*/ct_feature_audit.json")):
        feature_path = audit_path.parent / "rad_features.npz"
        audit = json.loads(audit_path.read_text())
        if sha(feature_path) != audit["features_sha256"]:
            raise ValueError("CT feature digest mismatch")
        with np.load(feature_path, allow_pickle=False) as data:
            if (
                str(data["patient_id"].item()) != audit["patient_id"]
                or str(data["series_uid"].item()) != audit["series_uid"]
            ):
                raise ValueError("CT bag/header identity mismatch")
            result = validate_strict_ct_layout(data, audit)
        records.append(
            {
                "patient_id": audit["patient_id"],
                "feature_sha256": sha(feature_path),
                "audit_sha256": sha(audit_path),
                "global_qc_hash_at_extraction": audit.get("qc_audit_sha256"),
                "global_qc_hash_matches_current_file": audit.get("qc_audit_sha256")
                == qc_hash,
                **result,
            }
        )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "technical_layout_checks_complete",
        "verified_bags": len(records),
        "records": records,
        "current_global_qc_sha256": qc_hash,
        "historical_global_qc_hash_mismatches": sum(
            not x["global_qc_hash_matches_current_file"] for x in records
        ),
        "interpretation": "Verifies strict feature layout/identity/source-index/physical-position/padding arithmetic only. Mutable historical global-QC digests are not immutable patient-QC proof; retain historical records and independently revalidate sources before admission.",
        "model_cohort_admitted": False,
        "prospective_prediction_eligible": False,
        "implementation_sha256": {
            name: sha(ROOT / name)
            for name in [
                "tools/audit_strict_ct_feature_layouts.py",
                "tools/independently_validate_image_cohort.py",
            ]
        },
    }
    output = ROOT / "RUN REPORT/strict-ct-feature-layout-audit-20261118.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "verified_bags": len(records),
                "historical_global_qc_hash_mismatches": report[
                    "historical_global_qc_hash_mismatches"
                ],
                "model_cohort_admitted": False,
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
