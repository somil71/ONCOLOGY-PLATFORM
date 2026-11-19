"""Check actual locked patient-fold integrity and deterministic assignment rules."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from tools.freeze_kirc_protocol import build_assignments


def main():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "config/kirc_retrospective_os_protocol.json").read_text()
    )
    source_path = root / protocol["clinical_source"]
    assert (
        hashlib.sha256(source_path.read_bytes()).hexdigest()
        == protocol["clinical_source_sha256"]
    )
    source = pd.read_csv(source_path)
    generated, counts = build_assignments(source)
    reordered, _ = build_assignments(source.sample(frac=1, random_state=123))
    changed_values = source.copy()
    changed_values["time"] *= 2
    changed_values["age_years"] = 999
    changed, _ = build_assignments(changed_values)
    tested = []
    for record in protocol["fold_ledgers"]:
        path = root / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        actual = pd.read_csv(path)
        assert set(actual.columns) == {"Patient_ID", "split"}
        pd.testing.assert_frame_equal(actual, generated[record["fold"]])
        pd.testing.assert_frame_equal(actual, reordered[record["fold"]])
        pd.testing.assert_frame_equal(actual, changed[record["fold"]])
        assert not actual.Patient_ID.duplicated().any()
        assert set(actual.split) == {"train", "validation", "test"}
        assert all(
            group["deaths"] >= 2 for group in counts[str(record["fold"])].values()
        )
        tested.extend(actual.loc[actual.split.eq("test"), "Patient_ID"])
    assert len(tested) == len(set(tested)) == len(source)
    assert set(tested) == set(source.Patient_ID)
    assert (
        protocol["hypotheses"]["primary_subset_panel"]
        and len(protocol["hypotheses"]["primary_subset_panel"]) == 15
    )
    assert (
        protocol["endpoint"]["prospective_prediction_time_eligibility_verified"]
        is False
    )
    for field, value in (
        ("Patient_ID", source.Patient_ID.iloc[1]),
        ("event", 2),
        ("time", 0),
    ):
        invalid = source.copy()
        invalid.loc[invalid.index[0], field] = value
        try:
            build_assignments(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid frozen cohort accepted: {field}")
    print(
        "PASS: all frozen hashes, exact original assignments, patient-disjoint train/validation/test, one outer test per patient, row-order/feature/followup invariance,15-subset panel and invalid-label/identity rejection."
    )


if __name__ == "__main__":
    main()
