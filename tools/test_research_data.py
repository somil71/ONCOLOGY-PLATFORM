"""Regression tests for baseline cohort selection and train-only transforms."""

import json
import hashlib
from io import BytesIO
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
from src.research_data import TrainOnlyClinicalTransform, validate_survival_frame
from tools.download_primary_rna import choose_primary_files, verified_download


def main():
    train = pd.DataFrame(
        {"age_years": [40.0, 60.0, np.nan], "pathologic_stage": [1.0, 3.0, np.nan]}
    )
    test = pd.DataFrame(
        {"age_years": [1000.0, np.nan], "pathologic_stage": [4.0, np.nan]}
    )
    transform = TrainOnlyClinicalTransform().fit(train)
    before = json.dumps(transform.parameters, sort_keys=True)
    matrix = transform.transform(test)
    assert json.dumps(transform.parameters, sort_keys=True) == before
    assert transform.parameters["median"]["age_years"] == 50.0
    assert matrix.shape == (2, 4) and np.isfinite(matrix).all()
    np.testing.assert_array_equal(matrix[:, 2:], [[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_array_equal(transform.transform(train).shape, (3, 4))
    for columns in (["event"], ["last_contact_days_to"], []):
        try:
            TrainOnlyClinicalTransform(columns)
        except ValueError:
            pass
        else:
            raise AssertionError("Unreviewed clinical columns accepted")
    row = {
        "Patient_ID": ["a"],
        "time": [100.0],
        "event": [1.0],
        "age_years": [50.0],
        "pathologic_stage": [2.0],
    }
    assert len(validate_survival_frame(pd.DataFrame(row))) == 1
    for field, bad in (("time", 0.0), ("event", 2.0), ("age_years", 150.0)):
        altered = dict(row)
        altered[field] = [bad]
        try:
            validate_survival_frame(pd.DataFrame(altered))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid {field} accepted")

    def file(uid, kind):
        return {
            "file_id": uid,
            "cases": [{"submitter_id": "A", "samples": [{"sample_type": kind}]}],
        }

    selected = choose_primary_files(
        [
            file("b", "Primary Tumor"),
            file("a", "Primary Tumor"),
            file("0", "Solid Tissue Normal"),
        ]
    )
    assert (
        len(selected) == 1
        and selected[0]["file_id"] == "a"
        and selected[0]["candidate_files"] == 2
    )
    assert choose_primary_files([file("a", "Primary Tumor")], {"B"}) == []
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        payload = b"ACGT"
        item = {
            "Patient_ID": "A",
            "file_id": "good",
            "file_size": len(payload),
            "md5sum": hashlib.md5(payload).hexdigest(),
        }
        with patch(
            "tools.download_primary_rna.urlopen",
            side_effect=lambda *a, **k: BytesIO(payload),
        ):
            assert verified_download(item, folder)["status"] == "verified_download"
        assert verified_download(item, folder)["status"] == "verified_cached"
        # An interrupted source resumes at the exact recorded byte boundary.
        resumed = {**item, "file_id": "resumed"}
        (folder / "resumed.partial").write_bytes(b"AC")

        class PartialResponse(BytesIO):
            status = 206
            headers = {"Content-Range": "bytes 2-3/4"}

        with patch(
            "tools.download_primary_rna.urlopen",
            side_effect=lambda *a, **k: PartialResponse(b"GT"),
        ) as request:
            assert verified_download(resumed, folder)["status"] == "verified_download"
            assert request.call_args[0][0].get_header("Range") == "bytes=2-"
        assert (folder / "resumed.tsv").read_bytes() == payload
        item = {**item, "file_id": "tampered"}
        with (
            patch(
                "tools.download_primary_rna.urlopen",
                side_effect=lambda *a, **k: BytesIO(b"TGCA"),
            ),
            patch("tools.download_primary_rna.time.sleep"),
        ):
            assert verified_download(item, folder)["status"] == "failed"
        assert not (folder / "tampered.tsv").exists()
    print(
        "PASS: train-only transforms, predictor contract, observed endpoints, deterministic tumour selection, verified downloads/resume, cached integrity and corrupted payload rejection"
    )


if __name__ == "__main__":
    main()
