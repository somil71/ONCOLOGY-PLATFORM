"""Identity, provenance and ambiguity checks for actual image availability."""

import json
from pathlib import Path
import tempfile
import numpy as np
from tools import audit_kirc_input_availability as audit


def main():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        case = root / "CASE-A"
        case.mkdir()
        path = case / "wsi_features.npz"
        record = case / "wsi_features.json"
        np.savez_compressed(
            path,
            features=np.ones((2, 768), dtype=np.float32),
            valid_items=np.ones(2, dtype=bool),
        )
        metadata = {
            "patient_id": "CASE-A",
            "file_id": "official-id",
            "source_md5": "checksum",
            "features_sha256": audit.sha256(path),
        }
        record.write_text(json.dumps(metadata))
        selected = {"CASE-A": {"file_id": "official-id", "md5sum": "checksum"}}
        original = audit.ROOT
        audit.ROOT = root
        try:
            valid = audit.verified_image_bags([root], "wsi", {"CASE-A"}, selected)
            assert valid["CASE-A"]["valid_items"] == 2
            record.write_text(json.dumps({**metadata, "patient_id": "WRONG"}))
            try:
                audit.verified_image_bags([root], "wsi", {"CASE-A"}, selected)
            except ValueError:
                pass
            else:
                raise AssertionError("Cross-patient features accepted")
            record.write_text(json.dumps(metadata))
            path.write_bytes(path.read_bytes() + b"corrupt")
            try:
                audit.verified_image_bags([root], "wsi", {"CASE-A"}, selected)
            except ValueError:
                pass
            else:
                raise AssertionError("Feature provenance mismatch accepted")
            np.savez_compressed(
                path, features=np.ones((2, 768)), valid_items=np.array([0.5, 1])
            )
            record.write_text(
                json.dumps({**metadata, "features_sha256": audit.sha256(path)})
            )
            try:
                audit.verified_image_bags([root], "wsi", {"CASE-A"}, selected)
            except ValueError:
                pass
            else:
                raise AssertionError("Nonbinary image mask accepted")
        finally:
            audit.ROOT = original
    print(
        "PASS: actual-feature identity, checksum and binary-item-mask guards; fixtures synthetic only."
    )


if __name__ == "__main__":
    main()
