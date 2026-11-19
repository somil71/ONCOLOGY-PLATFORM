"""Exercise the notebook audit with synthetic source tables and image files."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from src.data_audit import audit_modality_files, audit_outcomes


def main():
    with TemporaryDirectory() as folder:
        root = Path(folder)
        output = root / "audit"
        demographics = pd.DataFrame({
            "Patient_ID": ["CASE-A", "CASE-B", "CASE-C", "CASE-D"],
            "demographic_vital_status": ["Dead", "Alive", "Unknown", "Dead"],
            "demographic_days_to_death": [20, None, None, -1],
            "diagnoses_days_to_last_follow_up": [100, 120, 50, 90],
        })
        summary = audit_outcomes(demographics, output)
        assert summary["usable_rows"] == 2
        assert summary["usable_event_rows"] == 1
        assert summary["unmapped_status_rows"] == 1
        # An invalid death time is not replaced by follow-up.
        rows = pd.read_csv(output / "outcome_rows.csv")
        assert rows.loc[3, "time"] == -1 and not rows.loc[3, "usable"]

        case_folder = root / "CASE-A" / "MRI"
        case_folder.mkdir(parents=True)
        Image.new("RGB", (8, 8), "pink").save(case_folder / "converted.png")
        (case_folder / "corrupt.dcm").write_bytes(b"invalid image")
        metadata = FileMetaDataset()
        metadata.TransferSyntaxUID = ExplicitVRLittleEndian
        metadata.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        metadata.MediaStorageSOPInstanceUID = generate_uid()
        image = FileDataset(str(case_folder / "scan.dcm"), {}, file_meta=metadata, preamble=b"\0" * 128)
        image.Rows = image.Columns = 8
        image.SamplesPerPixel = 1
        image.PhotometricInterpretation = "MONOCHROME2"
        image.BitsAllocated = image.BitsStored = 16
        image.HighBit = 15
        image.PixelRepresentation = 0
        image.PixelData = np.arange(64, dtype="<u2").tobytes()
        image.save_as(case_folder / "scan.dcm")
        cohort = pd.DataFrame({
            "Patient_ID": ["CASE-A", "CASE-B"], "event": [1, 0], "time": [20, 120],
            "split": ["training", "validation"],
        })
        summary = audit_modality_files(cohort, root, ["CASE-A"], ["CASE-A", "CASE-B"], 20, output)
        assert summary["partitions"]["training"]["all_four_available_cases"] == 1
        assert summary["partitions"]["validation"]["rad_usable_cases"] == 0
        manifest = pd.read_csv(output / "modality_manifest.csv")
        assert manifest.loc[0, "raster_in_radiology_path"] == 1
        decode = pd.read_csv(output / "image_decode_audit.csv")
        assert decode["decoded"].sum() == 2
        assert decode["error"].notna().sum() == 1
        assert json.loads((output / "modality_summary.json").read_text()) == summary
    print("PASS: outcome exclusions, no fabricated times, successful DICOM decoding, corrupt-file reporting and suspicious MRI raster routing.")


if __name__ == "__main__":
    main()
