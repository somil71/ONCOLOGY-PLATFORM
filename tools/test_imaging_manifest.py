"""Manifest and DICOM geometry regressions; all fixtures are synthetic."""

import csv
from pathlib import Path
import tempfile
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
from src.imaging_manifest import (
    load_image_manifest,
    order_mr_series,
    select_evenly_spaced,
)


def main():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        series, frame = generate_uid(), generate_uid()

        def make(name, z, series_uid=series):
            path = root / name
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
            ds.Modality = "MR"
            ds.PatientID = "TCGA-TEST"
            ds.SeriesInstanceUID = series_uid
            ds.FrameOfReferenceUID = frame
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [0, 0, z]
            ds.Rows = 2
            ds.Columns = 2
            ds.PixelSpacing = [1, 1]
            ds.save_as(str(path), enforce_file_format=True)
            return path

        paths = [make("last.dcm", 4), make("first.dcm", 0), make("middle.dcm", 2)]
        assert [p.name for p in order_mr_series(paths)] == [
            "first.dcm",
            "middle.dcm",
            "last.dcm",
        ]
        assert len(select_evenly_spaced(paths, 2)) == 2
        for invalid in (
            [paths[0], make("duplicate.dcm", 4)],
            [paths[0], make("mixed.dcm", 8, generate_uid())],
            [*paths, make("irregular.dcm", 9)],
        ):
            try:
                order_mr_series(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Ambiguous MR sequence accepted")
        # Common normal spacing alone does not establish a regular 3D grid.
        shifted = make("laterally_shifted.dcm", 6)
        ds = pydicom.dcmread(str(shifted))
        ds.ImagePositionPatient = [0.01, 0, 6]
        ds.save_as(str(shifted), enforce_file_format=True)
        invalid_geometry = [[paths[0], shifted]]
        for field, value in (("PatientID", ""), ("SeriesInstanceUID", ""),
                             ("FrameOfReferenceUID", ""), ("Rows", 0), ("Columns", 0)):
            invalid_path = make(f"invalid_{field}.dcm", 6)
            ds = pydicom.dcmread(str(invalid_path))
            setattr(ds, field, value)
            ds.save_as(str(invalid_path), enforce_file_format=True)
            invalid_geometry.append([invalid_path])
        for invalid in invalid_geometry:
            try:
                order_mr_series(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid identity, dimensions or in-plane slice shift accepted")
        rounded = make("scanner_rounding.dcm", 6)
        ds = pydicom.dcmread(str(rounded))
        ds.ImagePositionPatient = [0.0005, 0, 6]
        ds.save_as(str(rounded), enforce_file_format=True)
        assert len(order_mr_series([paths[0], rounded])) == 2
        manifest = root / "manifest.csv"

        def write(rows):
            with manifest.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["patient_id", "modality", "relative_path", "identity_source"]
                )
                writer.writerows(rows)

        write([["TCGA-TEST", "rad", "first.dcm", "DICOM acquisition review"]])
        assert len(load_image_manifest(manifest, root)) == 1
        for invalid in (
            [["OTHER", "rad", "first.dcm", "reviewed"]],
            [["TCGA-TEST", "wsi", "../outside.jpg", "reviewed"]],
            [["TCGA-TEST", "wsi", "first.dcm", ""]],
        ):
            write(invalid)
            try:
                load_image_manifest(manifest, root)
            except ValueError:
                pass
            else:
                raise AssertionError("Unsafe manifest accepted")
    print(
        "PASS: physical MR ordering, nonempty identity, positive dimensions, in-plane origin consistency, scanner rounding tolerance, duplicate/spacing rejection, provenance and path containment."
    )


if __name__ == "__main__":
    main()
