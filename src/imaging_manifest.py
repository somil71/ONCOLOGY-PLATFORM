"""Validated manifest selection and geometric ordering for research image inputs.

Raster identity must come from a reviewed manifest, never from its extension.
DICOM sequences must be selected explicitly before ordering; this code does not
choose clinically appropriate MRI protocols or establish prediction-time eligibility.
"""

import csv
from pathlib import Path
import numpy as np


def load_image_manifest(manifest_path, image_root):
    root = Path(image_root).resolve()
    groups = {}
    seen = set()
    with Path(manifest_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"patient_id", "modality", "relative_path", "identity_source"} <= set(
            reader.fieldnames or []
        ):
            raise ValueError(
                "Image manifest requires patient_id, modality, relative_path, identity_source."
            )
        for row in reader:
            patient, modality = (
                row["patient_id"].strip(),
                row["modality"].strip().lower(),
            )
            if not patient or any(token in patient for token in ("/", "\\", "..")):
                raise ValueError("Invalid manifest patient identifier.")
            if modality not in ("wsi", "rad") or not row["identity_source"].strip():
                raise ValueError(
                    "Each image needs a reviewed modality and identity source."
                )
            relative = Path(row["relative_path"])
            path = (root / relative).resolve()
            if (
                relative.is_absolute()
                or not path.is_relative_to(root)
                or not path.is_file()
            ):
                raise ValueError("Manifest image must exist inside the image root.")
            if path in seen:
                raise ValueError(
                    "An image cannot be assigned twice or to multiple patients/modalities."
                )
            if modality == "rad":
                import pydicom

                header = pydicom.dcmread(str(path), stop_before_pixels=True)
                if str(header.get("PatientID", "")).strip() != patient:
                    raise ValueError(
                        "Radiology PatientID must match the reviewed case mapping."
                    )
            seen.add(path)
            groups.setdefault((patient, modality), []).append(path)
    return groups


def order_dicom_series(paths, modality):
    """Check one selected MR/CT series and sort by physical slice position."""
    if modality not in ("MR", "CT"):
        raise ValueError("Only reviewed MR or CT volumes are supported.")
    import pydicom

    if not paths:
        return []
    records = []
    for path in paths:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        if ds.get("Modality") != modality or int(ds.get("NumberOfFrames", 1)) != 1:
            raise ValueError(
                "Expected single-frame matching-modality DICOM; enhanced/multiframe needs a dedicated loader."
            )
        for name in (
            "PatientID",
            "SeriesInstanceUID",
            "FrameOfReferenceUID",
            "ImageOrientationPatient",
            "ImagePositionPatient",
            "Rows",
            "Columns",
            "PixelSpacing",
        ):
            if name not in ds:
                raise ValueError(f"DICOM geometry is missing {name}.")
        for name in ("PatientID", "SeriesInstanceUID", "FrameOfReferenceUID"):
            if not str(ds.get(name, "")).strip():
                raise ValueError(f"DICOM identity is empty: {name}.")
        if int(ds.Rows) <= 0 or int(ds.Columns) <= 0:
            raise ValueError("DICOM Rows and Columns must be positive.")
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        position = np.asarray(ds.ImagePositionPatient, dtype=float)
        spacing = np.asarray(ds.PixelSpacing, dtype=float)
        if (
            orientation.shape != (6,)
            or position.shape != (3,)
            or spacing.shape != (2,)
            or not np.isfinite(np.r_[orientation, position, spacing]).all()
            or (spacing <= 0).any()
        ):
            raise ValueError("Invalid DICOM geometry.")
        if (
            not np.isclose(np.linalg.norm(orientation[:3]), 1, atol=1e-4)
            or not np.isclose(np.linalg.norm(orientation[3:]), 1, atol=1e-4)
            or not np.isclose(np.dot(orientation[:3], orientation[3:]), 0, atol=1e-4)
        ):
            raise ValueError(
                "DICOM orientation must contain orthonormal direction cosines."
            )
        records.append((Path(path), ds, orientation, spacing, position))
    first = records[0]
    reference_normal = np.cross(first[2][:3], first[2][3:])
    reference_normal /= np.linalg.norm(reference_normal)
    for record in records[1:]:
        if (
            str(record[1].PatientID) != str(first[1].PatientID)
            or str(record[1].SeriesInstanceUID) != str(first[1].SeriesInstanceUID)
            or str(record[1].FrameOfReferenceUID) != str(first[1].FrameOfReferenceUID)
        ):
            raise ValueError("Select one DICOM series and frame of reference per bag.")
        if (
            (int(record[1].Rows), int(record[1].Columns))
            != (int(first[1].Rows), int(first[1].Columns))
            or not np.allclose(record[2], first[2], atol=1e-4)
            or not np.allclose(record[3], first[3], atol=1e-4)
        ):
            raise ValueError("Mixed DICOM orientation, dimensions or pixel spacing.")
        delta = record[4] - first[4]
        in_plane_shift = delta - float(delta @ reference_normal) * reference_normal
        if np.linalg.norm(in_plane_shift) > 1e-3:
            raise ValueError(
                "DICOM slice origins shift within the image plane; resample explicitly before volume feature extraction."
            )
    records.sort(key=lambda record: float(record[4] @ reference_normal))
    gaps = np.diff([float(record[4] @ reference_normal) for record in records])
    if (gaps <= 1e-4).any():
        raise ValueError(
            "Duplicate or overlapping slice positions; select a single acquisition."
        )
    if len(gaps) > 1 and not np.allclose(gaps, np.median(gaps), rtol=0.05, atol=0.1):
        raise ValueError(
            "Nonuniform slice spacing; resample explicitly before feature extraction."
        )
    return [record[0] for record in records]


def select_evenly_spaced(paths, limit):
    if limit < 1:
        raise ValueError("Image item limit must be positive.")
    if len(paths) <= limit:
        return list(paths)
    return [paths[index] for index in np.linspace(0, len(paths) - 1, limit, dtype=int)]


def order_mr_series(paths):
    """Validate selected MR data without inferring sequence identity."""
    return order_dicom_series(paths, "MR")


def order_ct_series(paths):
    """Validate selected CT data; intensity conversion is a separate contract."""
    return order_dicom_series(paths, "CT")
