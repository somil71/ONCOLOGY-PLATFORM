"""Independently verify one actual case-matched four-input engineering pilot.

This audit supplies no survival-performance claim. Feature hashes, source slide,
CT identity/geometry/pixels and the actual patient RNA row are checked directly.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pydicom-dir", type=Path)
    args = parser.parse_args()
    if args.pydicom_dir:
        sys.path.insert(0, str(args.pydicom_dir))
    sys.path.insert(0, str(ROOT / "runtime/imaging-deps"))
    import openslide
    import pydicom

    patient = "TCGA-B8-4148"
    folder = ROOT / "RUN REPORT/kirc-paired-pilot" / patient
    wsi = json.loads((folder / "wsi_features.json").read_text())
    ct = json.loads((folder / "ct_feature_audit.json").read_text())
    if wsi["patient_id"] != patient or ct["patient_id"] != patient:
        raise ValueError("Cross-patient feature sidecars")
    for name, sidecar in (("wsi_features.npz", wsi), ("rad_features.npz", ct)):
        if digest(folder / name) != sidecar["features_sha256"]:
            raise ValueError(f"Feature hash mismatch: {name}")
    with np.load(folder / "wsi_features.npz", allow_pickle=False) as x:
        wf, xy, wm = x["features"], x["coordinates_level0"], x["valid_items"]
    with np.load(folder / "rad_features.npz", allow_pickle=False) as x:
        rf, rm, positions = x["features"], x["item_mask"], x["positions_mm"]
        if str(x["patient_id"]) != patient or str(x["series_uid"]) != ct["series_uid"]:
            raise ValueError("CT feature identity differs from provenance")
    for features, mask, shape in ((wf, wm, (32, 768)), (rf, rm, (32, 512))):
        if (
            features.shape != shape
            or not np.isfinite(features).all()
            or mask.shape != (32,)
            or not mask.all()
        ):
            raise ValueError("Invalid image feature bag/padding contract")
        if not np.any(np.std(features, axis=0) > 0):
            raise ValueError("Every image item has an identical feature vector")
    manifest = json.loads(
        (ROOT / "data/tcga-kirc/wsi-diagnostic/selected-manifest.json").read_text()
    )
    selected = [x for x in manifest["files"] if x["Patient_ID"] == patient]
    if len(selected) != 1 or selected[0]["file_id"] != wsi["file_id"]:
        raise ValueError("WSI source not the selected patient diagnostic slide")
    item = selected[0]
    if len(item["cases"]) != 1 or item["cases"][0]["submitter_id"] != patient:
        raise ValueError("WSI case identity mismatch")
    if {s["sample_type"] for s in item["cases"][0]["samples"]} != {"Primary Tumor"}:
        raise ValueError("Pilot WSI is not primary tumour")
    slide_path = ROOT / "data/tcga-kirc/wsi-diagnostic" / (item["file_id"] + ".svs")
    if (
        slide_path.stat().st_size != item["file_size"]
        or digest(slide_path, "md5") != item["md5sum"]
    ):
        raise ValueError("Actual slide fails original GDC checksum")
    with openslide.OpenSlide(str(slide_path)) as slide:
        mpp = [
            float(slide.properties[k]) for k in ("openslide.mpp-x", "openslide.mpp-y")
        ]
        np.testing.assert_allclose(
            mpp, [wsi["qc"]["source_mpp_x"], wsi["qc"]["source_mpp_y"]], atol=0, rtol=0
        )
        extent = np.asarray(wsi["qc"]["level0_patch_extent"])
        if not (
            (xy >= 0).all() and (xy + extent <= np.asarray(slide.dimensions)).all()
        ):
            raise ValueError("Patch extends beyond actual slide boundaries")
        # Directly decode one recorded patch; full bag hashes are checked above.
        region = np.asarray(
            slide.read_region(tuple(map(int, xy[0])), 0, tuple(map(int, extent)))
        )
        if region.shape[:2] != tuple(extent[::-1]) or not np.any(
            region[..., :3].std(axis=(0, 1)) > 0
        ):
            raise ValueError("Actual source patch did not decode as nonconstant tissue")
    records = ct["slices"]
    if len(records) != len(rf):
        raise ValueError("Missing selected CT slice provenance")
    hu_range, sops = [np.inf, -np.inf], set()
    for n, record in enumerate(records):
        source = Path(record["source"])
        if digest(source) != record["sha256"]:
            raise ValueError("CT raw source hash mismatch")
        ds = pydicom.dcmread(source)
        if (
            str(ds.PatientID) != patient
            or str(ds.SeriesInstanceUID) != ct["series_uid"]
            or str(ds.Modality) != "CT"
        ):
            raise ValueError("DICOM patient/modality/series mismatch")
        sop = str(ds.SOPInstanceUID)
        if sop in sops:
            raise ValueError("Repeated CT source image")
        sops.add(sop)
        np.testing.assert_array_equal(
            np.asarray(ds.ImagePositionPatient, float), positions[n]
        )
        hu = ds.pixel_array.astype(float) * float(ds.RescaleSlope) + float(
            ds.RescaleIntercept
        )
        if not np.isfinite(hu).all():
            raise ValueError("Invalid decoded CT HU values")
        hu_range = [
            min(hu_range[0], float(hu.min())),
            max(hu_range[1], float(hu.max())),
        ]
    normal = np.cross(
        np.asarray(ds.ImageOrientationPatient, float)[:3],
        np.asarray(ds.ImageOrientationPatient, float)[3:],
    )
    if not (np.diff(positions @ normal) > 0).all():
        raise ValueError("Selected CT slices not in strictly increasing physical order")
    rna_path = ROOT / "data/processed/tcga_kirc_rna_logtpm.npz"
    rna_audit = json.loads(
        (ROOT / "data/processed/tcga_kirc_rna_audit.json").read_text()
    )
    if digest(rna_path) != rna_audit["sha256"]:
        raise ValueError("RNA matrix hash mismatch")
    with np.load(rna_path, allow_pickle=False) as matrix:
        matched = np.flatnonzero(matrix["patient_ids"].astype(str) == patient)
        if len(matched) != 1:
            raise ValueError("Patient missing or repeated in actual RNA matrix")
        rna = matrix["values"][matched[0]]
        if rna.shape != (19962,) or not np.isfinite(rna).all() or (rna < 0).any():
            raise ValueError("Patient RNA row violates logTPM feature contract")
    clinical = pd.read_csv(ROOT / "data/processed/tcga_kirc_clinical.csv")
    case = clinical.loc[clinical.Patient_ID.eq(patient)]
    if (
        len(case) != 1
        or not np.isfinite(case[["age_years", "pathologic_stage"]].to_numpy()).all()
    ):
        raise ValueError("Clinical predictors unavailable for pilot patient")
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": "One actual case-matched four-input engineering feasibility audit; no survival model trained or validated by this audit",
        "patient_id": patient,
        "actual_same_case_modalities": ["clinical", "RNA", "WSI", "CT"],
        "feature_shapes": {
            "WSI": list(wf.shape),
            "CT": list(rf.shape),
            "RNA": list(rna.shape),
        },
        "source_slide_md5_verified": True,
        "direct_wsi_patch_decode_and_mpp_verified": True,
        "ct_all_selected_slices_directly_decoded_and_source_hash_verified": True,
        "ct_selected_hu_range": hu_range,
        "rna_matrix_source_hash_and_patient_join_verified": True,
        "source_mpp_um_per_pixel": mpp,
        "target_mpp_um_per_pixel": wsi["qc"]["target_mpp"],
        "feature_sha256": {
            name: digest(folder / name)
            for name in ("wsi_features.npz", "rad_features.npz")
        },
        "clinical_use_eligible": False,
        "four_input_survival_performance_established": False,
        "remaining_gates": [
            "patient sample/assay and CT diagnosis-relative timing",
            "tumour ROI/visibility and CT phase review",
            "population acquisition/QC and immutable partitions",
            "trained model/baselines/ablations with observed-event evaluation",
        ],
    }
    (folder / "independent_matched_pilot_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
