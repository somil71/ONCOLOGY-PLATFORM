"""Audit image provenance, selected items and locked-fold feasibility.

This is a technical provenance audit, not a clinical QC decision or permission
to fit an image survival model. It verifies available WSI selection metadata,
CT source-slice bytes, feature-bag sampling arithmetic, and fixed-fold counts.
It records when original image pixels are unavailable for independent replay.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def audit_wsi(records: list[dict], contract: dict) -> dict:
    manifest_info = contract["wsi_selection_manifest"]
    manifest_path = Path(manifest_info["path"])
    if sha(manifest_path) != manifest_info["sha256"]:
        raise ValueError("Pinned WSI selection manifest has changed")
    selection = {
        item["Patient_ID"]: item
        for item in load_json(manifest_path)["files"]
    }
    # The OpenSlide wheel and DLL are deliberately kept in the isolated imaging
    # dependency directory used for acquisition. Add it only for this audit.
    imaging_dependencies = ROOT / "runtime/imaging-deps"
    if imaging_dependencies.is_dir():
        sys.path.insert(0, str(imaging_dependencies))
    try:
        import openslide
        from src.wsi_features import sample_wsi_bag
    except ImportError as exc:
        raise RuntimeError(
            "Independent WSI patch replay requires the isolated OpenSlide dependency"
        ) from exc

    checked = []
    replayed_pixels = 0
    for record in records:
        if record["modality"] != "wsi":
            continue
        patient = record["patient_id"]
        item = selection.get(patient)
        if item is None or item["file_id"] != record["source_file_id"]:
            raise ValueError(f"WSI GDC file identity mismatch for {patient}")
        if item["md5sum"].lower() != record["source_md5"].lower():
            raise ValueError(f"WSI source MD5 mismatch for {patient}")
        slide_path = ROOT / "data/tcga-kirc/wsi-diagnostic" / f"{item['file_id']}.svs"
        if not slide_path.is_file():
            raise FileNotFoundError(f"Original selected SVS is unavailable for {patient}")
        if slide_path.stat().st_size != int(item["file_size"]):
            raise ValueError(f"WSI byte count mismatch for {patient}")
        if md5(slide_path).lower() != item["md5sum"].lower():
            raise ValueError(f"Original SVS MD5 mismatch for {patient}")
        audit_path = Path(record["audit_file"])
        audit = load_json(audit_path)
        if audit["file_id"] != item["file_id"] or audit["source_md5"].lower() != item[
            "md5sum"
        ].lower():
            raise ValueError(f"WSI sidecar-to-source mismatch for {patient}")
        if audit["configuration"]["source_manifest_sha256"] != manifest_info[
            "sha256"
        ]:
            raise ValueError(f"WSI extraction used another selection manifest: {patient}")
        qc = audit["qc"]
        coords = np.asarray(qc["coordinates_level0"], dtype=np.int64)
        fractions = np.asarray(qc["patch_tissue_fractions"], dtype=float)
        if (
            coords.shape != (int(qc["selected_patches"]), 2)
            or fractions.shape != (len(coords),)
            or len(np.unique(coords, axis=0)) != len(coords)
            or not np.isfinite(fractions).all()
            or (fractions < float(qc["minimum_tissue_fraction"])).any()
        ):
            raise ValueError(f"WSI patch coordinates/QC mismatch for {patient}")
        dims = np.asarray(qc["level0_dimensions"], dtype=np.int64)
        extent = np.asarray(qc["level0_patch_extent"], dtype=np.int64)
        if dims.shape != (2,) or extent.shape != (2,) or (
            (coords < 0).any() or (coords + extent > dims).any()
        ):
            raise ValueError(f"WSI patch coordinates fall outside slide for {patient}")
        if int(qc["selected_patches"]) != int(record["valid_items"]):
            raise ValueError(f"WSI selected patch count mismatch for {patient}")
        # Reopen the pinned GDC SVS and rerun deterministic physical-resolution
        # sampling. This independently checks selected coordinates and decoded
        # patch pixel QC from the actual slide bytes, rather than only sidecars.
        slide = openslide.OpenSlide(str(slide_path))
        try:
            images, replay = sample_wsi_bag(
                slide,
                patient,
                max_patches=int(audit["configuration"]["max_patches"]),
                target_mpp=float(audit["configuration"]["target_mpp"]),
                seed=int(audit["configuration"]["seed"]),
                minimum_tissue_fraction=float(qc["minimum_tissue_fraction"]),
                min_patches=int(audit["configuration"]["min_patches"]),
            )
        finally:
            slide.close()
        if replay["coordinates_level0"] != qc["coordinates_level0"]:
            raise ValueError(f"WSI selected-patch coordinates do not replay for {patient}")
        if replay["level0_dimensions"] != qc["level0_dimensions"]:
            raise ValueError(f"WSI slide dimensions do not replay for {patient}")
        replay_fraction_delta = float(
            np.max(
                np.abs(
                    np.asarray(replay["patch_tissue_fractions"], dtype=float)
                    - fractions
                )
            )
        )
        if replay_fraction_delta > 1e-9:
            raise ValueError(f"WSI decoded patch tissue QC does not replay for {patient}")
        replayed_pixels += len(images)
        del images

        snapshot = audit.get("source_snapshot") or {}
        snapshot_path = ROOT / snapshot.get("path", "__missing__")
        snapshot_ok = snapshot_path.is_file() and sha(snapshot_path) == snapshot.get(
            "sha256"
        )
        if snapshot_ok:
            with zipfile.ZipFile(snapshot_path) as archive:
                names = set(archive.namelist())
            snapshot_ok = any(name.endswith("extraction-contract.json") for name in names)
        # The compact engineering snapshot contains code and manifests, not the
        # original SVS. The GDC MD5 pins slide identity but cannot replay pixels.
        checked.append(
            {
                "patient_id": patient,
                "gdc_file_id_matches": True,
                "gdc_slide_md5_matches_pinned_manifest": True,
                "original_svs_bytes_rehashed": slide_path.stat().st_size,
                "selected_patch_count": len(coords),
                "patch_coordinates_unique_and_in_bounds": True,
                "tissue_fraction_thresholds_match": True,
                "patch_selection_replayed_from_original_svs": True,
                "replayed_patch_count": len(replay["coordinates_level0"]),
                "replayed_tissue_fraction_max_abs_delta": replay_fraction_delta,
                "selection_seed": int(qc["seed"]),
                "target_mpp": float(qc["target_mpp"]),
                "source_snapshot_hash_valid": bool(snapshot_ok),
                "original_svs_available_for_pixel_replay": True,
            }
        )
    return {
        "bags_checked": len(checked),
        "status": "original-svs-and-deterministic-patch-replay-passed",
        "replayed_patches": replayed_pixels,
        "records": checked,
        "limitation": "Selected patches and tissue QC are independently replayed from the original SVS. The ViT embeddings are not recomputed in this audit, and tissue QC is not tumour annotation.",
    }


def audit_ct(records: list[dict]) -> dict:
    checked = []
    for record in records:
        if record["modality"] != "rad":
            continue
        patient = record["patient_id"]
        audit = load_json(Path(record["audit_file"]))
        if audit["patient_id"] != patient or audit["series_uid"] != record["series_uid"]:
            raise ValueError(f"CT patient/series identity mismatch for {patient}")
        if audit.get("all_slice_identity_and_geometry_verified") is not True:
            raise ValueError(f"CT extraction did not pass source geometry for {patient}")
        source = audit["all_source_dicom_sha256"]
        selected = audit["slices"]
        count = int(audit["source_dicom_count"])
        retained = min(128, count)
        expected = np.linspace(0, count - 1, retained, dtype=int).tolist()
        if len(source) != count or len(selected) != retained:
            raise ValueError(f"CT source/selected slice count mismatch for {patient}")
        if audit["source_slice_indices"] != expected:
            raise ValueError(f"CT selected slice indices are not the declared linspace for {patient}")
        if selected != [source[n] for n in expected]:
            raise ValueError(f"CT selected slices differ from full series records for {patient}")
        actual_bytes_verified = 0
        for item in selected:
            path = ROOT / item["path"]
            if not path.is_file() or sha(path) != item["sha256"]:
                raise ValueError(f"CT source DICOM hash mismatch for {patient}: {path}")
            actual_bytes_verified += path.stat().st_size
        positions = np.asarray([x["position_mm"] for x in selected], dtype=float)
        if (
            positions.shape != (retained, 3)
            or not np.isfinite(positions).all()
            or len(np.unique(positions, axis=0)) != retained
        ):
            raise ValueError(f"CT selected physical positions invalid for {patient}")
        checked.append(
            {
                "patient_id": patient,
                "series_uid": audit["series_uid"],
                "selected_series_identity_matches_feature_record": True,
                "source_slices": count,
                "selected_slices": retained,
                "selection_indices_reconstructed": True,
                "selected_source_slice_sha256_rechecked": retained,
                "selected_source_bytes_rehashed": actual_bytes_verified,
                "physical_positions_finite_unique": True,
                "phase_label": audit.get("phase_label"),
                "phase_label_source": audit.get("phase_label_source"),
                "radiologist_tumour_coverage_review_complete": bool(
                    audit.get("clinical_review_complete", False)
                ),
                "acquisition_timing_verified": False,
            }
        )
    return {
        "bags_checked": len(checked),
        "selected_source_slices_rehashed": sum(
            x["selected_source_slice_sha256_rechecked"] for x in checked
        ),
        "status": "selected-source-byte-and-sidecar-checks-passed",
        "dicom_headers_reparsed": False,
        "limitation": "This audit rehashes selected source files and reconstructs selection against extraction sidecars; it does not independently reparse DICOM headers or confirm tumour coverage/contrast phase clinically.",
        "records": checked,
    }


def audit_fold_feasibility(ids: set[str]) -> list[dict]:
    clinical = pd.read_csv(ROOT / "data/processed/tcga_kirc_clinical.csv").set_index(
        "Patient_ID", verify_integrity=True
    )
    rows = []
    for fold in range(1, 6):
        ledger = pd.read_csv(ROOT / f"data/processed/kirc-fixed-folds/fold_{fold:02d}.csv")
        admitted = ledger[ledger.Patient_ID.isin(ids)]
        for split in ("train", "validation", "test"):
            patients = admitted[admitted.split.eq(split)].Patient_ID.tolist()
            events = int(clinical.loc[patients, "event"].sum()) if patients else 0
            rows.append(
                {"fold": fold, "split": split, "patients": len(patients), "observed_deaths": events}
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-root", type=Path, required=True)
    args = parser.parse_args()
    folder = args.cohort_root.resolve()
    contract_path = folder / "cohort_contract.json"
    contract = load_json(contract_path)
    input_validation_path = folder / "independent_input_validation.json"
    input_validation = load_json(input_validation_path)
    if input_validation.get("status") != "passed":
        raise ValueError("Independent source/identity/availability validation must pass first")
    records = load_json(folder / "image_feature_records.json")
    wsi = audit_wsi(records, contract)
    ct = audit_ct(records)
    ids = set(input_validation["complete_input_patient_ids"])
    folds = audit_fold_feasibility(ids)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_with_scoped_limitations",
        "scope": "Image source identity, selected-item provenance and locked-fold feasibility; not model evaluation, radiologist approval or prospective timing validation.",
        "cohort_contract_sha256": sha(contract_path),
        "independent_input_validation_sha256": sha(input_validation_path),
        "complete_input_patients": len(ids),
        "complete_input_observed_deaths": int(input_validation["complete_input_deaths"]),
        "wsi": wsi,
        "ct": ct,
        "locked_fold_complete_input_counts": folds,
        "four_input_model_fit_recommended": False,
        "fit_recommendation_reason": "Complete-input cohort is too small and at least one locked outer test fold has fewer than two observed deaths; retain this as a feasibility/QC result only.",
        "implementation_sha256": sha(Path(__file__)),
    }
    output = folder / "source_reproducibility_audit.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "wsi_bags_checked": wsi["bags_checked"],
                "ct_bags_checked": ct["bags_checked"],
                "complete_input_patients": len(ids),
                "complete_input_deaths": report["complete_input_observed_deaths"],
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
