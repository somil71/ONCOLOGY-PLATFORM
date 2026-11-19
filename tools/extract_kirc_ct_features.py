"""Extract frozen ResNet18 CT slice bags from the locked, QC-passed KIRC series.

Uses at most128 evenly spaced slices after physical ordering, pads to128 with
explicit masks, records normalized positions and source hashes, and atomically
publishes each bag. No tumour segmentation/localization is claimed.
"""

from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image

from src.ct_preprocessing import DEFAULT_CT_WINDOWS, ct_window_channels
from src.imaging_manifest import order_ct_series
from tools.extract_ct_features import sha256

ROOT = Path(__file__).resolve().parents[1]
MAX_ITEMS = 128


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def feature_one(
    item, qc, model, transform, weight_path, torch, torchvision, nn, pydicom
):
    uid, patient = item["SeriesInstanceUID"], item["PatientID"]
    dicom_root = ROOT / "data/tcia-kirc/ct-cohort/dicom" / patient / uid
    paths = order_ct_series(list(dicom_root.rglob("*.dcm")))
    if len(paths) != qc["dicom_files"]:
        raise ValueError(
            "QC-passed DICOM file count changed before feature extraction."
        )
    source_records = []
    positions, physical_positions, images = [], [], []
    for path in paths:
        ds = pydicom.dcmread(path)
        if (
            str(ds.PatientID) != patient
            or str(ds.SeriesInstanceUID) != uid
            or str(ds.StudyInstanceUID) != item["StudyInstanceUID"]
        ):
            raise ValueError("Feature extraction identity mismatch.")
        array = ct_window_channels(ds)
        image = Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB")
        images.append(transform(image))
        positions.append(
            float(
                np.dot(
                    np.asarray(ds.ImagePositionPatient, dtype=float),
                    np.cross(
                        np.asarray(ds.ImageOrientationPatient[:3], dtype=float),
                        np.asarray(ds.ImageOrientationPatient[3:], dtype=float),
                    ),
                )
            )
        )
        physical_positions.append(
            np.asarray(ds.ImagePositionPatient, dtype=np.float64).tolist()
        )
        source_records.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "sop_instance_uid": str(ds.SOPInstanceUID),
                "position_mm": physical_positions[-1],
            }
        )
    order = np.argsort(np.asarray(positions), kind="stable")
    images = [images[i] for i in order]
    positions = np.asarray(positions, dtype=np.float32)[order]
    physical_positions = np.asarray(physical_positions, dtype=np.float64)[order]
    source_records = [source_records[i] for i in order]
    selected_indices = np.linspace(
        0, len(images) - 1, min(MAX_ITEMS, len(images)), dtype=int
    )
    selected_images = [images[i] for i in selected_indices]
    selected_physical_positions = physical_positions[selected_indices]
    selected_source_records = [source_records[i] for i in selected_indices]
    selected_positions = positions[selected_indices]
    if len(selected_positions) > 1:
        lo, hi = float(selected_positions[0]), float(selected_positions[-1])
        normalized_positions = (selected_positions - lo) / max(hi - lo, 1e-6)
    else:
        normalized_positions = np.zeros(len(selected_positions), dtype=np.float32)
    vectors = []
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(selected_images), 8):
            batch = torch.stack(selected_images[offset : offset + 8])
            vectors.append(model(batch).cpu().numpy().astype(np.float32))
    encoded = np.concatenate(vectors, axis=0)
    if encoded.shape != (len(selected_images), 512) or not np.isfinite(encoded).all():
        raise ValueError("Frozen ResNet18 returned invalid CT features.")
    features = np.zeros((MAX_ITEMS, 512), dtype=np.float32)
    mask = np.zeros(MAX_ITEMS, dtype=bool)
    pos = np.zeros((MAX_ITEMS, 1), dtype=np.float32)
    physical_pos = np.zeros((MAX_ITEMS, 3), dtype=np.float64)
    features[: len(encoded)] = encoded
    mask[: len(encoded)] = True
    pos[: len(encoded), 0] = normalized_positions
    physical_pos[: len(encoded)] = selected_physical_positions
    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        features=features,
        item_mask=mask,
        normalized_positions=pos,
        positions_mm=physical_pos,
        patient_id=np.asarray(patient),
        series_uid=np.asarray(uid),
        source_slice_indices=selected_indices.astype(np.int32),
    )
    out = ROOT / "data/processed/kirc_ct_features_v2" / patient
    out.mkdir(parents=True, exist_ok=True)
    feature_file = out / "rad_features.npz"
    atomic_bytes(feature_file, payload.getvalue())
    selection_sha = sha256(ROOT / "data/tcia-kirc/ct-cohort/selected-series.json")
    implementation = {
        p: sha256(ROOT / p)
        for p in [
            "tools/extract_kirc_ct_features.py",
            "src/ct_preprocessing.py",
            "src/imaging_manifest.py",
        ]
    }
    record = {
        "status": "features_verified",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "patient_id": patient,
        "series_uid": uid,
        "study_uid": item["StudyInstanceUID"],
        "phase_label": item["phase_label"],
        "phase_label_source": item["phase_label_source"],
        "qc_audit_sha256": sha256(
            ROOT / "data/tcia-kirc/ct-cohort/download-qc-audit.json"
        ),
        "selected_series_metadata": {
            k: item.get(k)
            for k in [
                "SeriesDescription",
                "StudyDesc",
                "ProtocolName",
                "ImageCount",
                "FileSize",
            ]
        },
        "source_dicom_count": len(paths),
        "retained_feature_items": len(encoded),
        "selected_slices": len(encoded),
        "padded_bag_shape": list(features.shape),
        "feature_shape": list(features.shape),
        "item_mask_true": int(mask.sum()),
        "selection": "Evenly spaced physical-order slice indices, deterministic linspace, max128; pad right with zero and explicit false mask.",
        "source_slice_indices": selected_indices.tolist(),
        "normalized_physical_position_range_mm": [
            float(selected_positions[0]),
            float(selected_positions[-1]),
        ],
        "slices": selected_source_records,
        "all_source_dicom_sha256": source_records,
        "all_slice_identity_and_geometry_verified": True,
        "selection_sha256": selection_sha,
        "hu_windows_center_width": DEFAULT_CT_WINDOWS,
        "input_encoding": "HU window channels -> rounded uint8 RGB -> official ImageNet ResNet18 transform",
        "encoder": "torchvision ResNet18 IMAGENET1K_V1 frozen, fc replaced by Identity",
        "weights_url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "transform": str(transform),
        "feature_sha256": sha256(feature_file),
        "features_sha256": sha256(feature_file),
        "feature_path": str(feature_file.relative_to(ROOT)),
        "weights_sha256": sha256(weight_path),
        "implementation_sha256": implementation,
        "window_encoding": "Three HU windows mapped to RGB channels, clipped, rounded to uint8, then ImageNet transform.",
        "clinical_review_complete": False,
        "device": "cpu",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "feature_inference_seconds": time.perf_counter() - start,
        "limitations": "Frozen natural-image encoder; no medical fine-tuning, physical 3D resampling, tumour localization or radiologist coverage confirmation. This is a feature bag, not a clinical prediction.",
    }
    atomic_bytes(
        out / "ct_feature_audit.json",
        json.dumps(record, indent=2, allow_nan=False).encode(),
    )
    return record


def main():
    import pydicom
    import torch
    import torchvision
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    torch.set_num_threads(2)
    torch.hub.set_dir(str(ROOT / "runtime/model-cache"))
    weights = ResNet18_Weights.IMAGENET1K_V1
    weight_path = (
        Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[1]
    )
    if not weight_path.exists() or not sha256(weight_path).startswith("f37072fd"):
        raise FileNotFoundError(
            "Pinned torchvision ResNet18 weights are unavailable or invalid."
        )
    model = resnet18(weights=None)
    model.load_state_dict(
        torch.load(weight_path, map_location="cpu", weights_only=True)
    )
    model.fc = nn.Identity()
    model.eval()
    transform = weights.transforms()

    selection = json.loads(
        (ROOT / "data/tcia-kirc/ct-cohort/selected-series.json").read_text()
    )
    qc = json.loads(
        (ROOT / "data/tcia-kirc/ct-cohort/download-qc-audit.json").read_text()
    )
    qc_by_uid = {x["series_uid"]: x for x in qc["results"]}
    rows = []
    for item in selection["files"]:
        uid = item["SeriesInstanceUID"]
        if uid not in qc_by_uid:
            continue
        output = (
            ROOT
            / "data/processed/kirc_ct_features_v2"
            / item["PatientID"]
            / "ct_feature_audit.json"
        )
        if output.exists():
            old = json.loads(output.read_text())
            if (
                old.get("series_uid") == uid
                and old.get("status") == "features_verified"
                and old.get("feature_shape") == [MAX_ITEMS, 512]
                and old.get("all_slice_identity_and_geometry_verified") is True
            ):
                continue
        try:
            record = feature_one(
                item,
                qc_by_uid[uid],
                model,
                transform,
                weight_path,
                torch,
                torchvision,
                nn,
                pydicom,
            )
            rows.append(record)
            print(
                json.dumps(
                    {
                        "patient": record["patient_id"],
                        "status": "features_verified",
                        "shape": record["padded_bag_shape"],
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "patient": item["PatientID"],
                        "series_uid": uid,
                        "status": "feature_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "processed_this_invocation": len(rows),
                "features_root": "data/processed/kirc_ct_features_v2",
            }
        )
    )


if __name__ == "__main__":
    main()
