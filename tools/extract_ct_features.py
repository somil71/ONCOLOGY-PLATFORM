"""Extract reproducible frozen CT slice features from one explicitly selected series.

This command verifies patient/series identity and geometry, applies fixed HU
windows, and uses cached ImageNet ResNet18 weights. It does not select a contrast
phase, segment a tumour, establish baseline timing, or train a survival model.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image

from src.ct_preprocessing import DEFAULT_CT_WINDOWS, ct_window_channels
from src.imaging_manifest import order_ct_series, select_evenly_spaced


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def publish(path, payload):
    """Write, flush, then atomically publish an artifact on the same volume."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract(args):
    import pydicom
    import torch
    import torchvision
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    if not 1 <= args.items <= 256 or not 1 <= args.batch_size <= 64:
        raise ValueError("Require 1–256 slices and 1–64 items per inference batch.")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    patient, series = selection["PatientID"], selection["SeriesInstanceUID"]
    if selection.get("Modality") != "CT":
        raise ValueError("An explicit CT selection is required.")
    paths = order_ct_series(list(args.dicom_root.rglob("*.dcm")))
    if len(paths) < 2 or len(paths) != int(selection["ImageCount"]):
        raise ValueError("Series slice count differs from the selected TCIA metadata.")
    headers = [pydicom.dcmread(path, stop_before_pixels=True) for path in paths]
    if any(
        str(ds.PatientID) != patient or str(ds.SeriesInstanceUID) != series
        for ds in headers
    ):
        raise ValueError("Decoded patient/series identity differs from the selection.")
    weights = ResNet18_Weights.IMAGENET1K_V1
    checkpoint = args.weights_root / weights.url.rsplit("/", 1)[1]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Cached official weights required: {checkpoint}")
    weights_hash = sha256(checkpoint)
    expected_prefix = checkpoint.stem.rsplit("-", 1)[1]
    if not weights_hash.startswith(expected_prefix):
        raise ValueError("Encoder weights fail the published torchvision hash prefix.")
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.fc = nn.Identity()
    model.eval()
    transform = weights.transforms()
    selected = select_evenly_spaced(paths, args.items)
    images, slices = [], []
    started = time.perf_counter()
    for path in selected:
        ds = pydicom.dcmread(path)
        channels = ct_window_channels(ds)
        if not np.any(channels.max(axis=(0, 1)) > channels.min(axis=(0, 1))):
            raise ValueError(
                "Selected CT slice has no variation after HU windowing; inspect the series before extraction."
            )
        image = Image.fromarray(np.rint(channels * 255).astype(np.uint8))
        images.append(transform(image))
        slices.append(
            {
                "source": str(path.resolve()),
                "sha256": sha256(path),
                "sop_instance_uid": str(ds.SOPInstanceUID),
                "position_mm": [float(v) for v in ds.ImagePositionPatient],
            }
        )
    features = []
    with torch.inference_mode():
        for start in range(0, len(images), args.batch_size):
            features.append(
                model(torch.stack(images[start : start + args.batch_size])).numpy()
            )
    features = np.concatenate(features).astype(np.float32)
    if features.shape != (len(selected), 512) or not np.isfinite(features).all():
        raise ValueError("Encoder produced invalid CT features.")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        features=features,
        patient_id=np.asarray(patient),
        series_uid=np.asarray(series),
        item_mask=np.ones(len(features), dtype=bool),
        positions_mm=np.asarray([s["position_mm"] for s in slices]),
    )
    feature_path = output / "rad_features.npz"
    publish(feature_path, payload.getvalue())
    first = headers[0]
    normal = np.cross(
        np.asarray(first.ImageOrientationPatient[:3], float),
        np.asarray(first.ImageOrientationPatient[3:], float),
    )
    positions = np.asarray([ds.ImagePositionPatient for ds in headers], float) @ normal
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "patient_id": patient,
        "series_uid": series,
        "selection_sha256": sha256(args.selection),
        "all_slice_identity_and_geometry_verified": True,
        "source_slices": len(paths),
        "selected_slices": len(selected),
        "sampling": "Evenly spaced indices after verified physical slice ordering; no tumour ROI selection.",
        "pixel_spacing_mm": [float(v) for v in first.PixelSpacing],
        "slice_spacing_mm": float(np.median(np.diff(positions))),
        "orientation_patient": [float(v) for v in first.ImageOrientationPatient],
        "hu_windows_center_width": DEFAULT_CT_WINDOWS,
        "window_encoding": "Three fixed HU window channels rounded to uint8 then official ImageNet transform; padding zeroed before transform.",
        "encoder": "Frozen ImageNet ResNet18; no medical fine-tuning",
        "weights_url": weights.url,
        "weights_sha256": weights_hash,
        "transform": str(transform),
        "feature_shape": list(features.shape),
        "feature_file": str(feature_path.resolve()),
        "features_sha256": sha256(feature_path),
        "implementation_sha256": {
            name: sha256(Path(name))
            for name in [
                "tools/extract_ct_features.py",
                "src/ct_preprocessing.py",
                "src/imaging_manifest.py",
            ]
        },
        "environment": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pydicom": pydicom.__version__,
            "device": "CPU",
            "threads": args.threads,
        },
        "decode_and_inference_seconds": time.perf_counter() - started,
        "slices": slices,
        "prediction_time_eligible": False,
        "efficacy_evaluation_eligible": False,
        "limitations": "Technical feature extraction only. Contrast phase, renal tumour visibility, full cohort selection and acquisition timing remain unreviewed. Physical resampling and tumour segmentation are not performed.",
    }
    publish(
        output / "ct_feature_audit.json",
        json.dumps(report, indent=2, allow_nan=False).encode(),
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "patient_id",
                    "source_slices",
                    "feature_shape",
                    "features_sha256",
                    "prediction_time_eligible",
                ]
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", type=Path, default=Path("data/tcia-kirc/pilot-ct-selection.json")
    )
    parser.add_argument(
        "--dicom-root",
        type=Path,
        default=Path("data/tcia-kirc/pilot-dicom/TCGA-B8-4148"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("RUN REPORT/kirc-paired-pilot/TCGA-B8-4148")
    )
    parser.add_argument(
        "--weights-root", type=Path, default=Path("runtime/model-cache/checkpoints")
    )
    parser.add_argument("--items", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, choices=range(1, 17), default=2)
    extract(parser.parse_args())


if __name__ == "__main__":
    main()
