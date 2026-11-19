"""Run real image decoding, QC, and optionally frozen ImageNet feature extraction.

This size-selected three-slide / one-series pilot tests engineering only. Missing
WSI physical resolution and limited sample/event counts prohibit efficacy claims.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from src.imaging_manifest import order_mr_series, select_evenly_spaced


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-features", action="store_true")
    parser.add_argument("--items", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.items <= 256:
        raise ValueError("Pilot items must lie in [1,256].")
    import openslide
    import pydicom

    out = Path("RUN REPORT/image-pilot-20261118")
    out.mkdir(exist_ok=True)
    records = []
    bags = {}
    manifest = []
    source = json.loads(
        Path("data/tcga-brca/wsi-pilot/pilot-manifest.json").read_text()
    )
    for entry in source["records"]:
        path = Path(entry["path"])
        patient = entry["Patient_ID"]
        dest = out / patient
        dest.mkdir(exist_ok=True)
        with openslide.OpenSlide(path) as slide:
            width, height = slide.dimensions
            thumbnail = slide.get_thumbnail((800, 800))
            thumbnail.save(dest / "slide_thumbnail.png")
            candidates = []
            for y in np.linspace(0, max(height - 224, 0), 16, dtype=int):
                for x in np.linspace(0, max(width - 224, 0), 16, dtype=int):
                    patch = slide.read_region((int(x), int(y)), 0, (224, 224)).convert(
                        "RGB"
                    )
                    array = np.asarray(patch, dtype=float) / 255
                    tissue = (
                        (array.max(2) - array.min(2) > 0.08) & (array.mean(2) < 0.9)
                    ).mean()
                    if tissue >= 0.30:
                        candidates.append((int(x), int(y), float(tissue), patch))
            selected = select_evenly_spaced(candidates, args.items)
            images = []
            tiles = []
            for index, (x, y, tissue, patch) in enumerate(selected):
                target = dest / f"patch_{index:03d}_x{x}_y{y}.png"
                patch.save(target)
                images.append(patch)
                tiles.append(
                    {
                        "x_level0": x,
                        "y_level0": y,
                        "tissue_fraction": tissue,
                        "path": str(target),
                    }
                )
                manifest.append(
                    [
                        patient,
                        "wsi",
                        target.as_posix(),
                        "Official GDC diagnostic slide " + entry["file_id"],
                    ]
                )
            mpp = {
                key: slide.properties.get(key)
                for key in (
                    "openslide.mpp-x",
                    "openslide.mpp-y",
                    "openslide.objective-power",
                )
            }
            records.append(
                {
                    "patient_id": patient,
                    "modality": "wsi",
                    "source_sha256": sha256(path),
                    "level0_dimensions": [width, height],
                    "physical_resolution": mpp,
                    "physical_resolution_verified": all(
                        mpp[k] for k in ("openslide.mpp-x", "openslide.mpp-y")
                    ),
                    "candidate_tiles": len(candidates),
                    "selected_tiles": len(selected),
                    "tiles": tiles,
                    "qc": "Heuristic tissue fraction; no tumor annotation or stain normalization; pixel-space feasibility only.",
                }
            )
            bags[(patient, "wsi")] = images
    selection = json.loads(
        Path("data/tcia-brca/pilot-series-selection.json").read_text()
    )[0]
    patient = selection["PatientID"]
    paths = list((Path("data/tcia-brca/pilot-dicom") / patient).rglob("*.dcm"))
    ordered = order_mr_series(paths)
    dest = out / patient
    dest.mkdir(exist_ok=True)
    images = []
    slices = []
    for index, path in enumerate(select_evenly_spaced(ordered, args.items)):
        ds = pydicom.dcmread(path)
        if str(ds.PatientID) != patient:
            raise ValueError("Wrong MRI patient identity")
        array = ds.pixel_array.astype(float) * float(ds.get("RescaleSlope", 1)) + float(
            ds.get("RescaleIntercept", 0)
        )
        if (
            array.ndim != 2
            or not np.isfinite(array).all()
            or array.max() <= array.min()
        ):
            raise ValueError("Invalid MRI pixels")
        # Pilot display normalization only; production protocol needs train-fixed volume normalization.
        array = np.clip((array - array.min()) / (array.max() - array.min()), 0, 1)
        if ds.get("PhotometricInterpretation") == "MONOCHROME1":
            array = 1 - array
        image = Image.fromarray((array * 255).astype("uint8")).convert("RGB")
        image.save(dest / f"slice_{index:03d}.png")
        images.append(image)
        slices.append(
            {
                "source": str(path),
                "source_sha256": sha256(path),
                "position": [float(v) for v in ds.ImagePositionPatient],
            }
        )
        manifest.append(
            [
                patient,
                "rad",
                path.as_posix(),
                "Official TCIA series " + selection["SeriesInstanceUID"],
            ]
        )
    records.append(
        {
            "patient_id": patient,
            "modality": "rad",
            "decoded_series_slices": len(ordered),
            "selected_slices": len(images),
            "series_uid": selection["SeriesInstanceUID"],
            "sequence": selection["SeriesDescription"],
            "slices": slices,
            "qc": "Geometric ordering and same-patient/series verified; protocol suitability and diagnosis-time eligibility unreviewed. Per-slice minmax is pilot display preprocessing.",
        }
    )
    bags[(patient, "rad")] = images
    with (out / "image_manifest.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "modality", "relative_path", "identity_source"])
        writer.writerows(manifest)
    feature_info = {}
    if args.extract_features:
        import torch
        import torchvision
        from torch import nn
        from torchvision.models import (
            vit_b_16,
            ViT_B_16_Weights,
            resnet18,
            ResNet18_Weights,
        )

        torch.set_num_threads(2)
        torch.hub.set_dir(str(Path("runtime/model-cache").resolve()))
        for modality, factory, weights in [
            ("wsi", vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
            ("rad", resnet18, ResNet18_Weights.IMAGENET1K_V1),
        ]:
            model = factory(weights=weights).eval()
            if modality == "wsi":
                model.heads = nn.Identity()
            else:
                model.fc = nn.Identity()
            transform = weights.transforms()
            feature_info[modality] = {
                "weights": str(weights),
                "url": weights.url,
                "transform": str(transform),
            }
            weight_path = (
                Path(torch.hub.get_dir())
                / "checkpoints"
                / weights.url.rsplit("/", 1)[1]
            )
            feature_info[modality]["weights_sha256"] = sha256(weight_path)
            for (patient, kind), images in bags.items():
                if kind != modality or not images:
                    continue
                features = []
                with torch.inference_mode():
                    for start in range(0, len(images), 8):
                        features.append(
                            model(
                                torch.stack(
                                    [
                                        transform(image)
                                        for image in images[start : start + 8]
                                    ]
                                )
                            ).numpy()
                        )
                values = np.concatenate(features)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite extracted image features")
                feature_path = out / patient / f"{modality}_features.npz"
                np.savez_compressed(feature_path, features=values)
                feature_info[f"{patient}_{modality}"] = {
                    "shape": list(values.shape),
                    "path": str(feature_path),
                    "sha256": sha256(feature_path),
                }
        feature_info["environment"] = {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": "CPU",
        }
    report = {
        "scope": "Real-file engineering feasibility only; not a survival experiment or validated clinical preprocessing.",
        "selection": source["selection"],
        "records": records,
        "feature_extraction": feature_info,
    }
    (out / "image-pilot-audit.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "records": len(records),
                "feature_extraction": feature_info,
                "output": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
