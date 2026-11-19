"""Acquire and extract KIRC WSI bags within an explicit raw-cache budget.

Default is one case, not bulk acquisition. Files are checksum-verified, features
are keyed by source/model/preprocessing, and failed cases remain explicit.
No raw source is deleted by this program. Tumour identity and clinical timing
remain independent review requirements.
"""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import re
from uuid import UUID
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def check_download_budget(
    folder,
    identity,
    expected_size,
    cache_budget_bytes,
    disk_reserve_bytes,
    free_bytes=None,
):
    """Bound final sources and resumable partials without deleting either."""
    cached = sum(
        path.stat().st_size
        for path in folder.iterdir()
        if path.is_file() and path.suffix in {".svs", ".partial", ".bulkpart"}
    )
    partial = folder / (identity + ".partial")
    partial_bytes = partial.stat().st_size if partial.exists() else 0
    additional = max(0, expected_size - min(expected_size, partial_bytes))
    if cached + additional > cache_budget_bytes:
        raise ValueError(
            "Raw WSI cache budget would be exceeded; sources and partials are retained"
        )
    free = shutil.disk_usage(folder).free if free_bytes is None else free_bytes
    if free - additional < disk_reserve_bytes:
        raise ValueError("Download would violate the configured disk reserve")
    return {
        "cached_source_and_partial_bytes": cached,
        "expected_additional_bytes": additional,
        "free_bytes_before": free,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patient", action="append", help="Explicit selected patient IDs"
    )
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--max-patches", type=int, default=128)
    parser.add_argument("--min-patches", type=int, default=16)
    parser.add_argument("--target-mpp", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--raw-cache-budget-gb", type=float, default=200)
    parser.add_argument("--disk-reserve-gb", type=float, default=75)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dependency-dir", action="append", default=[])
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/processed/kirc_wsi_features"
    )
    args = parser.parse_args()
    if (
        args.max_cases < 1
        or not math.isfinite(args.raw_cache_budget_gb)
        or not math.isfinite(args.disk_reserve_gb)
        or args.raw_cache_budget_gb <= 0
        or args.disk_reserve_gb < 0
        or not 1 <= args.min_patches <= args.max_patches <= 4096
        or not math.isfinite(args.target_mpp)
        or args.target_mpp <= 0
    ):
        raise ValueError("Invalid acquisition, physical-resolution or bag-size limits")
    for directory in args.dependency_dir:
        sys.path.insert(0, str(Path(directory).resolve()))
    import openslide
    import torch
    from torch import nn
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    from src.wsi_features import sample_wsi_bag
    from tools.download_primary_rna import verified_download

    folder = ROOT / "data/tcga-kirc/wsi-diagnostic"
    manifest_path = folder / "selected-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("candidate_count") != len(manifest["files"]):
        raise ValueError("Selected WSI manifest count mismatch")
    if len({item["Patient_ID"] for item in manifest["files"]}) != len(
        manifest["files"]
    ):
        raise ValueError("Duplicate selected WSI patients")
    for item in manifest["files"]:
        UUID(item["file_id"])
        if not re.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", item["Patient_ID"]):
            raise ValueError("Unsafe selected WSI patient identifier")
        if item["file_size"] <= 0 or not re.fullmatch(r"[a-f0-9]{32}", item["md5sum"]):
            raise ValueError("Invalid selected WSI size/checksum")
    requested = set(args.patient or [])
    items = [
        item
        for item in manifest["files"]
        if not requested or item["Patient_ID"] in requested
    ]
    if requested - {item["Patient_ID"] for item in items}:
        raise ValueError("Unknown selected KIRC patient")
    items = items[: args.max_cases]
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.hub.set_dir(str(ROOT / "runtime/model-cache"))
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.heads = nn.Identity()
    model.eval()
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    model.to(device)
    normalization = weights.transforms()
    mean = torch.tensor(normalization.mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(normalization.std, dtype=torch.float32).view(3, 1, 1)

    def transform(image):
        # Physical extraction already yields exactly224pixels; a second resize/
        # center-crop would silently change the field-of-view and effectiveMPP.
        array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - mean) / std

    weight_path = (
        Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[1]
    )
    configuration = {
        "source_manifest_sha256": digest(manifest_path),
        "weights": str(weights),
        "weight_sha256": digest(weight_path),
        "transform": {
            "input_size": [224, 224],
            "rescale": "uint8/255",
            "mean": list(normalization.mean),
            "std": list(normalization.std),
            "resize_or_crop": "None after physical-resolution extraction",
        },
        "target_mpp": args.target_mpp,
        "max_patches": args.max_patches,
        "min_patches": args.min_patches,
        "seed": args.seed,
        "feature_code_sha256": digest(ROOT / "src/wsi_features.py"),
        "driver_code_sha256": digest(Path(__file__)),
        "device": device,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "Pillow": importlib.metadata.version("Pillow"),
            "openslide_python": openslide.__version__,
            "openslide_library": openslide.__library_version__,
        },
    }
    snapshots = args.output / "source-snapshots"
    snapshots.mkdir(exist_ok=True)
    snapshot = (
        snapshots
        / f"{configuration['driver_code_sha256'][:12]}-{configuration['feature_code_sha256'][:12]}.zip"
    )
    if not snapshot.exists():
        temporary_snapshot = snapshot.with_suffix(".zip.tmp")
        with zipfile.ZipFile(
            temporary_snapshot, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for relative in (
                "tools/extract_kirc_wsi_features.py",
                "src/wsi_features.py",
                "tools/download_primary_rna.py",
            ):
                archive.write(ROOT / relative, relative)
            archive.write(manifest_path, "input/selected-manifest.json")
            archive.writestr(
                "extraction-contract.json", json.dumps(configuration, indent=2)
            )
        temporary_snapshot.replace(snapshot)
    source_snapshot = {"path": str(snapshot), "sha256": digest(snapshot)}
    started = time.perf_counter()
    results = []
    run_started_utc = datetime.now(timezone.utc).isoformat()

    def publish_progress(active_case=None, stage=None, completed=False):
        atomic_json(
            args.output / "extraction-audit.json",
            {
                "started_utc": run_started_utc,
                "process_id": os.getpid(),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "status": "completed" if completed else "running",
                "selected_patients": len(items),
                "completed_patients": len(results),
                "valid_patients": sum(
                    row["status"].startswith("verified") for row in results
                ),
                "failed_patients": sum(row["status"] == "failed" for row in results),
                "active_patient": active_case,
                "active_stage": stage,
                "results": results,
                "elapsed_seconds": time.perf_counter() - started,
                "configuration": configuration,
                "source_snapshot": source_snapshot,
                "resource_limits": {
                    "raw_cache_budget_gb": args.raw_cache_budget_gb,
                    "minimum_free_disk_gb": args.disk_reserve_gb,
                },
            },
        )

    publish_progress(stage="initializing")
    for item in items:
        patient, identity = item["Patient_ID"], item["file_id"]
        source = folder / (identity + ".svs")
        record = {
            "patient_id": patient,
            "file_id": identity,
            "source_md5": item["md5sum"],
        }
        case_folder = args.output / patient
        case_folder.mkdir(exist_ok=True)
        stage = "verify_source"
        publish_progress(patient, stage)
        try:
            if (
                not source.exists()
                or source.stat().st_size != item["file_size"]
                or digest(source, "md5") != item["md5sum"]
            ):
                if not args.download:
                    raise ValueError(
                        "Verified source missing; explicit --download required"
                    )
                record["download_budget"] = check_download_budget(
                    folder,
                    identity,
                    item["file_size"],
                    args.raw_cache_budget_gb * 10**9,
                    args.disk_reserve_gb * 10**9,
                )
                stage = "download_source"
                publish_progress(patient, stage)
                result = verified_download(
                    item,
                    folder,
                    suffix=".svs",
                    minimum_free_bytes=args.disk_reserve_gb * 10**9,
                )
                if not result["status"].startswith("verified"):
                    raise RuntimeError(str(result))
            cache_key = hashlib.sha256(
                json.dumps(
                    {"source_md5": item["md5sum"], "config": configuration},
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            target = case_folder / "wsi_features.npz"
            audit_path = case_folder / "wsi_features.json"
            if target.is_file() and audit_path.is_file():
                previous = json.loads(audit_path.read_text())
                if previous.get("cache_key") == cache_key and previous.get(
                    "features_sha256"
                ) == digest(target):
                    results.append(
                        {
                            **record,
                            "status": "verified_cached_features",
                            "path": str(target),
                        }
                    )
                    publish_progress()
                    continue
            stage = "sample_and_qc"
            publish_progress(patient, stage)
            with openslide.OpenSlide(source) as slide:
                if hasattr(openslide, "OpenSlideCache"):
                    slide.set_cache(openslide.OpenSlideCache(32 * 1024**2))
                images, qc = sample_wsi_bag(
                    slide,
                    patient,
                    args.max_patches,
                    args.target_mpp,
                    args.seed,
                    min_patches=args.min_patches,
                )
                thumbnail_path = case_folder / "wsi-thumbnail.png"
                slide.get_thumbnail((800, 800)).save(thumbnail_path)
            features = []
            stage = "encode_features"
            publish_progress(patient, stage)
            with torch.inference_mode():
                for start in range(0, len(images), 8):
                    batch = torch.stack(
                        [transform(image) for image in images[start : start + 8]]
                    ).to(device)
                    features.append(model(batch).cpu().numpy())
            values = np.concatenate(features).astype(np.float32)
            if values.shape != (len(images), 768) or not np.isfinite(values).all():
                raise ValueError("Invalid extracted ViT features")
            temporary = target.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    features=values,
                    coordinates_level0=np.asarray(
                        qc["coordinates_level0"], dtype=np.int64
                    ),
                    valid_items=np.ones(len(values), dtype=bool),
                )
            temporary.replace(target)
            audit = {
                **record,
                "status": "verified_extracted_features",
                "cache_key": cache_key,
                "features_sha256": digest(target),
                "feature_shape": list(values.shape),
                "configuration": configuration,
                "source_snapshot": source_snapshot,
                "qc": qc,
                "scope": "Image feature extraction; no survival evaluation, annotated tumour or acquisition-time eligibility claim.",
            }
            atomic_json(audit_path, audit)
            record.update(
                {
                    "status": audit["status"],
                    "path": str(target),
                    "shape": list(values.shape),
                }
            )
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "failure_stage": stage,
                    "failure_kind": "source_acquisition"
                    if stage in ("verify_source", "download_source")
                    else "image_quality_or_resolution"
                    if stage == "sample_and_qc"
                    else "feature_encoding",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            atomic_json(
                case_folder / "wsi_features_failure.json",
                {
                    **record,
                    "failed_utc": datetime.now(timezone.utc).isoformat(),
                    "configuration": configuration,
                    "scope": "Explicit exclusion/failure for this extraction contract; prior valid outputs are retained.",
                },
            )
        results.append(record)
        print(record, flush=True)
        publish_progress()
    publish_progress(completed=True)
    if any(record["status"] == "failed" for record in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
