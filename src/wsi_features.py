"""Deterministic physical-resolution WSI bags with explicit, limited tissue QC.

Tissue heuristics do not identify tumour. Physical resolution must be recorded in
standard OpenSlide or Aperio metadata; objective magnification is not substituted
for microns-per-pixel. All selection is independent of survival outcomes.
"""

from __future__ import annotations

import hashlib
import math
import numpy as np
from PIL import Image


def tissue_fraction(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    return float(((rgb.max(2) - rgb.min(2) > 0.08) & (rgb.mean(2) < 0.9)).mean())


def physical_patch_spec(slide, target_mpp=0.5, patch_pixels=224):
    if not math.isfinite(target_mpp) or target_mpp <= 0 or patch_pixels < 1:
        raise ValueError(
            "Require positive finite target microns-per-pixel and patch size."
        )
    properties = slide.properties
    x_key = "openslide.mpp-x" if properties.get("openslide.mpp-x") else "aperio.MPP"
    y_key = "openslide.mpp-y" if properties.get("openslide.mpp-y") else "aperio.MPP"
    try:
        mpp_x, mpp_y = float(properties[x_key]), float(properties[y_key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Recorded physical pixel resolution is required; no objective-power guess."
        ) from exc
    if not all(math.isfinite(x) and x > 0 for x in (mpp_x, mpp_y)):
        raise ValueError("Invalid source microns-per-pixel.")
    if target_mpp < max(mpp_x, mpp_y) * 0.95:
        raise ValueError(
            "Target resolution would invent a finer physical scan resolution."
        )
    desired_downsample = min(target_mpp / mpp_x, target_mpp / mpp_y)
    level = int(slide.get_best_level_for_downsample(desired_downsample))
    downsample = float(slide.level_downsamples[level])
    if not math.isfinite(downsample) or downsample <= 0:
        raise ValueError("Invalid slide level downsample.")
    extent_x = max(1, round(patch_pixels * target_mpp / mpp_x))
    extent_y = max(1, round(patch_pixels * target_mpp / mpp_y))
    level_size = (
        max(1, round(extent_x / downsample)),
        max(1, round(extent_y / downsample)),
    )
    return {
        "source_mpp_x": mpp_x,
        "source_mpp_y": mpp_y,
        "mpp_source_fields": [x_key, y_key],
        "target_mpp": float(target_mpp),
        "patch_pixels": int(patch_pixels),
        "level": level,
        "level_downsample": downsample,
        "level0_patch_extent": [extent_x, extent_y],
        "read_size_at_level": list(level_size),
    }


def sample_wsi_bag(
    slide,
    patient_id,
    max_patches=128,
    target_mpp=0.5,
    seed=42,
    minimum_tissue_fraction=0.30,
    min_patches=16,
):
    if (
        not 1 <= min_patches <= max_patches <= 4096
        or not 0 < minimum_tissue_fraction <= 1
    ):
        raise ValueError("Invalid bag size or tissue threshold.")
    spec = physical_patch_spec(slide, target_mpp)
    width, height = map(int, slide.dimensions)
    extent_x, extent_y = spec["level0_patch_extent"]
    if width < extent_x or height < extent_y:
        raise ValueError("Slide is smaller than one physical patch.")
    thumb = slide.get_thumbnail((1024, 1024)).convert("RGB")
    pixels = np.asarray(thumb, dtype=np.float32) / 255
    tissue = (pixels.max(2) - pixels.min(2) > 0.08) & (pixels.mean(2) < 0.9)
    xs = np.arange(0, width - extent_x + 1, extent_x, dtype=np.int64)
    ys = np.arange(0, height - extent_y + 1, extent_y, dtype=np.int64)
    xx, yy = np.meshgrid(xs, ys)
    candidates = np.c_[xx.ravel(), yy.ravel()]
    tx = np.minimum(
        ((candidates[:, 0] + extent_x / 2) / width * thumb.width).astype(int),
        thumb.width - 1,
    )
    ty = np.minimum(
        ((candidates[:, 1] + extent_y / 2) / height * thumb.height).astype(int),
        thumb.height - 1,
    )
    candidates = candidates[tissue[ty, tx]]
    patient_seed = int.from_bytes(
        hashlib.sha256(f"{patient_id}:{seed}".encode()).digest()[:8], "little"
    )
    rng = np.random.default_rng(patient_seed)
    if len(candidates):
        candidates = candidates[rng.permutation(len(candidates))]
    images, coordinates, fractions = [], [], []
    examined = 0
    for x, y in candidates[: max_patches * 10]:
        examined += 1
        rgba = slide.read_region(
            (int(x), int(y)), spec["level"], tuple(spec["read_size_at_level"])
        )
        if rgba.mode == "RGBA":
            image = Image.new("RGB", rgba.size, "white")
            image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            image = rgba.convert("RGB")
        image = image.resize(
            (spec["patch_pixels"], spec["patch_pixels"]), Image.Resampling.BILINEAR
        )
        fraction = tissue_fraction(image)
        if fraction < minimum_tissue_fraction:
            continue
        images.append(image)
        coordinates.append([int(x), int(y)])
        fractions.append(fraction)
        if len(images) == max_patches:
            break
    if len(images) < min_patches:
        raise ValueError(
            f"Only {len(images)} qualifying tissue patches; require {min_patches}."
        )
    audit = {
        **spec,
        "level0_dimensions": [width, height],
        "seed": int(seed),
        "patient_seed": patient_seed,
        "candidate_tissue_centers": len(candidates),
        "examined_patches": examined,
        "selected_patches": len(images),
        "minimum_tissue_fraction": float(minimum_tissue_fraction),
        "coordinates_level0": coordinates,
        "patch_tissue_fractions": fractions,
        "limitations": "Heuristic background QC only; no tumour annotation or stain normalization.",
    }
    return images, audit
