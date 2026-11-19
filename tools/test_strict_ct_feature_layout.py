"""Analytic padded/source-selection CT layout checks; synthetic fixtures only."""

import copy
import hashlib

import numpy as np

from tools.independently_validate_image_cohort import validate_strict_ct_layout


def fixture(total):
    count = min(total, 128)
    indices = np.linspace(0, total - 1, count, dtype=int)
    source = [
        {
            "sop_instance_uid": f"SOP-{n}",
            "sha256": hashlib.sha256(str(n).encode()).hexdigest(),
            "position_mm": [0.0, 0.0, float(n * 5)],
        }
        for n in range(total)
    ]
    mask = np.arange(128) < count
    features = np.zeros((128, 512), dtype="float32")
    features[mask] = 1
    positions = np.zeros((128, 3))
    positions[mask] = np.array([source[n]["position_mm"] for n in indices])
    normalized = np.zeros((128, 1), dtype="float32")
    normalized[mask, 0] = indices / (total - 1)
    bag = {
        "features": features,
        "item_mask": mask,
        "positions_mm": positions,
        "normalized_positions": normalized,
        "source_slice_indices": indices.astype("int32"),
    }
    audit = {
        "source_dicom_count": total,
        "retained_feature_items": count,
        "selected_slices": count,
        "item_mask_true": count,
        "padded_bag_shape": [128, 512],
        "feature_shape": [128, 512],
        "source_slice_indices": indices.tolist(),
        "all_source_dicom_sha256": source,
        "slices": [source[n] for n in indices],
    }
    return bag, audit


def rejects(bag, audit):
    try:
        validate_strict_ct_layout(bag, audit)
    except (ValueError, AssertionError):
        return
    raise AssertionError("Malformed strict CT feature layout accepted")


def main():
    for n in [25, 200]:
        bag, audit = fixture(n)
        assert validate_strict_ct_layout(bag, audit)["retained_items"] == min(128, n)
        wrong = copy.deepcopy(bag)
        wrong["normalized_positions"][5] = 0.99
        rejects(wrong, audit)
        wrong = copy.deepcopy(bag)
        wrong["source_slice_indices"][1] = 0
        rejects(wrong, audit)
        wrong = copy.deepcopy(bag)
        wrong["positions_mm"][2, 2] += 1
        rejects(wrong, audit)
    bag, audit = fixture(25)
    wrong = copy.deepcopy(bag)
    wrong["item_mask"][26] = True
    rejects(wrong, audit)
    wrong = copy.deepcopy(bag)
    wrong["features"][26] = 1
    rejects(wrong, audit)
    print(
        "PASS:128-item right-padding,25/200-source deterministic selection, physical-position normalization and corrupted mask/index/coordinate/padding rejection; synthetic fixtures only."
    )


if __name__ == "__main__":
    main()
