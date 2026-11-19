"""Fixed, explicit CT Hounsfield-window contract for renal imaging research.

This is separate from MRI display normalization. It does not select contrast
phase, locate renal tumours or establish acquisition before the survival origin.
Window choices are a prespecified engineering representation, not a finding.
"""

import numpy as np

DEFAULT_CT_WINDOWS = ((40.0, 400.0), (40.0, 1000.0), (400.0, 1800.0))


def ct_hounsfield_pixels(dataset):
    if dataset.get("Modality") != "CT":
        raise ValueError("Hounsfield conversion requires verified CT modality.")
    if int(dataset.get("NumberOfFrames", 1)) != 1 or "ModalityLUTSequence" in dataset:
        raise ValueError(
            "Multiframe or non-linear modality LUT needs a dedicated CT loader."
        )
    if "RescaleSlope" not in dataset or "RescaleIntercept" not in dataset:
        raise ValueError("CT must record slope/intercept; HU cannot be assumed.")
    if str(dataset.get("RescaleType", "HU")).strip().upper() != "HU":
        raise ValueError("CT rescale units are not Hounsfield units.")
    slope, intercept = float(dataset.RescaleSlope), float(dataset.RescaleIntercept)
    if not np.isfinite([slope, intercept]).all() or slope <= 0:
        raise ValueError("Invalid CT rescale parameters.")
    raw = dataset.pixel_array
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("CT pixels must be a finite single-frame matrix.")
    valid = np.ones(raw.shape, dtype=bool)
    if "PixelPaddingValue" in dataset:
        padding = float(dataset.PixelPaddingValue)
        bound = float(dataset.get("PixelPaddingRangeLimit", padding))
        if not np.isfinite([padding, bound]).all():
            raise ValueError("Invalid CT padding values.")
        valid = ~((raw >= min(padding, bound)) & (raw <= max(padding, bound)))
    with np.errstate(over="ignore", invalid="ignore"):
        values = (raw.astype(np.float64) * slope + intercept).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("CT rescaling produced nonfinite Hounsfield pixels.")
    return values, valid


def ct_window_channels(dataset, windows=DEFAULT_CT_WINDOWS):
    values, valid = ct_hounsfield_pixels(dataset)
    channels = []
    if len(windows) != 3:
        raise ValueError(
            "The RGB feature encoder requires exactly three explicit windows."
        )
    for center, width in windows:
        if not np.isfinite([center, width]).all() or width <= 0:
            raise ValueError(
                "CT window center/width must be finite with positive width."
            )
        channel = np.clip((values - (center - width / 2)) / width, 0, 1)
        channels.append(np.where(valid, channel, 0))
    return np.stack(channels, axis=-1).astype(np.float32)
