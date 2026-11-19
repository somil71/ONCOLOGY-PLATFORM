"""Outcome-blind renal CT triage and technical volume QC.

Descriptions are candidate filters, never proof of renal tumour visibility,
contrast timing or acquisition before diagnosis. No outcomes enter this module.
"""

from datetime import datetime
import re

import numpy as np

from src.imaging_manifest import order_ct_series


POLICY_VERSION = "kirc-retrospective-ct-technical-v1"


def _text(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _abdomen(text):
    return bool(re.search(r"\b(abd\w*|ab|a p|a pelvis|kidney\w*|renal|liver|cap|c a p)\b", text))


def candidate_reason(series):
    """Return a metadata rejection reason; an empty result allows header QC."""
    if series.get("Modality") != "CT" or series.get("Collection") != "TCGA-KIRC":
        return "wrong_collection_or_modality"
    try:
        count = int(series.get("ImageCount", 0))
    except (TypeError, ValueError):
        return "invalid_image_count"
    if count < 20:
        return "fewer_than_20_images"
    description = _text(series.get("SeriesDescription"))
    if re.search(r"\b(scout|topogram|locali[sz]er|smart prep|bolus|monitor\w*|dose|cor|coronal\w*|sagittal\w*|sag|mpr|mip)\b", description):
        return "localizer_monitor_or_nonaxial_description"
    if re.search(r"\b(chest|thorax|thoracic)\b", description) and not _abdomen(description):
        return "chest_only_series_description"
    combined = " ".join([description, _text(series.get("ProtocolName")),
                         _text(series.get("StudyDesc"))])
    if not _abdomen(combined):
        return "no_abdominal_or_renal_text_evidence"
    if not str(series.get("PatientID", "")).strip() or not str(series.get("SeriesInstanceUID", "")).strip():
        return "missing_series_or_patient_identity"
    return None


def phase_label(series, annotated_phases=None):
    """Use a pinned external phase annotation when present, else label a heuristic."""
    uid = series["SeriesInstanceUID"]
    if annotated_phases and uid in annotated_phases:
        label = annotated_phases[uid]
        if label not in {"arterial", "venous", "delayed"}:
            raise ValueError("Unsupported RCC-AID phase annotation")
        return label, "RCC-AID-v2-author-annotation"
    text = _text(series.get("SeriesDescription"))
    for label, expression in [
        ("unenhanced", r"\b(pre contrast|precontrast|non contrast|noncontrast|without contrast|unenhanced|pre liver)\b"),
        ("nephrographic", r"\b(nephro\w*|delay nephro)\b"),
        ("venous", r"\b(venous|portal)\b"),
        ("arterial", r"\b(arterial|corticomedullary|art)\b"),
        ("delayed", r"\b(delay\w*|[0-9]+ min)\b"),
        ("postcontrast_unspecified", r"\b(post contrast|postcontrast|enhanced)\b"),
    ]:
        if re.search(expression, text):
            return label, "series-description-heuristic-unreviewed"
    return "unknown", "unknown"


def _date(value):
    if not value:
        return "9999-12-31"
    raw = str(value).split(" ", 1)[0]
    for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            pass
    return "9999-12-31"


def ranked_candidates(metadata, patient_ids, annotated_phases=None):
    """Prefer earliest eligible study, then phase; break ties by source UID.

    Size, event status, follow-up and model outcomes never rank candidates.
    Metadata phases are recorded as unreviewed unless an author label exists.
    """
    ids = set(patient_ids)
    groups, rejected = {patient: [] for patient in sorted(ids)}, []
    seen = set()
    priority = {"nephrographic": 0, "venous": 1, "postcontrast_unspecified": 2,
                "arterial": 3, "delayed": 4, "unknown": 5, "unenhanced": 6}
    for entry in metadata:
        patient = entry.get("PatientID")
        if patient not in ids:
            continue
        uid = entry.get("SeriesInstanceUID")
        if uid in seen:
            raise ValueError("Duplicate series UID in pinned CT metadata.")
        seen.add(uid)
        reason = candidate_reason(entry)
        if reason:
            rejected.append({"patient_id": patient, "series_uid": uid, "reason": reason})
            continue
        label, source = phase_label(entry, annotated_phases)
        date = _date(entry.get("StudyDate"))
        groups[patient].append({**entry, "phase_label": label, "phase_source": source,
                                "within_patient_study_date": date})
    for patient in groups:
        groups[patient].sort(key=lambda entry: (entry["within_patient_study_date"],
                                              priority[entry["phase_label"]],
                                              entry["SeriesInstanceUID"]))
    return groups, rejected


def verify_volume(paths, patient_id, series_uid):
    """Require one regular axial CT acquisition with adequate physical coverage."""
    import pydicom

    ordered = order_ct_series(paths)
    if len(ordered) < 20:
        raise ValueError("Technical cohort requires at least 20 slices.")
    headers = [pydicom.dcmread(path, stop_before_pixels=True) for path in ordered]
    if any(str(ds.PatientID) != patient_id or str(ds.SeriesInstanceUID) != series_uid for ds in headers):
        raise ValueError("Selected and decoded CT patient/series identities differ.")
    first = headers[0]
    normal = np.cross(np.asarray(first.ImageOrientationPatient[:3], float),
                      np.asarray(first.ImageOrientationPatient[3:], float))
    normal /= np.linalg.norm(normal)
    if abs(float(normal[2])) < .95:
        raise ValueError("Selected volume is not approximately axial.")
    positions = np.asarray([ds.ImagePositionPatient for ds in headers], float) @ normal
    gaps = np.diff(positions)
    if gaps.max() > 5.1:
        raise ValueError("CT slice spacing exceeds the prespecified 5 mm plus 0.1 mm tolerance.")
    coverage = float(positions[-1] - positions[0] + np.median(gaps))
    if coverage < 100:
        raise ValueError("CT physical coverage is below 100 mm.")
    if len({str(ds.SOPInstanceUID) for ds in headers}) != len(headers):
        raise ValueError("Duplicate CT SOP instance UID.")
    return ordered, {"slices": len(ordered), "slice_spacing_mm": float(np.median(gaps)),
                     "coverage_mm": coverage, "axial_normal": normal.tolist(),
                     "pixel_spacing_mm": [float(v) for v in first.PixelSpacing],
                     "tumour_visibility_reviewed": False, "prediction_time_eligible": False}
