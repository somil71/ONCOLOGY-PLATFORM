"""Audit selected sample timing from official file-linked GDC metadata.

Offsets are reported as recorded. This tool does not redefine OS, fit a model,
assume unknown timing is baseline, or automatically establish a clinical landmark.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
import pandas as pd

FIELDS = [
    "file_id",
    "file_name",
    "cases.case_id",
    "cases.submitter_id",
    "cases.index_date",
    "cases.samples.sample_id",
    "cases.samples.submitter_id",
    "cases.samples.sample_type",
    "cases.samples.days_to_collection",
    "cases.samples.days_to_sample_procurement",
    "cases.diagnoses.diagnosis_id",
    "cases.diagnoses.submitter_id",
    "cases.diagnoses.days_to_diagnosis",
    "cases.diagnoses.year_of_diagnosis",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cancer", choices=["KIRC", "BRCA"], default="KIRC")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / f"data/tcga-{args.cancer.lower()}"
    inventory = folder / "inventory"
    inventory.mkdir(exist_ok=True)
    identities = {}
    manifest_hashes = {}
    for kind, name in [("rna", "rna-primary"), ("wsi", "wsi-diagnostic")]:
        path = folder / name / "selected-manifest.json"
        if not path.exists():
            continue
        contents = path.read_bytes()
        manifest_hashes[kind] = hashlib.sha256(contents).hexdigest()
        for item in json.loads(contents)["files"]:
            if item["file_id"] in identities:
                raise ValueError("Duplicate selected file UUID")
            identities[item["file_id"]] = {
                "patient_id": item["Patient_ID"],
                "modality": kind,
            }
    if not identities:
        raise ValueError("No selected manifests")
    source = inventory / "selected-file-sample-timing-metadata.json"
    query = inventory / "selected-file-sample-timing-query.json"
    if args.refresh or not source.exists():
        with urlopen(
            "https://api.gdc.cancer.gov/files/_mapping", timeout=60
        ) as response:
            mapping = json.load(response)
        if set(FIELDS) - set(mapping["fields"]):
            raise ValueError(
                "Requested timing fields absent in current official GDC schema"
            )
        payload = {
            "filters": {
                "op": "in",
                "content": {"field": "file_id", "value": sorted(identities)},
            },
            "fields": ",".join(FIELDS),
            "size": 1000,
            "sort": "file_id:asc",
        }
        hits = []
        offset = 0
        while True:
            request = Request(
                "https://api.gdc.cancer.gov/files",
                data=json.dumps({**payload, "from": offset}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "CancerFusion selected-sample timing audit",
                },
            )
            with urlopen(request, timeout=90) as response:
                result = json.load(response)
            page = result["data"]["hits"]
            hits.extend(page)
            offset += len(page)
            if offset >= result["data"]["pagination"]["total"]:
                break
            if not page:
                raise ValueError("Timing metadata pagination stopped early")
        source.write_text(
            json.dumps(
                {"data": {"hits": hits, "pagination": {"total": len(hits)}}}, indent=2
            ),
            encoding="utf-8",
        )
        query.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    hits = json.loads(source.read_bytes())["data"]["hits"]
    if {item["file_id"] for item in hits} != set(identities) or len(hits) != len(
        identities
    ):
        raise ValueError(
            "Pinned timing source does not exactly cover current selected files; explicit --refresh required"
        )
    outcomes = pd.read_csv(
        root / f"data/processed/tcga_{args.cancer.lower()}_clinical.csv"
    ).set_index("Patient_ID")
    rows = []
    for item in hits:
        expected = identities[item["file_id"]]
        if (
            len(item.get("cases", [])) != 1
            or item["cases"][0]["submitter_id"] != expected["patient_id"]
        ):
            raise ValueError(
                "Selected file/patient identity differs in timing metadata"
            )
        case = item["cases"][0]
        samples = case.get("samples", [])
        if not samples:
            raise ValueError("Selected file has no linked sample metadata")
        for sample in samples:
            collection = sample.get("days_to_collection")
            procurement = sample.get("days_to_sample_procurement")
            status = "unknown"
            if procurement is not None:
                status = (
                    "recorded_procurement_after_index"
                    if float(procurement) > 0
                    else "recorded_procurement_at_or_before_index"
                )
            time = float(outcomes.loc[expected["patient_id"], "time"])
            rows.append(
                {
                    **expected,
                    "file_id": item["file_id"],
                    "sample_id": sample.get("sample_id"),
                    "sample_submitter_id": sample.get("submitter_id"),
                    "sample_type": sample.get("sample_type"),
                    "index_date": case.get("index_date"),
                    "days_to_collection": collection,
                    "days_to_sample_procurement": procurement,
                    "os_days": time,
                    "recorded_center_receipt_after_os": collection is not None
                    and float(collection) > time,
                    "timing_status": status,
                    "clinical_prediction_time_eligibility_verified": False,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(inventory / "selected-sample-timing.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": f"TCGA-{args.cancer}",
        "endpoint": "Recorded TCGA-CDR OS; unchanged",
        "selected_files": len(identities),
        "sample_rows": len(frame),
        "manifest_sha256": manifest_hashes,
        "metadata_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "groups": {},
        "official_field_definition_source": "https://raw.githubusercontent.com/NCI-GDC/gdcdictionary/develop/src/gdcdictionary/schemas/_terms.yaml",
        "field_semantics": {
            "days_to_collection": "Receipt of sample by BCR/processing center relative to index, not tissue removal",
            "days_to_sample_procurement": "Sample procedure/removal relative to index",
        },
        "interpretation": "Unknown procurement timing is not baseline eligibility. Center receipt after diagnosis or OS is not evidence of postdiagnosis/postdeath tissue removal. Clinical/source-origin review required; no automatic exclusions, landmark, or residual survival transformation.",
    }
    for kind, group in frame.groupby("modality"):
        known = group.days_to_collection.dropna()
        summary["groups"][kind] = {
            "files": int(group.file_id.nunique()),
            "patients": int(group.patient_id.nunique()),
            "index_dates": dict(Counter(group.index_date)),
            "known_center_receipt_offsets": len(known),
            "center_receipt_offset_min": float(known.min()) if len(known) else None,
            "center_receipt_offset_max": float(known.max()) if len(known) else None,
            "known_procurement_offsets": int(
                group.days_to_sample_procurement.notna().sum()
            ),
            "positive_center_receipt_offsets": int(known.gt(0).sum()),
            "center_receipt_after_os": int(
                group.recorded_center_receipt_after_os.sum()
            ),
            "timing_status_counts": dict(Counter(group.timing_status)),
        }
    (inventory / "selected-sample-timing-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
