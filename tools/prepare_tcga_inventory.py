"""Fetch public TCGA-BRCA metadata and join its coverage to curated outcomes.

This downloads file metadata only. Image and expression data remain on the GDC.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.request import Request, urlopen


ENDPOINT = "https://api.gdc.cancer.gov/files"
PUBLICATION = "https://gdc.cancer.gov/about-data/publications/pancanatlas"
FIELDS = (
    "file_id,file_name,file_size,md5sum,access,data_type,data_format,experimental_strategy,"
    "analysis.workflow_type,cases.case_id,cases.submitter_id,cases.samples.sample_type"
)


def fetch_files(conditions):
    filters = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
        {"op": "in", "content": {"field": "access", "value": ["open"]}},
        *conditions,
    ]}
    hits = []
    offset = 0
    while True:
        payload = json.dumps({"filters": filters, "fields": FIELDS, "format": "JSON",
                              "from": offset, "size": 1000, "sort": "file_id:asc"}).encode()
        request = Request(ENDPOINT, data=payload, headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urlopen(request, timeout=60) as response:
                    data = json.load(response)["data"]
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        hits.extend(data["hits"])
        offset += len(data["hits"])
        if offset >= data["pagination"]["total"]:
            return hits, filters
        if not data["hits"]:
            raise ValueError("GDC pagination stopped before the reported total.")


def condition(field, value):
    return {"op": "in", "content": {"field": field, "value": [value]}}


def usable_os(row):
    try:
        duration = float(row["OS.time"])
    except (ValueError, TypeError):
        return False
    return row["OS"] in ("0", "1") and math.isfinite(duration) and duration > 0


def write_inventory(output_dir, outcome_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with outcome_path.open(encoding="utf-8-sig", newline="") as source:
        outcomes = list(csv.DictReader(source))
    ids = [row["bcr_patient_barcode"] for row in outcomes]
    if len(ids) != len(set(ids)):
        raise ValueError("Resolve duplicated outcome case identifiers before joining.")
    groups = {
        "rna": [condition("data_type", "Gene Expression Quantification"),
                condition("analysis.workflow_type", "STAR - Counts")],
        "diagnostic_slides": [condition("data_type", "Slide Image"),
                              condition("experimental_strategy", "Diagnostic Slide")],
    }
    summary = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "project": "TCGA-BRCA",
               "endpoint": ENDPOINT, "outcome_source": PUBLICATION,
               "outcome_csv_sha256": hashlib.sha256(outcome_path.read_bytes()).hexdigest(),
               "interpretation": "File availability from metadata. Decode quality and eligible prediction-date samples require review.",
               "groups": {}}
    available = {}
    for name, conditions in groups.items():
        hits, filters = fetch_files(conditions)
        (output_dir / f"{name}-metadata.json").write_text(json.dumps(hits, indent=2), encoding="utf-8")
        available[name] = {case["submitter_id"] for hit in hits for case in hit.get("cases", [])}
        with (output_dir / f"{name}-gdc-manifest.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["id", "filename", "md5", "size", "state"])
            for hit in hits:
                writer.writerow([hit["file_id"], hit["file_name"], hit.get("md5sum", ""), hit["file_size"], "released"])
        summary["groups"][name] = {"files": len(hits), "cases": len(available[name]),
                                  "total_bytes": sum(hit["file_size"] for hit in hits), "filter": filters}
    rows = [{**row, **{f"has_{key}_metadata": row["bcr_patient_barcode"] in values
                      for key, values in available.items()}} for row in outcomes]
    with (output_dir / "cdr-case-coverage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    both = [row for row in rows if all(row[f"has_{key}_metadata"] for key in available)]
    eligible = [row for row in both if usable_os(row)]
    summary["cdr_overlap"] = {"cdr_cases": len(rows), "rna_and_slides": len(both),
                              "with_positive_os_duration": len(eligible),
                              "observed_deaths_with_positive_os_duration": sum(row["OS"] == "1" for row in eligible)}
    (output_dir / "inventory-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "data" / "tcga-brca" / "inventory")
    parser.add_argument("--outcomes", type=Path, default=root / "data" / "tcga-brca" / "TCGA-BRCA-survival-outcomes.csv")
    args = parser.parse_args()
    print(json.dumps(write_inventory(args.output, args.outcomes), indent=2))
