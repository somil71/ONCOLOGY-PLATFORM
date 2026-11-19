"""Prepare a separate KIRC candidate cohort, file inventory and storage budget.

This command fetches metadata only. WSI bulk acquisition requires a separately
reviewed storage/feature-extraction plan; BRCA and KIRC outputs remain separate.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

import pandas as pd

from src.research_data import tcga_clinical_cohort
from tools.download_primary_rna import choose_primary_files
from tools.prepare_tcga_inventory import FIELDS


def fetch_project_files(data_type, strategy=None):
    terms = [
        {
            "op": "in",
            "content": {"field": "cases.project.project_id", "value": ["TCGA-KIRC"]},
        },
        {"op": "in", "content": {"field": "access", "value": ["open"]}},
        {"op": "in", "content": {"field": "data_type", "value": [data_type]}},
    ]
    if data_type == "Gene Expression Quantification":
        terms.append(
            {
                "op": "in",
                "content": {
                    "field": "analysis.workflow_type",
                    "value": ["STAR - Counts"],
                },
            }
        )
    if strategy:
        terms.append(
            {
                "op": "in",
                "content": {"field": "experimental_strategy", "value": [strategy]},
            }
        )
    files = []
    offset = 0
    while True:
        request = Request(
            "https://api.gdc.cancer.gov/files",
            data=json.dumps(
                {
                    "filters": {"op": "and", "content": terms},
                    "fields": FIELDS,
                    "size": 1000,
                    "from": offset,
                    "sort": "file_id:asc",
                }
            ).encode(),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "CancerFusion separate KIRC study",
            },
        )
        with urlopen(request, timeout=90) as response:
            result = json.load(response)
        hits = result["data"]["hits"]
        files.extend(hits)
        offset += len(hits)
        if offset >= result["data"]["pagination"]["total"]:
            break
        if not hits:
            raise ValueError("GDC metadata pagination stopped prematurely")
    return files


def choose_primary_diagnostic_slides(metadata, allowed_ids):
    """One original primary-tumour diagnostic WSI per eligible case.

    Additional/new primary, recurrent and unspecified sample types are excluded;
    those slides cannot silently stand in for the original diagnosis specimen.
    """
    grouped = {}
    for item in metadata:
        cases = item.get("cases", [])
        if (
            len(cases) != 1
            or item.get("data_type") != "Slide Image"
            or item.get("experimental_strategy") != "Diagnostic Slide"
        ):
            continue
        case = cases[0]
        kinds = {sample.get("sample_type") for sample in case.get("samples", [])}
        if kinds != {"Primary Tumor"} or case["submitter_id"] not in allowed_ids:
            continue
        grouped.setdefault(case["submitter_id"], []).append(item)
    return [
        {
            "Patient_ID": patient,
            "candidate_files": len(items),
            **min(items, key=lambda item: item["file_id"]),
        }
        for patient, items in sorted(grouped.items())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-metadata",
        action="store_true",
        help="Explicitly replace the pinned metadata snapshot from GDC",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / "data/tcga-kirc"
    inventory = folder / "inventory"
    inventory.mkdir(parents=True, exist_ok=True)
    candidate_source = (
        root / "data/separate-cancer-feasibility/TCGA-KIRC-candidate-overlap.csv"
    )
    candidates = pd.read_csv(candidate_source)
    if candidates.bcr_patient_barcode.duplicated().any():
        raise ValueError("Duplicate KIRC candidate IDs")
    clinical, excluded = tcga_clinical_cohort(
        root / "data/tcga-brca/TCGA-CDR-SupplementalTableS1.xlsx",
        cancer_type="KIRC",
        patient_ids=set(candidates.bcr_patient_barcode),
    )
    if len(clinical) != len(candidates) or len(excluded):
        raise ValueError("KIRC candidate outcomes failed original CDR verification")
    comparison = candidates.merge(
        clinical,
        left_on="bcr_patient_barcode",
        right_on="Patient_ID",
        validate="one_to_one",
    )
    if not (
        comparison["OS.time"].eq(comparison.time) & comparison.OS.eq(comparison.event)
    ).all():
        raise ValueError("Candidate endpoint values differ from original CDR")
    rna_path = inventory / "rna-metadata.json"
    slide_path = inventory / "diagnostic_slides-metadata.json"
    if args.refresh_metadata or not rna_path.exists() or not slide_path.exists():
        rna = fetch_project_files("Gene Expression Quantification")
        slides = fetch_project_files("Slide Image", "Diagnostic Slide")
        rna_path.write_text(json.dumps(rna, indent=2))
        slide_path.write_text(json.dumps(slides, indent=2))
    else:
        rna = json.loads(rna_path.read_text())
        slides = json.loads(slide_path.read_text())
    selected_rna = choose_primary_files(rna, set(clinical.Patient_ID))
    selected_wsi = choose_primary_diagnostic_slides(slides, set(clinical.Patient_ID))
    both = {x["Patient_ID"] for x in selected_rna} & {
        x["Patient_ID"] for x in selected_wsi
    }
    complete = clinical[clinical.Patient_ID.isin(both)].copy()
    complete.to_csv(root / "data/processed/tcga_kirc_clinical.csv", index=False)
    clinical.assign(
        has_primary_rna=clinical.Patient_ID.isin(
            {x["Patient_ID"] for x in selected_rna}
        ),
        has_diagnostic_wsi=clinical.Patient_ID.isin(
            {x["Patient_ID"] for x in selected_wsi}
        ),
    ).to_csv(inventory / "candidate-coverage.csv", index=False)
    stamp = datetime.now(timezone.utc).isoformat()
    for name, items in [
        ("rna-primary", selected_rna),
        ("wsi-diagnostic", selected_wsi),
    ]:
        output = folder / name
        output.mkdir(exist_ok=True)
        (output / "selected-manifest.json").write_text(
            json.dumps(
                {
                    "created_utc": stamp,
                    "project": "TCGA-KIRC",
                    "selection": "Separate authorized cancer study; one lexicographically first UUID per candidate patient; Primary Tumor-only RNA and Primary Tumor-only Diagnostic Slide strategy; selection independent of endpoint values",
                    "candidate_count": len(items),
                    "selected_bytes": sum(x["file_size"] for x in items),
                    "files": items,
                },
                indent=2,
            )
        )
    report = {
        "retrieved_utc": stamp,
        "project": "TCGA-KIRC",
        "scope": "Separate renal clear-cell study; never pool with BRCA",
        "rna_metadata_sha256": hashlib.sha256(rna_path.read_bytes()).hexdigest(),
        "wsi_metadata_sha256": hashlib.sha256(slide_path.read_bytes()).hexdigest(),
        "primary_wsi_filter": "Exactly Primary Tumor sample type and Diagnostic Slide strategy; Additional/New Primary excluded",
        "candidate_manifest_sha256": hashlib.sha256(
            candidate_source.read_bytes()
        ).hexdigest(),
        "candidate_patients": len(clinical),
        "candidate_deaths": int(clinical.event.sum()),
        "primary_rna_patients": len(selected_rna),
        "diagnostic_wsi_patients": len(selected_wsi),
        "paired_selected_patients": len(complete),
        "paired_selected_deaths": int(complete.event.sum()),
        "selected_rna_bytes": sum(x["file_size"] for x in selected_rna),
        "selected_wsi_bytes": sum(x["file_size"] for x in selected_wsi),
        "free_disk_bytes": shutil.disk_usage(root).free,
        "clinical_features": ["age_years", "pathologic_stage"],
        "clinical_missing": complete[["age_years", "pathologic_stage"]]
        .isna()
        .sum()
        .to_dict(),
        "warning": "File metadata overlap is not proof of baseline timing, decode quality or a validated four-input survival experiment. WSI bulk download not performed by this command.",
    }
    (inventory / "inventory-summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
