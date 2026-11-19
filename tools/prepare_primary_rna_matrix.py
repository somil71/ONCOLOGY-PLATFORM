"""Build a verified primary-tumour expression matrix without outcome-fitted transforms."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="For smoke checks only; output reports incomplete cohort",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=0,
        help="Bounded wait for in-progress downloads before declaring missing selected files",
    )
    parser.add_argument(
        "--cancer",
        choices=["BRCA", "KIRC"],
        default="BRCA",
        help="Keep cancer cohorts in separate data folders and outputs",
    )
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_seconds
    root = Path(__file__).resolve().parents[1]
    folder = root / f"data/tcga-{args.cancer.lower()}/rna-primary"
    manifest = json.loads((folder / "selected-manifest.json").read_text())
    selected = manifest["files"]
    ids = []
    vectors = []
    genes = None
    symbols = None
    missing = []
    hashes = []
    for item in selected:
        path = folder / (item["file_id"] + ".tsv")
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(min(5, max(0, deadline - time.monotonic())))
        if not path.exists():
            missing.append(item["Patient_ID"])
            continue
        if (
            path.stat().st_size != item["file_size"]
            or hashlib.md5(path.read_bytes()).hexdigest() != item["md5sum"]
        ):
            raise ValueError(f"Unverified RNA source file {item['file_id']}")
        d = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            usecols=["gene_id", "gene_name", "gene_type", "tpm_unstranded"],
        )
        d = d[d.gene_type.eq("protein_coding")].copy()
        current = d.gene_id.str.replace(r"\.\d+$", "", regex=True).to_numpy(dtype=str)
        if len(set(current)) != len(current):
            raise ValueError("Duplicate stripped Ensembl IDs require review")
        if genes is None:
            genes = current
            symbols = d.gene_name.fillna("").to_numpy(dtype=str)
        elif not np.array_equal(genes, current) or not np.array_equal(
            symbols, d.gene_name.fillna("").to_numpy(dtype=str)
        ):
            raise ValueError("Inconsistent gene order/annotation across source files")
        values = pd.to_numeric(d.tpm_unstranded, errors="raise").to_numpy(
            dtype=np.float32
        )
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Invalid TPM values")
        vectors.append(np.log1p(values))
        ids.append(item["Patient_ID"])
        hashes.append(item["md5sum"])
    if missing and not args.allow_partial:
        raise ValueError(
            f"{len(missing)} selected source files missing; finish download or use explicit smoke option"
        )
    if not ids:
        raise ValueError("No verified expression files")
    out = root / "data/processed"
    out.mkdir(exist_ok=True)
    suffix = "_partial" if missing else ""
    path = out / (f"tcga_{args.cancer.lower()}_rna_logtpm" + suffix + ".npz")
    np.savez_compressed(
        path,
        patient_ids=np.asarray(ids),
        gene_ids=genes,
        gene_symbols=symbols,
        values=np.stack(vectors),
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_file": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "patients": len(ids),
        "genes": len(genes),
        "selected_patients": len(selected),
        "missing_patients": missing,
        "complete_selected_cohort": not missing,
        "representation": "GENCODEv36 protein_coding; stripped Ensembl version; log1p(unstranded TPM); no imputation/gene selection/scaling fit",
        "source_manifest_sha256": hashlib.sha256(
            (folder / "selected-manifest.json").read_bytes()
        ).hexdigest(),
        "training_requirement": "Fit any feature filtering/selection, dimensionality reduction and standardization using training patients only; join patient_ids to frozen cohort split",
    }
    (out / (f"tcga_{args.cancer.lower()}_rna_audit" + suffix + ".json")).write_text(
        json.dumps(report, indent=2)
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "missing_patients"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
