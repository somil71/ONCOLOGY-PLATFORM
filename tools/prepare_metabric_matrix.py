"""Prepare public METABRIC microarray matrix without refitting normalization."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "data/metabric/data_mrna_illumina_microarray.txt"
    s = pd.read_csv(
        root / "data/metabric/data_clinical_sample.txt", sep="\t", comment="#"
    )
    eligible = set(
        pd.read_csv(root / "data/processed/metabric_clinical.csv").Patient_ID
    )
    if s.SAMPLE_ID.duplicated().any():
        raise ValueError("Resolve duplicate sample IDs")
    sample_to_patient = s.set_index("SAMPLE_ID").PATIENT_ID.to_dict()
    header = pd.read_csv(source, sep="\t", nrows=0).columns
    sample_columns = [x for x in header[2:] if sample_to_patient.get(x) in eligible]
    ids = [sample_to_patient[x] for x in sample_columns]
    if len(set(ids)) != len(ids):
        raise ValueError("Resolve multiple expression samples per patient")
    chunks = []
    symbols = []
    entrez = []
    for d in pd.read_csv(
        source,
        sep="\t",
        usecols=["Hugo_Symbol", "Entrez_Gene_Id", *sample_columns],
        chunksize=500,
    ):
        symbols.extend(d.Hugo_Symbol.fillna("").astype(str))
        entrez.extend(d.Entrez_Gene_Id.fillna("").astype(str))
        chunks.append(d.loc[:, sample_columns].to_numpy(dtype=np.float32))
    values = np.concatenate(chunks, axis=0).T
    out = root / "data/processed"
    target = out / "metabric_microarray.npz"
    np.savez_compressed(
        target,
        patient_ids=np.asarray(ids),
        gene_symbols=np.asarray(symbols),
        entrez_ids=np.asarray(entrez),
        values=values,
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_file": str(target.relative_to(root)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "matrix_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "patients": len(ids),
        "genes": values.shape[1],
        "nonfinite_values": int((~np.isfinite(values)).sum()),
        "duplicate_gene_symbols": len(symbols) - len(set(symbols)),
        "representation": "cBioPortal METABRIC normalized Illumina microarray values preserved; no additional log transform or outcome-fitted transforms",
        "warning": "RNA-seq logTPM and normalized microarray values are not directly interchangeable. Shared-gene representation and train-only harmonization require explicit validation before external multimodal inference.",
    }
    (out / "metabric_microarray_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
