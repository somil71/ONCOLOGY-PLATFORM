"""Size-selected diagnostic slide pilot for loader feasibility, not model evaluation."""

from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd
from tools.download_primary_rna import verified_download


def main():
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads(
        (root / "data/tcga-brca/inventory/diagnostic_slides-metadata.json").read_text()
    )
    allowed = set(
        pd.read_csv(root / "data/processed/tcga_brca_clinical.csv").Patient_ID
    )
    eligible = [
        x
        for x in metadata
        if len(x.get("cases", [])) == 1 and x["cases"][0]["submitter_id"] in allowed
    ]
    selected = []
    seen = set()
    for x in sorted(eligible, key=lambda x: (x["file_size"], x["file_id"])):
        patient = x["cases"][0]["submitter_id"]
        if patient in seen:
            continue
        selected.append({"Patient_ID": patient, **x})
        seen.add(patient)
        if len(selected) == 3:
            break
    folder = root / "data/tcga-brca/wsi-pilot"
    folder.mkdir(exist_ok=True)
    results = []
    for item in selected:
        result = verified_download(item, folder, suffix=".svs")
        results.append(
            {
                **result,
                "modality": "wsi",
                "path": str((folder / (item["file_id"] + ".svs")).relative_to(root)),
                "md5sum": item["md5sum"],
                "file_size": item["file_size"],
                "source_url": "https://api.gdc.cancer.gov/data/" + item["file_id"],
            }
        )
        print(result, flush=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "Three smallest diagnostic-slide files on different OS-eligible TCGA BRCA patients; feasibility pilot only. Size selection is biased and must not define a performance cohort.",
        "records": results,
    }
    (folder / "pilot-manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
