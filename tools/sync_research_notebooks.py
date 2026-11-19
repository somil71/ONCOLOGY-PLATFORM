"""Refresh standalone Kaggle cells from canonical source modules.

The dataset/orchestration remains notebook-specific. Generated definitions are
verified by tools.test_research_objectives so fixes cannot silently diverge.
"""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "CancerFusion_Phase2_PanCancer.ipynb"
    document = json.loads(path.read_bytes())
    architecture = (root / "src/architecture.py").read_text(encoding="utf-8-sig")
    document["cells"][5]["source"] = architecture.splitlines(keepends=True)
    dataset = "".join(document["cells"][4]["source"]).split(
        "class PanCancerDataset", 1
    )[1]
    dataset = "class PanCancerDataset" + dataset
    if "        # PanCancer Double Directory Fix" in dataset:
        start = dataset.index("        # PanCancer Double Directory Fix")
        end = dataset.index("    def __len__(self):", start)
        dataset = (
            dataset[:start]
            + "        self.img_folder = CFG['dir_multimodal']\n        self.image_manifest = load_image_manifest(CFG['image_manifest_csv'], self.img_folder)\n\n"
            + dataset[end:]
        )
    imaging = (root / "src/imaging_manifest.py").read_text(encoding="utf-8-sig")
    audit = (
        (root / "src/data_audit.py")
        .read_text(encoding="utf-8-sig")
        .replace(
            "            from src.imaging_manifest import order_mr_series, select_evenly_spaced\n",
            "",
        )
    )
    document["cells"][4]["source"] = (
        imaging + "\n\n" + audit + "\n\n" + dataset
    ).splitlines(keepends=True)
    for index in (4, 5):
        code = "".join(document["cells"][index]["source"])
        document["cells"][index]["metadata"]["generated_source_sha256"] = (
            hashlib.sha256(code.encode()).hexdigest()
        )
    contents = (
        (json.dumps(document, indent=1, ensure_ascii=False) + "\n")
        .replace("\n", "\r\n")
        .encode()
    )
    path.write_bytes(contents)
    (root / "notebooks" / path.name).write_bytes(contents)
    print("Synchronized canonical architecture and image auditing into both notebooks.")


if __name__ == "__main__":
    main()
