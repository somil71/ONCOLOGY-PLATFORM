"""Verify a prepared dataset/notebook bundle without submitting a Kaggle run."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from tools.run_survival_experiments import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    dataset = bundle / "dataset"
    manifest = json.loads((dataset / "bundle_manifest.json").read_text())
    if any("\\" in name for name in manifest["sha256"]):
        raise ValueError(
            "Input manifest paths must use POSIX separators for the Linux Kaggle runtime"
        )
    for relative, expected in manifest["sha256"].items():
        path = (dataset / relative).resolve()
        if (
            not path.is_relative_to(dataset)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Dataset checksum/path failure: {relative}")
    notebook = json.loads((bundle / "CancerFusion_Validated_Study.ipynb").read_text())
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell{i}", "exec")
    notebook_source = "\n".join("".join(c["source"]) for c in notebook["cells"])
    if (
        "RUN_IMAGE_FOUR_INPUT_STUDY = False" not in notebook_source
        or manifest.get("image_four_input_execution_enabled") is not False
    ):
        raise ValueError("Image/four-input execution must remain explicitly disabled")
    repo = Path(__file__).resolve().parents[1]
    runtime = repo / "runtime"
    runtime.mkdir(exist_ok=True)
    tests = []
    code_files = 0
    archive_files = 0
    with tempfile.TemporaryDirectory(prefix="bundle-qa-", dir=runtime) as tmp:
        extract = Path(tmp).resolve()
        if not extract.is_relative_to(runtime.resolve()):
            raise ValueError(
                "Temporary extraction directory escaped the workspace runtime"
            )
        code_root = dataset / "cancerfusion-study-code"
        if not code_root.is_dir():
            raise ValueError("Dataset must contain the Kaggle-extracted code directory")
        for item in sorted(path for path in code_root.rglob("*") if path.is_file()):
            relative = item.relative_to(code_root).as_posix()
            source = repo / relative
            if not source.is_file() or item.read_bytes() != source.read_bytes():
                raise ValueError(
                    f"Packaged source/license/environment differs from the workspace: {relative}"
                )
            archive_files += 1
            if item.suffix == ".py":
                compile(item.read_text(encoding="utf-8"), relative, "exec")
                code_files += 1
        shutil.copytree(code_root, extract, dirs_exist_ok=True)
        for relative in manifest.get("project_layout_inputs", []):
            target = (extract / relative).resolve()
            if "\\" in relative or not target.is_relative_to(extract):
                raise ValueError("Invalid portable project input layout")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dataset / "study-data" / relative, target)
        for module in [
            "tools.test_survival_models",
            "tools.test_survival_metrics",
            "tools.test_published_baselines",
        ]:
            result = subprocess.run(
                [sys.executable, "-m", module],
                cwd=extract,
                capture_output=True,
                text=True,
                check=True,
            )
            tests.append(
                {
                    "module": module,
                    "exit_code": result.returncode,
                    "output": result.stdout.strip(),
                }
            )
        if "TCGA-KIRC" in manifest["cohorts"]:
            preflight = """from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
from tools.run_survival_experiments import locked_patient_assignments,verify_study_protocol
from tools import run_author_multisurv_rna,validate_author_multisurv_rna
root=Path.cwd()
protocol_path=Path('config/kirc_retrospective_os_protocol.json')
protocol=json.loads(protocol_path.read_text())
clinical=pd.read_csv(protocol['clinical_source'])
for record in protocol['fold_ledgers']:
    path=Path(record['path'])
    locked_patient_assignments(path,clinical.Patient_ID)
    verify_study_protocol(protocol_path,clinical,['age_years'],path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
with np.load('data/processed/tcga_kirc_rna_logtpm.npz',allow_pickle=False) as matrix:
    assert set(matrix['patient_ids'].astype(str))==set(clinical.Patient_ID)
original=pd.read_excel('data/tcga-brca/TCGA-CDR-SupplementalTableS1.xlsx',sheet_name='TCGA-CDR')
original=original.loc[original.type.eq('KIRC')].set_index('bcr_patient_barcode')
np.testing.assert_array_equal(original.loc[clinical.Patient_ID,'OS.time'],clinical.time)
np.testing.assert_array_equal(original.loc[clinical.Patient_ID,'OS'],clinical.event)
assert run_author_multisurv_rna.ROOT==root
assert run_author_multisurv_rna.REFERENCE==root/'RUN REPORT/kirc-cli-rna-20261118-fixed5fold-v1'
assert validate_author_multisurv_rna.REFERENCE==run_author_multisurv_rna.REFERENCE
assert validate_author_multisurv_rna.RUN==run_author_multisurv_rna.OUTPUT
print(json.dumps({'status':'passed','patients':len(clinical),'deaths':int(clinical.event.sum()),'locked_folds':5,'original_cdr_outcomes_match':True,'separate_author_runner_paths_match':True,'model_fitting_performed':False}))
"""
            result = subprocess.run(
                [sys.executable, "-c", preflight],
                cwd=extract,
                capture_output=True,
                text=True,
                check=True,
            )
            tests.append(
                {
                    "module": "portable_kirc_protocol_and_author_paths_preflight",
                    "exit_code": result.returncode,
                    "output": result.stdout.strip(),
                }
            )
    with zipfile.ZipFile(bundle / "dataset-upload.zip") as archive:
        if archive.testzip() is not None:
            raise ValueError("Upload archive CRC failure")
        expected_names = {
            str(p.relative_to(dataset)).replace("\\", "/")
            for p in dataset.rglob("*")
            if p.is_file()
        }
        if set(archive.namelist()) != expected_names:
            raise ValueError(
                "Upload archive members do not match the verified dataset folder"
            )
        for name in archive.namelist():
            if (
                hashlib.sha256(archive.read(name)).hexdigest()
                != hashlib.sha256((dataset / name).read_bytes()).hexdigest()
            ):
                raise ValueError(f"Upload archive payload differs: {name}")
    report = {
        "status": "passed",
        "scope": "Local prepared-bundle integrity/import/analytic checks; no Kaggle execution or submission performed",
        "dataset_files_verified": len(manifest["sha256"]),
        "current_source_files_compiled_and_matched": code_files,
        "all_source_license_environment_files_matched": archive_files,
        "cohorts": manifest["cohorts"],
        "image_modalities_bundled": manifest["image_modalities_bundled"],
        "image_four_input_execution_enabled": False,
        "manifest_paths_posix_portable": True,
        "notebook_cells_compiled": len(notebook["cells"]),
        "upload_zip_bytes": (bundle / "dataset-upload.zip").stat().st_size,
        "upload_zip_sha256": hashlib.sha256(
            (bundle / "dataset-upload.zip").read_bytes()
        ).hexdigest(),
        "extracted_code_tests": tests,
    }
    dump(bundle / "bundle_qa.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
