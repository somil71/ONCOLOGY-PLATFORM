"""Build an offline Kaggle study notebook, code archive and verified input bundle.

Build locally; upload the resulting dataset directory as a PRIVATE Kaggle dataset
with source-compatible access/terms and attach it to the generated notebook.
No Kaggle upload or training submission is performed by this command.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/CancerFusion_Validated_Study.ipynb"


def write_package_readme(bundle):
    bundle = Path(bundle)
    upload = bundle / "dataset-upload.zip"
    checksum = hashlib.sha256(upload.read_bytes()).hexdigest()
    readme = f"""# CancerFusion Kaggle reproduction package

Prepared and checked locally. Kaggle upload and runtime execution are still pending; local QA is not a Kaggle training result.

1. Create a **private Kaggle dataset** from `dataset-upload.zip`. Kaggle extracts the archive and preserves its folders. Verify the resulting input contains `cancerfusion-study-code/`, `bundle_manifest.json` and `study-data/`.
2. Import `CancerFusion_Validated_Study.ipynb` into a Kaggle notebook and attach that private input dataset. `kernel-metadata.json` intentionally has no dataset identifier until the real dataset exists.
3. Use the notebook's CPU configuration and run the cells in order. Network access is unnecessary after attachment. The preflight checks hashes, pinned author source and recorded outcomes. Library versions are recorded; a different runtime is not guaranteed to reproduce every floating-point bit.
4. Download `/kaggle/working/cancerfusion_validated_outputs.zip` after successful completion. Keep predictions, checkpoints, source hashes, independent audits and figures together. Existing run folders are never overwritten; use a fresh session/output workspace for another full run.

The notebook reproduces BRCA clinical/RNA experiments and separate METABRIC shared-rank transport. When KIRC inputs are included, it then runs **five locked KIRC clinical/RNA folds**, followed by the **separate author MultiSurv mRNA shared-cohort adaptation**. The latter retains the pinned author network/loss and MIT license; cohort,1000gene preprocessing and optimization are disclosed adaptations. It is not the original pan-cancer or full multimodal MultiSurv study.

**Images/four-input execution is OFF.** This package contains no WSI/CT feature matrices or admitted image-cohort lock. Image efficacy, the four-input15-subset hypothesis, full MCAT/PathomicFusion and validated evidential confidence cannot be demonstrated by this package. Procurement timing remains unverified; the KIRC analyses are retrospective associations.

Preserve the negative findings: paired multimodal improvements were not established and disagreement scores did not establish calibrated confidence. `study-data/verified_local_results.json` contains local reference evidence, not Kaggle results or promised accuracy.

Upload ZIP: {upload.stat().st_size:,} bytes. SHA256:

`{checksum}`

See `bundle_qa.json` for the completed local checks and `dataset/bundle_manifest.json` for exact inputs/code. Official documentation: [datasets](https://www.kaggle.com/docs/datasets), [notebooks](https://www.kaggle.com/docs/notebooks).
"""
    (bundle / "README.md").write_text(readme, encoding="utf-8")
    (bundle / "README.txt").write_text(readme, encoding="utf-8")


def write_notebook():
    cells = []

    def md(text):
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": text.splitlines(keepends=True),
            }
        )

    def code(text):
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": text.splitlines(keepends=True),
            }
        )

    md("""# CancerFusion: leakage-restricted survival study

This notebook runs the corrected shared experiment engine, not the legacy pilot head. **Validated means engineering checks, not clinical validation.** Labels are recorded OS/time/censoring; no outcome-generated classification labels are used. Age is the strict diagnosis-time clinical baseline. Pathologic-stage analyses are exploratory because the recorded OS origin and staging-completion landmark are not guaranteed aligned. Earlier legacy runs used overlapping TCGA patients, so the new held-out split is not a globally untouched external cohort.

Attach the private dataset produced by `python -m tools.build_kaggle_study_bundle`. It contains a checksummed `cancerfusion-study-code/` tree, processed patient inputs and checksum manifest. Network access is unnecessary once inputs are attached. This engine uses CPU full-batch training for its approximately1080patient cohort; allocating a GPU does not automatically change its device. No accuracy or C-index is promised. Raw image encoders and four-modality claims require real verified image inputs and are not fabricated here.
""")
    code("""from pathlib import Path
import hashlib, json, os, sys, zipfile, platform, shutil
import numpy as np, pandas as pd, scipy, sklearn, torch, openpyxl
print({'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'openpyxl':openpyxl.__version__})
manifests = list(Path('/kaggle/input').rglob('bundle_manifest.json'))
if len(manifests) != 1:
    raise ValueError(f'Attach exactly one prepared CancerFusion dataset; found {len(manifests)} manifests.')
INPUT = manifests[0].parent
manifest = json.loads((INPUT/'bundle_manifest.json').read_text())
for relative, expected in manifest['sha256'].items():
    path = (INPUT/relative).resolve()
    if not path.is_relative_to(INPUT.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Input checksum/path failure: {relative}')
WORK = Path('/kaggle/working/cancerfusion-study')
WORK.mkdir(parents=True, exist_ok=True)
CODE = INPUT/'cancerfusion-study-code'
if not CODE.is_dir():
    raise ValueError(f'Expected extracted, checksummed code directory: {CODE}')
shutil.copytree(CODE, WORK, dirs_exist_ok=True)
os.chdir(WORK)
sys.path.insert(0,str(WORK))
DATA = INPUT/'study-data'
# Preserve the original relative source layout required by the locked protocol
# and the separately audited author-model runner. Files remain byte-identical.
for relative in manifest.get('project_layout_inputs',[]):
    source = DATA/relative
    target = WORK/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
print('Verified private input bundle:',manifest['built_at_utc'])
""")
    md("""## Configuration and engineering checks

Three training seeds42/123/2026; training-only preprocessing and gene selection. BRCA uses its existing60/20/20 patient split. The separate KIRC study uses the locked five outer folds and fixed inner validation assignments described below. Event counts, available inputs and unsupported horizons are exported. Conditional patient-bootstrap intervals do not include retraining uncertainty or overlap among development folds. Library versions are recorded; identical input/code hashes do not guarantee bitwise equality on a different runtime. Do not choose the best test configuration and claim superiority.
""")
    code("""import subprocess
from datetime import datetime, timezone
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
EPOCHS = 300
RNA_GENES = 256
RUN_CLINICAL = True
RUN_CLI_RNA = True
RUN_EXTERNAL_RANK_TRANSPORT = True
RUN_KIRC_CLI_RNA = 'TCGA-KIRC' in manifest['cohorts']
RUN_KIRC_AUTHOR_MULTISURV_RNA = RUN_KIRC_CLI_RNA
# Image acquisition alone is insufficient: a reviewed QC-cohort lock and
# verified feature matrices are required before any image efficacy experiment.
RUN_IMAGE_FOUR_INPUT_STUDY = False
OUT = WORK/'RUN REPORT'
OUT.mkdir(exist_ok=True)
def run_module(module, *arguments):
    command = [sys.executable,'-m',module,*map(str,arguments)]
    print('Running:',command)
    subprocess.run(command,check=True)
run_module('tools.test_survival_models')
run_module('tools.test_survival_metrics')
if RUN_KIRC_AUTHOR_MULTISURV_RNA:
    run_module('tools.test_published_baselines')
""")
    md("""## A. Recorded-outcome clinical baselines

Age-only Cox is the strict shared baseline. Age+pathologic-stage Cox is an exploratory sensitivity analysis; source staging date versus OS origin must be reviewed before a clinical landmark claim. A conventional censored-hazard neural model and between-seed disagreement are evaluated, without claiming calibrated evidential uncertainty. METABRIC external predictions use a frozen age-only model; target outcomes estimate only the evaluation censoring nuisance distribution, not model parameters.
""")
    code("""if RUN_CLINICAL:
    run_module('tools.run_survival_experiments',
        '--input',DATA/'tcga_brca_clinical.csv',
        '--external',DATA/'metabric_confirmed_carcinoma_clinical.csv',
        '--epochs',EPOCHS,'--output',OUT/f'clinical-{RUN_ID}')
""")
    md("""## B. Actual matched clinical+RNA experiments and ablations

This uses full downloaded and checksum-verified primary-tumour RNA inputs, not outcomes alone. The clinical branch uses age only. Top256variable genes are selected on training patients only. Compare unimodal, concatenation, mean and attention fusion; no-mask and training-dropout0.1/0.3/0.5; separately retrained leave-one-input-out models; and inference withholding on identical complete-input held-out cases. With only two actual modalities these are **two-modality experiments**, not three/four-modality results. No WSI or radiology performance is inferred.
""")
    code("""if RUN_CLI_RNA:
    run_module('tools.run_survival_experiments',
        '--input',DATA/'tcga_brca_clinical.csv','--clinical-columns','age_years',
        '--rna-npz',DATA/'tcga_brca_rna_logtpm.npz','--rna-genes',RNA_GENES,
        '--complete-cohort','--epochs',EPOCHS,'--output',OUT/f'cli-rna-{RUN_ID}')
""")
    md("""## C. Separate exploratory cross-assay external transfer

Raw METABRIC microarray values cannot pass through an RNA-seq logTPM scaler. This is a **different representation and different trained model**: source/target shared gene symbols are fixed, duplicate symbols averaged within patient, patient-level ranks computed, then gene selection/scaling fitted only on TCGA training. The model is frozen before METABRIC outcomes are evaluated. This is an exploratory portability study, not proof that rank normalization removes domain shift.
""")
    code("""if RUN_EXTERNAL_RANK_TRANSPORT:
    run_module('tools.run_survival_transport_experiment',
        '--source-clinical',DATA/'tcga_brca_clinical.csv',
        '--source-rna',DATA/'tcga_brca_rna_logtpm.npz',
        '--target-clinical',DATA/'metabric_confirmed_carcinoma_clinical.csv',
        '--target-rna',DATA/'metabric_microarray.npz',
        '--genes',RNA_GENES,'--epochs',EPOCHS,'--output',OUT/f'rank-transfer-{RUN_ID}')
""")
    md("""## D. Separate retrospective KIRC clinical + RNA study: five locked folds

The `--include-kirc` bundle contains217 patients/66 recorded OS deaths, the original CDR workbook, protocol and all five immutable patient ledgers. Original OS/time are retained. RNA procurement timing is unknown; this is retrospective association, without a prospective diagnosis-time claim. GDC `days_to_collection` is center receipt and is never substituted as tissue procurement or a landmark. No receipt-date exclusions are applied. Kidney and breast cohorts remain separate.

Each fold compares matched age-only Cox, clinical/RNA unimodal networks, concatenation, mean and attention fusion, masking/dropout0/0.1/0.3/0.5, retrained leave-one-input-out models and inference withholding. Top256variable genes and scalers are fitted only on that fold's training patients. The locked primary estimator is the equal mean of within-fold C, with paired patient resampling **within** fixed folds. Secondary pooled OOF scores use negative mean predicted survival at365/1095/1825days for every model, including Cox; unscaled Cox linear predictors are never pooled. The two-input three-subset experiment does not test the planned four-input15-subset hypothesis.
""")
    code("""if RUN_KIRC_CLI_RNA:
    # This fixed relative output identifier is also the frozen reference used
    # by the separately executed author-model driver. A rerun never overwrites it.
    KIRC_OUT = OUT/'kirc-cli-rna-20261118-fixed5fold-v1'
    KIRC_OUT.mkdir(exist_ok=False)
    for fold in range(1,6):
        run = KIRC_OUT/f'fold_{fold:02d}'
        run_module('tools.run_survival_experiments',
            '--input',WORK/'data/processed/tcga_kirc_clinical.csv',
            '--clinical-columns','age_years',
            '--rna-npz',WORK/'data/processed/tcga_kirc_rna_logtpm.npz',
            '--rna-genes',256,'--complete-cohort',
            '--split-csv',WORK/f'data/processed/kirc-fixed-folds/fold_{fold:02d}.csv',
            '--study-protocol',WORK/'config/kirc_retrospective_os_protocol.json',
            '--bootstrap',1000,'--bootstrap-seed',20261118,
            '--epochs',300,'--output',run)
        run_module('tools.independently_validate_survival_run','--run',run)
    run_module('tools.summarize_survival_folds','--runs-root',KIRC_OUT)
    run_module('tools.independently_validate_survival_folds','--root',KIRC_OUT)
    run_module('tools.plot_survival_fold_summary','--runs-root',KIRC_OUT)
else:
    print('Separate KIRC clinical/RNA study skipped: optional verified inputs were not bundled.')
""")
    md("""## E. Separate adapted published baseline: author MultiSurv mRNA network/loss

This executes a separate RNA-only baseline from the pinned author repository commit `deb35e12502f61a59726bfb543210735406d5959`; original MIT source/license are included and checked. It retains the unmodified author mRNA network,30annual conditional-survival outputs and original midpoint-censor likelihood. The217-case KIRC cohort, locked folds, log1pTPM top1000training-variance genes/scaling, Adam0.0001 and early-stopping policy are explicit shared-cohort adaptations. This is not the original pan-cancer study, nine-category clinical pathway, ResNeXt WSI pathway, CT model or a four-input MultiSurv reproduction. The benchmark was added after the initial CLI/RNA analysis and is secondary. Different gene budgets and training policies prevent a clean architectural-superiority claim.

The author-model driver and validator use the original project-relative paths and therefore remain unmodified. Run the KIRC reference step first. The independent audit reconstructs checkpoints, preprocessing, likelihood, original CDR outcomes, predictions and paired fold summaries. Original float32 CSV rendering can round decimal text; the validator writes a separate full-precision reconstruction and preserves the original artifact/hash.
""")
    code("""if RUN_KIRC_AUTHOR_MULTISURV_RNA:
    if not RUN_KIRC_CLI_RNA or not (OUT/'kirc-cli-rna-20261118-fixed5fold-v1/five_fold_summary.json').is_file():
        raise ValueError('Run and validate the separate locked KIRC reference study first.')
    run_module('tools.run_author_multisurv_rna')
    run_module('tools.validate_author_multisurv_rna')
else:
    print('Adapted author MultiSurv RNA study skipped: KIRC reference inputs are absent.')
""")
    md("""## Interpretation and uncertainty: preserve the negative findings

The independently audited local KIRC study did not establish multimodal superiority: all paired improvement intervals versus age-only Cox crossed zero. For the exploratory two-input missingness panel, dropout0.3 minus0 was0.00789,95% conditional CI[−0.00396,0.02013]. Attention's5-year error/disagreement correlations varied approximately−0.299 to+0.131 across folds and its selective-risk curves were inconsistent. These are three-seed disagreement scores; calibrated confidence and evidential-survival uncertainty remain unestablished. The planned four-input hypothesis remains untested.

Recompute findings from this run; these local observations are not acceptance targets or promised accuracy. Images/four-input execution stays OFF until a reviewed QC-cohort lock and verified feature matrices are available. Source availability and encoder extraction do not establish tumour coverage, diagnosis-time eligibility or clinical validation.
""")
    md("""## Results, figures and downloadable artifacts

Read results from exported predictions and metric files. Clinical classification accuracy is not an appropriate replacement for censored survival metrics. Intervals crossing zero in paired differences do not establish improvement. Ensemble disagreement/error association is descriptive and may fail to identify poor predictions. All data-constrained or unsupported metrics must remain explicit.
""")
    code("""rows=[]
for folder in sorted(OUT.iterdir()):
    if (folder/'five_fold_summary.json').is_file():
        combined=json.loads((folder/'five_fold_summary.json').read_text())
        for name,value in combined['primary_locked_within_fold_summary'].items():
            rows.append({'run':folder.name,'model':name,'scope':'primary_mean_within_fold_c','patients':combined['oof_patients'],'events':combined['oof_observed_deaths'],'harrell_c':value['mean_within_fold_c'],'bootstrap_95_ci':value['conditional_fold_stratified_patient_bootstrap_95_ci']})
        for name,value in combined['secondary_pooled_oof_common_probability_summary'].items():
            rows.append({'run':folder.name,'model':name,'scope':'secondary_pooled_oof_common_probability_c','patients':combined['oof_patients'],'events':combined['oof_observed_deaths'],'harrell_c':value['harrell_c'],'bootstrap_95_ci':value['bootstrap_95_ci']})
        continue
    if folder.name=='kirc-multisurv-author-rna-20261118-v1':
        author=json.loads((folder/'independent_validation.json').read_text())['aggregate']
        rows.append({'run':folder.name,'model':'MultiSurv-author-mRNA shared-cohort adaptation','scope':'secondary_mean_within_fold_c','patients':author['patients'],'events':author['deaths'],'harrell_c':author['mean_outer_fold_c'],'bootstrap_95_ci':author['conditional_patient_bootstrap_95_ci']})
        continue
    if not (folder/'metrics.json').exists():continue
    measured=json.loads((folder/'metrics.json').read_text())
    for name,result in measured.items():
        for scope in ('test','internal','external'):
            value=result.get(scope)
            if isinstance(value,dict) and 'harrell_c' in value:
                rows.append({'run':folder.name,'model':name,'scope':scope,'patients':value.get('patients'),'events':value.get('observed_events'),'harrell_c':value['harrell_c'],'bootstrap_95_ci':value.get('bootstrap_95_ci')})
    if (folder/'test_predictions.csv').exists():
        run_module('tools.independently_validate_survival_run','--run',folder)
        if (folder/'inference_withholding_predictions.csv').exists():
            run_module('tools.audit_multimodal_robustness','--run',folder)
        run_module('tools.plot_survival_results','--run',folder)
    elif (folder/'predictions.csv').exists():
        run_module('tools.independently_validate_transport_run','--run',folder)
        run_module('tools.plot_transport_results','--run',folder)
pd.DataFrame(rows).to_csv(OUT/'all_run_results.csv',index=False)
display(pd.DataFrame(rows))
output_zip=Path('/kaggle/working/cancerfusion_validated_outputs.zip')
with zipfile.ZipFile(output_zip,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in OUT.rglob('*'):
        if path.is_file():archive.write(path,path.relative_to(WORK))
print('Download:',output_zip)
""")
    md("""Primary methods: [Kvamme and Borgan2021, censored discrete hazards](https://pmc.ncbi.nlm.nih.gov/articles/PMC8536596/); recorded endpoint source [TCGA-CDR](https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018); external source [cBioPortal METABRIC](https://github.com/cBioPortal/datahub/tree/master/public/brca_metabric); [pinned author MultiSurv source](https://github.com/luisvalesilva/multisurv/tree/deb35e12502f61a59726bfb543210735406d5959). The explicit mRNA-only author adaptation above does not reproduce the full MultiSurv paper. MCAT/PathomicFusion, image-cohort efficacy and evidential-survival uncertainty remain separate requirements.
""")
    document = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, cell in enumerate(cells):
        cell["id"] = f"validated-study-{i:02}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell{i}", "exec")
    NOTEBOOK.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/kaggle-validated-study"
    )
    parser.add_argument("--notebook-only", action="store_true")
    parser.add_argument(
        "--include-kirc",
        action="store_true",
        help="Include the complete verified separate KIRC clinical/RNA cohort; never infer image availability.",
    )
    args = parser.parse_args()
    write_notebook()
    if args.notebook_only:
        print(f"Wrote and compiled notebook: {NOTEBOOK}")
        return
    code_files = [
        *ROOT.glob("src/survival_*.py"),
        ROOT / "src/clinical_contract.py",
        ROOT / "src/image_feature_cohort.py",
        ROOT / "src/published_baselines.py",
        ROOT / "tools/run_survival_experiments.py",
        ROOT / "tools/run_survival_external_evaluation.py",
        ROOT / "tools/run_survival_transport_experiment.py",
        ROOT / "tools/test_survival_models.py",
        ROOT / "tools/test_survival_metrics.py",
        ROOT / "tools/plot_survival_results.py",
        ROOT / "tools/plot_transport_results.py",
        ROOT / "tools/independently_validate_survival_run.py",
        ROOT / "tools/independently_validate_transport_run.py",
        ROOT / "tools/audit_multimodal_robustness.py",
        ROOT / "tools/summarize_survival_folds.py",
        ROOT / "tools/independently_validate_survival_folds.py",
        ROOT / "tools/plot_survival_fold_summary.py",
        ROOT / "tools/run_author_multisurv_rna.py",
        ROOT / "tools/validate_author_multisurv_rna.py",
        ROOT / "tools/test_published_baselines.py",
        ROOT / "requirements-research-tested.txt",
    ]
    vendor_root = ROOT / "third_party/multisurv"
    vendor_manifest = json.loads((vendor_root / "source_manifest.json").read_text())
    if vendor_manifest["commit"] != "deb35e12502f61a59726bfb543210735406d5959":
        raise ValueError("Unexpected MultiSurv source revision")
    for record in vendor_manifest["files"]:
        path = (vendor_root / record["path"]).resolve()
        if (
            not path.is_relative_to(vendor_root.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
        ):
            raise ValueError("Pinned MultiSurv source/license checksum mismatch")
        code_files.append(path)
    code_files.append(vendor_root / "source_manifest.json")
    project_layout_inputs = {}
    data_files = [
        ROOT / "data/processed" / name
        for name in [
            "tcga_brca_clinical.csv",
            "tcga_brca_rna_logtpm.npz",
            "metabric_confirmed_carcinoma_clinical.csv",
            "metabric_microarray.npz",
        ]
    ]
    if args.include_kirc:
        import numpy as np
        import pandas as pd

        clinical_path = ROOT / "data/processed/tcga_kirc_clinical.csv"
        rna_path = ROOT / "data/processed/tcga_kirc_rna_logtpm.npz"
        audit = json.loads(
            (ROOT / "data/processed/tcga_kirc_rna_audit.json").read_text()
        )
        source_manifest = ROOT / "data/tcga-kirc/rna-primary/selected-manifest.json"
        if (
            not audit.get("complete_selected_cohort")
            or audit.get("missing_patients")
            or hashlib.sha256(rna_path.read_bytes()).hexdigest() != audit["sha256"]
            or hashlib.sha256(source_manifest.read_bytes()).hexdigest()
            != audit["source_manifest_sha256"]
        ):
            raise ValueError("KIRC matrix/source audit is incomplete or changed")
        clinical = pd.read_csv(clinical_path)
        with np.load(rna_path, allow_pickle=False) as rna:
            ids = rna["patient_ids"].astype(str)
            if (
                clinical.Patient_ID.duplicated().any()
                or len(set(ids)) != len(ids)
                or set(clinical.Patient_ID) != set(ids)
                or not np.isfinite(rna["values"]).all()
            ):
                raise ValueError(
                    "KIRC requires unique, complete, matched verified clinical/RNA inputs"
                )
        protocol_path = ROOT / "config/kirc_retrospective_os_protocol.json"
        protocol = json.loads(protocol_path.read_text())
        if (
            hashlib.sha256(clinical_path.read_bytes()).hexdigest()
            != protocol["clinical_source_sha256"]
        ):
            raise ValueError("KIRC clinical source changed after protocol lock")
        project_layout_inputs = {
            "config/kirc_retrospective_os_protocol.json": protocol_path,
            "data/processed/tcga_kirc_clinical.csv": clinical_path,
            "data/processed/tcga_kirc_rna_logtpm.npz": rna_path,
            "data/tcga-brca/TCGA-CDR-SupplementalTableS1.xlsx": ROOT
            / "data/tcga-brca/TCGA-CDR-SupplementalTableS1.xlsx",
        }
        for record in protocol["fold_ledgers"]:
            path = ROOT / record["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(
                    "KIRC fixed patient assignments changed after protocol lock"
                )
            project_layout_inputs[record["path"]] = path
    for path in [*code_files, *data_files]:
        if not path.is_file():
            raise FileNotFoundError(f"Complete verified input required: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = args.output / "dataset"
    dataset.mkdir()
    (dataset / "study-data").mkdir()
    import shutil

    code_root = dataset / "cancerfusion-study-code"
    for path in code_files:
        target = code_root / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    for path in data_files:
        shutil.copy2(path, dataset / "study-data" / path.name)
    for relative, path in project_layout_inputs.items():
        target = dataset / "study-data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    if args.include_kirc:
        study = ROOT / "RUN REPORT/kirc-cli-rna-20261118-fixed5fold-v1"
        author = ROOT / "RUN REPORT/kirc-multisurv-author-rna-20261118-v1"
        local_summary = json.loads((study / "five_fold_summary.json").read_text())
        local_audit = json.loads(
            (study / "independent_five_fold_summary.json").read_text()
        )
        author_audit = json.loads((author / "independent_validation.json").read_text())
        author_run = json.loads((author / "run_manifest.json").read_text())
        if (
            local_audit["independent_summary_comparison"]["status"] != "passed"
            or author_audit["status"] != "passed"
        ):
            raise ValueError(
                "The locally referenced KIRC studies require completed independent audits"
            )
        for relative, expected in local_summary["training_code_sha256"].items():
            if hashlib.sha256(Path(relative).read_bytes()).hexdigest() != expected:
                raise ValueError(
                    "Current shared experiment source differs from the audited local KIRC execution"
                )
        for relative, expected in author_run["code_sha256"].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(
                    "Current author-model source differs from its audited local execution"
                )
        evidence = {
            "scope": "Reference observations from audited local runs; not Kaggle outputs or future acceptance targets",
            "kirc_primary_fold_summary": local_summary[
                "primary_locked_within_fold_summary"
            ],
            "kirc_secondary_oof_summary": local_summary[
                "secondary_pooled_oof_common_probability_summary"
            ],
            "exploratory_two_input_missingness": local_summary["missing_input_panel"],
            "uncertainty_by_fold": {
                k: v
                for k, v in local_summary["uncertainty_by_fold"].items()
                if k.endswith("/fusion_attention")
            },
            "author_multisurv_secondary_adaptation": author_audit["aggregate"],
            "adaptation_notice": author_run["adaptation_notice"],
            "source_artifact_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    study / "five_fold_summary.json",
                    study / "independent_five_fold_summary.json",
                    author / "run_manifest.json",
                    author / "independent_validation.json",
                ]
            },
        }
        (dataset / "study-data/verified_local_results.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8"
        )
    for name in [
        "clinical_cohort_audit.json",
        "tcga_brca_rna_audit.json",
        "metabric_microarray_audit.json",
        "metabric_histology_eligibility.json",
        *(["tcga_kirc_rna_audit.json"] if args.include_kirc else []),
    ]:
        path = ROOT / "data/processed" / name
        if path.exists():
            shutil.copy2(path, dataset / "study-data" / name)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Offline prepared research inputs; no Kaggle submission performed",
        "cohorts": ["TCGA-BRCA", "METABRIC"]
        + (["TCGA-KIRC"] if args.include_kirc else []),
        "image_modalities_bundled": [],
        "image_four_input_execution_enabled": False,
        "project_layout_inputs": list(project_layout_inputs),
        "multisurv_author_source": {
            "repository": vendor_manifest["repository"],
            "commit": vendor_manifest["commit"],
            "license": vendor_manifest["license"],
        },
        "reproduction_steps": [
            "BRCA clinical",
            "BRCA clinical/RNA",
            "BRCA/METABRIC shared-rank external transfer",
        ]
        + (
            [
                "KIRC clinical/RNA five locked folds",
                "Separate adapted author MultiSurv RNA five locked folds",
            ]
            if args.include_kirc
            else []
        ),
        "sha256": {
            path.relative_to(dataset).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in dataset.rglob("*")
            if path.is_file()
        },
    }
    (dataset / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    with zipfile.ZipFile(
        args.output / "dataset-upload.zip", "w", compression=zipfile.ZIP_STORED
    ) as archive:
        for path in dataset.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(dataset))
    shutil.copy2(NOTEBOOK, args.output / NOTEBOOK.name)
    kernel = {
        "id": "somiljha/cancerfusion-leakage-restricted-study",
        "title": "CancerFusion leakage-restricted survival study",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (args.output / "kernel-metadata.json").write_text(
        json.dumps(kernel, indent=2), encoding="utf-8"
    )
    write_package_readme(args.output)
    print(
        json.dumps(
            {
                "bundle": str(args.output),
                "dataset_bytes": sum(
                    p.stat().st_size for p in dataset.rglob("*") if p.is_file()
                ),
                "notebook_cells": len(write_notebook()["cells"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
