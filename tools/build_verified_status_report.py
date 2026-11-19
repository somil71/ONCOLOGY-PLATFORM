"""Build the current handoff tables from completed, independently audited outputs."""

from datetime import datetime, timezone
import json
from pathlib import Path

from tools.extract_ct_features import publish, sha256


def main():
    main_run = Path("RUN REPORT/clean-cli-rna-1080-20261118-v3")
    transfer = Path("RUN REPORT/rank-transfer-cli-rna-20261118-v2")
    metrics = json.loads((main_run / "metrics.json").read_text())
    external = json.loads((transfer / "metrics.json").read_text())
    main_audit = json.loads(
        (main_run / "independent_prediction_reconstruction.json").read_text()
    )
    external_audit = json.loads(
        (transfer / "independent_transport_reconstruction.json").read_text()
    )
    kirc_run = Path("RUN REPORT/kirc-cli-rna-20261118-fixed5fold-v1")
    kirc_audit = json.loads(
        (kirc_run / "independent_five_fold_summary.json").read_text()
    )
    if main_audit["status"] != "passed" or external_audit["status"] != "passed":
        raise ValueError(
            "Independent reconstruction must pass before publishing results."
        )
    if kirc_audit["status"] != "independently_reconstructed":
        raise ValueError("Independent KIRC reconstruction is required.")
    manifest = json.loads((main_run / "run_manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Main run is not complete.")
    availability_path = Path("data/processed/kirc_actual_input_availability.json")
    availability = json.loads(availability_path.read_text())
    wsi_audit_path = Path("data/processed/kirc_wsi_features/extraction-audit.json")
    wsi_audit = json.loads(wsi_audit_path.read_text()) if wsi_audit_path.exists() else {}
    labels = {
        "clinical_cox": "Age-only Cox",
        "unimodal_cli": "Age-only neural ensemble",
        "unimodal_rna": "RNA neural ensemble",
        "fusion_concat": "Clinical + RNA concatenation",
        "fusion_mean": "Clinical + RNA mean fusion",
        "fusion_attention": "Clinical + RNA attention",
        "attention_no_mask": "Attention, no masking",
        "attention_dropout_01": "Attention, dropout 0.1",
        "attention_dropout_03": "Attention, dropout 0.3",
        "attention_dropout_05": "Attention, dropout 0.5",
        "without_cli": "Retrained without clinical",
        "without_rna": "Retrained without RNA",
        "age_only_cox": "Age-only Cox",
    }
    lines = [
        "# CancerFusion — current verified research status",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} from saved outputs and independent reconstruction reports. Documentation is not treated as experimental proof.",
        "",
        "The study now has real censored-survival experiments and real matched imaging feasibility. **All seven coordinator concerns are not fixed.** A completed four-input survival evaluation is still absent.",
        "",
        "## Coordinator concerns",
        "",
        "| Concern | Verified progress | Remaining requirement |",
        "| --- | --- | --- |",
        "| 1. Survival evaluation | Recorded CDR OS/events; patient splits; censored objectives; reference-checked metrics; independent checkpoint/prediction reconstruction. | Clinical/biological sample timing must match the intended prediction moment. Prior overlapping-patient use limits internal holdout claims. |",
        "| 2. Patients and events | BRCA CLI/RNA 1080/151; matched METABRIC external 1960/1138. Separate KIRC clinical/RNA 217/66 downloaded and verified. | KIRC metadata overlap is not decoded imaging availability: actual four-input cohort is currently one patient, zero deaths. |",
        "| 3. Baselines | Age Cox, unimodal, concat, mean and attention completed on consistent matched patients. | WSI/CT baselines and faithful compatible published-method comparisons. |",
        "| 4. Ablations | Separately trained two-input subsets, attention removal, masking/dropout and controlled withholding completed; paired bootstrap audit. | Separately trained three/four-input models and uncertainty-component ablations on adequate real cohorts. |",
        "| 5. Uncertainty | Censoring-adjusted disagreement/error and selective-prediction analyses executed. | Reliable confidence is not demonstrated. Legacy EDL objective is not a validated survival uncertainty solution. |",
        "| 6. Novelty | Current prior art and a falsifiable missing-input hypothesis reviewed. | Dropout/attention benefit is unsupported by present experiments. A demonstrable contribution remains required. |",
        "| 7. Preprocessing/reproducibility | Train-only genes/scalers; hashes; explicit image identity/geometry; physical WSI pilot; CT HU windows; checkpoint audits and notebook parity. | Full imaging QC/protocol/timing; clean Kaggle reproduction; independent external four-input validation. |",
        "",
        "## Completed BRCA clinical/RNA experiment",
        "",
        "The matched cohort has 1080 patients and 151 observed OS deaths. Train: 648/91 deaths; validation: 216/30; test: 216/30. Clinical input is age only; RNA uses 256 variance-selected genes fitted on training patients. Neural entries are three-seed ensembles (42, 123, 2026). Selection/early stopping uses validation. Intervals use 1000 patient bootstrap resamples.",
        "",
        "**These are survival C-indices, not classification accuracy, precision, recall or F1.** No cancer-detection task or fabricated binary target is evaluated. One-year horizon metrics are not estimable in this test split because no observed death occurs by one year.",
        "",
        "| Model | Test C-index | 95% bootstrap CI | 3-year IPCW AUC | 5-year IPCW AUC |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for name, result in metrics.items():
        score = result["test"]
        lo, hi = score["bootstrap_95_ci"]
        horizon = {int(row["horizon"]): row for row in score["horizons"]}
        lines.append(
            f"| {labels[name]} | {score['harrell_c']:.4f} | {lo:.4f}–{hi:.4f} | {horizon[1095]['ipcw_auc']:.4f} | {horizon[1825]['ipcw_auc']:.4f} |"
        )
    lines += [
        "",
        "Concatenation has the highest observed test C-index among these configurations. Its difference versus matched age Cox is +0.1063, paired 95% CI −0.0605 to +0.2654. This does not establish superiority. Attention is lower than concatenation. Aggregate missing-input dropout improvements also have intervals crossing zero. These comparisons are exploratory and unadjusted for multiplicity.",
        "",
        "Attention's five-year disagreement/error correlation is approximately −0.0304; lower disagreement does not reliably select lower-error predictions. Do not label disagreement as calibrated confidence or a validated clinical uncertainty estimate.",
        "",
        f"Evidence: [{main_run.name}](../{main_run.as_posix()}/metrics.json), [independent reconstruction](../{main_run.as_posix()}/independent_prediction_reconstruction.json), [paired robustness audit](../{main_run.as_posix()}/independent_robustness_audit.json). The earlier v2 metrics/runtime/manifest were zero-filled and are preserved as invalid artifacts; v3 is a fresh completed run.",
        "",
        "## Separate shared-rank METABRIC transfer experiment",
        "",
        "This is a separately trained representation/model. It harmonizes shared gene symbols, averages duplicated symbols within each patient, converts expression to within-patient ranks, and fits feature selection/scaling/model parameters on TCGA only. It is **not** external application of the preceding logTPM model. Matched external set: 1960 patients, 1138 OS deaths. Evaluation censoring estimates use external outcomes only as a nuisance reference; prediction parameters remain frozen.",
        "",
        "| Model | External C-index | 95% bootstrap CI |",
        "| --- | ---: | --- |",
    ]
    for name, result in external.items():
        score = result["external"]
        lo, hi = score["bootstrap_95_ci"]
        lines.append(
            f"| {labels[name]} | {score['harrell_c']:.4f} | {lo:.4f}–{hi:.4f} |"
        )
    lines += [
        "",
        "External concatenation minus age Cox is +0.03794 (paired 95% CI +0.01601 to +0.05859). This is an exploratory, unadjusted comparison and does not demonstrate attention superiority, prospective clinical utility, calibration or four-input benefit. The transport run omitted its original code hash; independent numerical reconstruction passes, and this provenance limitation remains disclosed. Future runs record code/configuration hashes.",
        "",
        f"Evidence: [transport metrics](../{transfer.as_posix()}/metrics.json), [independent transport reconstruction](../{transfer.as_posix()}/independent_transport_reconstruction.json).",
        "",
        "## Separate KIRC clinical/RNA experiment",
        "",
        "This retrospectively locked five-fold study covers 217 TCGA-KIRC patients and 66 recorded OS deaths. Each patient appears in one outer test fold; folds contain 43–44 patients and 13–14 deaths. Clinical inputs are age and prespecified pathologic stage, which is retrospective. RNA procurement timing is unknown. Center receipt dates are not used to exclude patients or shift outcome time. No diagnosis-time prediction claim is made.",
        "",
        "The primary summary is the equal-weight mean of the five outer-fold C-indices. Confidence intervals are conditional patient-bootstrap intervals on saved outer-fold predictions and do not include model refitting variation.",
        "",
        "| Model | Mean outer-fold C | Paired delta vs age Cox, 95% CI |",
        "| --- | ---: | --- |",
    ]
    kirc_labels = {
        "clinical_cox": "Age Cox",
        "unimodal_cli": "Clinical neural",
        "unimodal_rna": "RNA neural",
        "fusion_concat": "Clinical + RNA concatenation",
        "fusion_mean": "Clinical + RNA mean",
        "fusion_attention": "Clinical + RNA attention",
        "attention_no_mask": "Attention without masking",
        "attention_dropout_01": "Attention dropout 0.1",
        "attention_dropout_03": "Attention dropout 0.3",
        "attention_dropout_05": "Attention dropout 0.5",
        "without_cli": "Retrained without clinical",
        "without_rna": "Retrained without RNA",
    }
    for name, score in kirc_audit["primary_equal_fold_summary"].items():
        lo, hi = score["conditional_paired_delta_95_ci"]
        lines.append(
            f"| {kirc_labels[name]} | {score['mean_outer_fold_c']:.4f} | {score['mean_fold_delta_vs_clinical_cox']:+.4f} [{lo:+.4f}, {hi:+.4f}] |"
        )
    lines += [
        "",
        "Concatenation mean fold C=0.7077 versus age Cox 0.6494; paired ΔC +0.0582, 95% CI [−0.0316, +0.1478]. This does not establish improvement. Attention benefit is uncertain; the two-input dropout comparison also crosses zero. Uncertainty analyses do not show reliable confidence. This experiment used no image inputs.",
        "",
        f"Evidence: [{kirc_run.name}](../RUN%20REPORT/{kirc_run.name}/five_fold_summary.json), [independent reconstruction](../RUN%20REPORT/{kirc_run.name}/independent_five_fold_summary.json), [locked protocol](../config/kirc_retrospective_os_protocol.json).",
        "",
        "## Separate KIRC four-input study",
        "",
        "TCGA-KIRC remains separate from BRCA and METABRIC. The metadata candidate cohort has 217 patients and 66 OS deaths. All selected RNA files are downloaded/MD5-verified and form a 217 × 19962 finite gene matrix. One incorrectly selected additional-primary pathology sample was replaced with an eligible primary-tumour slide.",
        "",
        "| Modality availability | Actual processed patients |",
        "| --- | ---: |",
    ]
    for modality, count in availability["actual_feature_patients"].items():
        lines.append(f"| {modality.upper()} | {count} |")
    lines += [
        f"| All four, patient-matched | {availability['actual_all_four_patients']} |",
        "",
        f"The complete-input pilot TCGA-B8-4148 has zero observed deaths. Its 32 × 768 pathology bag and 32 × 512 CT bag derive from real source images, verified identity, direct decoding and recorded preprocessing. It is engineering feasibility only, excluded from efficacy claims. Full selected WSI raw acquisition is approximately 223.5 GB. Current extraction journal: status `{wsi_audit.get('status', 'not_started')}`, {wsi_audit.get('valid_patients', 1)} verified WSI bags, {wsi_audit.get('failed_patients', 0)} failed cases; active case {wsi_audit.get('active_patient', 'none')} at `{wsi_audit.get('active_stage', 'none')}`. A rejected source MD5/size mismatch remains failed and is not a feature input.",
        "",
        "GDC `days_to_collection` describes processing-centre receipt, not removal of tissue from the patient. Do not exclude patients or construct landmarks from that receipt field. RNA procurement timing remains unknown; selected WSI procurement is recorded at diagnosis. TCIA dates are deidentified; source-specific timing review is still required. No prospective prediction claim is supported.",
        "",
        "RCC-AID v2 supplies a verified annotated/phase-labelled subset, but only 58 of our candidate patients (9 deaths) overlap; the venous subset has 22 patients/4 deaths. This subset supports imaging QC rather than resolving event scarcity. Masks belong to exact series IDs and cannot be moved across contrast phases without registration/verification.",
        "",
        "Evidence: [actual availability](../data/processed/kirc_actual_input_availability.json), [same-case independent audit](../RUN%20REPORT/kirc-paired-pilot/TCGA-B8-4148/independent_matched_pilot_audit.json), [KIRC source/protocol audit](../RUN%20REPORT/kirc-protocol-audit-20261118.json), [acquisition evidence](KIRC_DATA_ACQUISITION_EVIDENCE.md).",
        "",
        "## Figures, code and execution",
        "",
        f"- [BRCA scientific figures](../{main_run.as_posix()}/paper_figures): 15 PNG/PDF pairs and figure provenance.",
        f"- [External transfer figures](../{transfer.as_posix()}/paper_figures): 6 PNG/PDF pairs and provenance.",
        "- [Validated study notebook](../notebooks/CancerFusion_Validated_Study.ipynb): corrected survival workflows; KIRC execution defaults off pending protocol admission. Images are not bundled.",
        "- [Kaggle package v5](../artifacts/kaggle-validated-study-20261118-v5): input/source hashes, archive checks, cell compilation and extracted-code tests pass. No new Kaggle submission or completed Kaggle result is claimed.",
        "- [Full coordinator response](MENTOR_POINT_BY_POINT_RESPONSE.md) and [research comparison](RESEARCH_COMPARISON_AND_VALIDATION.md) retain study limitations and unresolved evidence gates.",
        "",
        "Historical results must remain separated: the earlier 100% scores used generated targets, and the old 0.8425 C-index used a follow-up predictor. Neither is valid paper performance. The older 1083-patient clinical-only split is also different from this matched 1080-patient experiment; its Cox score must not be used as this table's comparator.",
        "",
        "## Report inputs",
        "",
        "| Artifact | SHA-256 |",
        "| --- | --- |",
    ]
    for source in [
        main_run / "metrics.json",
        main_run / "run_manifest.json",
        main_run / "independent_prediction_reconstruction.json",
        transfer / "metrics.json",
        transfer / "independent_transport_reconstruction.json",
        availability_path,
        *([wsi_audit_path] if wsi_audit_path.exists() else []),
        kirc_run / "five_fold_summary.json",
        kirc_run / "independent_five_fold_summary.json",
    ]:
        lines.append(f"| {source.as_posix()} | `{sha256(source)}` |")
    document = ("\n".join(lines) + "\n").replace(
        "](../RUN REPORT/", "](../RUN%20REPORT/"
    )
    payload = document.encode("utf-8")
    publish(Path("docs/CURRENT_VERIFIED_STATUS.md"), payload)
    print("Built docs/CURRENT_VERIFIED_STATUS.md from independently audited artifacts.")


if __name__ == "__main__":
    main()
