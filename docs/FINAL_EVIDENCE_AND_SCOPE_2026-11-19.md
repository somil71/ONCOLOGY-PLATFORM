# CancerFusion AI — evidence, remaining work, and paper scope

**Status date:** 19 November 2026
**Authority:** saved research outputs, source-level audits, and independent reconstruction. This report supersedes older cohort and platform summaries where they differ. Documentation text alone is not treated as proof.

## Executive status

CancerFusion AI currently has a four-input **research prototype**, plus independently reconstructed clinical/RNA survival studies. It does **not** have validated four-input survival performance, calibrated confidence, prospective clinical validation, or an image-based diagnosis result. The paper should report the clinical/RNA experiments and describe the image work as engineering feasibility only.

The original pilot's perfect classification results used outcomes generated from the model's own risk score. They are invalid and withdrawn. The old `run_25_review_only` confusion matrix, ROC, precision–recall, Kaplan–Meier, and reliability plots are not clinical evidence and must not be used as final performance figures.

## Measured survival results

| Cohort and design | Model | Patients / OS deaths | Survival C-index (95% conditional interval) | Reading |
| --- | --- | ---: | ---: | --- |
| TCGA-KIRC, locked five-fold, clinical + RNA | Age-only Cox | 217 / 66 | 0.6494 (0.5784–0.7204) | Reference |
| TCGA-KIRC, same folds | RNA-only neural | 217 / 66 | 0.6865 (0.6091–0.7566) | Paired gain over Cox not established |
| TCGA-KIRC, same folds | Age + RNA concatenation | 217 / 66 | 0.7077 (0.6315–0.7804) | Highest point estimate; ΔC +0.0582, paired 95% CI −0.0316 to +0.1478 |
| TCGA-KIRC, same folds | Age + RNA attention | 217 / 66 | 0.7029 (0.6260–0.7683) | Below concatenation by point estimate; no established gain over Cox |
| TCGA-KIRC, same folds | Adapted author-source MultiSurv mRNA model | 217 / 66 | 0.6792 (0.6042–0.7438) | Secondary adapted RNA benchmark, not full paper reproduction |
| TCGA-BRCA, held-out clinical/RNA | Age-only Cox | 216 / 30 | 0.5360 (0.3878–0.6834) | Matched reference |
| TCGA-BRCA, same held-out patients | Age + RNA concatenation | 216 / 30 | 0.6423 (0.5298–0.7489) | Paired gain over Cox not established |
| TCGA-BRCA, same held-out patients | Age + RNA attention | 216 / 30 | 0.5983 (0.4793–0.7161) | Below concatenation by point estimate |

These are survival ranking metrics, **not** accuracy, precision, recall, F1, or cancer-detection rates. The confidence intervals condition on fitted models and omit model-refitting and model-selection variability. Do not merge the BRCA and KIRC values into a single performance claim.

Public aggregate evidence: [survival results](evidence/aggregate_survival_results.json). Detailed local run exports remain in the working copy and are excluded from Git because they contain patient-level predictions. See the [published-method comparison](RESEARCH_COMPARISON_AND_VALIDATION.md) for methods and limitations.

## Four-input image evidence and acquisition

The rebuilt KIRC image feature preview contains 217 patients / 66 recorded deaths overall, with **43 WSI feature bags and 70 radiology CT feature bags**. An independent source/identity/valid-item-mean audit passed; strict complete-input availability is **27 patients / 7 deaths**. Seven events remain too sparse for a credible four-input survival evaluation. No four-input model was fit.

The 19 November source audit now reopens the actual selected diagnostic SVS files and deterministically replays patch selection:

- **37 of the originally audited WSI bags:** original SVS byte count and GDC MD5 checked; all **4,736 selected patches** re-extracted at the recorded coordinates; coordinates and tissue fractions match the sidecars exactly. The additional six feature bags passed the independent cohort reconstruction; the expanded 43-bag source replay is pending.
- **70 of 70 CT bags:** 5,081 selected source DICOM files rehashed; CT series identity, slice selection indices, and physical-position sidecars reconstructed.
- CT headers were not independently reparsed by this additional audit. Existing acquisition QC reports DICOM identity, geometry, pixel decoding, and HU conversion, but clinical tumour coverage and acquisition timing remain unreviewed.
- The WSI replay verifies source identity and patch selection/pixel QC, not ViT embedding replay, tumour annotation, stain normalization, or clinical-time eligibility.

Public aggregate evidence: [imaging availability and replay counts](evidence/aggregate_imaging_audit.json). Detailed local audit files contain patient-level identifiers and remain excluded from Git.

The outcome-blind CT metadata selection has 171 patients / 50 recorded deaths, while the current CT feature cohort has 70 patients / 15 deaths. A bounded attempt to acquire additional outcome-blind WSI slides stalled on a public slide download and was stopped; it did not complete the remaining matched acquisitions. The resulting local rebuild increased WSI features from 37 to 43 and complete-input availability from 22 to 27 patients, still with 7 deaths. This is a technical input-count update only, not a four-input model result.

## Fusion benefit and novelty

The four-input primary hypothesis in the locked protocol is that training-time modality dropout plus explicit masking improves missing-input robustness over the otherwise matched no-dropout attention model across all 15 nonempty four-input subsets, without harming complete-input performance. **It has not been tested.**

The available clinical/RNA comparisons do not establish a fusion advantage: paired intervals against the matched Cox reference include no difference. The smaller two-input missingness/dropout comparison also does not establish a benefit. Attention and modality masking are prior methods, so their implementation alone is not a novelty claim.

**Paper scope decision:** report the verified clinical/RNA evidence as the quantitative study; describe pathology/CT integration as a prototype and data-QC feasibility result. Do not claim four-input efficacy or novelty unless a suitably powered, independently evaluated image cohort is later available and the locked subset protocol is completed.

## Uncertainty and calibration

The current ensemble standard deviation is **between-seed disagreement**, not calibrated confidence. Its error association was weak and inconclusive. At five years, KIRC attention disagreement/error correlation was 0.0321 (conditional interval −0.1401 to 0.2247). With RNA withheld from the KIRC concatenation model, Brier error worsened by 0.0912 (paired interval 0.0460 to 0.1340) while disagreement decreased by 0.0136 (−0.0166 to −0.0108). Lower disagreement can accompany worse predictions.

No independent calibration mapping for a probability of prediction error has been validated. The app and paper must call the number an **unvalidated raw research score** or **ensemble disagreement**, never “confidence,” a confidence percentage, or a clinical uncertainty estimate. A future validation requires a calibration partition used only to fit the mapping and a separate untouched patient-level test partition, with censoring-aware error targets and calibration/error-ranking intervals.

Public aggregate evidence: [uncertainty findings](evidence/aggregate_uncertainty_findings.json). Detailed local uncertainty reports and patient-level predictions remain excluded from Git.

## Kaggle run and artifacts

The latest Kaggle command visible in the supplied log starts the clinical survival runner on TCGA-BRCA with METABRIC external evaluation. A completed metrics export, log tail, run manifest, and checkpoint from that specific run are **not present in this workspace snapshot**, so its completion and checkpoint cannot be verified here. Kaggle CLI inspection is unavailable in this environment: the executable is blocked by Windows Application Control, and no Kaggle API credentials are configured. Do not cite that run as final until its Kaggle output page is inspected and its downloaded artifacts are reconstructed against the input files and protocol.

The previously completed local KIRC and BRCA reports listed above remain the current verified performance evidence. They are not substitutes for the missing Kaggle run export.

## Platform language and boundaries

The React site and local API are a research demonstration around the Phase 2 four-input checkpoint. Workspace cases may be synthetic or contain user-supplied feature arrays. A checkpoint's scores, evidence score, and attention weights do not predict clinical outcomes for a real patient. The platform must state that it is not for diagnosis or care, label the score as unvalidated, and separate the four-input architecture diagram from the clinical/RNA results.

The platform's research figures should use source-linked clinical/RNA and uncertainty outputs. Generated-label confusion, ROC, PR, Kaplan–Meier, and calibration figures are excluded from the main research gallery. Historical files are retained for audit only.

## Action ledger

| Request | Status | Evidence / next gate |
| --- | --- | --- |
| Finish and independently verify Kaggle run; export metrics/log/checkpoint | **Not verifiable in this session** | Need accessible Kaggle run output and export; local project has no completed artifact for the latest command |
| Expand matched, QC-passed imaging cohort | **Partially completed; acquisition stalled** | Available WSI bags increased 37→43 and the independently reconstructed complete intersection increased 22→27 patients (7 deaths). Additional public GDC download stalled; no four-input model fit. |
| Run 15 four-input comparisons or limit paper to clinical/RNA | **Paper scope limited to clinical/RNA** | No four-input model was fit; complete-input n=27 / 7 deaths makes the locked fold evaluation unfit for the intended claim |
| Validate uncertainty with independent calibration and test data | **Not complete** | Disagreement tested on held-out outcomes, but calibrated confidence/error probability not validated; preserve explicit no-confidence language |
| Revise paper/platform claims | **Revised locally** | This scope report, React evidence page, and figure API point to verified clinical/RNA evidence and exclude generated-label plots |
