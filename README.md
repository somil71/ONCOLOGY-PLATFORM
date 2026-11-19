> **19 November 2026 — current evidence and public scope:** Read [the current evidence and paper-scope report](docs/FINAL_EVIDENCE_AND_SCOPE_2026-11-19.md). The four-input checkpoint is a research prototype; the verified survival results use clinical and RNA inputs only. No four-input survival result, calibrated confidence or clinical validation is established.

# CancerFusion AI

CancerFusion AI is a multimodal oncology research platform prototype. The Phase 2 checkpoint accepts pathology, radiology, omics and clinical feature arrays and supports masked-input software demonstrations. Its raw scores and pooling weights are not validated clinical predictions. The completed quantitative survival studies currently cover clinical and RNA inputs only; see the evidence report for cohorts, results, limitations and the withdrawn generated-label pilot scores.

This B.Tech Computer Science and Engineering major project (PT079) is developed at IILM University, Greater Noida, under the supervision of Dr. Gunjan Mittal Roy.

## Run the React site

From the repository root, install the Python dependencies and build the frontend. Use Python 3.10 or newer and a Node.js version supported by the Vite package in `web/`.

```powershell
python -m pip install -r requirements.txt
npm --prefix web ci
npm --prefix web run build
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open [the local site](http://127.0.0.1:8000/). For frontend development, keep the API running and use `npm --prefix web run dev`. The Vite server forwards API requests to port 8000.

The checkpoint is not included in the public source package. Set `CANCERFUSION_CHECKPOINT` to a locally supplied checkpoint path before using the case workspace. Saved research cases and comparison records use `runtime/cases.sqlite3` by default. Use synthetic or de-identified research features. The site is a research demonstration, not for diagnosis or treatment. The legacy Streamlit entry point is `app.py`.

## What the workspace demonstrates

1. Create a case from synthetic demonstration inputs or a preprocessed feature JSON that follows the model input contract.
2. Select available modalities and run the Phase 2 research checkpoint.
3. Compare the all-supplied-input result with the selected-input result, including each modality's pooling weight.
4. Reopen the saved run and download a report with its model and input provenance.

The checkpoint expects preprocessed feature arrays with the exact trained feature order. Verify every feature contract before uploading. The relative-risk and evidence scores have no validated survival horizon, calibrated probability, or patient-care interpretation. A single-case input-masking comparison is not a cohort-level performance evaluation.

## Research evidence

The original 122-case Kaggle pilot generated validation labels from model scores; its perfect accuracy and concordance are withdrawn. Generated-label confusion, ROC, PR, Kaplan–Meier and reliability plots are archived, not valid performance evidence.

Current KIRC and BRCA survival metrics, intervals, source audits and remaining experimental gates are documented in [the current evidence report](docs/FINAL_EVIDENCE_AND_SCOPE_2026-11-19.md). The latest Kaggle run does not have a verified local result export; do not cite it as complete until its metrics, log, manifest and checkpoint have been downloaded and independently reconstructed.

See the [current evidence and paper scope](docs/FINAL_EVIDENCE_AND_SCOPE_2026-11-19.md). Older handoff drafts and generated-label outputs are retained locally for audit; they are not current performance evidence.

## Main files

| File or directory | Purpose |
| --- | --- |
| `web/` | React website and case workspace |
| `src/api.py` | Local Python API and built-site serving |
| `src/architecture.py` | PyTorch model |
| `src/case_inference.py` and `src/case_store.py` | Feature validation, checkpoint inference, and persisted comparisons |
| `CancerFusion_Phase2_PanCancer.ipynb` | Kaggle data audit, training, and evaluation |
| `docs/FINAL_EVIDENCE_AND_SCOPE_2026-11-19.md` | Current measurements, imaging audit, scope, and remaining work |

Patient-level run exports and source imaging are excluded from the public Git repository. The evidence report links to sanitized aggregate results; reproduce the studies using the acquisition and analysis tools with the recorded manifests.
