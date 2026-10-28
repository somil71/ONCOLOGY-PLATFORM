# CancerFusion AI: Multimodal Precision Oncology Platform
**Phase 1 Technical Documentation & Implementation Guide**

---

## 1. Abstract
In modern oncology, clinical decision-making relies on diverse patient data, including radiological scans, histopathology slides, genomic sequencing, and clinical history. However, current AI diagnostic systems largely analyze these modalities in isolation, missing critical cross-modal correlations. Furthermore, real-world clinical datasets suffer from Missing Not At Random (MNAR) data, causing standard multimodal algorithms to fail when a patient lacks a specific test. 

**CancerFusion AI** is an advanced academic research prototype that solves these limitations. It introduces a modular, attention-based fusion architecture that dynamically aggregates Gigapixel Whole Slide Images (WSI), RNA-Seq profiles, and tabular clinical data. The system utilizes dynamic attention-masking to maintain predictive integrity even when modalities are missing, and employs Evidential Deep Learning (EDL) to quantify diagnostic uncertainty.

---

## 2. System Architecture

The pipeline processes raw data through four distinct stages:

### A. Unimodal Encoders
* **Histopathology (WSI):** Raw tissue patches are passed through a pre-trained **Vision Transformer (ViT-B/16)** to extract 768-dimensional global feature vectors. These are condensed via Multiple Instance Learning (MIL) into a 16-dimensional concept bottleneck.
* **Genomics (RNA-Seq):** Deep Multi-Layer Perceptrons utilizing `AlphaDropout` and `SELU` (Scaled Exponential Linear Unit) activations, optimized for self-normalization on high-dimensional genomic arrays.
* **Clinical Records:** Standard `BatchNorm1d` and `ReLU` networks engineered to process structured demographic and diagnostic histories.

### B. Safe Attention Fusion Engine
Instead of rigid feature concatenation, the model employs a learned Cross-Attention Transformer layer. 
* **Dynamic Masking:** If a modality is absent (e.g., no genomic data available), the network applies a negative-infinity (`-inf`) mask prior to the softmax activation. This assigns a mathematical weight of `0.0` to the missing data, forcing the network to dynamically re-weight the remaining available modalities.

### C. Prognostic Prediction Model
* **Cox Proportional Hazards:** The network is optimized using a Cox Negative Log-Likelihood loss function, outputting a time-to-event survival risk score (Hazard Ratio) rather than a simple binary classification.

### D. Evidential Deep Learning (EDL) Head
* Instead of standard probability outputs, the final layer uses a Softplus-clamped Dirichlet distribution. This allows the model to output a distinct **Uncertainty Bound** alongside its prediction, alerting clinicians to out-of-distribution or highly ambiguous data.

---

## 3. Technology Stack

| Component | Technology / Library | Description |
| :--- | :--- | :--- |
| **Core Framework** | `PyTorch` (CUDA 12.1) | Deep learning engine with TF32 & cuDNN acceleration. |
| **Vision Backbone** | `Torchvision` (ViT-B/16) | State-of-the-art transformer for medical image extraction. |
| **Data Processing** | `Pandas`, `NumPy`, `PIL` | High-efficiency dataset parsing and image manipulation. |
| **Survival Analysis** | `Lifelines` | Generation of Kaplan-Meier baselines and statistical validation. |
| **Visualization** | `Seaborn`, `Matplotlib` | Generation of publication-ready XAI charts and survival curves. |
| **API Export** | Python `json` | Serializes model weights and explainability metrics for backend systems. |

---

## 4. Key Research Novelties (Gaps Addressed)

1. **The MNAR Problem:** Successfully handles "Missing Not At Random" data without requiring hallucinated data imputation.
2. **The "Black Box" Problem:** Implements automated Modality Importance mapping, demonstrating precisely which data type (WSI, RNA, Clinical) drove the prognosis.
3. **Overconfidence:** Calculates Epistemic Uncertainty via Dirichlet parameters, establishing clinical trust.

---

## 5. Execution & Deployment Guide

This implementation is optimized for the **Kaggle Cloud Environment** (Dual T4 GPU Accelerator).

### Prerequisites
1. Upload the `CancerFusion_Advanced_Kaggle.ipynb` to a new Kaggle session.
2. Mount the dataset: `breast-cancer-vision-and-genomic-fusion-ml-ready` to the `/kaggle/input/` directory.
3. Ensure the Kaggle session is set to **GPU T4 x2**.

### Configuration Parameters
The `CFG` dictionary at the top of the notebook controls the runtime:
```python
CFG = {
    'batch_size': 32,
    'epochs': 50,
    'lr': 1e-3,
    'max_patches': 100,  # Max WSI patches processed per patient
    'num_workers': 0,    # MUST REMAIN 0 to prevent CUDA Multiprocessing forks
}
```

### Execution Steps
1. Click **Run All**.
2. **Cell 1** will automatically install `lifelines` and configure the CUDA environment.
3. **Cell 4** will dynamically parse the datasets and align the `Patient_IDs`.
4. **Cell 9** will execute the 50-epoch training loop, providing live Cox Loss metrics via `tqdm`.
5. **Cell 10** will generate the three publication-ready charts (Kaplan-Meier, Attention Weights, Uncertainty) and save `backend_response.json` and `CancerFusion_Weights.pth` to the `/kaggle/working/` directory.

---
*Documentation Prepared for Phase 1 B.Tech Major Project Submission.*
