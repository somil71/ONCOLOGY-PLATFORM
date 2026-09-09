import nbformat as nbf

# Load the existing notebook
notebook_path = 'e:/MAJOR PROJECT/CancerFusion_Advanced_Kaggle.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# --- CELL 7: Markdown for Visualizations ---
nb.cells.append(nbf.v4.new_markdown_cell("""
---
# 📊 Research Paper Visualizations
The following cell generates publication-quality graphs required for medical AI journals (Nature Medicine, IEEE).
It plots Kaplan-Meier survival curves, Attention Distributions, and Evidential Uncertainty bounds.
"""))

# --- CELL 8: Advanced Visualizations Code ---
nb.cells.append(nbf.v4.new_code_cell("""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure lifelines is installed for KM curves
try:
    from lifelines import KaplanMeierFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("Run !pip install lifelines for Kaplan-Meier curves.")

def generate_research_plots(risk_scores, times, events, uncertainties, attention_weights):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig = plt.figure(figsize=(18, 5))
    
    # 1. Kaplan-Meier Survival Analysis
    ax1 = plt.subplot(1, 3, 1)
    if HAS_LIFELINES:
        kmf = KaplanMeierFitter()
        # Split into High Risk vs Low Risk based on median
        median_risk = np.median(risk_scores)
        high_risk = risk_scores > median_risk
        
        kmf.fit(times[~high_risk], event_observed=events[~high_risk], label="Low Risk (AI)")
        kmf.plot_survival_function(ax=ax1, color='blue', linewidth=2)
        
        kmf.fit(times[high_risk], event_observed=events[high_risk], label="High Risk (AI)")
        kmf.plot_survival_function(ax=ax1, color='red', linewidth=2)
        
        ax1.set_title("Kaplan-Meier Survival Curves", fontweight='bold')
        ax1.set_xlabel("Time (Months)")
        ax1.set_ylabel("Survival Probability")
    else:
        ax1.text(0.5, 0.5, "Lifelines not installed", ha='center', va='center')
        
    # 2. Modality Attention Distribution
    ax2 = plt.subplot(1, 3, 2)
    avg_attention = np.mean(attention_weights, axis=0)
    sns.barplot(x=['Histopathology\\n(WSI)', 'Genomics\\n(RNA)', 'Clinical'], 
                y=avg_attention, palette="viridis", ax=ax2)
    ax2.set_title("Average Modality Importance (Attention)", fontweight='bold')
    ax2.set_ylabel("Attention Weight")
    
    # 3. Uncertainty Correlation (Evidential Deep Learning)
    ax3 = plt.subplot(1, 3, 3)
    sns.regplot(x=risk_scores, y=uncertainties, scatter_kws={'alpha':0.5, 'color':'purple'}, 
                line_kws={'color':'red'}, ax=ax3)
    ax3.set_title("Prediction vs. Uncertainty Bounds", fontweight='bold')
    ax3.set_xlabel("Predicted Risk Score")
    ax3.set_ylabel("Evidential Uncertainty")
    
    plt.tight_layout()
    plt.savefig("CancerFusion_Research_Plots.png", dpi=300, bbox_inches='tight')
    plt.show()

# Mock data to demonstrate the graphs before full training
print("[*] Generating Sample Publication Graphs...")
generate_research_plots(
    risk_scores=np.random.beta(2, 5, 200),
    times=np.random.exponential(50, 200),
    events=np.random.binomial(1, 0.7, 200),
    uncertainties=np.random.uniform(0.1, 0.8, 200),
    attention_weights=np.random.dirichlet((5, 3, 2), 200)
)
"""))

# --- CELL 9: Markdown for Backend ---
nb.cells.append(nbf.v4.new_markdown_cell("""
---
# ⚙️ Backend API Export
This cell saves the trained PyTorch weights and generates a standard JSON response. 
Your FastAPI backend (Pillar 3) will read this JSON to render the React dashboard.
"""))

# --- CELL 10: Backend Export Code ---
nb.cells.append(nbf.v4.new_code_cell("""
import json

def export_for_backend(model, sample_patient_id="TCGA-BRCA-9999"):
    # 1. Save Weights
    weight_path = "CancerFusion_Hybrid_Weights.pth"
    torch.save(model.state_dict(), weight_path)
    print(f"✅ Saved Production Weights to: {weight_path}")
    
    # 2. Generate Sample JSON for FastAPI
    # In production, this would be generated dynamically during inference.
    api_payload = {
        "patient_id": sample_patient_id,
        "prediction": {
            "risk_score_normalized": 0.842,
            "evidential_uncertainty": 0.125,
            "clinical_status": "HIGH RISK"
        },
        "explainability": {
            "attention_weights": {
                "histopathology": 0.58,
                "genomics": 0.32,
                "clinical": 0.10
            },
            "wsi_concepts_detected": [
                "High Cellularity",
                "Necrosis",
                "Lymphocyte Infiltration"
            ]
        },
        "missing_data_handled": {
            "was_genomics_missing": False,
            "imputation_method": "-inf Attention Masking"
        }
    }
    
    json_path = "backend_api_response.json"
    with open(json_path, 'w') as f:
        json.dump(api_payload, f, indent=4)
        
    print(f"✅ Generated API Payload for Frontend Dashboard: {json_path}")
    return api_payload

# Run export
payload = export_for_backend(model)
print(json.dumps(payload, indent=2))
"""))

# Save the appended notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
