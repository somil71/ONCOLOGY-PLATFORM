import torch
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import numpy as np
import uvicorn
from typing import List, Optional

# Import the architecture we defined earlier
from CancerFusion_Advanced_Architecture import CancerFusionPlatform

app = FastAPI(title="CancerFusion AI API", description="Backend Inference Server for Multimodal Precision Oncology")

# Load model (Mocking the load for now)
DEVICE = 'cpu'
model = CancerFusionPlatform().to(DEVICE)
model.eval()

# Concept labels based on our Concept Bottleneck
CONCEPT_LABELS = [
    "Cellularity", "Necrosis", "Lymphocyte Infiltration", "Stromal Density", 
    "Nuclear Atypia", "Mitotic Count", "Vascular Invasion", "Fibrosis"
]

class ClinicalData(BaseModel):
    age: float
    stage: int
    biomarker_1: float
    biomarker_2: float
    # ... other variables

@app.post("/predict")
async def predict_survival(
    clinical_data: str = Form(...), # JSON string of clinical features
    genomics_file: Optional[UploadFile] = File(None),
    wsi_embeddings: Optional[UploadFile] = File(None)
):
    """
    Inference endpoint. Handles missing modalities dynamically.
    """
    # 1. Process Clinical Data (Always expected)
    # Mocking clinical tensor (1 patient, 15 features)
    cli_tensor = torch.randn(1, 15) 
    
    # 2. Process Genomics
    if genomics_file:
        # e.g., np.load(genomics_file.file)
        rna_tensor = torch.randn(1, 1000)
        rna_present = 1.0
    else:
        rna_tensor = torch.zeros(1, 1000)
        rna_present = 0.0
        
    # 3. Process WSI Embeddings
    if wsi_embeddings:
        # e.g., np.load(wsi_embeddings.file)
        wsi_tensor = torch.randn(1, 100, 768) # 100 patches
        wsi_present = 1.0
    else:
        wsi_tensor = torch.zeros(1, 100, 768)
        wsi_present = 0.0
        
    # Missing Mask
    missing_mask = torch.tensor([[wsi_present, rna_present, 1.0]], dtype=torch.float32)

    # 4. Model Inference
    with torch.no_grad():
        risk_score, concepts, alphas = model(wsi_tensor, rna_tensor, cli_tensor, missing_mask)
        
    # 5. Format Output
    concept_scores = concepts[0].numpy().tolist()
    
    # Map the first 8 concepts to human-readable biological traits
    extracted_concepts = {
        label: round(score, 3) 
        for label, score in zip(CONCEPT_LABELS, concept_scores[:8])
    }
    
    # Extract Evidential Uncertainty (Higher Alpha = More Certainty)
    uncertainty_mass = {
        "WSI_Evidence": round(float(alphas[0][0]), 3) if wsi_present else 0.0,
        "Genomics_Evidence": round(float(alphas[0][1]), 3) if rna_present else 0.0,
        "Clinical_Evidence": round(float(alphas[0][2]), 3),
        "Total_Joint_Certainty": round(float(torch.sum(alphas[0])), 3)
    }

    return {
        "status": "success",
        "predicted_risk_score": round(float(risk_score[0]), 4),
        "biological_concepts": extracted_concepts,
        "evidential_uncertainty": uncertainty_mass,
        "missing_modalities": {
            "WSI": bool(not wsi_present),
            "Genomics": bool(not rna_present)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
