from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import json

app = FastAPI(title="NeuroSense AI Clinical API", version="1.0.0")

# Allow React Frontend to communicate with this Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Placeholder for our PyTorch Model
class NeuroSenseInferenceEngine:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_loaded = False
        print(f"[*] AI Engine Initializing on {self.device.upper()}...")
        # TODO: Load the .pt weights from ../model_weights/ here

    def predict(self, mri_bytes=None, speech_data=None):
        # TODO: Run the exact PyTorch architecture and return probabilities
        return {"status": "Success", "prediction": "Alzheimer's (Simulated)", "confidence": 0.94}

engine = NeuroSenseInferenceEngine()

@app.get("/")
def read_root():
    return {"message": "NeuroSense AI Backend is Online."}

@app.post("/api/analyze")
async def analyze_patient(
    mri_file: UploadFile = File(None),
    speech_csv: UploadFile = File(None)
):
    """
    Endpoint for the React frontend to send patient data.
    """
    print(f"Received API Request...")
    if mri_file: print(f" - MRI Uploaded: {mri_file.filename}")
    if speech_csv: print(f" - Speech Uploaded: {speech_csv.filename}")
    
    # Run Inference
    results = engine.predict()
    
    return results

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
