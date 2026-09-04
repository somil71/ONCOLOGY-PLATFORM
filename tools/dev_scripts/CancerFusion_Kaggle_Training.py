# %% [markdown]
# # CancerFusion AI: Multimodal Fusion Training Notebook (Kaggle)
# This notebook is designed to run on Kaggle using the TCGA-BRCA Multi-Modal Fusion Dataset.
# It implements a dual-stream architecture (Histopathology + Genomics) fused via a Transformer cross-attention mechanism.
# 
# **Instructions for Kaggle:**
# 1. Open Kaggle and create a new Notebook.
# 2. Add the Dataset: "Breast Cancer: Multi-Modal Fusion Dataset"
# 3. Copy these cells into your Kaggle Notebook and run!

# %% [code]
# Cell 1: Imports & Environment Setup
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [code]
# Cell 2: Dataset Definition
# Note: Adjust the file paths based on the exact structure of the Kaggle dataset you attach.
class CancerFusionDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        """
        Loads pre-computed Histopathology embeddings (WSI) and RNA-Seq data.
        In a real Kaggle environment, you will point this to the attached dataset directory.
        """
        self.data_dir = data_dir
        self.is_train = is_train
        
        # MOCK DATA LOADER (Replace with actual pandas/numpy loading logic)
        # Assuming 500 patients for training, 100 for testing
        num_samples = 500 if is_train else 100
        
        # WSI Embeddings (e.g., from a frozen ResNet/ViT): Shape [num_patches, embed_dim]
        # We assume 100 patches per patient, 768-D embeddings
        self.wsi_embeddings = [torch.randn(100, 768) for _ in range(num_samples)]
        
        # Genomic Data (RNA-seq): Shape [num_genes]
        # We assume top 1000 highly variant genes
        self.rna_embeddings = [torch.randn(1000) for _ in range(num_samples)]
        
        # Clinical Survival Data: [Survival Time (months), Event Status (1=Deceased, 0=Censored)]
        self.survival_time = torch.abs(torch.randn(num_samples) * 50 + 10) 
        self.event_status = torch.randint(0, 2, (num_samples,)).float()
        
    def __len__(self):
        return len(self.survival_time)
    
    def __getitem__(self, idx):
        return {
            'wsi': self.wsi_embeddings[idx],
            'rna': self.rna_embeddings[idx],
            'time': self.survival_time[idx],
            'event': self.event_status[idx]
        }

# Initialize DataLoaders
train_dataset = CancerFusionDataset(data_dir='/kaggle/input/brca-dataset/', is_train=True)
val_dataset = CancerFusionDataset(data_dir='/kaggle/input/brca-dataset/', is_train=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
print("DataLoaders initialized!")

# %% [code]
# Cell 3: Neural Network Architecture
class GenomicEncoder(nn.Module):
    def __init__(self, input_dim=1000, hidden_dim=256):
        super().__init__()
        # SNN (Self-Normalizing Network) for Genomics
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SELU(),
            nn.AlphaDropout(p=0.1),
            nn.Linear(512, hidden_dim),
            nn.SELU()
        )
    def forward(self, x):
        return self.net(x)

class WSIEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        # Compress 768-D ViT patches to our common hidden dimension
        self.projection = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        # x shape: [batch, num_patches, input_dim]
        return self.projection(x)

class MultimodalFusionTransformer(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        super().__init__()
        self.genomic_encoder = GenomicEncoder(input_dim=1000, hidden_dim=hidden_dim)
        self.wsi_encoder = WSIEncoder(input_dim=768, hidden_dim=hidden_dim)
        
        # Cross-Attention: Genomics queries the WSI patches
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Survival Prediction Head
        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1) # Outputs a single risk score
        )
        
    def forward(self, wsi, rna):
        # 1. Encode Modalities
        rna_feat = self.genomic_encoder(rna) # [batch, hidden_dim]
        wsi_feat = self.wsi_encoder(wsi)     # [batch, num_patches, hidden_dim]
        
        # 2. Cross-Attention Fusion
        # RNA acts as Query (unsqueeze to add sequence dimension)
        query = rna_feat.unsqueeze(1)        # [batch, 1, hidden_dim]
        # WSI acts as Key and Value
        attn_out, attn_weights = self.cross_attention(query=query, key=wsi_feat, value=wsi_feat)
        attn_out = attn_out.squeeze(1)       # [batch, hidden_dim]
        
        # 3. Concatenate and Predict Risk
        fused_representation = torch.cat([rna_feat, attn_out], dim=1) # [batch, hidden_dim * 2]
        risk_score = self.survival_head(fused_representation)
        
        return risk_score, attn_weights

model = MultimodalFusionTransformer().to(device)
print("Model Architecture built!")

# %% [code]
# Cell 4: Cox Proportional Hazards Loss Function
def cox_nll_loss(risk_scores, survival_times, events):
    """
    Negative Log-Likelihood for the Cox Proportional Hazards model.
    Handles right-censored data (where event=0).
    """
    # Sort patients by survival time descending
    sorted_indices = torch.argsort(survival_times, descending=True)
    risk_scores = risk_scores[sorted_indices]
    events = events[sorted_indices]
    
    # Compute log-sum-exp of risk scores for the denominator (risk set)
    hazard_ratio = torch.exp(risk_scores)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
    
    # Uncensored patients (event=1) contribute to the loss
    uncensored_likelihood = risk_scores - log_risk
    loss = -torch.sum(uncensored_likelihood * events) / (torch.sum(events) + 1e-8)
    
    return loss

# %% [code]
# Cell 5: Training Loop
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
epochs = 20

print("Starting Training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        wsi = batch['wsi'].to(device)
        rna = batch['rna'].to(device)
        time = batch['time'].to(device)
        event = batch['event'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        risk_scores, _ = model(wsi, rna)
        
        # Calculate loss
        loss = cox_nll_loss(risk_scores, time, event)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    # Validation Phase
    model.eval()
    val_risk_scores = []
    val_times = []
    val_events = []
    
    with torch.no_grad():
        for batch in val_loader:
            wsi = batch['wsi'].to(device)
            rna = batch['rna'].to(device)
            val_times.append(batch['time'])
            val_events.append(batch['event'])
            
            risk_scores, _ = model(wsi, rna)
            val_risk_scores.append(risk_scores.cpu())
            
    # Calculate Concordance Index (C-index)
    val_risk_scores = torch.cat(val_risk_scores).numpy().flatten()
    val_times = torch.cat(val_times).numpy()
    val_events = torch.cat(val_events).numpy()
    
    # Note: C-index expects survival times, but model predicts hazard risk. 
    # Higher risk = lower survival time, so we invert the risk score.
    c_index = concordance_index(val_times, -val_risk_scores, val_events)
    
    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {total_loss/len(train_loader):.4f} | Val C-Index: {c_index:.4f}")

print("Training Complete! The model is ready for deeper experimentation.")
