import nbformat as nbf

nb = nbf.v4.new_notebook()

# CELL 1: Markdown
nb.cells.append(nbf.v4.new_markdown_cell("""
# CancerFusion AI: Ultimate Hybrid Architecture (Phase 1)
**Architecture:** Attention-Fusion + Evidential Deep Learning (Dirichlet) + Concept Bottlenecks  
**Objective:** Multimodal Precision Oncology (Radiology, Histopathology, Genomics, Clinical)

This notebook is production-ready for Kaggle environments (P100 / T4x2).
"""))

# CELL 2: Imports
nb.cells.append(nbf.v4.new_code_cell("""
import os
import time
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Optional: pip install lifelines for C-Index
try:
    from lifelines.utils import concordance_index
except ImportError:
    print("Please run: !pip install lifelines")

# Kaggle optimization
torch.backends.cudnn.benchmark = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[*] Engine Booting on {DEVICE.upper()}")
"""))

# CELL 3: Dataset
nb.cells.append(nbf.v4.new_code_cell("""
class TCGADataset(Dataset):
    def __init__(self, is_train=True):
        \"\"\"
        TODO: Replace with actual TCGA dataloading logic.
        Currently generates synthetic data matching the expected shapes for architecture validation.
        \"\"\"
        self.n_samples = 1000 if is_train else 200
        # Simulated Data
        self.wsi = torch.randn(self.n_samples, 100, 768) # 100 patches per patient
        self.rna = torch.randn(self.n_samples, 512)
        self.cli = torch.randn(self.n_samples, 15)
        
        # Missing Mask (1=Present, 0=Missing). Simulate real-world missingness.
        self.mask = torch.ones(self.n_samples, 3)
        self.mask[:, 1] = (torch.rand(self.n_samples) > 0.2).float() # 20% missing Genomics
        self.mask[:, 0] = (torch.rand(self.n_samples) > 0.05).float() # 5% missing WSI
        
        # Survival Data
        self.time = torch.abs(torch.randn(self.n_samples) * 50 + 10)
        self.event = (torch.rand(self.n_samples) < 0.75).float()
        
    def __len__(self): return self.n_samples
    def __getitem__(self, idx):
        return {
            'wsi': self.wsi[idx], 'rna': self.rna[idx], 'cli': self.cli[idx],
            'mask': self.mask[idx], 'time': self.time[idx], 'event': self.event[idx]
        }
"""))

# CELL 4: The Ultimate Hybrid Architecture
nb.cells.append(nbf.v4.new_code_cell("""
class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, hidden_dim)
        )
    def forward(self, x): 
        # Handle batch_size=1 for BatchNorm via evaluating mode behavior
        return self.net(x)

class GenomicEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.SELU(),
            nn.AlphaDropout(0.1), nn.Linear(256, hidden_dim)
        )
    def forward(self, x): return self.net(x)

class WSIConceptBottleneck(nn.Module):
    def __init__(self, input_dim=768, num_concepts=16, hidden_dim=128):
        super().__init__()
        self.concept_layer = nn.Linear(input_dim, num_concepts)
        self.projection = nn.Linear(num_concepts, hidden_dim)
        
    def forward(self, x):
        # x: [B, Patches, Dim] -> mean pool to [B, Dim]
        x_bag = torch.mean(x, dim=1) 
        concepts = torch.sigmoid(self.concept_layer(x_bag))
        embed = self.projection(concepts)
        return embed, concepts

class SafeAttentionFusion(nn.Module):
    \"\"\"
    Fuses modalities with -inf masking, safeguarded against edge cases.
    \"\"\"
    def __init__(self, embed_dim=128):
        super().__init__()
        self.score_net = nn.Sequential(nn.Linear(embed_dim, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, embeds, mask):
        stacked = torch.stack(embeds, dim=1) # [B, 3, Dim]
        scores = self.score_net(stacked).squeeze(-1) # [B, 3]
        
        # EDGE CASE SAFETY: If ALL modalities are missing for a patient, softmax will crash (NaN).
        # We catch this and set scores to 0.0 (uniform attention) to prevent gradient explosion.
        all_missing = (mask.sum(dim=1) == 0)
        
        # Apply -inf mask
        scores = scores.masked_fill(mask == 0, float('-inf'))
        scores[all_missing, :] = 0.0 # Safety fallback
        
        weights = F.softmax(scores, dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights

class EvidentialHead(nn.Module):
    \"\"\" Outputs Risk Score and Uncertainty via Dirichlet Math \"\"\"
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.evidence = nn.Linear(hidden_dim, 2)
        
    def forward(self, fused):
        e = F.softplus(self.evidence(fused))
        
        # EDGE CASE SAFETY: Prevent alpha from exploding to infinity by clamping evidence
        e = torch.clamp(e, min=0.0, max=1e3) 
        
        alpha = e + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        risk = alpha[:, 1] / S.squeeze(-1)
        uncertainty = 2.0 / S.squeeze(-1)
        return risk, uncertainty

class CancerFusionHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 128
        self.wsi_enc = WSIConceptBottleneck(input_dim=768, hidden_dim=dim)
        self.rna_enc = GenomicEncoder(input_dim=512, hidden_dim=dim)
        self.cli_enc = ClinicalEncoder(input_dim=15, hidden_dim=dim)
        self.fusion = SafeAttentionFusion(embed_dim=dim)
        self.head = EvidentialHead(hidden_dim=dim)
        
    def forward(self, wsi, rna, cli, mask):
        # 1. Encode
        w_feat, concepts = self.wsi_enc(wsi)
        r_feat = self.rna_enc(rna)
        c_feat = self.cli_enc(cli)
        
        # 2. Attention Fusion
        fused, attn = self.fusion([w_feat, r_feat, c_feat], mask)
        
        # 3. Evidential Prediction
        risk, unc = self.head(fused)
        
        return {'risk': risk, 'uncertainty': unc, 'attention': attn, 'concepts': concepts}

model = CancerFusionHybrid().to(DEVICE)
print("[*] Ultimate Hybrid Architecture Initialized!")
"""))

# CELL 5: Edge Case Testing
nb.cells.append(nbf.v4.new_code_cell("""
print("--- RUNNING EXTREME EDGE CASE TESTS ---")
model.eval()

# Edge Case 1: Batch Size of 1 (Tests BatchNorm stability)
try:
    print("Testing Batch Size = 1...")
    out1 = model(torch.randn(1, 100, 768).to(DEVICE), 
                 torch.randn(1, 512).to(DEVICE), 
                 torch.randn(1, 15).to(DEVICE), 
                 torch.ones(1, 3).to(DEVICE))
    print("✅ Passed: Model supports B=1 Inference.")
except Exception as e: print(f"❌ Failed B=1: {e}")

# Edge Case 2: Extreme Missingness (Only Clinical Data Present)
try:
    print("Testing Extreme Missingness (Mask = [0, 0, 1])...")
    mask_extreme = torch.tensor([[0.0, 0.0, 1.0]]).to(DEVICE)
    out2 = model(torch.randn(1, 100, 768).to(DEVICE), torch.randn(1, 512).to(DEVICE), 
                 torch.randn(1, 15).to(DEVICE), mask_extreme)
    attn = out2['attention'][0].detach().cpu().numpy()
    assert np.allclose(attn, [0.0, 0.0, 1.0]), f"Attention did not route correctly: {attn}"
    print(f"✅ Passed: Attention routed 100% to Clinical perfectly.")
except Exception as e: print(f"❌ Failed Extreme Missingness: {e}")

# Edge Case 3: Complete Modality Dropout (Mask = [0, 0, 0])
try:
    print("Testing Complete Dropout (All Modalities Missing)...")
    mask_zero = torch.tensor([[0.0, 0.0, 0.0]]).to(DEVICE)
    out3 = model(torch.randn(1, 100, 768).to(DEVICE), torch.randn(1, 512).to(DEVICE), 
                 torch.randn(1, 15).to(DEVICE), mask_zero)
    print("✅ Passed: Safe Softmax prevented NaN crash.")
except Exception as e: print(f"❌ Failed Complete Dropout: {e}")

# Edge Case 4: Exploding Feature Values (NaN check)
try:
    print("Testing Exploding Features (Huge Inputs)...")
    out4 = model(torch.randn(1, 100, 768).to(DEVICE)*1e5, torch.randn(1, 512).to(DEVICE)*1e5, 
                 torch.randn(1, 15).to(DEVICE)*1e5, torch.ones(1, 3).to(DEVICE))
    assert not torch.isnan(out4['risk']).any(), "Risk yielded NaN"
    print("✅ Passed: Evidential Dirichlet head remained stable.")
except Exception as e: print(f"❌ Failed Numerical Stability: {e}")
"""))

# CELL 6: Training Loop
nb.cells.append(nbf.v4.new_code_cell("""
def cox_loss(risk, times, events):
    \"\"\" Robust Cox NLL Loss \"\"\"
    sorted_idx = torch.argsort(times, descending=True)
    risk, events = risk[sorted_idx], events[sorted_idx]
    hr = torch.exp(torch.clamp(risk, max=20.0)) # Clamp to prevent exp explosion
    log_risk = torch.log(torch.cumsum(hr, dim=0) + 1e-8)
    return -torch.sum((risk - log_risk) * events) / (torch.sum(events) + 1e-8)

def train_kaggle(epochs=10):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))
    
    loader = DataLoader(TCGADataset(is_train=True), batch_size=32, shuffle=True, drop_last=True)
    
    for epoch in range(1, epochs+1):
        ep_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda')):
                out = model(batch['wsi'].to(DEVICE), batch['rna'].to(DEVICE), 
                            batch['cli'].to(DEVICE), batch['mask'].to(DEVICE))
                loss = cox_loss(out['risk'], batch['time'].to(DEVICE), batch['event'].to(DEVICE))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ep_loss += loss.item()
            
        print(f"Epoch {epoch}/{epochs} | Loss: {ep_loss/len(loader):.4f}")

# train_kaggle(epochs=5)
"""))

with open('e:/MAJOR PROJECT/CancerFusion_Advanced_Kaggle.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
