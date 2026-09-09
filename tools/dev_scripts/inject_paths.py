import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell("# CancerFusion AI: Ultimate Hybrid Architecture\\n**Architecture:** Attention-Fusion + Evidential Deep Learning (Dirichlet) + Concept Bottlenecks\\n**Features:** Automated Kaplan-Meier Research Plots, API Backend Export, Missing Data (MNAR) Handling"))

nb.cells.append(nbf.v4.new_code_cell("""import os, json, time, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from lifelines import KaplanMeierFitter
    HAS_LIFELINES = True
except ImportError:
    import os
    os.system('pip install lifelines')
    from lifelines import KaplanMeierFitter
    HAS_LIFELINES = True

torch.backends.cudnn.benchmark = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[*] Engine Booting on {DEVICE.upper()}")"""))

nb.cells.append(nbf.v4.new_code_cell("""# =====================================================================
# 🚀 REAL DATASET CONFIGURATION
# =====================================================================
# Set this to True when you are ready to train on the 350GB data!
USE_REAL_DATA = False 

# Your exact mounted Kaggle directories:
DIR_MULTI_MODAL = "/kaggle/input/datasets/sepehreslamimoghadam/breast-cancer-vision-and-genomic-fusion-ml-ready"
DIR_CPTAC_PDA = "/kaggle/input/datasets/pietracaforia/cptac-pda-71"
DIR_BRCA_SURV = "/kaggle/input/datasets/jmalagontorres/tcga-brca-survival-analysis"
DIR_LUAD = "/kaggle/input/datasets/nahin333/tcgaluad"

class TCGADataset(Dataset):
    def __init__(self, is_train=True):
        self.n_samples = 1000 if is_train else 200
        
        if USE_REAL_DATA:
            print(f"[*] Mounting massive 350GB Kaggle Datasets from:\\n{DIR_BRCA_SURV}")
            
            # --- STEP 1: LOAD CLINICAL DATA ---
            # Replace 'clinical_data.csv' with the exact filename inside DIR_BRCA_SURV
            cli_df = pd.read_csv(f"{DIR_BRCA_SURV}/clinical_data.csv").fillna(0)
            
            # Extract target variables (replace 'survival_time' and 'event' with exact column names)
            self.time = torch.tensor(cli_df['survival_time'].values, dtype=torch.float32)
            self.event = torch.tensor(cli_df['event'].values, dtype=torch.float32)
            
            # Drop targets to get features
            cli_features = cli_df.drop(columns=['survival_time', 'event']).select_dtypes(include=[np.number])
            self.cli = torch.tensor(cli_features.values, dtype=torch.float32)
            
            # --- STEP 2: LOAD GENOMICS (RNA) ---
            rna_df = pd.read_csv(f"{DIR_BRCA_SURV}/rna_data.csv").fillna(0)
            self.rna = torch.tensor(rna_df.select_dtypes(include=[np.number]).values, dtype=torch.float32)
            
            # --- STEP 3: LOAD WSI IMAGE PATCHES ---
            # (Assuming patches are loaded/preprocessed via dataloader in production)
            self.wsi = torch.randn(len(cli_df), 100, 768) # Placeholder for WSI loader
            
            # Missing mask (Assume all present for real data unless explicitly missing)
            self.mask = torch.ones(len(cli_df), 3)
            self.n_samples = len(cli_df)
            
        else:
            # SYNTHETIC FALLBACK (To test the architecture and see the graphs instantly)
            self.wsi = torch.randn(self.n_samples, 100, 768)
            self.rna = torch.randn(self.n_samples, 512)
            self.cli = torch.randn(self.n_samples, 15)
            self.mask = torch.ones(self.n_samples, 3)
            self.mask[:, 1] = (torch.rand(self.n_samples) > 0.2).float() # 20% missing Genomics
            self.time = torch.abs(torch.randn(self.n_samples) * 50 + 10)
            self.event = (torch.rand(self.n_samples) < 0.75).float()
            
    def __len__(self): return self.n_samples
    def __getitem__(self, idx):
        return {'wsi': self.wsi[idx], 'rna': self.rna[idx], 'cli': self.cli[idx],
                'mask': self.mask[idx], 'time': self.time[idx], 'event': self.event[idx]}"""))

nb.cells.append(nbf.v4.new_code_cell("""class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim=15):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 128))
    def forward(self, x): return self.net(x)

class GenomicEncoder(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.SELU(), nn.AlphaDropout(0.1), nn.Linear(256, 128))
    def forward(self, x): return self.net(x)

class WSIConceptBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.concept_layer = nn.Linear(768, 16)
        self.projection = nn.Linear(16, 128)
    def forward(self, x):
        x_bag = torch.mean(x, dim=1) 
        concepts = torch.sigmoid(self.concept_layer(x_bag))
        return self.projection(concepts), concepts

class SafeAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.score_net = nn.Sequential(nn.Linear(128, 32), nn.Tanh(), nn.Linear(32, 1))
    def forward(self, embeds, mask):
        stacked = torch.stack(embeds, dim=1) 
        scores = self.score_net(stacked).squeeze(-1) 
        scores = scores.masked_fill(mask == 0, float('-inf'))
        scores[(mask.sum(dim=1) == 0), :] = 0.0
        weights = F.softmax(scores, dim=1)
        return (stacked * weights.unsqueeze(-1)).sum(dim=1), weights

class EvidentialHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.evidence = nn.Linear(128, 2)
    def forward(self, fused):
        e = torch.clamp(F.softplus(self.evidence(fused)), min=0.0, max=1e3) 
        alpha = e + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        return alpha[:, 1] / S.squeeze(-1), 2.0 / S.squeeze(-1)

class CancerFusionHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.wsi_enc, self.rna_enc, self.cli_enc = WSIConceptBottleneck(), GenomicEncoder(512), ClinicalEncoder(15)
        self.fusion, self.head = SafeAttentionFusion(), EvidentialHead()
    def forward(self, wsi, rna, cli, mask):
        w, c_scores = self.wsi_enc(wsi)
        fused, attn = self.fusion([w, self.rna_enc(rna), self.cli_enc(cli)], mask)
        risk, unc = self.head(fused)
        return risk, unc, attn, c_scores

model = CancerFusionHybrid().to(DEVICE)"""))

nb.cells.append(nbf.v4.new_code_cell("""def export_and_visualize(model, test_loader):
    print("\\n" + "="*50)
    print("🚀 GENERATING PUBLICATION CHARTS & BACKEND EXPORT")
    print("="*50)
    
    model.eval()
    all_risks, all_times, all_events, all_uncs, all_attns = [], [], [], [], []
    with torch.no_grad():
        for b in test_loader:
            risk, unc, attn, _ = model(b['wsi'].to(DEVICE), b['rna'].to(DEVICE), b['cli'].to(DEVICE), b['mask'].to(DEVICE))
            all_risks.extend(risk.cpu().numpy()); all_times.extend(b['time'].numpy())
            all_events.extend(b['event'].numpy()); all_uncs.extend(unc.cpu().numpy())
            all_attns.extend(attn.cpu().numpy())
    
    all_risks, all_times, all_events = np.array(all_risks), np.array(all_times), np.array(all_events)
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig = plt.figure(figsize=(18, 5))
    
    ax1 = plt.subplot(1, 3, 1)
    if HAS_LIFELINES:
        kmf = KaplanMeierFitter()
        high_risk = all_risks > np.median(all_risks)
        kmf.fit(all_times[~high_risk], event_observed=all_events[~high_risk], label="Low Risk (AI)")
        kmf.plot_survival_function(ax=ax1, color='blue', linewidth=2)
        kmf.fit(all_times[high_risk], event_observed=all_events[high_risk], label="High Risk (AI)")
        kmf.plot_survival_function(ax=ax1, color='red', linewidth=2)
    ax1.set_title("Kaplan-Meier Survival Curves", fontweight='bold')
    
    ax2 = plt.subplot(1, 3, 2)
    sns.barplot(x=['WSI', 'Genomics', 'Clinical'], y=np.mean(all_attns, axis=0), palette="viridis", ax=ax2)
    ax2.set_title("Modality Importance (Attention)", fontweight='bold')
    
    ax3 = plt.subplot(1, 3, 3)
    sns.regplot(x=all_risks, y=all_uncs, scatter_kws={'alpha':0.5, 'color':'purple'}, line_kws={'color':'red'}, ax=ax3)
    ax3.set_title("Prediction vs. Uncertainty", fontweight='bold')
    
    plt.tight_layout(); plt.savefig("CancerFusion_Publication_Graphs.png", dpi=300); plt.show()

    torch.save(model.state_dict(), "CancerFusion_Weights.pth")
    api_payload = {
        "patient_id": "TCGA-BRCA-LATEST",
        "prediction": {"risk_score": float(all_risks[0]), "uncertainty": float(all_uncs[0])},
        "explainability": {"attention_weights": {"wsi": float(all_attns[0][0]), "rna": float(all_attns[0][1]), "cli": float(all_attns[0][2])}}
    }
    with open("backend_response.json", 'w') as f: json.dump(api_payload, f, indent=4)
    print("✅ Saved CancerFusion_Weights.pth and backend_response.json!")"""))

nb.cells.append(nbf.v4.new_code_cell("""def cox_loss(risk, times, events):
    sorted_idx = torch.argsort(times, descending=True)
    risk, events = risk[sorted_idx], events[sorted_idx]
    hr = torch.exp(torch.clamp(risk, max=20.0))
    log_risk = torch.log(torch.cumsum(hr, dim=0) + 1e-8)
    return -torch.sum((risk - log_risk) * events) / (torch.sum(events) + 1e-8)

def run_full_pipeline(epochs=5):
    print("Starting CancerFusion Training Engine...")
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TCGADataset(is_train=True), batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(TCGADataset(is_train=False), batch_size=32, shuffle=False)
    
    for epoch in range(1, epochs+1):
        ep_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            risk, _, _, _ = model(batch['wsi'].to(DEVICE), batch['rna'].to(DEVICE), batch['cli'].to(DEVICE), batch['mask'].to(DEVICE))
            loss = cox_loss(risk, batch['time'].to(DEVICE), batch['event'].to(DEVICE))
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        print(f"Epoch {epoch}/{epochs} | Loss: {ep_loss/len(loader):.4f}")
        
    export_and_visualize(model, test_loader)

run_full_pipeline(epochs=10)"""))

with open('e:/MAJOR PROJECT/CancerFusion_Advanced_Kaggle.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
