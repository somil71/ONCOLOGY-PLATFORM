import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell("# CancerFusion AI: Ultimate Hybrid Architecture (PRODUCTION RUN)\\n**Architecture:** Attention-Fusion + Evidential Deep Learning (Dirichlet) + Concept Bottlenecks + Vision Transformer (ViT)\\n**Features:** Automated Kaplan-Meier Research Plots, API Backend Export, Missing Data (MNAR) Handling"))

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
from PIL import Image
import torchvision.transforms as T
from torchvision.models import vit_b_16, ViT_B_16_Weights

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
print(f"[*] Engine Booting on {DEVICE.upper()}")

# Initialize Vision Transformer for real WSI pixel extraction
print("[*] Downloading Vision Transformer (ViT-B/16)...")
vision_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(DEVICE)
vision_model.heads = nn.Identity() # Strip classifier head for raw 768 features
vision_model.eval()

wsi_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
"""))

nb.cells.append(nbf.v4.new_code_cell("""# =====================================================================
# 🚀 REAL DATASET CONFIGURATION (PRODUCTION WSI LOADER)
# =====================================================================
DIR_MULTI_MODAL = "/kaggle/input/datasets/sepehreslamimoghadam/breast-cancer-vision-and-genomic-fusion-ml-ready"

class TCGADataset(Dataset):
    def __init__(self, is_train=True):
        print(f"[*] Mounting REAL 350GB Breast Cancer Data...")
        
        # --- 1. LOAD CLINICAL DATA ---
        clinical_path = f"{DIR_MULTI_MODAL}/Clinical_Demographic_Data.csv"
        cli_df = pd.read_csv(clinical_path).fillna(0)
        
        # --- 2. LOAD GENOMICS DATA ---
        rna_path = f"{DIR_MULTI_MODAL}/RNA_CNV_ModelReady.csv"
        rna_df = pd.read_csv(rna_path).fillna(0)
        
        print("Successfully loaded CSV files!")
        
        # Extract Time & Event (TCGA Standard Columns)
        time_dead = pd.to_numeric(cli_df.get('demographic_days_to_death', pd.Series()), errors='coerce')
        time_alive = pd.to_numeric(cli_df.get('diagnoses_days_to_last_follow_up', pd.Series()), errors='coerce')
        
        # Robust fallback mechanism for Survival Time
        if time_dead.empty and time_alive.empty:
            print("⚠️ Attempting to auto-detect 'survival_time'...")
            time_vals = pd.to_numeric(cli_df.get('survival_time', cli_df.iloc[:, 1]), errors='coerce').fillna(0)
            event_vals = pd.to_numeric(cli_df.get('event', cli_df.iloc[:, 2]), errors='coerce').fillna(0)
        else:
            time_vals = time_dead.fillna(time_alive).fillna(0) 
            event_vals = (cli_df['demographic_vital_status'].astype(str).str.strip().str.lower() == 'dead').astype(float)
            
        self.time = torch.tensor(time_vals.values, dtype=torch.float32)
        self.event = torch.tensor(event_vals.values, dtype=torch.float32)
        
        # Drop targets to get numeric clinical features
        drop_cols = ['demographic_days_to_death', 'diagnoses_days_to_last_follow_up', 'demographic_vital_status', 'survival_time', 'event']
        cli_features = cli_df.drop(columns=[c for c in drop_cols if c in cli_df.columns]).select_dtypes(include=[np.number]).fillna(0)
        
        self.cli = torch.tensor(cli_features.values, dtype=torch.float32)
        self.rna = torch.tensor(rna_df.select_dtypes(include=[np.number]).values, dtype=torch.float32)
        
        # Map Patient IDs to the WSI Folder
        self.patient_ids = cli_df.get('Patient_ID', cli_df.iloc[:, 0]).astype(str).values
        self.wsi_folder = f"{DIR_MULTI_MODAL}/MRI_and_SVS_Patches"
        
        self.n_samples = len(cli_df)
        self.mask = torch.ones(self.n_samples, 3)

    def __len__(self): return self.n_samples
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        # DYNAMIC VISION EXTRACTION (The Heavy Lifting)
        patch_paths = glob.glob(f"{self.wsi_folder}/*{patient_id}*.*")
        wsi_tensor = torch.zeros(100, 768) 
        
        if len(patch_paths) > 0:
            patches_to_load = patch_paths[:100]
            loaded_images = []
            for p in patches_to_load:
                try:
                    img = Image.open(p).convert('RGB')
                    loaded_images.append(wsi_transform(img))
                except Exception: pass
            
            if len(loaded_images) > 0:
                with torch.no_grad():
                    img_batch = torch.stack(loaded_images).to(DEVICE)
                    features = vision_model(img_batch).cpu()
                    wsi_tensor[:features.size(0), :] = features

        return {'wsi': wsi_tensor, 'rna': self.rna[idx], 'cli': self.cli[idx],
                'mask': self.mask[idx], 'time': self.time[idx], 'event': self.event[idx]}
"""))

nb.cells.append(nbf.v4.new_code_cell("""class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 128))
    def forward(self, x): return self.net(x)

class GenomicEncoder(nn.Module):
    def __init__(self, input_dim):
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
    def __init__(self, rna_dim, cli_dim):
        super().__init__()
        self.wsi_enc = WSIConceptBottleneck()
        self.rna_enc = GenomicEncoder(rna_dim)
        self.cli_enc = ClinicalEncoder(cli_dim)
        self.fusion = SafeAttentionFusion()
        self.head = EvidentialHead()
    def forward(self, wsi, rna, cli, mask):
        w, c_scores = self.wsi_enc(wsi)
        fused, attn = self.fusion([w, self.rna_enc(rna), self.cli_enc(cli)], mask)
        risk, unc = self.head(fused)
        return risk, unc, attn, c_scores

print("[*] Auto-detecting real dataset dimensions...")
temp_ds = TCGADataset(is_train=True)
REAL_RNA_DIM = temp_ds.rna.shape[1]
REAL_CLI_DIM = temp_ds.cli.shape[1]

model = CancerFusionHybrid(rna_dim=REAL_RNA_DIM, cli_dim=REAL_CLI_DIM).to(DEVICE)
print(f"[*] Success! Architecture initialized with RNA={REAL_RNA_DIM}, Clinical={REAL_CLI_DIM}")
"""))

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
    print("✅ Saved CancerFusion_Weights.pth and backend_response.json!")
"""))

nb.cells.append(nbf.v4.new_code_cell("""def cox_loss(risk, times, events):
    sorted_idx = torch.argsort(times, descending=True)
    risk, events = risk[sorted_idx], events[sorted_idx]
    hr = torch.exp(torch.clamp(risk, max=20.0))
    log_risk = torch.log(torch.cumsum(hr, dim=0) + 1e-8)
    return -torch.sum((risk - log_risk) * events) / (torch.sum(events) + 1e-8)

def run_full_pipeline(epochs=50):
    print("Starting CancerFusion Heavy-Duty Training Engine...")
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

# 🔥 FULL PRODUCTION RUN (50 EPOCHS) 🔥
run_full_pipeline(epochs=50)
"""))

with open('e:/MAJOR PROJECT/CancerFusion_Advanced_Kaggle.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
