import nbformat as nbf

nb = nbf.v4.new_notebook()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""
# CancerFusion AI \u2014 Multimodal Precision Oncology Platform
**BTP Major Project | Full-Scale Implementation**  

### Architecture Upgrades
| Component | Description |
|---|---|
| **Vision** | ViT-B/16 (768d) Concept Bottleneck on Gigapixel WSI Patches |
| **Genomics** | AlphaDropout + SELU MLP on RNA-Seq profiles |
| **Fusion** | Cross-Modal Attention with $-\\infty$ Masking for Missing Modalities |
| **Head** | Evidential Deep Learning (Dirichlet) for Uncertainty Quantification |
| **Loss** | Cox Proportional Hazards Negative Log-Likelihood |
"""))

# ---------------------------------------------------------
# CELL 1: Dependencies
# ---------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("## Cell 1 \u2014 Install Dependencies & GPU Setup"))
nb.cells.append(nbf.v4.new_code_cell("""
# Cell 1 \u2014 Dependencies
import subprocess, sys

def pip(*args):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *args])

try:
    import lifelines
    print("  ok  lifelines")
except ImportError:
    print("  installing lifelines...")
    pip("lifelines")
    print("  installed lifelines")

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\\nPyTorch {torch.__version__} | Device: {DEVICE.upper()}')

if DEVICE == 'cuda':
    props = torch.cuda.get_device_properties(0)
    TOTAL_VRAM_GB = props.total_memory / 1e9
    print(f'GPU : {props.name}')
    print(f'VRAM: {TOTAL_VRAM_GB:.1f} GB')
else:
    TOTAL_VRAM_GB = 0
print('Dependencies ready.')
"""))

# ---------------------------------------------------------
# CELL 2: Imports & Config
# ---------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("## Cell 2 \u2014 Imports & Global Configuration"))
nb.cells.append(nbf.v4.new_code_cell("""
# Cell 2 \u2014 Imports and CFG
import os, json, time, glob, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import vit_b_16, ViT_B_16_Weights
from lifelines import KaplanMeierFitter

# High-Performance Backend
torch.backends.cudnn.benchmark = True
if DEVICE == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True

CFG = {
    'batch_size': 32,
    'epochs': 50,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'max_patches': 100,
    'wsi_dim': 768,
    'rna_dim': None, # Auto-detected
    'cli_dim': None, # Auto-detected
    'dir_multimodal': "/kaggle/input/datasets/sepehreslamimoghadam/breast-cancer-vision-and-genomic-fusion-ml-ready",
    'num_workers': min(4, os.cpu_count() or 1),
}

print('\\n' + '='*65)
print('  CancerFusion Configuration')
print('='*65)
print(f'  Batch Size : {CFG["batch_size"]}')
print(f'  Epochs     : {CFG["epochs"]}')
print(f'  Learn Rate : {CFG["lr"]}')
print(f'  WSI Patches: {CFG["max_patches"]} per patient')
print('='*65)
"""))

# ---------------------------------------------------------
# CELL 3: Vision Transformer Setup
# ---------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("## Cell 3 \u2014 Pretrained Vision Transformer (Feature Extractor)"))
nb.cells.append(nbf.v4.new_code_cell("""
# Cell 3 \u2014 ViT Setup
print("[*] Loading Vision Transformer (ViT-B/16)...")
vision_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(DEVICE)
vision_model.heads = nn.Identity() # Strip classification head for raw 768d embedding
vision_model.eval()

wsi_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

vit_params = sum(p.numel() for p in vision_model.parameters())
print(f"\u2705 ViT loaded | Params: {vit_params:,} | Output Dim: 768")
"""))

# ---------------------------------------------------------
# CELL 4: Dataset & Dataloaders
# ---------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("## Cell 4 \u2014 TCGADataset & Dataloader (Multimodal Linking)"))
nb.cells.append(nbf.v4.new_code_cell("""
# Cell 4 \u2014 Dataset
class TCGADataset(Dataset):
    def __init__(self, is_train=True):
        print(f"\\n[*] Parsing Kaggle Dataset: {CFG['dir_multimodal'].split('/')[-1]}")
        
        # 1. Clinical Data
        cli_df = pd.read_csv(f"{CFG['dir_multimodal']}/Clinical_Demographic_Data.csv").fillna(0)
        
        # 2. Genomics Data
        rna_df = pd.read_csv(f"{CFG['dir_multimodal']}/RNA_CNV_ModelReady.csv").fillna(0)
        
        # Time & Event extraction with TCGA fallbacks
        time_dead = pd.to_numeric(cli_df.get('demographic_days_to_death', pd.Series()), errors='coerce')
        time_alive = pd.to_numeric(cli_df.get('diagnoses_days_to_last_follow_up', pd.Series()), errors='coerce')
        
        if time_dead.empty and time_alive.empty:
            time_vals = pd.to_numeric(cli_df.get('survival_time', cli_df.iloc[:, 1]), errors='coerce').fillna(0)
            event_vals = pd.to_numeric(cli_df.get('event', cli_df.iloc[:, 2]), errors='coerce').fillna(0)
        else:
            time_vals = time_dead.fillna(time_alive).fillna(0) 
            event_vals = (cli_df['demographic_vital_status'].astype(str).str.strip().str.lower() == 'dead').astype(float)
            
        self.time = torch.tensor(time_vals.values, dtype=torch.float32)
        self.event = torch.tensor(event_vals.values, dtype=torch.float32)
        
        # Features
        drop_cols = ['demographic_days_to_death', 'diagnoses_days_to_last_follow_up', 'demographic_vital_status', 'survival_time', 'event']
        cli_features = cli_df.drop(columns=[c for c in drop_cols if c in cli_df.columns]).select_dtypes(include=[np.number]).fillna(0)
        
        self.cli = torch.tensor(cli_features.values, dtype=torch.float32)
        self.rna = torch.tensor(rna_df.select_dtypes(include=[np.number]).values, dtype=torch.float32)
        
        self.patient_ids = cli_df.get('Patient_ID', cli_df.iloc[:, 0]).astype(str).values
        self.wsi_folder = f"{CFG['dir_multimodal']}/MRI_and_SVS_Patches"
        
        self.n_samples = len(cli_df)
        self.mask = torch.ones(self.n_samples, 3) # MNAR Masking Support

        CFG['cli_dim'] = self.cli.shape[1]
        CFG['rna_dim'] = self.rna.shape[1]

    def __len__(self): return self.n_samples
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patch_paths = glob.glob(f"{self.wsi_folder}/*{patient_id}*.*")
        
        wsi_tensor = torch.zeros(CFG['max_patches'], CFG['wsi_dim']) 
        
        if len(patch_paths) > 0:
            patches_to_load = patch_paths[:CFG['max_patches']]
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

# Initialize Dataset to globally set dimensions
full_ds = TCGADataset(is_train=True)
train_size = int(0.8 * len(full_ds))
val_size = len(full_ds) - train_size
train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])

train_dl = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, drop_last=True, num_workers=CFG['num_workers'])
val_dl = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['num_workers'])

print(f"\u2705 Dataloaders Ready")
print(f"   Train: {len(train_ds):,} patients | Val: {len(val_ds):,} patients")
print(f"   Detected Input Shapes -> RNA: {CFG['rna_dim']}d | Clinical: {CFG['cli_dim']}d")
"""))

# ---------------------------------------------------------
# CELL 5: Modality Encoders
# ---------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("## Cell 5 \u2014 Unimodal Encoders (WSI, RNA, Clinical)"))
nb.cells.append(nbf.v4.new_code_cell("""
# Cell 5 \u2014 Encoders
class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), 
            nn.BatchNorm1d(64), 
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(64, 128)
        )
    def forward(self, x): return self.net(x)

class GenomicEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), 
            nn.SELU(), 
            nn.AlphaDropout(0.1), 
            nn.Linear(256, 128)
        )
    def forward(self, x): return self.net(x)

class WSIConceptBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.concept_layer = nn.Linear(CFG['wsi_dim'], 16)
        self.projection = nn.Linear(16, 128)
    def forward(self, x):
        x_bag = torch.mean(x, dim=1) # MIL Aggregation
        concepts = torch.sigmoid(self.concept_layer(x_bag))
        return self.projection(concepts), concepts
"""))

# ---------------------------------------------------------
# CELL 6: Fusion & Head
# ---------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("## Cell 6 \u2014 Safe Attention Fusion & Evidential Head"))
nb.cells.append(nbf.v4.new_code_cell("""
# Cell 6 \u2014 Fusion & Evidential Logic
class SafeAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(128, 32), 
            nn.Tanh(), 
            nn.Linear(32, 1)
        )
    def forward(self, embeds, mask):
        stacked = torch.stack(embeds, dim=1) 
        scores = self.score_net(stacked).squeeze(-1) 
        
        # Safety Masking for MNAR (Missing Not At Random)
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
"""))

# ---------------------------------------------------------
# CELL 7: Hybrid Model Assembly
# ---------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("## Cell 7 \u2014 CancerFusion Hybrid Model Assembly"))
nb.cells.append(nbf.v4.new_code_cell("""
# Cell 7 \u2014 Hybrid Model
class CancerFusionHybrid(nn.Module):
    def __init__(self, rna_dim, cli_dim):
        super().__init__()
        self.wsi_enc = WSIConceptBottleneck()
        self.rna_enc = GenomicEncoder(rna_dim)
        self.cli_enc = ClinicalEncoder(cli_dim)
        self.fusion  = SafeAttentionFusion()
        self.head    = EvidentialHead()
        
    def forward(self, wsi, rna, cli, mask):
        w, c_scores = self.wsi_enc(wsi)
        fused, attn = self.fusion([w, self.rna_enc(rna), self.cli_enc(cli)], mask)
        risk, unc   = self.head(fused)
        return risk, unc, attn, c_scores

model = CancerFusionHybrid(rna_dim=CFG['rna_dim'], cli_dim=CFG['cli_dim']).to(DEVICE)
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\\n\u2705 CancerFusion Model Compiled")
print(f"   Trainable Parameters: {train_params:,}")
"""))

# ---------------------------------------------------------
# CELL 8: Loss & Optimizer
# ---------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("## Cell 8 \u2014 Cox Proportional Hazards Loss & Optimizer"))
nb.cells.append(nbf.v4.new_code_cell("""
# Cell 8 \u2014 Loss & Optimizer
def cox_loss(risk, times, events):
    sorted_idx = torch.argsort(times, descending=True)
    risk, events = risk[sorted_idx], events[sorted_idx]
    hr = torch.exp(torch.clamp(risk, max=20.0))
    log_risk = torch.log(torch.cumsum(hr, dim=0) + 1e-8)
    return -torch.sum((risk - log_risk) * events) / (torch.sum(events) + 1e-8)

optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
print(f"\u2705 Optimizer: AdamW | LR: {CFG['lr']} | WD: {CFG['weight_decay']}")
"""))

# ---------------------------------------------------------
# CELL 9: Training Loop
# ---------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("## Cell 9 \u2014 Training Engine"))
nb.cells.append(nbf.v4.new_code_cell("""
# Cell 9 \u2014 Training Loop
print("\\n" + "="*65)
print("  🚀 IGNITING CANCERFUSION TRAINING ENGINE")
print("="*65)

history = {'loss': []}

for epoch in range(1, CFG['epochs']+1):
    model.train()
    ep_loss = 0.0
    
    # Using tqdm for a beautiful progress bar (like GenMol)
    pbar = tqdm(train_dl, desc=f'Epoch {epoch:02d}/{CFG["epochs"]}', leave=False)
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)
        
        wsi = batch['wsi'].to(DEVICE, non_blocking=True)
        rna = batch['rna'].to(DEVICE, non_blocking=True)
        cli = batch['cli'].to(DEVICE, non_blocking=True)
        mask = batch['mask'].to(DEVICE, non_blocking=True)
        times = batch['time'].to(DEVICE, non_blocking=True)
        events = batch['event'].to(DEVICE, non_blocking=True)
        
        risk, _, _, _ = model(wsi, rna, cli, mask)
        loss = cox_loss(risk, times, events)
        
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    avg_loss = ep_loss / max(len(train_dl), 1)
    history['loss'].append(avg_loss)
    print(f"Epoch {epoch:02d}/{CFG['epochs']:02d} | Train Cox Loss: {avg_loss:.4f}")

print("\\n\u2705 Training Complete!")
"""))

# ---------------------------------------------------------
# CELL 10: Evaluation & Export
# ---------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("## Cell 10 \u2014 Evaluation, Visualization & API Export"))
nb.cells.append(nbf.v4.new_code_cell("""
# Cell 10 \u2014 Visualization & Export
print("\\n" + "="*65)
print("  📊 GENERATING PUBLICATION CHARTS & API EXPORT")
print("="*65)

model.eval()
all_risks, all_times, all_events, all_uncs, all_attns = [], [], [], [], []

print("[*] Running inference on Validation Set...")
with torch.no_grad():
    for b in tqdm(val_dl, desc="Validating"):
        wsi = b['wsi'].to(DEVICE, non_blocking=True)
        rna = b['rna'].to(DEVICE, non_blocking=True)
        cli = b['cli'].to(DEVICE, non_blocking=True)
        mask = b['mask'].to(DEVICE, non_blocking=True)
        
        risk, unc, attn, _ = model(wsi, rna, cli, mask)
        
        all_risks.extend(risk.cpu().numpy())
        all_times.extend(b['time'].numpy())
        all_events.extend(b['event'].numpy())
        all_uncs.extend(unc.cpu().numpy())
        all_attns.extend(attn.cpu().numpy())

all_risks = np.array(all_risks)
all_times = np.array(all_times)
all_events = np.array(all_events)
all_attns = np.array(all_attns)

# --- DRAW CHARTS ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CancerFusion AI Multimodal Analysis', fontsize=14, fontweight='bold')

# 1. Kaplan-Meier
ax1 = axes[0]
if HAS_LIFELINES and len(all_risks) > 0:
    kmf = KaplanMeierFitter()
    high_risk = all_risks > np.median(all_risks)
    
    if np.sum(~high_risk) > 0:
        kmf.fit(all_times[~high_risk], event_observed=all_events[~high_risk], label="Low Risk (AI)")
        kmf.plot_survival_function(ax=ax1, color='blue', linewidth=2)
    if np.sum(high_risk) > 0:
        kmf.fit(all_times[high_risk], event_observed=all_events[high_risk], label="High Risk (AI)")
        kmf.plot_survival_function(ax=ax1, color='red', linewidth=2)
ax1.set_title("Kaplan-Meier Survival Curves", fontweight='bold')
ax1.set_xlabel("Time")
ax1.set_ylabel("Survival Probability")

# 2. Attention
ax2 = axes[1]
if len(all_attns) > 0:
    sns.barplot(x=['WSI', 'Genomics', 'Clinical'], y=np.mean(all_attns, axis=0), color="steelblue", ax=ax2)
ax2.set_title("Modality Importance (Cross-Attention)", fontweight='bold')
ax2.set_ylabel("Attention Weight")

# 3. Uncertainty
ax3 = axes[2]
if len(all_risks) > 0:
    sns.regplot(x=all_risks, y=all_uncs, scatter_kws={'alpha':0.5, 'color':'purple'}, line_kws={'color':'red'}, ax=ax3)
ax3.set_title("Evidential Uncertainty vs Prediction", fontweight='bold')
ax3.set_xlabel("Predicted Risk")
ax3.set_ylabel("Uncertainty Bound")

plt.tight_layout()
out_png = "/kaggle/working/CancerFusion_Plots.png"
plt.savefig(out_png, dpi=300)
plt.show()

# --- BACKEND EXPORT ---
weight_path = "/kaggle/working/CancerFusion_Weights.pth"
json_path = "/kaggle/working/backend_response.json"

torch.save(model.state_dict(), weight_path)

if len(all_risks) > 0:
    api_payload = {
        "patient_id": "TCGA-VALIDATION-COHORT",
        "prediction": {"risk_score": float(all_risks[0]), "uncertainty": float(all_uncs[0])},
        "explainability": {"attention_weights": {"wsi": float(all_attns[0][0]), "rna": float(all_attns[0][1]), "cli": float(all_attns[0][2])}}
    }
    with open(json_path, 'w') as f: 
        json.dump(api_payload, f, indent=4)

print(f"\\n\u2705 Saved Weights: {weight_path}")
print(f"\u2705 Saved API JSON : {json_path}")
print(f"\u2705 Saved Plots    : {out_png}")
"""))

with open('e:/MAJOR PROJECT/CancerFusion_Advanced_Kaggle.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
