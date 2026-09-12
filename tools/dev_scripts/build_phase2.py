import nbformat as nbf
import os

print("Generating Phase 2 Architecture Notebook...")

nb = nbf.v4.new_notebook()

# ---------------------------------------------------------
# Cell 1: Setup & Imports
# ---------------------------------------------------------
cell_1 = nbf.v4.new_code_cell('''!pip install -q lifelines pydicom
import os, glob, json, math
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights, resnet18, ResNet18_Weights
from lifelines import KaplanMeierFitter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Version: {torch.__version__} | Device: {DEVICE}")''')

# ---------------------------------------------------------
# Cell 2: CFG & Pan-Cancer Settings
# ---------------------------------------------------------
cell_2 = nbf.v4.new_code_cell('''CFG = {
    'dataset': 'breast-cancer-vision-and-genomic-fusion-ml-ready', # Pan-Cancer Switch
    'dir_multimodal': '/kaggle/input/datasets/sepehreslamimoghadam/breast-cancer-vision-and-genomic-fusion-ml-ready',
    'batch_size': 32,
    'epochs': 50,
    'lr': 1e-3,
    'max_patches': 20, # Reduced to fit 4 modalities in RAM
    'num_workers': 0
}''')

# ---------------------------------------------------------
# Cell 3: Load Backbones (ViT + ResNet18)
# ---------------------------------------------------------
cell_3 = nbf.v4.new_code_cell('''print("[*] Loading WSI Backbone (ViT-B/16)...")
wsi_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
wsi_model.heads = nn.Identity()
wsi_model = wsi_model.to(DEVICE).eval()

print("[*] Loading Radiology Backbone (ResNet18)...")
rad_model = resnet18(weights=ResNet18_Weights.DEFAULT)
rad_model.fc = nn.Identity()
rad_model = rad_model.to(DEVICE).eval()

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])''')

# ---------------------------------------------------------
# Cell 4: 4-Modality Dataset
# ---------------------------------------------------------
cell_4 = nbf.v4.new_code_cell('''class PanCancerDataset(Dataset):
    def __init__(self, df, rna_df, cli_df):
        self.df = df
        self.rna_df = rna_df
        self.cli_df = cli_df
        # PanCancer Double Directory Fix
        self.img_folder = f"{CFG['dir_multimodal']}/MRI_and_SVS_Patches/MRI_and_SVS_Patches"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_id = self.df.iloc[idx]['Patient_ID']
        
        # Initialize Tensors (1 WSI, 2 RAD, 3 RNA, 4 CLI)
        wsi_tensor = torch.zeros((CFG['max_patches'], 768))
        rad_tensor = torch.zeros((CFG['max_patches'], 512))
        mask = torch.zeros(4) # [WSI, RAD, RNA, CLI]
        
        # 1. Image Modalities (WSI & Radiology)
        all_files = glob.glob(f"{self.img_folder}/{patient_id}/**/*", recursive=True)
        all_files = [p for p in all_files if os.path.isfile(p)]
        
        wsi_files = [p for p in all_files if p.endswith('.jpg') or p.endswith('.png')]
        rad_files = [p for p in all_files if not p.endswith('.jpg') and not p.endswith('.png')]
        
        # Process WSI
        if len(wsi_files) > 0:
            mask[0] = 1.0
            loaded = []
            for p in wsi_files[:CFG['max_patches']]:
                try:
                    loaded.append(img_transform(Image.open(p).convert('RGB')))
                except: pass
            if loaded:
                with torch.no_grad():
                    wsi_tensor[:len(loaded)] = wsi_model(torch.stack(loaded).to(DEVICE)).cpu()
                    
        # Process Radiology (DICOM)
        if len(rad_files) > 0:
            mask[1] = 1.0
            loaded = []
            for p in rad_files[:CFG['max_patches']]:
                try:
                    ds = pydicom.dcmread(p)
                    arr = ds.pixel_array.astype(float)
                    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
                    arr = (arr * 255).astype(np.uint8)
                    img = Image.fromarray(arr if len(arr.shape)==2 else arr[0]).convert('RGB')
                    loaded.append(img_transform(img))
                except: pass
            if loaded:
                with torch.no_grad():
                    rad_tensor[:len(loaded)] = rad_model(torch.stack(loaded).to(DEVICE)).cpu()

        # 2 & 3. RNA & Clinical
        rna = torch.tensor(self.rna_df.loc[patient_id].values, dtype=torch.float32) if patient_id in self.rna_df.index else torch.zeros(self.rna_df.shape[1])
        cli = torch.tensor(self.cli_df.loc[patient_id].values, dtype=torch.float32) if patient_id in self.cli_df.index else torch.zeros(self.cli_df.shape[1])
        if patient_id in self.rna_df.index: mask[2] = 1.0
        if patient_id in self.cli_df.index: mask[3] = 1.0
        
        # Target
        time = self.df.iloc[idx]['time']
        event = self.df.iloc[idx]['event']
        
        return {'wsi': wsi_tensor, 'rad': rad_tensor, 'rna': rna, 'cli': cli, 'mask': mask, 'time': time, 'event': event}''')

# ---------------------------------------------------------
# Cell 5: True 4-Way Architecture
# ---------------------------------------------------------
cell_5 = nbf.v4.new_code_cell('''class SafeAttentionFusion(nn.Module):
    def __init__(self, embed_dim=16, num_modalities=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, features, mask):
        B = features.size(0)
        q = self.query.expand(B, -1, -1)
        k = self.key(features)
        v = self.value(features)
        
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(k.size(-1))
        scores.masked_fill_(mask.unsqueeze(1) == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        
        fused = torch.bmm(attn, v).squeeze(1)
        return fused, attn.squeeze(1)

class EvidentialHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 2)
    def forward(self, x):
        evidence = F.softplus(self.fc(x))
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        risk = alpha[:, 1] / S.squeeze()
        uncertainty = 2 / S.squeeze()
        return risk, uncertainty

class CancerFusionPhase2(nn.Module):
    def __init__(self, rna_dim, cli_dim):
        super().__init__()
        self.wsi_proj = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, 16))
        self.rad_proj = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 16))
        self.rna_enc  = nn.Sequential(nn.Linear(rna_dim, 256), nn.AlphaDropout(0.1), nn.SELU(), nn.Linear(256, 16))
        self.cli_enc  = nn.Sequential(nn.Linear(cli_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 16))
        
        self.fusion = SafeAttentionFusion(embed_dim=16, num_modalities=4)
        self.head = EvidentialHead(in_features=16)

    def forward(self, wsi, rad, rna, cli, mask):
        f_wsi = self.wsi_proj(wsi.mean(dim=1))
        f_rad = self.rad_proj(rad.mean(dim=1))
        f_rna = self.rna_enc(rna)
        f_cli = self.cli_enc(cli)
        
        stacked = torch.stack([f_wsi, f_rad, f_rna, f_cli], dim=1)
        fused, attn = self.fusion(stacked, mask)
        risk, unc = self.head(fused)
        
        return risk, unc, attn

def cox_loss(risk, times, events):
    events = events.float()
    ix = torch.argsort(times, descending=True)
    risk, events = risk[ix], events[ix]
    log_risk = torch.log(risk + 1e-8)
    cum_risk = torch.logcumsumexp(log_risk, dim=0)
    loss = -torch.sum((log_risk - cum_risk) * events) / (torch.sum(events) + 1e-8)
    return loss''')

# ---------------------------------------------------------
# Cell 6 & 7: Loading CSVs & Train Loop
# ---------------------------------------------------------
cell_6 = nbf.v4.new_code_cell('''print("[*] Processing Phase 2 Pan-Cancer CSVs...")
cli_path = f"{CFG['dir_multimodal']}/Clinical_Treatment_Data.csv"
rna_path = f"{CFG['dir_multimodal']}/RNA_CNV_ModelReady.csv"

cli_df = pd.read_csv(cli_path, encoding='latin1').set_index('Patient_ID')
cli_df = cli_df.select_dtypes(include=[np.number]).fillna(0)

rna_df = pd.read_csv(rna_path).set_index('Patient_ID')
rna_df = rna_df.select_dtypes(include=[np.number]).fillna(0)

# Build Survival Target DataFrame
demo_path = f"{CFG['dir_multimodal']}/Clinical_Demographic_Data.csv"
demo_df = pd.read_csv(demo_path, encoding='latin1')
demo_df['event'] = demo_df['vital_status'].apply(lambda x: 1.0 if str(x).lower() == 'dead' else 0.0)
demo_df['time'] = demo_df['demographic_days_to_death'].fillna(demo_df['diagnoses_days_to_last_follow_up']).fillna(0)
target_df = demo_df[['Patient_ID', 'time', 'event']].drop_duplicates(subset=['Patient_ID'])

# Valid Patients
valid_patients = list(set(target_df['Patient_ID']) & (set(cli_df.index) | set(rna_df.index)))
target_df = target_df[target_df['Patient_ID'].isin(valid_patients)].reset_index(drop=True)

train_df = target_df.iloc[:int(0.8*len(target_df))]
val_df   = target_df.iloc[int(0.8*len(target_df)):]

train_dl = DataLoader(PanCancerDataset(train_df, rna_df, cli_df), batch_size=CFG['batch_size'], shuffle=True, num_workers=CFG['num_workers'])
val_dl   = DataLoader(PanCancerDataset(val_df, rna_df, cli_df), batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['num_workers'])

model = CancerFusionPhase2(rna_dim=rna_df.shape[1], cli_dim=cli_df.shape[1]).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-4)
print("✅ Phase 2 Data & Engine Ready!")''')

cell_7 = nbf.v4.new_code_cell('''print("\\n" + "="*65)
print("  🚀 IGNITING PHASE 2: 4-MODALITY TRAINING ENGINE")
print("="*65)

model.train()
for epoch in range(1, CFG['epochs']+1):
    ep_loss = 0
    pbar = tqdm(train_dl, desc=f'Epoch {epoch:02d}/{CFG["epochs"]}', leave=False)
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)
        wsi = batch['wsi'].to(DEVICE)
        rad = batch['rad'].to(DEVICE)
        rna = batch['rna'].to(DEVICE)
        cli = batch['cli'].to(DEVICE)
        mask= batch['mask'].to(DEVICE)
        times = batch['time'].to(DEVICE)
        events= batch['event'].to(DEVICE)
        
        risk, _, _ = model(wsi, rad, rna, cli, mask)
        loss = cox_loss(risk, times, events)
        
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            
    print(f"Epoch {epoch:02d}/{CFG['epochs']:02d} | Train Cox Loss: {ep_loss/max(len(train_dl),1):.4f}")

torch.save(model.state_dict(), '/kaggle/working/CancerFusion_Phase2_Weights.pth')
print("✅ Phase 2 Training Complete!")''')

# ---------------------------------------------------------
# Assemble and Save
# ---------------------------------------------------------
nb.cells = [
    nbf.v4.new_markdown_cell("# Phase 2: True 4-Modality Pan-Cancer Platform\nThis notebook implements the massive Phase 2 structural upgrade. It utilizes a dedicated ResNet18 backbone for Radiology, a Vision Transformer for Histopathology, and fuses 4 independent modalities via Cross-Attention."),
    cell_1, cell_2, cell_3, cell_4, cell_5, cell_6, cell_7
]

with open("e:/MAJOR PROJECT/CancerFusion_Phase2_PanCancer.ipynb", 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
    
print("Successfully generated Phase 2 Notebook.")
