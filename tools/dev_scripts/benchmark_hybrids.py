import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

# --- 1. THE FLAWED DATASET GENERATOR ---
def generate_flawed_dataset(n_samples, is_test=False):
    torch.manual_seed(42 if not is_test else 99)
    
    # Latent true risk
    true_risk = torch.rand(n_samples)
    
    # Base features
    wsi = torch.randn(n_samples, 128) * true_risk.unsqueeze(1)
    rna = torch.randn(n_samples, 256) * true_risk.unsqueeze(1)
    cli = torch.randn(n_samples, 15)
    
    # Bias: Demographic (cli feature 0)
    # Train: 85% Group A (0). Test: 50% Group B (1)
    demo_prob = 0.5 if is_test else 0.15
    demographics = (torch.rand(n_samples) < demo_prob).float()
    cli[:, 0] = demographics
    
    # If Group B (1), inject baseline shift to RNA
    rna[demographics == 1.0] += 2.0 
    
    # Flaw 1: MNAR (Missing Not At Random)
    # If a patient is high risk (>0.7), 60% chance they miss Genomic (RNA) sequencing
    missing_mask = torch.ones(n_samples, 3)
    high_risk_idx = true_risk > 0.7
    missing_rna_probs = torch.rand(high_risk_idx.sum())
    missing_mask[high_risk_idx, 1] = (missing_rna_probs > 0.6).float()
    
    # Zero out dropped data
    rna = rna * missing_mask[:, 1].unsqueeze(1)
    
    # Flaw 2: Sensor Noise in Test Set (WSI Artifacts)
    if is_test:
        noise_idx = torch.rand(n_samples) < 0.10
        wsi[noise_idx] += torch.randn(noise_idx.sum(), 128) * 5.0
        
    # Survival Time Generation (Higher risk = shorter time)
    time_arr = 100.0 - (true_risk * 80.0) + torch.randn(n_samples)*5
    event = (torch.rand(n_samples) < 0.8).float() # 80% mortality event rate
    
    return wsi, rna, cli, missing_mask, time_arr, event

train_data = generate_flawed_dataset(3000, is_test=False)
test_data = generate_flawed_dataset(1000, is_test=True)

# --- 2. THE ARCHITECTURES ---
class BaseEncoders(nn.Module):
    def __init__(self):
        super().__init__()
        self.wsi_enc = nn.Linear(128, 64)
        self.rna_enc = nn.Linear(256, 64)
        self.cli_enc = nn.Linear(15, 64)
    def forward(self, wsi, rna, cli):
        return F.relu(self.wsi_enc(wsi)), F.relu(self.rna_enc(rna)), F.relu(self.cli_enc(cli))

# Model A: Baseline Concatenation
class ModelA_Concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = BaseEncoders()
        self.head = nn.Linear(64*3, 1)
    def forward(self, wsi, rna, cli, mask):
        w, r, c = self.enc(wsi, rna, cli)
        out = self.head(torch.cat([w, r, c], dim=1))
        return out.squeeze()

# Model B: Pure Attention
class ModelB_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = BaseEncoders()
        self.attn = nn.Linear(64, 1)
        self.head = nn.Linear(64, 1)
    def forward(self, wsi, rna, cli, mask):
        w, r, c = self.enc(wsi, rna, cli)
        stacked = torch.stack([w, r, c], dim=1)
        scores = self.attn(stacked).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(fused).squeeze()

# Model C: Hybrid (Attention + Evidential)
class ModelC_Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = BaseEncoders()
        self.attn = nn.Linear(64, 1)
        self.evidential = nn.Linear(64, 2)
    def forward(self, wsi, rna, cli, mask):
        w, r, c = self.enc(wsi, rna, cli)
        stacked = torch.stack([w, r, c], dim=1)
        scores = self.attn(stacked).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        
        evidence = F.softplus(self.evidential(fused))
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        risk = alpha[:, 1] / S.squeeze()
        return risk

# --- 3. TRAINING AND EVALUATION ---
def cox_loss(risk, times, events):
    sorted_idx = torch.argsort(times, descending=True)
    risk, events = risk[sorted_idx], events[sorted_idx]
    hr = torch.exp(risk)
    log_risk = torch.log(torch.cumsum(hr, dim=0))
    return -torch.sum((risk - log_risk) * events) / (torch.sum(events) + 1e-8)

def c_index_fast(risk_scores, times, events):
    # Quick O(N^2) concordance index implementation
    n = len(times)
    concordant = 0
    permissible = 0
    for i in range(n):
        if events[i] == 1:
            for j in range(n):
                if times[i] < times[j]:
                    permissible += 1
                    # Note: risk is inverted (higher risk = lower time)
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] == risk_scores[j]:
                        concordant += 0.5
    return concordant / permissible if permissible > 0 else 0.5

def train_and_eval(model, name, epochs=15):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        risk = model(train_data[0], train_data[1], train_data[2], train_data[3])
        loss = cox_loss(risk, train_data[4], train_data[5])
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        test_risk = model(test_data[0], test_data[1], test_data[2], test_data[3])
    
    # Calculate C-Index using our fast function
    c_idx = c_index_fast(test_risk.numpy(), test_data[4].numpy(), test_data[5].numpy())
        
    return c_idx

print("Running Benchmark on Highly Biased, Flawed Test Set...")
print("="*60)
score_a = train_and_eval(ModelA_Concat(), "Model A (Baseline)")
print(f"Model A (Baseline Concat)   C-Index: {score_a:.4f}")

score_b = train_and_eval(ModelB_Attention(), "Model B (Attention)")
print(f"Model B (Attention Fusion)  C-Index: {score_b:.4f}")

score_c = train_and_eval(ModelC_Hybrid(), "Model C (Hybrid)")
print(f"Model C (Ultimate Hybrid)   C-Index: {score_c:.4f}")
print("="*60)
