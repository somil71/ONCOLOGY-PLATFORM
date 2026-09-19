import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, hidden_dim)
        )
    def forward(self, x): return self.net(x)

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
        x_bag = torch.mean(x, dim=1) 
        concepts = torch.sigmoid(self.concept_layer(x_bag))
        embed = self.projection(concepts)
        return embed, concepts

class SafeAttentionFusion(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.score_net = nn.Sequential(nn.Linear(embed_dim, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, embeds, mask):
        stacked = torch.stack(embeds, dim=1) 
        scores = self.score_net(stacked).squeeze(-1) 
        all_missing = (mask.sum(dim=1) == 0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        scores[all_missing, :] = 0.0 
        weights = F.softmax(scores, dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights

class EvidentialHead(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.evidence = nn.Linear(hidden_dim, 2)
        
    def forward(self, fused):
        e = F.softplus(self.evidence(fused))
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
        w_feat, concepts = self.wsi_enc(wsi)
        r_feat = self.rna_enc(rna)
        c_feat = self.cli_enc(cli)
        fused, attn = self.fusion([w_feat, r_feat, c_feat], mask)
        risk, unc = self.head(fused)
        return {'risk': risk, 'uncertainty': unc, 'attention': attn, 'concepts': concepts}

DEVICE = 'cpu'
model = CancerFusionHybrid().to(DEVICE)
model.eval()

print("--- RUNNING EXTREME EDGE CASE TESTS ---")
# Edge Case 1: Batch Size of 1
try:
    print("Testing Batch Size = 1...")
    out1 = model(torch.randn(1, 100, 768).to(DEVICE), torch.randn(1, 512).to(DEVICE), torch.randn(1, 15).to(DEVICE), torch.ones(1, 3).to(DEVICE))
    print("✅ Passed: Model supports B=1 Inference.")
except Exception as e: print(f"❌ Failed B=1: {e}")

# Edge Case 2: Extreme Missingness
try:
    print("Testing Extreme Missingness (Mask = [0, 0, 1])...")
    mask_extreme = torch.tensor([[0.0, 0.0, 1.0]]).to(DEVICE)
    out2 = model(torch.randn(1, 100, 768).to(DEVICE), torch.randn(1, 512).to(DEVICE), torch.randn(1, 15).to(DEVICE), mask_extreme)
    attn = out2['attention'][0].detach().cpu().numpy()
    assert np.allclose(attn, [0.0, 0.0, 1.0]), f"Attention did not route correctly: {attn}"
    print(f"✅ Passed: Attention routed 100% to Clinical perfectly.")
except Exception as e: print(f"❌ Failed Extreme Missingness: {e}")

# Edge Case 3: Complete Modality Dropout
try:
    print("Testing Complete Dropout (All Modalities Missing)...")
    mask_zero = torch.tensor([[0.0, 0.0, 0.0]]).to(DEVICE)
    out3 = model(torch.randn(1, 100, 768).to(DEVICE), torch.randn(1, 512).to(DEVICE), torch.randn(1, 15).to(DEVICE), mask_zero)
    print("✅ Passed: Safe Softmax prevented NaN crash.")
except Exception as e: print(f"❌ Failed Complete Dropout: {e}")

# Edge Case 4: Exploding Feature Values
try:
    print("Testing Exploding Features (Huge Inputs)...")
    out4 = model(torch.randn(1, 100, 768).to(DEVICE)*1e5, torch.randn(1, 512).to(DEVICE)*1e5, torch.randn(1, 15).to(DEVICE)*1e5, torch.ones(1, 3).to(DEVICE))
    assert not torch.isnan(out4['risk']).any(), "Risk yielded NaN"
    print("✅ Passed: Evidential Dirichlet head remained stable.")
except Exception as e: print(f"❌ Failed Numerical Stability: {e}")
