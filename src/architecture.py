import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =====================================================================
# 🔬 CANCERFUSION RESEARCH PLATFORM: TIER-1 MULTIMODAL ONCOLOGY ARCHITECTURE
# =====================================================================

class GatedAttentionMIL(nn.Module):
    """
    Gated Attention Multiple Instance Learning (Ilse et al., ICML 2018 / CLAM).
    Learns to attend to invasive malignant tumor patches while downweighting normal stroma.
    """
    def __init__(self, in_features=768, hidden_dim=256, out_dim=64, num_concepts=16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)
        
        # Histological Concept Bottleneck (Interpretable cellular phenotypes)
        self.concept_layer = nn.Linear(hidden_dim, num_concepts)
        self.concept_proj = nn.Linear(num_concepts, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: [B, N_patches, 768]
        B, N, D = x.shape
        h = self.proj(x)  # [B, N, 256]
        
        # Gated Attention Scoring: A = softmax(w^T * (tanh(Vh) * sigm(Uh)))
        A_V = self.attention_V(h)  # [B, N, 128]
        A_U = self.attention_U(h)  # [B, N, 128]
        raw_scores = self.attention_weights(A_V * A_U).squeeze(-1)  # [B, N]
        
        attn_weights = F.softmax(raw_scores, dim=1)  # [B, N]
        
        # Attentive Bag Pooling
        bag_repr = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # [B, 256]
        
        # Concept Bottleneck Activation
        concepts = torch.sigmoid(self.concept_layer(bag_repr))  # [B, num_concepts]
        out = self.out_norm(self.concept_proj(concepts))  # [B, out_dim]
        
        return out, concepts, attn_weights


class VolumetricSliceTransformer(nn.Module):
    """
    3D-Aware Positional Slice Transformer for Radiology (MRI DICOM Volumes).
    Models z-axis anatomical lesion continuity across medical scan depth.
    """
    def __init__(self, in_features=512, hidden_dim=256, out_dim=64, max_slices=32, nheads=4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, max_slices, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nheads, dim_feedforward=512,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.slice_attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # x: [B, S_slices, 512]
        B, S, D = x.shape
        h = self.proj(x)  # [B, S, 256]
        
        # Add 3D anatomical depth positional embeddings
        h = h + self.pos_embedding[:, :S, :]
        
        # Cross-slice context encoding
        h_trans = self.transformer(h)  # [B, S, 256]
        
        # Attentive slice pooling
        slice_scores = self.slice_attention(h_trans).squeeze(-1)  # [B, S]
        slice_weights = F.softmax(slice_scores, dim=1)  # [B, S]
        
        vol_repr = torch.bmm(slice_weights.unsqueeze(1), h_trans).squeeze(1)  # [B, 256]
        out = self.out_proj(vol_repr)  # [B, out_dim]
        
        return out, slice_weights


class PathwayGenomicSNN(nn.Module):
    """
    Self-Normalizing Neural Network (SNN) with SELU and AlphaDropout for High-Dimensional RNA-Seq.
    Incorporates a Sparse Biological Hallmark Concept Bottleneck.
    """
    def __init__(self, in_features=1000, hidden_dim=256, out_dim=64, num_pathways=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.SELU(),
            nn.AlphaDropout(p=0.1),
            nn.Linear(512, hidden_dim),
            nn.SELU()
        )
        # Biological Pathway Concept Layer (e.g., DNA Repair, Angiogenesis, Apoptosis)
        self.pathway_layer = nn.Linear(hidden_dim, num_pathways)
        self.out_proj = nn.Sequential(
            nn.Linear(num_pathways, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # x: [B, D_rna]
        h = self.encoder(x)
        pathway_activations = torch.sigmoid(self.pathway_layer(h))  # [B, num_pathways]
        out = self.out_proj(pathway_activations)  # [B, out_dim]
        return out, pathway_activations


class ClinicalDenseEncoder(nn.Module):
    """
    LayerNorm Tabular Encoder with Residual Feed-Forward Blocks for Clinical/Demographic Data.
    """
    def __init__(self, in_features=15, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.net(x)


class MultiModalCoAttentionTransformer(nn.Module):
    """
    Multi-Head Cross-Modal Co-Attention Engine with Safe Dynamic Masking for MNAR Data.
    Fuses 4 heterogeneous diagnostic streams with information-theoretic attention gating.
    """
    def __init__(self, embed_dim=64, num_modalities=4, nheads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        # Learnable Modality Type Embeddings
        self.modality_embeddings = nn.Parameter(torch.randn(1, num_modalities, embed_dim) * 0.02)
        
        # Multi-Head Co-Attention
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nheads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Modality Gating Network (Learns adaptive modality reliability weights)
        self.gating = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, modality_tokens, mask):
        # modality_tokens: [B, 4, embed_dim]
        # mask: [B, 4] (1.0 = present, 0.0 = missing)
        B, M, D = modality_tokens.shape
        
        # Add Modality Type Embeddings
        tokens = modality_tokens + self.modality_embeddings
        
        # Prepare Key Padding Mask for nn.MultiheadAttention (True = ignored/masked)
        key_padding_mask = (mask == 0.0)  # [B, 4]
        
        # Safety fallback: If all modalities are missing for an extreme edge case
        all_missing = key_padding_mask.all(dim=1)
        if all_missing.any():
            key_padding_mask[all_missing, :] = False  # Prevent NaN division
            
        # Cross-Modal Multi-Head Self-Attention
        attn_out, attn_weights = self.mha(tokens, tokens, tokens, key_padding_mask=key_padding_mask)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.ffn(tokens) + tokens  # [B, 4, D]
        
        # Learned Modality Gating (Which modality drives final prognosis)
        gate_logits = self.gating(tokens).squeeze(-1)  # [B, 4]
        # Mask out absent modalities with -inf
        gate_logits = gate_logits.masked_fill(mask == 0.0, float('-inf'))
        if all_missing.any():
            gate_logits[all_missing, :] = 0.0
            
        modality_importance = F.softmax(gate_logits, dim=1)  # [B, 4]
        # Zero out NaNs if any
        modality_importance = torch.where(torch.isnan(modality_importance), torch.zeros_like(modality_importance), modality_importance)
        
        # Weighted Fusion
        fused = (tokens * modality_importance.unsqueeze(-1)).sum(dim=1)  # [B, D]
        
        return fused, modality_importance, attn_weights


class EvidentialSurvivalHead(nn.Module):
    """
    Dempster-Shafer Subjective Logic Head with Dirichlet Distribution.
    Quantifies Epistemic (Model Ambiguity) and Aleatoric (Inherent Data Variance) Uncertainty.
    """
    def __init__(self, in_features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )

    def forward(self, fused):
        # Evidence non-negative parameterization
        evidence = F.softplus(self.net(fused))
        alpha = evidence + 1.0  # Dirichlet parameters
        S = torch.sum(alpha, dim=1, keepdim=True)  # Dirichlet strength
        
        # Risk Score (Relative expectation of survival event)
        risk = alpha[:, 1] / S.squeeze(-1)
        
        # Epistemic Uncertainty: U = K / S (where K=2)
        epistemic_unc = 2.0 / S.squeeze(-1)
        
        # Aleatoric Uncertainty: Variance of the Dirichlet distribution
        aleatoric_unc = (alpha[:, 0] * alpha[:, 1]) / ((S.squeeze(-1) ** 2) * (S.squeeze(-1) + 1.0))
        
        return risk, epistemic_unc, aleatoric_unc, alpha


class CancerFusionResearchPlatform(nn.Module):
    """
    Complete Tier-1 Research Architecture:
    - Gated-MIL Histopathology (WSI)
    - 3D Volumetric Positional Transformer (Radiology MRI)
    - Pathway SNN with Sparse Concept Bottleneck (Genomics RNA)
    - LayerNorm Dense Encoder (Clinical)
    - Multi-Head Cross-Modal Co-Attention Engine
    - Dirichlet Evidential Head (Dual Uncertainty)
    """
    def __init__(self, rna_dim=1000, cli_dim=15, embed_dim=64, max_slices=32):
        super().__init__()
        self.wsi_encoder = GatedAttentionMIL(in_features=768, out_dim=embed_dim)
        self.rad_encoder = VolumetricSliceTransformer(in_features=512, out_dim=embed_dim, max_slices=max_slices)
        self.rna_encoder = PathwayGenomicSNN(in_features=rna_dim, out_dim=embed_dim)
        self.cli_encoder = ClinicalDenseEncoder(in_features=cli_dim, out_dim=embed_dim)
        
        self.fusion = MultiModalCoAttentionTransformer(embed_dim=embed_dim, num_modalities=4, nheads=4)
        self.head = EvidentialSurvivalHead(in_features=embed_dim)

    def forward(self, wsi, rad, rna, cli, mask):
        # 1. Unimodal Encoding
        f_wsi, wsi_concepts, wsi_patch_attns = self.wsi_encoder(wsi)
        f_rad, rad_slice_attns = self.rad_encoder(rad)
        f_rna, rna_pathways = self.rna_encoder(rna)
        f_cli = self.cli_encoder(cli)
        
        # 2. Modality Stack [B, 4, D]
        tokens = torch.stack([f_wsi, f_rad, f_rna, f_cli], dim=1)
        
        # 3. Safe Cross-Modal Fusion
        fused, modality_weights, cross_attns = self.fusion(tokens, mask)
        
        # 4. Evidential Prognosis
        risk, epistemic_unc, aleatoric_unc, alpha = self.head(fused)
        
        explainability = {
            'modality_weights': modality_weights,       # [B, 4]
            'cross_attns': cross_attns,                 # [B, 4, 4]
            'wsi_patch_attns': wsi_patch_attns,         # [B, N_patches]
            'wsi_concepts': wsi_concepts,               # [B, 16]
            'rad_slice_attns': rad_slice_attns,         # [B, S_slices]
            'rna_pathways': rna_pathways,               # [B, 16]
            'unimodal_tokens': tokens,                  # [B, 4, 64]
            'aleatoric_unc': aleatoric_unc              # [B]
        }
        
        return risk, epistemic_unc, modality_weights, explainability, alpha


# =====================================================================
# 📐 COMPOSITE MULTI-TASK LOSS SUITE
# =====================================================================

def continuous_cox_loss(risk, times, events):
    """
    Breslow-approximated Negative Partial Log-Likelihood for Cox Proportional Hazards.
    """
    events = events.float()
    ix = torch.argsort(times, descending=True)
    risk, events = risk[ix], events[ix]
    log_risk = torch.log(risk + 1e-8)
    cum_risk = torch.logcumsumexp(log_risk, dim=0)
    loss = -torch.sum((log_risk - cum_risk) * events) / (torch.sum(events) + 1e-8)
    return loss

def modality_orthogonality_loss(unimodal_tokens):
    """
    Frobenius Orthogonality Constraint across modalities to prevent feature redundancy:
    L_orth = sum_{i != j} || z_i^T z_j ||_F^2
    """
    B, M, D = unimodal_tokens.shape
    norm_tokens = F.normalize(unimodal_tokens, p=2, dim=-1)
    gram = torch.bmm(norm_tokens, norm_tokens.transpose(1, 2))
    identity = torch.eye(M, device=unimodal_tokens.device).unsqueeze(0).expand(B, -1, -1)
    orth_loss = torch.norm(gram - identity, p='fro') / B
    return orth_loss

def evidential_kl_regularization(alpha, events):
    """
    KL Divergence Regularization on Dirichlet prior for out-of-distribution control.
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    kl = torch.sum(alpha - 1.0, dim=1) / (S.squeeze(-1) + 1e-8)
    return torch.mean(kl)

def composite_research_loss(risk, times, events, alpha, unimodal_tokens, rna_pathways,
                            lambda_edl=0.01, lambda_orth=0.05, lambda_sparse=0.005):
    """
    Unified Tier-1 Objective Function:
    L_total = L_cox + lambda_edl * L_kl + lambda_orth * L_orth + lambda_sparse * L_l1
    """
    l_cox = continuous_cox_loss(risk, times, events)
    l_edl = evidential_kl_regularization(alpha, events)
    l_orth = modality_orthogonality_loss(unimodal_tokens)
    l_sparse = torch.mean(torch.sum(torch.abs(rna_pathways), dim=-1))
    
    total_loss = l_cox + (lambda_edl * l_edl) + (lambda_orth * l_orth) + (lambda_sparse * l_sparse)
    return total_loss, {'l_cox': l_cox.item(), 'l_edl': l_edl.item(), 'l_orth': l_orth.item(), 'l_sparse': l_sparse.item()}
