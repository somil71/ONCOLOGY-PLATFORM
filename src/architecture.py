import torch
import torch.nn as nn
import torch.nn.functional as F


def _item_mask(x, valid_items=None):
    """Validate bag membership; infer legacy zero padding when unspecified."""
    if x.ndim != 3 or x.shape[1] == 0:
        raise ValueError("Image features require nonempty [batch, items, features].")
    valid = x.ne(0).any(dim=-1) if valid_items is None else valid_items
    if valid.shape != x.shape[:2] or not torch.all((valid == 0) | (valid == 1)):
        raise ValueError("Item mask must be binary with shape [batch, items].")
    valid = valid.bool()
    if not torch.isfinite(x.masked_select(valid.unsqueeze(-1).expand_as(x))).all():
        raise ValueError("Available image features must be finite.")
    return valid


def _safe_attention_mask(valid):
    safe = valid.clone()
    safe[~safe.any(dim=1), 0] = True
    return safe


# =====================================================================
# 🔬 CANCERFUSION RESEARCH PLATFORM: TIER-1 MULTIMODAL ONCOLOGY ARCHITECTURE
# =====================================================================


class GatedAttentionMIL(nn.Module):
    """
    Gated Attention Multiple Instance Learning (Ilse et al., ICML 2018 / CLAM).
    Learns patch pooling weights; tissue semantics require annotation-based validation.
    """

    def __init__(self, in_features=768, hidden_dim=256, out_dim=64, num_concepts=16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)

        # Unsupervised latent bottleneck (no annotated cellular semantics)
        self.concept_layer = nn.Linear(hidden_dim, num_concepts)
        self.concept_proj = nn.Linear(num_concepts, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x, valid_items=None):
        valid = _item_mask(x, valid_items)
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        # x: [B, N_patches, 768]
        B, N, D = x.shape
        h = self.proj(x)  # [B, N, 256]

        # Gated Attention Scoring: A = softmax(w^T * (tanh(Vh) * sigm(Uh)))
        A_V = self.attention_V(h)  # [B, N, 128]
        A_U = self.attention_U(h)  # [B, N, 128]
        raw_scores = self.attention_weights(A_V * A_U).squeeze(-1)  # [B, N]

        attn_weights = F.softmax(
            raw_scores.masked_fill(~_safe_attention_mask(valid), float("-inf")), dim=1
        )
        attn_weights = attn_weights.masked_fill(~valid, 0)

        # Attentive Bag Pooling
        bag_repr = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # [B, 256]

        # Concept Bottleneck Activation
        concepts = torch.sigmoid(self.concept_layer(bag_repr))  # [B, num_concepts]
        out = self.out_norm(self.concept_proj(concepts))  # [B, out_dim]

        out = out.masked_fill(~valid.any(dim=1, keepdim=True), 0)
        return out, concepts, attn_weights


class VolumetricSliceTransformer(nn.Module):
    """
    3D-Aware Positional Slice Transformer for Radiology (MRI DICOM Volumes).
    Uses sequence positions; physical anatomical ordering must be established in preprocessing.
    """

    def __init__(
        self, in_features=512, hidden_dim=256, out_dim=64, max_slices=32, nheads=4
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, max_slices, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.slice_attention = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim)
        )

    def forward(self, x, valid_items=None):
        valid = _item_mask(x, valid_items)
        if x.shape[1] > self.pos_embedding.shape[1]:
            raise ValueError("Slice count exceeds positional embedding length.")
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        # x: [B, S_slices, 512]
        B, S, D = x.shape
        h = self.proj(x)  # [B, S, 256]

        # Add 3D anatomical depth positional embeddings
        h = h + self.pos_embedding[:, :S, :]

        # Cross-slice context encoding
        h_trans = self.transformer(h, src_key_padding_mask=~_safe_attention_mask(valid))

        # Attentive slice pooling
        slice_scores = self.slice_attention(h_trans).squeeze(-1)  # [B, S]
        slice_weights = F.softmax(
            slice_scores.masked_fill(~_safe_attention_mask(valid), float("-inf")), dim=1
        )
        slice_weights = slice_weights.masked_fill(~valid, 0)

        vol_repr = torch.bmm(slice_weights.unsqueeze(1), h_trans).squeeze(1)  # [B, 256]
        out = self.out_proj(vol_repr)  # [B, out_dim]

        out = out.masked_fill(~valid.any(dim=1, keepdim=True), 0)
        return out, slice_weights


class PathwayGenomicSNN(nn.Module):
    """
    Self-Normalizing Neural Network (SNN) with SELU and AlphaDropout for High-Dimensional RNA-Seq.
    Uses an unsupervised latent bottleneck, not annotated biological pathways.
    """

    def __init__(self, in_features=1000, hidden_dim=256, out_dim=64, num_pathways=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.SELU(),
            nn.AlphaDropout(p=0.1),
            nn.Linear(512, hidden_dim),
            nn.SELU(),
        )
        # Unsupervised latent layer (no pathway supervision)
        self.pathway_layer = nn.Linear(hidden_dim, num_pathways)
        self.out_proj = nn.Sequential(
            nn.Linear(num_pathways, out_dim), nn.LayerNorm(out_dim)
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
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MultiModalCoAttentionTransformer(nn.Module):
    """
    Multi-head fusion with observed availability masks.
    Masking alone does not establish robustness to informative clinical missingness.
    """

    def __init__(self, embed_dim=64, num_modalities=4, nheads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities

        # Learnable Modality Type Embeddings
        self.modality_embeddings = nn.Parameter(
            torch.randn(1, num_modalities, embed_dim) * 0.02
        )

        # Multi-Head Co-Attention
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=nheads, dropout=0.1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Modality Gating Network (Learns adaptive modality reliability weights)
        self.gating = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.Tanh(), nn.Linear(32, 1)
        )

    def forward(self, modality_tokens, mask):
        # modality_tokens: [B, 4, embed_dim]
        # mask: [B, 4] (1.0 = present, 0.0 = missing)
        B, M, D = modality_tokens.shape

        if mask.shape != (B, M) or M != self.num_modalities:
            raise ValueError("Modality mask and token dimensions must match branches.")
        if not torch.all((mask == 0) | (mask == 1)) or not mask.bool().any(dim=1).all():
            raise ValueError(
                "Each case requires a binary mask and at least one available modality."
            )
        modality_tokens = modality_tokens.masked_fill(~mask.bool().unsqueeze(-1), 0)
        if not torch.isfinite(modality_tokens).all():
            raise ValueError("Available modality tokens must be finite.")

        # Add Modality Type Embeddings
        tokens = modality_tokens + self.modality_embeddings

        # Prepare Key Padding Mask for nn.MultiheadAttention (True = ignored/masked)
        key_padding_mask = mask == 0.0  # [B, 4]

        # Cross-Modal Multi-Head Self-Attention
        attn_out, attn_weights = self.mha(
            tokens, tokens, tokens, key_padding_mask=key_padding_mask
        )
        tokens = self.norm1(tokens + attn_out)
        tokens = self.ffn(tokens) + tokens  # [B, 4, D]

        # Learned Modality Gating (Which modality drives final prognosis)
        gate_logits = self.gating(tokens).squeeze(-1)  # [B, 4]
        # Mask out absent modalities with -inf
        gate_logits = gate_logits.masked_fill(mask == 0.0, float("-inf"))

        modality_importance = F.softmax(gate_logits, dim=1)  # [B, 4]
        # Zero out NaNs if any
        modality_importance = torch.where(
            torch.isnan(modality_importance),
            torch.zeros_like(modality_importance),
            modality_importance,
        )

        # Weighted Fusion
        fused = (tokens * modality_importance.unsqueeze(-1)).sum(dim=1)  # [B, D]

        return fused, modality_importance, attn_weights


class EvidentialSurvivalHead(nn.Module):
    """
    Dempster-Shafer Subjective Logic Head with Dirichlet Distribution.
    Legacy concentration diagnostics; uncertainty interpretation is unvalidated under censoring.
    """

    def __init__(self, in_features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32), nn.GELU(), nn.Linear(32, 2)
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
        aleatoric_unc = (alpha[:, 0] * alpha[:, 1]) / (
            (S.squeeze(-1) ** 2) * (S.squeeze(-1) + 1.0)
        )

        return risk, epistemic_unc, aleatoric_unc, alpha


class CancerFusionResearchPlatform(nn.Module):
    """
    Research prototype architecture:
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
        self.rad_encoder = VolumetricSliceTransformer(
            in_features=512, out_dim=embed_dim, max_slices=max_slices
        )
        self.rna_encoder = PathwayGenomicSNN(in_features=rna_dim, out_dim=embed_dim)
        self.cli_encoder = ClinicalDenseEncoder(in_features=cli_dim, out_dim=embed_dim)

        self.fusion = MultiModalCoAttentionTransformer(
            embed_dim=embed_dim, num_modalities=4, nheads=4
        )
        self.head = EvidentialSurvivalHead(in_features=embed_dim)

    def forward(self, wsi, rad, rna, cli, mask, wsi_valid=None, rad_valid=None):
        if mask.shape != (wsi.shape[0], 4):
            raise ValueError("Availability mask must have shape [batch, 4].")
        if not torch.all((mask == 0) | (mask == 1)) or not mask.bool().any(dim=1).all():
            raise ValueError(
                "Each case requires a binary mask and at least one available modality."
            )
        inputs = [wsi, rad, rna, cli]
        image_masks = [wsi_valid, rad_valid]
        for index, tensor in enumerate(inputs):
            keep = mask[:, index].bool().reshape((-1,) + (1,) * (tensor.ndim - 1))
            inputs[index] = tensor.masked_fill(~keep, 0)
            if index < 2:
                valid = (
                    _item_mask(inputs[index], image_masks[index])
                    & mask[:, index : index + 1].bool()
                )
                if (mask[:, index].bool() & ~valid.any(dim=1)).any():
                    raise ValueError(
                        "An available image modality must contain at least one valid item."
                    )
                image_masks[index] = valid
                inputs[index] = inputs[index].masked_fill(~valid.unsqueeze(-1), 0)
            if not torch.isfinite(inputs[index]).all():
                raise ValueError("Available feature values must be finite.")
        wsi, rad, rna, cli = inputs
        wsi_valid, rad_valid = image_masks
        # 1. Unimodal Encoding
        f_wsi, wsi_concepts, wsi_patch_attns = self.wsi_encoder(wsi, wsi_valid)
        f_rad, rad_slice_attns = self.rad_encoder(rad, rad_valid)
        f_rna, rna_pathways = self.rna_encoder(rna)
        f_cli = self.cli_encoder(cli)

        # 2. Modality Stack [B, 4, D]
        tokens = torch.stack([f_wsi, f_rad, f_rna, f_cli], dim=1)

        # 3. Safe Cross-Modal Fusion
        fused, modality_weights, cross_attns = self.fusion(tokens, mask)

        # 4. Evidential Prognosis
        risk, epistemic_unc, aleatoric_unc, alpha = self.head(fused)

        explainability = {
            "modality_weights": modality_weights,  # [B, 4]
            "cross_attns": cross_attns,  # [B, 4, 4]
            "wsi_patch_attns": wsi_patch_attns,  # [B, N_patches]
            "wsi_concepts": wsi_concepts,  # [B, 16]
            "rad_slice_attns": rad_slice_attns,  # [B, S_slices]
            "rna_pathways": rna_pathways,  # [B, 16]
            "unimodal_tokens": tokens,  # [B, 4, 64]
            "aleatoric_unc": aleatoric_unc,  # [B]
        }

        return risk, epistemic_unc, modality_weights, explainability, alpha


# =====================================================================
# 📐 COMPOSITE MULTI-TASK LOSS SUITE
# =====================================================================


def continuous_cox_loss(risk, times, events):
    """Batchwise negative Cox partial likelihood with Breslow handling of ties.

    The public interface accepts positive risk scores. Its logarithm is used as
    the Cox log-relative hazard. Censoring is indicated by event=0.
    """
    # Descending times put every risk set in a prefix of this batch.
    order = torch.argsort(times, descending=True)
    ordered_times = times[order]
    ordered_events = events[order].float()
    log_risk = torch.log(risk[order].clamp_min(1e-8))
    log_risk_sum = torch.logcumsumexp(log_risk, dim=0)
    # Breslow ties share the denominator containing all rows with that time.
    _, tie_counts = torch.unique_consecutive(ordered_times, return_counts=True)
    group_end = tie_counts.cumsum(0) - 1
    full_risk_set_logsum = log_risk_sum[group_end.repeat_interleave(tie_counts)]
    return -(
        (log_risk - full_risk_set_logsum) * ordered_events
    ).sum() / ordered_events.sum().clamp_min(1)


def modality_orthogonality_loss(unimodal_tokens, modality_mask=None):
    """
    Mean squared cosine similarity over available off-diagonal modality pairs.
    A single available modality contributes zero; missing branches are excluded.
    """
    B, M, D = unimodal_tokens.shape
    available = (
        torch.ones((B, M), dtype=torch.bool, device=unimodal_tokens.device)
        if modality_mask is None
        else modality_mask.bool()
    )
    if available.shape != (B, M):
        raise ValueError("Orthogonality mask must match token branches.")
    safe_tokens = unimodal_tokens.masked_fill(~available.unsqueeze(-1), 0)
    norm_tokens = F.normalize(safe_tokens, p=2, dim=-1)
    gram = torch.bmm(norm_tokens, norm_tokens.transpose(1, 2))
    pairs = available.unsqueeze(2) & available.unsqueeze(1)
    pairs = pairs & ~torch.eye(M, device=gram.device, dtype=torch.bool).unsqueeze(0)
    return gram.square().masked_fill(~pairs, 0).sum() / pairs.sum().clamp_min(1)


def evidence_strength_penalty(alpha):
    """Mean evidence fraction (S - K) / S for K Dirichlet concentrations.

    This is the pilot's existing evidence penalty. A censored survival uncertainty
    likelihood and clinical validation remain separate research tasks.
    """
    strength = alpha.sum(dim=1)
    return ((alpha - 1.0).sum(dim=1) / (strength + 1e-8)).mean()


def evidential_kl_regularization(alpha, events):
    """Compatibility entry point for the legacy evidence-strength penalty.

    The events argument is retained for existing callers. This function computes
    the same event-independent quantity as the pilot implementation.
    """
    return evidence_strength_penalty(alpha)


def composite_research_loss(
    risk,
    times,
    events,
    alpha,
    unimodal_tokens,
    rna_pathways,
    lambda_edl=0.01,
    lambda_orth=0.05,
    lambda_sparse=0.005,
    modality_mask=None,
):
    """
    Pilot composite objective with an evidence-strength penalty.
    The lambda_edl and l_edl names are retained for existing training callers.
    """
    l_cox = continuous_cox_loss(risk, times, events)
    l_edl = evidence_strength_penalty(alpha)
    l_orth = modality_orthogonality_loss(unimodal_tokens, modality_mask)
    per_case_sparse = torch.sum(torch.abs(rna_pathways), dim=-1)
    if modality_mask is None:
        l_sparse = per_case_sparse.mean()
    else:
        has_rna = modality_mask[:, 2].bool()
        l_sparse = per_case_sparse.masked_fill(
            ~has_rna, 0
        ).sum() / has_rna.sum().clamp_min(1)

    total_loss = (
        l_cox
        + (lambda_edl * l_edl)
        + (lambda_orth * l_orth)
        + (lambda_sparse * l_sparse)
    )
    return total_loss, {
        "l_cox": l_cox.item(),
        "l_edl": l_edl.item(),
        "l_orth": l_orth.item(),
        "l_sparse": l_sparse.item(),
    }
