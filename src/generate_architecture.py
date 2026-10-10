import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create high-resolution figure
fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Styling constants
COLOR_INPUT = "#ebf5fb"
COLOR_ENC   = "#d4efdf"
COLOR_FUS   = "#f5eef8"
COLOR_HEAD  = "#fdebd0"
COLOR_LOSS  = "#f2d7d5"
BORDER_DARK = "#2c3e50"

def draw_box(x, y, w, h, title, subtitle, bg_color, border_color="#2c3e50", title_color="#1a252f"):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.15",
                                 facecolor=bg_color, edgecolor=border_color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.65, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=title_color)
    if subtitle:
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=7.8, color="#566573", style='italic')

def draw_arrow(x1, y1, x2, y2, label=None, label_pos=(0, 0)):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#34495e", lw=1.5, mutation_scale=14))
    if label:
        ax.text(label_pos[0], label_pos[1], label, fontsize=7.5, color="#2c3e50", fontweight='bold', ha='center')

# --- HEADER TITLE ---
ax.text(8, 9.6, "CancerFusion AI: 4-Modality Precision Oncology Architecture",
        ha='center', va='center', fontsize=15, fontweight='bold', color="#1a252f")
ax.text(8, 9.25, "End-to-End Deep Representation Learning, Safe Co-Attention Fusion & Evidential Uncertainty",
        ha='center', va='center', fontsize=10, color="#566573")

# --- COLUMN BACKGROUND PANELS (LIFECYCLE STAGES) ---
stages = [
    ("Stage 1: Multimodal Intake", 0.5, 0.5, 3.2, 8.3, "#f8f9fa"),
    ("Stage 2: Unimodal Encoders", 4.1, 0.5, 3.6, 8.3, "#fdfefe"),
    ("Stage 3: Cross-Modal Co-Attention", 8.1, 0.5, 3.8, 8.3, "#fbfcfc"),
    ("Stage 4: Evidential Head & Loss", 12.3, 0.5, 3.2, 8.3, "#fef9e7")
]

for name, px, py, pw, ph, bg in stages:
    panel = patches.FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0.1,rounding_size=0.2",
                                  facecolor=bg, edgecolor="#bdc3c7", linewidth=1.2, linestyle="--")
    ax.add_patch(panel)
    ax.text(px + pw/2, py + ph - 0.35, name, ha='center', va='center', fontsize=10, fontweight='bold', color="#34495e")

# --- ROW 1: HISTOPATHOLOGY (WSI) ---
draw_box(0.8, 6.9, 2.6, 1.2, "Whole Slide Images (WSI)", "Gigapixel Patches (N x 768)", COLOR_INPUT)
draw_box(4.4, 6.9, 3.0, 1.2, "Gated-MIL & Concept Layer", "Attention Pooling (Ilse et al.)", COLOR_ENC)
draw_arrow(3.4, 7.5, 4.4, 7.5)

# --- ROW 2: RADIOLOGY (MRI) ---
draw_box(0.8, 4.9, 2.6, 1.2, "Radiology (MRI DICOM)", "3D Scans (S x 512, Slices)", COLOR_INPUT)
draw_box(4.4, 4.9, 3.0, 1.2, "3D Positional Transformer", "z-axis Depth Attention (P in R^S)", COLOR_ENC)
draw_arrow(3.4, 5.5, 4.4, 5.5)

# --- ROW 3: GENOMICS (RNA-SEQ) ---
draw_box(0.8, 2.9, 2.6, 1.2, "Transcriptomics (RNA-Seq)", "High-D Gene Expression", COLOR_INPUT)
draw_box(4.4, 2.9, 3.0, 1.2, "Self-Normalizing SNN", "SELU + Hallmark Pathways (K=16)", COLOR_ENC)
draw_arrow(3.4, 3.5, 4.4, 3.5)

# --- ROW 4: CLINICAL DEMOGRAPHICS ---
draw_box(0.8, 0.9, 2.6, 1.2, "Clinical Records (EHR)", "Stage, Age, Demographics", COLOR_INPUT)
draw_box(4.4, 0.9, 3.0, 1.2, "LayerNorm Dense Network", "Residual Linear Projections", COLOR_ENC)
draw_arrow(3.4, 1.5, 4.4, 1.5)

# --- STAGE 3: FUSION ENGINE ---
draw_box(8.4, 5.0, 3.2, 2.8, "4-Head Cross-Modal\nCo-Attention Transformer",
         "Multi-Head Dot-Product Attention\nwith Learned Modality Embeddings", COLOR_FUS, border_color="#8e44ad")

draw_box(8.4, 1.5, 3.2, 2.8, "Safe Dynamic Masking\nEngine (MNAR Safeguard)",
         "Injected -inf Attention Gating:\nA_{i,j} == 0 for Missing Scans\nDynamic Weight Re-Routing", COLOR_FUS, border_color="#8e44ad")

# Arrows from encoders to fusion
draw_arrow(7.4, 7.5, 8.4, 6.8, "e_wsi (R^64)", (7.9, 7.2))
draw_arrow(7.4, 5.5, 8.4, 5.8, "e_rad (R^64)", (7.9, 5.8))
draw_arrow(7.4, 3.5, 8.4, 3.8, "e_rna (R^64)", (7.9, 3.8))
draw_arrow(7.4, 1.5, 8.4, 2.2, "e_cli (R^64)", (7.9, 1.9))

# Link between Fusion boxes
draw_arrow(10.0, 4.3, 10.0, 5.0)

# --- STAGE 4: OUTPUT HEAD & LOSS ---
draw_box(12.6, 5.5, 2.6, 2.4, "Evidential Survival Head",
         "Dirichlet Prior Dir(alpha)\nRisk: R_hat = alpha_1 / S\nUncertainty: U = 2 / S\nAleatoric Var: Var(p_1)", COLOR_HEAD, border_color="#d35400")

draw_box(12.6, 1.2, 2.6, 3.2, "Multi-Task Loss Suite",
         "L_total = L_Cox\n+ lambda_edl * L_KL\n+ lambda_orth * L_orth\n+ lambda_sparse * ||c||_1", COLOR_LOSS, border_color="#c0392b")

# Arrow from fusion to evidential head
draw_arrow(11.6, 6.5, 12.6, 6.5, "h_fused (R^64)", (12.1, 6.8))

# Arrow from head to loss
draw_arrow(13.9, 5.5, 13.9, 4.4, "Predictions", (14.5, 5.0))

plt.tight_layout()
arch_path = "Figure_System_Architecture.png"
plt.savefig(arch_path, bbox_inches='tight')
plt.savefig("paper_figures/Figure_System_Architecture.png", bbox_inches='tight')
plt.close(fig)

print(f"[SUCCESS] High-resolution architecture diagram generated at: {arch_path}")
