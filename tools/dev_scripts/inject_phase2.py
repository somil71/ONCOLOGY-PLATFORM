import nbformat as nbf
import os

notebook_path = "e:/MAJOR PROJECT/CancerFusion_Advanced_Kaggle.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Define the new Phase 2 Cell
phase_2_cell = nbf.v4.new_code_cell('''# =================================================================
# 🔎 PHASE 2: MICRO-EXPLAINABILITY (GRAD-CAM / SALIENCY)
# =================================================================
print("\\n" + "="*65)
print("  🔎 PHASE 2: PIXEL-LEVEL & GENE-LEVEL EXPLAINABILITY")
print("="*65)

model.eval()

# Grab a single patient from the validation set
sample = next(iter(val_dl))

# Isolate the tensors and track gradients for explainability
wsi_sample = sample['wsi'][0:1].clone().requires_grad_(True).to(DEVICE)
rna_sample = sample['rna'][0:1].clone().requires_grad_(True).to(DEVICE)
cli_sample = sample['cli'][0:1].clone().requires_grad_(True).to(DEVICE)
mask_sample = sample['mask'][0:1].clone().to(DEVICE)

# Forward pass
risk, uncertainty, attn = model(wsi_sample, rna_sample, cli_sample, mask_sample)

# Backward pass to calculate Gradients (How much did each pixel/gene affect the risk score?)
model.zero_grad()
risk.backward()

# 1. Process Image Saliency Map (Approximation of Grad-CAM)
# We take the absolute gradient of the image pixels across all channels
img_saliency = wsi_sample.grad[0, 0].abs().mean(dim=0).cpu().numpy()
original_img = wsi_sample[0, 0].detach().cpu().permute(1, 2, 0).numpy()
# Normalize for visualization
original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8)

# 2. Process Genomic Feature Importance (Approximation of SHAP)
gene_importance = rna_sample.grad[0].abs().cpu().numpy()
top_gene_indices = np.argsort(gene_importance)[-5:] # Top 5 driver genes

# 3. Plotting the Phase 2 Dashboard
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot A: Original Image
axes[0].imshow(original_img)
axes[0].set_title("Original Tumour Scan")
axes[0].axis('off')

# Plot B: Saliency Heatmap (Where the AI looked)
im = axes[1].imshow(img_saliency, cmap='hot')
axes[1].set_title("Radiology/WSI Saliency Map")
axes[1].axis('off')
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

# Plot C: Genomic SHAP Importance
axes[2].barh([f"Gene_Idx_{i}" for i in top_gene_indices], gene_importance[top_gene_indices], color='#2ecc71')
axes[2].set_title("Top 5 Genomic Drivers (Risk Influence)")
axes[2].set_xlabel("Gradient Importance")

plt.tight_layout()
plt.savefig("/kaggle/working/Phase2_MicroExplainability.png", dpi=300, bbox_inches='tight')
plt.show()

print("✅ Phase 2 Micro-Explainability Maps Generated Successfully.")''')

# Check if Phase 2 is already added to avoid duplicates
if "PHASE 2: MICRO-EXPLAINABILITY" not in nb.cells[-1].source:
    # Add a markdown header for the cell
    md_cell = nbf.v4.new_markdown_cell("## Cell 11 — Phase 2: Micro-Explainability (Grad-CAM & Genomic SHAP)\nThis cell unlocks Phase 2. It calculates the backpropagated gradients to determine exactly which pixels in the tumour image, and which specific genetic features, caused the AI to predict the high/low risk score.")
    nb.cells.extend([md_cell, phase_2_cell])

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Successfully injected Phase 2 Micro-Explainability into the Notebook.")
