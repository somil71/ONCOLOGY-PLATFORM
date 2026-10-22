import json

notebook_path = "CancerFusion_Phase2_PanCancer.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 2: Auto-detecting directory
cell_2_code = """CFG = {
    'dataset': 'breast-cancer-vision-and-genomic-fusion-ml-ready',
    'dir_multimodal': '/kaggle/input/breast-cancer-vision-and-genomic-fusion-ml-ready',
    'batch_size': 32,
    'epochs': 50,
    'lr': 1e-3,
    'max_patches': 20, # Optimized to fit 4 modalities smoothly in GPU RAM
    'num_workers': 0
}

# Auto-Locate Dataset Path in Kaggle (handles both direct and nested mounts)
possible_paths = [
    CFG['dir_multimodal'],
    '/kaggle/input/datasets/sepehreslamimoghadam/breast-cancer-vision-and-genomic-fusion-ml-ready',
    '/kaggle/input/sepehreslamimoghadam/breast-cancer-vision-and-genomic-fusion-ml-ready',
] + glob.glob('/kaggle/input/*breast-cancer*') + glob.glob('/kaggle/input/*/*breast-cancer*')

for p in possible_paths:
    if os.path.exists(p):
        CFG['dir_multimodal'] = p
        break

print(f"[*] Multimodal Dataset Root Auto-Detected At: {CFG['dir_multimodal']}")
"""

nb['cells'][2]['source'] = [line + '\n' for line in cell_2_code.split('\n')[:-1]] + [cell_2_code.split('\n')[-1]]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated Cell 2 in CancerFusion_Phase2_PanCancer.ipynb with dynamic auto-mount discovery.")
