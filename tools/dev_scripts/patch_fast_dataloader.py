import nbformat as nbf

notebook_path = "e:/MAJOR PROJECT/CancerFusion_Advanced_Kaggle.ipynb"

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Patch the glob search to be O(1) folder targeting instead of O(N) recursive
for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'patch_paths = glob.glob(f"{self.wsi_folder}/**/*{patient_id}*/**/*"' in cell.source:
            old_code = '''patch_paths = glob.glob(f"{self.wsi_folder}/**/*{patient_id}*/**/*", recursive=True)'''
            new_code = '''# Extremely fast targeted folder search
        patch_paths = glob.glob(f"{self.wsi_folder}/{patient_id}/**/*", recursive=True)'''
            cell.source = cell.source.replace(old_code, new_code)

# Write it back
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
    
print("Successfully patched dataloader for high-speed I/O.")
