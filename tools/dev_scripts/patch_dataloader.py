import nbformat as nbf
import re

notebook_path = "e:/MAJOR PROJECT/CancerFusion_Advanced_Kaggle.ipynb"

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Patch the glob search
for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'patch_paths = glob.glob' in cell.source:
            old_line = 'patch_paths = glob.glob(f"{self.wsi_folder}/*{patient_id}*.*")'
            new_lines = '''patch_paths = glob.glob(f"{self.wsi_folder}/**/*{patient_id}*/**/*", recursive=True)
        patch_paths = [p for p in patch_paths if os.path.isfile(p)]'''
            cell.source = cell.source.replace(old_line, new_lines)

# Write it back
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
    
print("Successfully patched dataloader glob pattern.")
