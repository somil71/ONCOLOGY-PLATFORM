import nbformat as nbf

notebook_path = "e:/MAJOR PROJECT/CancerFusion_Advanced_Kaggle.ipynb"

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Patch HAS_LIFELINES -> True
for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'HAS_LIFELINES' in cell.source:
            cell.source = cell.source.replace('if HAS_LIFELINES and len(all_risks) > 0:', 'if True and len(all_risks) > 0:')

# Write it back
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
    
print("Successfully patched lifelines.")
