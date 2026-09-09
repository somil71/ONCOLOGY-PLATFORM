import nbformat as nbf

notebook_path = "e:/MAJOR PROJECT/CancerFusion_Advanced_Kaggle.ipynb"

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Patch the num_workers to 0
for cell in nb.cells:
    if cell.cell_type == 'code':
        if "'num_workers': min(4, os.cpu_count() or 1)" in cell.source:
            cell.source = cell.source.replace(
                "'num_workers': min(4, os.cpu_count() or 1)", 
                "'num_workers': 0"
            )

# Write it back
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
    
print("Successfully forced num_workers to 0.")
