import nbformat as nbf

notebook_path = "e:/MAJOR PROJECT/CancerFusion_Advanced_Kaggle.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Patch the folder path to account for the double 'MRI_and_SVS_Patches' directory
for cell in nb.cells:
    if cell.cell_type == 'code':
        if "self.wsi_folder = f\"{CFG['dir_multimodal']}/MRI_and_SVS_Patches\"" in cell.source:
            cell.source = cell.source.replace(
                "self.wsi_folder = f\"{CFG['dir_multimodal']}/MRI_and_SVS_Patches\"",
                "self.wsi_folder = f\"{CFG['dir_multimodal']}/MRI_and_SVS_Patches/MRI_and_SVS_Patches\""
            )

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
    
print("Successfully patched double folder path.")
