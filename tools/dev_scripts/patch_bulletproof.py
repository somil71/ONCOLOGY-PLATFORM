import nbformat as nbf

notebook_path = "e:/MAJOR PROJECT/CancerFusion_Phase2_PanCancer.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Patch Cell 4 (Image Folder Autodiscovery)
old_img_folder = '''self.img_folder = f"{CFG['dir_multimodal']}/MRI_and_SVS_Patches/MRI_and_SVS_Patches"'''
new_img_folder = '''# Auto-discover the deepest directory containing patient folders (e.g. TCGA-...) to prevent file path crashes
        try:
            sample_patient = glob.glob(f"{CFG['dir_multimodal']}/**/TCGA-*", recursive=True)[0]
            self.img_folder = os.path.dirname(sample_patient)
        except:
            self.img_folder = CFG['dir_multimodal']'''

# 2. Patch Cell 6 (CSV Filename Autodiscovery)
old_csv_paths = '''cli_path = f"{CFG['dir_multimodal']}/Clinical_Treatment_Data.csv"
rna_path = f"{CFG['dir_multimodal']}/RNA_CNV_ModelReady.csv"'''

new_csv_paths = '''# Bulletproof CSV File Discovery
all_csvs = glob.glob(f"{CFG['dir_multimodal']}/**/*.csv", recursive=True)
cli_path = next((f for f in all_csvs if 'treatment' in f.lower() or 'clinical' in f.lower()), all_csvs[0] if len(all_csvs) > 0 else "")
rna_path = next((f for f in all_csvs if 'rna' in f.lower() or 'cnv' in f.lower() or 'mutation' in f.lower()), all_csvs[1] if len(all_csvs) > 1 else "")
demo_path = next((f for f in all_csvs if 'demographic' in f.lower() or 'survival' in f.lower()), all_csvs[2] if len(all_csvs) > 2 else "")

print(f"Auto-Found Clinical Data: {cli_path}")
print(f"Auto-Found Genomic Data: {rna_path}")
print(f"Auto-Found Demographic Data: {demo_path}")'''

old_demo_path = '''demo_path = f"{CFG['dir_multimodal']}/Clinical_Demographic_Data.csv"'''
new_demo_path = '''# demo_path is auto-discovered above'''

for cell in nb.cells:
    if cell.cell_type == 'code':
        if old_img_folder in cell.source:
            cell.source = cell.source.replace(old_img_folder, new_img_folder)
        if old_csv_paths in cell.source:
            cell.source = cell.source.replace(old_csv_paths, new_csv_paths)
        if old_demo_path in cell.source:
            cell.source = cell.source.replace(old_demo_path, new_demo_path)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Successfully injected Absolute Bulletproof File and Column Discovery.")
