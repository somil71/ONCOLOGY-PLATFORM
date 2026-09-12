import nbformat as nbf

notebook_path = "e:/MAJOR PROJECT/CancerFusion_Phase2_PanCancer.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

old_img_discovery = '''try:
            sample_patient = glob.glob(f"{CFG['dir_multimodal']}/**/TCGA-*", recursive=True)[0]
            self.img_folder = os.path.dirname(sample_patient)
        except:
            self.img_folder = CFG['dir_multimodal']'''

new_img_discovery = '''try:
            # FAST O(1) Discovery (prevents Kaggle network drive I/O stall)
            paths = glob.glob(f"{CFG['dir_multimodal']}/*/TCGA-*") + \\
                    glob.glob(f"{CFG['dir_multimodal']}/*/*/TCGA-*") + \\
                    glob.glob(f"{CFG['dir_multimodal']}/*/*/*/TCGA-*")
            self.img_folder = os.path.dirname(paths[0]) if paths else CFG['dir_multimodal']
        except:
            self.img_folder = CFG['dir_multimodal']'''

old_csv_discovery = '''all_csvs = glob.glob(f"{CFG['dir_multimodal']}/**/*.csv", recursive=True)'''

new_csv_discovery = '''# FAST O(1) CSV Discovery (prevents Kaggle network drive I/O stall)
all_csvs = glob.glob(f"{CFG['dir_multimodal']}/*.csv") + glob.glob(f"{CFG['dir_multimodal']}/*/*.csv")'''

for cell in nb.cells:
    if cell.cell_type == 'code':
        if old_img_discovery in cell.source:
            cell.source = cell.source.replace(old_img_discovery, new_img_discovery)
        if old_csv_discovery in cell.source:
            cell.source = cell.source.replace(old_csv_discovery, new_csv_discovery)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Successfully patched I/O stall issue.")
