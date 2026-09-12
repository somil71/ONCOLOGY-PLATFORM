import nbformat as nbf

notebook_path = "e:/MAJOR PROJECT/CancerFusion_Phase2_PanCancer.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

old_cell_6_content = '''cli_df = pd.read_csv(cli_path, encoding='latin1').set_index('Patient_ID')
cli_df = cli_df.select_dtypes(include=[np.number]).fillna(0)

rna_df = pd.read_csv(rna_path).set_index('Patient_ID')
rna_df = rna_df.select_dtypes(include=[np.number]).fillna(0)

# Build Survival Target DataFrame
demo_path = f"{CFG['dir_multimodal']}/Clinical_Demographic_Data.csv"
demo_df = pd.read_csv(demo_path, encoding='latin1')'''

new_cell_6_content = '''# Bulletproof Pan-Cancer ID Standardizer
def standardize_id(df):
    for col in ['Patient_ID', 'bcr_patient_barcode', 'patient_id', 'Patient ID', 'PATIENT_ID']:
        if col in df.columns:
            df = df.rename(columns={col: 'Patient_ID'})
            return df.set_index('Patient_ID')
    return df.set_index(df.columns[0]) # Fallback to first column

cli_df = standardize_id(pd.read_csv(cli_path, encoding='latin1'))
cli_df = cli_df.select_dtypes(include=[np.number]).fillna(0)

rna_df = standardize_id(pd.read_csv(rna_path, encoding='latin1'))
rna_df = rna_df.select_dtypes(include=[np.number]).fillna(0)

# Build Survival Target DataFrame
demo_path = f"{CFG['dir_multimodal']}/Clinical_Demographic_Data.csv"
demo_df = pd.read_csv(demo_path, encoding='latin1')
demo_df = standardize_id(demo_df).reset_index()'''

for cell in nb.cells:
    if cell.cell_type == 'code':
        if "cli_df = pd.read_csv(cli_path" in cell.source:
            cell.source = cell.source.replace(old_cell_6_content, new_cell_6_content)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Successfully patched Pan-Cancer CSV Loader.")
