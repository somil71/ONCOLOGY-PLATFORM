import nbformat as nbf
import os

notebook_path = "e:/MAJOR PROJECT/CancerFusion_Phase2_PanCancer.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

old_discovery = '''cli_path = next((f for f in all_csvs if 'treatment' in f.lower() or 'clinical' in f.lower()), all_csvs[0] if len(all_csvs) > 0 else "")
rna_path = next((f for f in all_csvs if 'rna' in f.lower() or 'cnv' in f.lower() or 'mutation' in f.lower()), all_csvs[1] if len(all_csvs) > 1 else "")
demo_path = next((f for f in all_csvs if 'demographic' in f.lower() or 'survival' in f.lower()), all_csvs[2] if len(all_csvs) > 2 else "")'''

new_discovery = '''# Smart Priority CSV Discovery
def get_best_csv(primary_kw, secondary_kw, files):
    for f in files:
        if primary_kw in f.lower() and secondary_kw in f.lower(): return f
    for f in files:
        if primary_kw in f.lower(): return f
    return files[0] if files else ""

cli_path = get_best_csv('clinical', 'treatment', all_csvs)
demo_path = get_best_csv('clinical', 'demographic', all_csvs)
rna_path = get_best_csv('rna', 'modelready', all_csvs)
if not rna_path: rna_path = get_best_csv('rna', 'raw', all_csvs)'''

for cell in nb.cells:
    if cell.cell_type == 'code':
        if "cli_path = next(" in cell.source:
            cell.source = cell.source.replace(old_discovery, new_discovery)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Successfully patched CSV file targeting.")
