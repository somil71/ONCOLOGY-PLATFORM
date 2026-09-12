import nbformat as nbf

notebook_path = "e:/MAJOR PROJECT/CancerFusion_Phase2_PanCancer.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

old_split = '''train_df = target_df.iloc[:int(0.8*len(target_df))]
val_df   = target_df.iloc[int(0.8*len(target_df)):]'''

new_split = '''# Shuffle the dataset before splitting! (Medical datasets are often sorted by survival outcome)
shuffled_df = target_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df = shuffled_df.iloc[:int(0.8*len(shuffled_df))]
val_df   = shuffled_df.iloc[int(0.8*len(shuffled_df)):]'''

for cell in nb.cells:
    if cell.cell_type == 'code':
        if "train_df = target_df.iloc[:int(0.8*len(target_df))]" in cell.source:
            cell.source = cell.source.replace(old_split, new_split)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Successfully patched dataset shuffling.")
