import nbformat as nbf

notebook_path = "e:/MAJOR PROJECT/CancerFusion_Phase2_PanCancer.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

old_cli_enc = '''self.cli_enc  = nn.Sequential(nn.Linear(cli_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 16))'''
new_cli_enc = '''self.cli_enc  = nn.Sequential(nn.Linear(cli_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 16))'''

old_dataloader = '''train_dl = DataLoader(PanCancerDataset(train_df, rna_df, cli_df), batch_size=CFG['batch_size'], shuffle=True, num_workers=CFG['num_workers'])'''
new_dataloader = '''train_dl = DataLoader(PanCancerDataset(train_df, rna_df, cli_df), batch_size=CFG['batch_size'], shuffle=True, drop_last=True, num_workers=CFG['num_workers'])'''


for cell in nb.cells:
    if cell.cell_type == 'code':
        if old_cli_enc in cell.source:
            cell.source = cell.source.replace(old_cli_enc, new_cli_enc)
        if old_dataloader in cell.source:
            cell.source = cell.source.replace(old_dataloader, new_dataloader)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Successfully patched BatchNorm crash.")
