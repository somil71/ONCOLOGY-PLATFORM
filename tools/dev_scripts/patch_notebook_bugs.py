import json

def patch_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # 1. Patch Cell 5: composite_research_loss return signature
    cell5_text = ''.join(nb['cells'][5]['source'])
    old_ret = 'return total_loss, l_cox'
    new_ret = '''loss_dict = {
        'l_cox': l_cox.item() if torch.is_tensor(l_cox) else float(l_cox),
        'l_edl': l_edl.item() if torch.is_tensor(l_edl) else float(l_edl),
        'l_orth': l_orth.item() if torch.is_tensor(l_orth) else float(l_orth),
        'l_sparse': l_sparse.item() if torch.is_tensor(l_sparse) else float(l_sparse)
    }
    return total_loss, loss_dict'''
    if old_ret in cell5_text:
        cell5_text = cell5_text.replace(old_ret, new_ret)
        nb['cells'][5]['source'] = [l + '\n' for l in cell5_text.splitlines()]
        print(f'[{path}] Patched Cell 5 composite_research_loss return!')
    else:
        print(f'[{path}] Cell 5 already has correct return or old_ret not found.')

    # 2. Patch Cell 6: deduplicate rna_df/cli_df & safe drop_last
    cell6_text = ''.join(nb['cells'][6]['source'])
    dedup_target = 'rna_df = rna_df.select_dtypes(include=[np.number]).fillna(0)'
    dedup_replacement = '''rna_df = rna_df.select_dtypes(include=[np.number]).fillna(0)

# Deduplicate clinical and genomic indices to ensure robust 1D vector indexing
cli_df = cli_df[~cli_df.index.duplicated(keep='first')]
rna_df = rna_df[~rna_df.index.duplicated(keep='first')]'''
    if dedup_target in cell6_text and 'cli_df.index.duplicated' not in cell6_text:
        cell6_text = cell6_text.replace(dedup_target, dedup_replacement)
        print(f'[{path}] Patched Cell 6 index deduplication!')
        
    dl_target = "train_dl = DataLoader(PanCancerDataset(train_df, rna_df, cli_df), batch_size=CFG['batch_size'], shuffle=True, drop_last=True, num_workers=CFG['num_workers'])"
    dl_replacement = '''eff_bs = min(CFG['batch_size'], max(4, len(train_df)))
safe_drop_last = (len(train_df) >= eff_bs * 2)
train_dl = DataLoader(PanCancerDataset(train_df, rna_df, cli_df), batch_size=eff_bs, shuffle=True, drop_last=safe_drop_last, num_workers=CFG['num_workers'])'''
    if dl_target in cell6_text:
        cell6_text = cell6_text.replace(dl_target, dl_replacement)
        print(f'[{path}] Patched Cell 6 safe DataLoader drop_last!')

    nb['cells'][6]['source'] = [l + '\n' for l in cell6_text.splitlines()]

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f'[{path}] Saved successfully.')

if __name__ == '__main__':
    patch_notebook('CancerFusion_Phase2_PanCancer.ipynb')
    patch_notebook('notebooks/CancerFusion_Phase2_PanCancer.ipynb')
