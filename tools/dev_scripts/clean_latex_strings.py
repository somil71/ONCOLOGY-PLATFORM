import json

def patch(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for idx, l in enumerate(nb['cells'][9]['source']):
        if '(G) Cross-Modal Co-Attention' in l:
            nb['cells'][9]['source'][idx] = 'axes[2,0].set_title(r"(G) Cross-Modal Co-Attention $\\mathbf{A}$", fontweight=\'bold\')\n'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f'[{path}] Direct line replacement done!')

patch('CancerFusion_Phase2_PanCancer.ipynb')
patch('notebooks/CancerFusion_Phase2_PanCancer.ipynb')
