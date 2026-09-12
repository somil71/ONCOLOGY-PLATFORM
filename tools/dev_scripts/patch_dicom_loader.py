import nbformat as nbf

notebook_path = "e:/MAJOR PROJECT/CancerFusion_Advanced_Kaggle.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Patch Cell 1 to install pydicom
for cell in nb.cells:
    if cell.cell_type == 'code' and 'lifelines' in cell.source:
        if 'pydicom' not in cell.source:
            cell.source = cell.source.replace('print("  installed lifelines")', 
            'print("  installed lifelines")\n    try:\n        import pydicom\n    except:\n        pip("pydicom")')

# 2. Patch Cell 4 to use the Universal Medical Image Loader
for cell in nb.cells:
    if cell.cell_type == 'code' and 'class TCGADataset' in cell.source:
        old_loader = '''            for p in patches_to_load:
                try:
                    img = Image.open(p).convert('RGB')
                    loaded_images.append(wsi_transform(img))
                except Exception: pass'''
        
        new_loader = '''            for p in patches_to_load:
                try:
                    # Attempt 1: Standard Image Formats (JPG/PNG)
                    img = Image.open(p).convert('RGB')
                    loaded_images.append(wsi_transform(img))
                except Exception:
                    try:
                        # Attempt 2: Raw Medical DICOM Formats (TCIA standard)
                        import pydicom
                        ds = pydicom.dcmread(p)
                        arr = ds.pixel_array.astype(float)
                        # Normalize raw medical pixels to RGB
                        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
                        arr = (arr * 255).astype(np.uint8)
                        
                        if len(arr.shape) == 2:
                            img = Image.fromarray(arr).convert('RGB')
                        elif len(arr.shape) > 2:
                            img = Image.fromarray(arr[0]).convert('RGB') # Take first slice if 3D MRI
                        
                        loaded_images.append(wsi_transform(img))
                    except Exception:
                        pass'''
        
        if old_loader in cell.source:
            cell.source = cell.source.replace(old_loader, new_loader)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
    
print("Successfully patched Universal Medical Image Loader (DICOM).")
