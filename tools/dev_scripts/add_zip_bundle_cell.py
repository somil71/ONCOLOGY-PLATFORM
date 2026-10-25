import json

notebook_path = "CancerFusion_Phase2_PanCancer.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 10 Code: 1-Click Submission Zip Package
cell_10_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# =====================================================================\n",
        "# 📦 CELL 10: 1-CLICK FINAL SUBMISSION BUNDLE (WEIGHTS + ALL GRAPHS)\n",
        "# =====================================================================\n",
        "import os, shutil, glob\n",
        "print(\"\\n\" + \"=\"*65)\n",
        "print(\"  📦 PACKAGING ALL SUBMISSION ARTIFACTS INTO 1-CLICK ZIP BUNDLE\")\n",
        "print(\"=\"*65)\n",
        "\n",
        "zip_target = '/kaggle/working/CancerFusion_Final_Submission_Package'\n",
        "package_dir = '/kaggle/working/final_submission'\n",
        "os.makedirs(package_dir, exist_ok=True)\n",
        "\n",
        "# Copy core artifacts into submission staging directory\n",
        "shutil.copy('/kaggle/working/CancerFusion_Phase2_Weights.pth', f'{package_dir}/CancerFusion_Phase2_Weights.pth')\n",
        "shutil.copy('/kaggle/working/Figure_Complete_Research_Paper_Results.png', f'{package_dir}/Figure_Complete_Research_Paper_Results.png')\n",
        "shutil.copy('/kaggle/working/complete_metrics_summary.json', f'{package_dir}/complete_metrics_summary.json')\n",
        "shutil.copy('/kaggle/working/backend_response.json', f'{package_dir}/backend_response.json')\n",
        "\n",
        "# Copy all individual figures\n",
        "shutil.copytree('/kaggle/working/paper_figures', f'{package_dir}/paper_figures', dirs_exist_ok=True)\n",
        "\n",
        "# Create the zip archive\n",
        "shutil.make_archive(zip_target, 'zip', package_dir)\n",
        "\n",
        "zip_size_mb = os.path.getsize(f'{zip_target}.zip') / (1024 * 1024)\n",
        "print(f\"\\n🎉 [ALL DELIVERABLES READY] Created: {zip_target}.zip ({zip_size_mb:.2f} MB)\")\n",
        "print(\"   Contains:\")\n",
        "print(\"   1. CancerFusion_Phase2_Weights.pth (Model Weights)\")\n",
        "print(\"   2. Figure_Complete_Research_Paper_Results.png (Master 9-Panel 300 DPI)\")\n",
        "print(\"   3. complete_metrics_summary.json (ROC-AUC, C-Index, F1, Precision, Accuracy)\")\n",
        "print(\"   4. backend_response.json (Streamlit Platform API response)\")\n",
        "print(\"   5. paper_figures/ (All 12 individual publication figures)\")\n",
        "print(\"\\n👉 In Kaggle right-hand panel, go to Output -> click download on 'CancerFusion_Final_Submission_Package.zip'!\")\n"
    ]
}

# If cell 10 already exists, replace it; else append
if len(nb['cells']) > 10:
    nb['cells'][10] = cell_10_code
else:
    nb['cells'].append(cell_10_code)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Appended Cell 10: 1-Click Submission Zip Package to CancerFusion_Phase2_PanCancer.ipynb.")
