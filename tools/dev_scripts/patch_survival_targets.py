import nbformat as nbf

notebook_path = "e:/MAJOR PROJECT/CancerFusion_Phase2_PanCancer.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

old_logic = '''demo_df['event'] = demo_df['vital_status'].apply(lambda x: 1.0 if str(x).lower() == 'dead' else 0.0)
demo_df['time'] = demo_df['demographic_days_to_death'].fillna(demo_df['diagnoses_days_to_last_follow_up']).fillna(0)
target_df = demo_df[['Patient_ID', 'time', 'event']].drop_duplicates(subset=['Patient_ID'])'''

new_logic = '''# Bulletproof Pan-Cancer Survival Target Locator
event_col = next((c for c in demo_df.columns if any(x in c.lower() for x in ['status', 'event', 'vital', 'os_'])), demo_df.columns[-1])
death_col = next((c for c in demo_df.columns if 'death' in c.lower() or 'os' in c.lower()), None)
follow_col = next((c for c in demo_df.columns if 'follow' in c.lower()), None)

# Parse Event (1.0 = Dead, 0.0 = Alive)
demo_df['event'] = demo_df[event_col].apply(lambda x: 1.0 if any(d in str(x).lower() for d in ['dead', 'deceased', '1']) else 0.0)

# Parse Time (Days/Months to Event)
if death_col and follow_col:
    demo_df['time'] = pd.to_numeric(demo_df[death_col], errors='coerce').fillna(pd.to_numeric(demo_df[follow_col], errors='coerce'))
elif death_col:
    demo_df['time'] = pd.to_numeric(demo_df[death_col], errors='coerce')
elif follow_col:
    demo_df['time'] = pd.to_numeric(demo_df[follow_col], errors='coerce')
else:
    time_col = next((c for c in demo_df.columns if 'time' in c.lower() or 'days' in c.lower()), demo_df.columns[-2])
    demo_df['time'] = pd.to_numeric(demo_df[time_col], errors='coerce')

demo_df['time'] = demo_df['time'].fillna(0)
target_df = demo_df[['Patient_ID', 'time', 'event']].drop_duplicates(subset=['Patient_ID'])'''

for cell in nb.cells:
    if cell.cell_type == 'code':
        if "demo_df['event'] = demo_df['vital_status']" in cell.source:
            cell.source = cell.source.replace(old_logic, new_logic)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Successfully patched Pan-Cancer Survival Logic.")
