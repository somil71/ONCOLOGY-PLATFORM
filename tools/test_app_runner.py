import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
from streamlit.testing.v1 import AppTest

print("="*70)
print("  AUTOMATED END-TO-END CLINICAL TEST SUITE (CancerFusion AI)")
print("="*70)

# TEST 1: Home View
print("\n[Test 1] Testing '🏠 Home' Landing Page...")
at_home = AppTest.from_file("app.py", default_timeout=30)
at_home.session_state["main_navbar"] = "🏠 Home"
at_home.session_state["current_nav"] = "🏠 Home"
at_home.run()

if at_home.exception:
    print(f"FAILED in Home View: {at_home.exception}")
    sys.exit(1)

hero_found = any("Deep Multimodal Precision Oncology" in m.value for m in at_home.markdown)
pillars_found = any("Gated-MIL Pathology" in m.value for m in at_home.markdown)
print(f"   • Hero Banner Present:    {hero_found}")
print(f"   • Core 4 Pillars Present: {pillars_found}")
assert hero_found and pillars_found, "Home page content missing!"
print("PASSED: [🏠 Home] View verified with 0 exceptions!")

# TEST 2: Clinical Intake View & Benchmark Loading
print("\n[Test 2] Testing '📝 Clinical Case Intake' (Benchmark Preset)...")
at_intake = AppTest.from_file("app.py", default_timeout=30)
at_intake.session_state["main_navbar"] = "📝 Clinical Case Intake"
at_intake.session_state["current_nav"] = "📝 Clinical Case Intake"
at_intake.run()

if at_intake.exception:
    print(f"FAILED in Intake View: {at_intake.exception}")
    sys.exit(1)

run_btns = [b for b in at_intake.button if "Execute Multimodal" in b.label]
assert len(run_btns) > 0, "Execute button not found in Intake view!"
run_btns[0].click().run()

if at_intake.exception:
    print(f"FAILED when clicking Execute: {at_intake.exception}")
    sys.exit(1)

print(f"   • Navigated to: {at_intake.session_state.current_nav}")
print(f"   • Has Run Flag: {at_intake.session_state.has_run}")
print("PASSED: [📝 Clinical Case Intake] Benchmark inference executed cleanly!")

# TEST 3: Results Dashboard View
print("\n[Test 3] Testing '🔬 Analysis & Model Results' Dashboard...")
at_results = AppTest.from_file("app.py", default_timeout=30)
at_results.session_state["main_navbar"] = "🔬 Analysis & Model Results"
at_results.session_state["current_nav"] = "🔬 Analysis & Model Results"
at_results.session_state["has_run"] = True
# Seed patient data
at_results.session_state["patient_data"] = {
    "id": "TCGA-AR-A1AQ",
    "stage": "Stage IIIB Locally Advanced",
    "tnm": "T3 N2a M0",
    "grade": "Nottingham Grade 3",
    "subtype": "Triple-Negative (TNBC)",
    "receptors": {"ER": "Negative (-)", "PR": "Negative (-)", "HER2": "Negative (0)", "Ki67": 78},
    "age": 49,
    "risk": 0.7640,
    "unc": 0.2810,
    "weights": {"wsi": 0.385, "rad": 0.275, "rna": 0.230, "cli": 0.110},
    "mask": [True, True, True, True],
    "genes": {"BRCA1": "+3.4σ", "TP53": "+3.8σ", "PIK3CA": "+0.4σ", "ESR1": "-2.8σ", "PTEN": "-2.1σ"},
    "hallmarks": {"DNA Damage Repair Defect": 2.95, "Cell Cycle Proliferation": 3.10, "EMT": 2.45}
}
at_results.run()

if at_results.exception:
    print(f"FAILED in Results Dashboard: {at_results.exception}")
    sys.exit(1)

md_texts = [m.value for m in at_results.markdown]
has_hazard = any("Predicted Survival Hazard" in t for t in md_texts)
has_unc = any("Dirichlet Uncertainty" in t for t in md_texts)
has_cindex = any("Harrell's Concordance Index" in t for t in md_texts)
print(f"   • Predicted Hazard KPI:     {has_hazard}")
print(f"   • Dirichlet Uncertainty KPI: {has_unc}")
print(f"   • Harrell's C-Index KPI:    {has_cindex}")
assert has_hazard and has_unc and has_cindex, "Core KPIs missing in Results dashboard!"
print("PASSED: [🔬 Analysis & Model Results] View verified with 0 exceptions!")

# TEST 4: Research Benchmarks View
print("\n[Test 4] Testing '📊 Research Benchmarks' View...")
at_bench = AppTest.from_file("app.py", default_timeout=30)
at_bench.session_state["main_navbar"] = "📊 Research Benchmarks"
at_bench.session_state["current_nav"] = "📊 Research Benchmarks"
at_bench.run()

if at_bench.exception:
    print(f"FAILED in Benchmarks View: {at_bench.exception}")
    sys.exit(1)

has_table = len(at_bench.dataframe) >= 2
print(f"   • Benchmark DataFrames Present: {has_table} ({len(at_bench.dataframe)} tables)")
assert has_table, "Benchmark data tables missing!"
print("PASSED: [📊 Research Benchmarks] View verified with 0 exceptions!")

# TEST 5: Tumor Board Dossier View
print("\n[Test 5] Testing '📄 Tumor Board Dossier' View...")
at_dossier = AppTest.from_file("app.py", default_timeout=30)
at_dossier.session_state["main_navbar"] = "📄 Tumor Board Dossier"
at_dossier.session_state["current_nav"] = "📄 Tumor Board Dossier"
at_dossier.run()

if at_dossier.exception:
    print(f"FAILED in Dossier View: {at_dossier.exception}")
    sys.exit(1)

has_dossier_text = any("CLINICAL PROGNOSTIC DOSSIER" in m.value for m in at_dossier.markdown)
has_dl_button = len(at_dossier.get("download_button")) > 0
print(f"   • Clinical Dossier Present:      {has_dossier_text}")
print(f"   • Official Download Button Active: {has_dl_button}")
assert has_dossier_text and has_dl_button, "Dossier or download button missing!"
print("PASSED: [📄 Tumor Board Dossier] View verified with 0 exceptions!")

print("\n" + "="*70)
print("  🎉 COMPLETE TEST SUITE: 5/5 VIEWS 100% PASSED WITH ZERO EXCEPTIONS!")
print("="*70)
