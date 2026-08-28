import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# Streamlit Config
st.set_page_config(page_title="CancerFusion AI", layout="wide", page_icon="🧬")

st.title("🧬 CancerFusion AI: Clinical Dashboard")
st.markdown("### Multimodal Precision Oncology with Evidential Explanations")

st.sidebar.header("Upload Patient Data")
st.sidebar.markdown("Upload the available modalities. The system dynamically adapts if any test is missing.")

# Clinical Inputs
st.sidebar.subheader("1. Clinical Data (Required)")
age = st.sidebar.slider("Patient Age", 20, 100, 60)
stage = st.sidebar.selectbox("Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])

# Modality Uploads
st.sidebar.subheader("2. Genomics / RNA-Seq (Optional)")
genomics_file = st.sidebar.file_uploader("Upload RNA-Seq (.csv or .npy)", type=["csv", "npy"])

st.sidebar.subheader("3. Pathology / WSI (Optional)")
wsi_file = st.sidebar.file_uploader("Upload WSI Embeddings (.npy)", type=["npy"])

if st.sidebar.button("Run AI Inference"):
    with st.spinner("Running Evidential Fusion Model..."):
        
        # Mocking the API request to the backend we just built
        # In production: requests.post("http://localhost:8000/predict", files=...)
        
        # We will directly query the backend mock logic here for demonstration purposes if the server isn't running
        try:
            files = {}
            if genomics_file: files['genomics_file'] = genomics_file.getvalue()
            if wsi_file: files['wsi_embeddings'] = wsi_file.getvalue()
            
            res = requests.post(
                "http://localhost:8000/predict", 
                data={"clinical_data": json.dumps({"age": age, "stage": stage})},
                files=files
            )
            data = res.json()
        except requests.exceptions.ConnectionError:
            st.error("Backend server is not running. Please start it with `python backend.py`.")
            st.stop()

        # Layout: 3 Columns
        col1, col2, col3 = st.columns(3)
        
        # --- Column 1: Prediction & Missing Data ---
        with col1:
            st.subheader("Survival Risk Prediction")
            risk = data["predicted_risk_score"]
            
            if risk > 0.7:
                st.error(f"High Risk Score: {risk:.2f}")
            elif risk > 0.4:
                st.warning(f"Moderate Risk Score: {risk:.2f}")
            else:
                st.success(f"Low Risk Score: {risk:.2f}")
                
            st.markdown("---")
            st.subheader("Data Modalities Used")
            st.write("✅ Clinical Records")
            if data["missing_modalities"]["Genomics"]:
                st.write("❌ Genomics (Missing - Ignored by Model)")
            else:
                st.write("✅ Genomics")
                
            if data["missing_modalities"]["WSI"]:
                st.write("❌ Histopathology (Missing - Ignored by Model)")
            else:
                st.write("✅ Histopathology")

        # --- Column 2: Concept Bottleneck (Explanation Layer) ---
        with col2:
            st.subheader("Biological Concept Explanations")
            st.markdown("*What the AI saw in the Histopathology image:*")
            
            if not data["missing_modalities"]["WSI"]:
                concepts = data["biological_concepts"]
                df_concepts = pd.DataFrame({
                    "Concept": list(concepts.keys()),
                    "Activation Score": list(concepts.values())
                })
                
                fig = px.bar(
                    df_concepts, x="Activation Score", y="Concept", orientation='h',
                    color="Activation Score", color_continuous_scale="Reds"
                )
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("WSI Image missing. Biological concept extraction disabled.")

        # --- Column 3: Evidential Uncertainty ---
        with col3:
            st.subheader("Dempster-Shafer Evidence")
            st.markdown("*How confident is the model based on provided tests?*")
            
            evidence = data["evidential_uncertainty"]
            labels = ["WSI Evidence", "Genomics Evidence", "Clinical Evidence"]
            values = [evidence["WSI_Evidence"], evidence["Genomics_Evidence"], evidence["Clinical_Evidence"]]
            
            fig2 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig2.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig2, use_container_width=True)
            
            st.metric("Total Joint Certainty", f"{evidence['Total_Joint_Certainty']:.1f}")

st.markdown("---")
st.markdown("*CancerFusion AI Research Platform - Powered by Evidential Deep Learning*")
