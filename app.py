import streamlit as st
import pandas as pd
import joblib
import json
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AdSmart AI: CTR Predictor", page_icon="📈")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # 1. Get the current directory of the app
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Define the exact paths to your files
    # Note: We added '.json' because that is the name on your GitHub
    model_path = os.path.join(current_dir, 'ctr_lgbm_model.joblib')
    columns_path = os.path.join(current_dir, 'ctr_model_columns.json')

    # 3. Load Model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Critical Error: Could not load model. Details: {e}")
        st.stop()
        
    # 4. Load Columns
    columns = []
    try:
        with open(columns_path, 'r') as f:
            columns = json.load(f)
    except Exception as e:
        st.error(f"❌ Critical Error: Could not load columns. Details: {e}")
        st.stop()
            
    # Handle case where columns might be a dict
    if isinstance(columns, dict):
        if "data_columns" in columns:
            columns = columns["data_columns"]
        else:
            columns = list(columns.keys())
            
    return model, columns
# Load them now
model, model_columns = load_assets()

# --- UI LAYOUT ---
st.title("📈 AdSmart AI")
st.markdown("### Click-Through Rate (CTR) Prediction Engine")
st.markdown("Enter the ad parameters below to predict the probability of a user clicking.")
st.markdown("---")

# --- DYNAMIC INPUT FORM ---
# We split inputs into 2 columns for a cleaner look
col1, col2 = st.columns(2)
input_data = {}

# Loop through columns and create inputs
for i, col in enumerate(model_columns):
    clean_name = col.replace('_', ' ').title()
    
    # Place in col1 or col2 alternating
    with col1 if i % 2 == 0 else col2:
        # Heuristic: IDs are usually integers, others are floats
        if 'id' in col.lower() or 'count' in col.lower() or 'category' in col.lower():
            input_data[col] = st.number_input(clean_name, value=0, step=1)
        else:
            input_data[col] = st.number_input(clean_name, value=0.0)

st.markdown("---")

# --- PREDICTION LOGIC ---
if st.button("🚀 Predict CTR Probability", use_container_width=True):
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure correct column order
    input_df = input_df[model_columns]
    
    try:
        # Predict Probability (Class 1)
        prediction_prob = model.predict_proba(input_df)[0][1]
        prediction_percent = prediction_prob * 100
        
        # Display Result
        st.metric(label="Likelihood of Click", value=f"{prediction_percent:.2f}%")
        
        # Interpretation
        if prediction_prob > 0.7:
            st.success("🌟 **High Potential:** This ad is very likely to be clicked!")
        elif prediction_prob > 0.4:
            st.info("⚠️ **Moderate Potential:** Standard performance expected.")
        else:
            st.error("📉 **Low Potential:** Consider optimizing ad parameters.")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
