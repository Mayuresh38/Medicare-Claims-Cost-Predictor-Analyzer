import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
import tensorflow as tf

from src.explainability import get_shap_explainer, explain_single_prediction

# Paths
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
COMP_CSV = os.path.join(MODELS_DIR, "model_comparison.csv")
BEST_INFO_JSON = os.path.join(MODELS_DIR, "best_model_info.json")
FEATURES_JSON = os.path.join(MODELS_DIR, "features.json")
SCALER_PKL = os.path.join(MODELS_DIR, "scaler.pkl")

# Styling Config
st.set_page_config(
    page_title="Medicare Claim Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    
    .card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #E5E7EB;
        margin-bottom: 1.5rem;
    }
    
    .dark-card {
        background: #1F2937;
        color: #F9FAFB;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #10B981;
    }
    
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9CA3AF;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check if models are trained
def is_pipeline_trained():
    return (
        os.path.exists(COMP_CSV) and
        os.path.exists(BEST_INFO_JSON) and
        os.path.exists(FEATURES_JSON) and
        os.path.exists(SCALER_PKL) and
        os.path.exists(os.path.join(MODELS_DIR, "ridge_regression.pkl")) and
        os.path.exists(os.path.join(MODELS_DIR, "ann.keras"))
    )

# Load pipeline files
@st.cache_resource
def load_pipeline_artifacts():
    with open(FEATURES_JSON, "r") as f:
        feat_info = json.load(f)
    with open(BEST_INFO_JSON, "r") as f:
        best_info = json.load(f)
    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)
        
    # Load Ridge model
    with open(os.path.join(MODELS_DIR, "ridge_regression.pkl"), "rb") as f:
        ridge_model = pickle.load(f)
        
    # Load ANN model
    ann_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "ann.keras"))
    
    # Load best model
    best_file = best_info["best_model_file"]
    m_type = best_info["type"]
    best_model = ann_model if m_type == "keras" else ridge_model
        
    return feat_info, best_info, scaler, best_model, ridge_model, ann_model

@st.cache_data
def load_comparison_metrics():
    if os.path.exists(COMP_CSV):
        return pd.read_csv(COMP_CSV)
    return None

# Title
st.markdown('<div class="main-title">🏥 Medicare Claims Cost Predictor & Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Using Deep Learning, Regression Analysis, and SHAP Explainability on CMS DE-SynPUF Data</div>', unsafe_allow_html=True)

if not is_pipeline_trained():
    st.warning("⚠️ Pipeline models have not been trained yet. Please wait a moment for the training process to finish and refresh.")
    st.info("The training pipeline is currently executing in the background. Once it finishes, the dashboard will load the trained models.")
    st.stop()

# Load models
feat_info, best_info, scaler, best_model, ridge_model, ann_model = load_pipeline_artifacts()
feature_names = feat_info["features"]
max_seq_len = feat_info["max_seq_len"]

# Sidebar model configuration selector
with st.sidebar:
    st.markdown('<div class="card" style="padding: 1rem; border-left: 5px solid #8B5CF6; margin-bottom: 1rem;"><h3>Model Settings</h3></div>', unsafe_allow_html=True)
    selected_model_name = st.selectbox(
        "Select Prediction Model",
        ["Best Model (Auto)", "Ridge Regression (Regression Analysis)", "Keras ANN (Deep Learning)"]
    )
    
    st.markdown("---")
    st.markdown("### Model Overview")
    st.markdown(
        "Use this toggle to compare the behavior of **Ridge Regression** (Regression Analysis) "
        "and **Keras ANN** (Deep Learning) on the same patient profile."
    )

# Map selection to active model
if selected_model_name == "Ridge Regression (Regression Analysis)":
    active_model = ridge_model
    m_type = "sklearn"
    model_display_name = "Ridge Regression"
elif selected_model_name == "Keras ANN (Deep Learning)":
    active_model = ann_model
    m_type = "keras"
    model_display_name = "ANN"
else:
    active_model = best_model
    m_type = best_info["type"]
    model_display_name = best_info["best_model_name"]

# Layout columns
col_inputs, col_results = st.columns([1, 2])

with col_inputs:
    st.markdown('<div class="card"><h3>Patient Profile Input</h3></div>', unsafe_allow_html=True)
    
    st.subheader("Demographics")
    age = st.slider("Beneficiary Age", min_value=65, max_value=100, value=75)
    sex = st.selectbox("Biological Sex", ["Female", "Male"])
    race = st.selectbox("Race Category", ["White", "Black", "Hispanic", "Others"])
    
    st.subheader("Chronic Conditions")
    chronic_selections = []
    
    # 11 Chronic conditions
    c_labels = {
        "SP_ALZHDMTA": "Alzheimer / Dementia",
        "SP_CHF": "Heart Failure (CHF)",
        "SP_CHRNKIDN": "Chronic Kidney Disease",
        "SP_CNCR": "Cancer",
        "SP_COPD": "COPD",
        "SP_DEPRESSN": "Depression",
        "SP_DIABETES": "Diabetes",
        "SP_ISCHMCHT": "Ischemic Heart Disease",
        "SP_OSTEOPRS": "Osteoporosis",
        "SP_RA_OA": "Rheumatoid Arthritis / OA",
        "SP_STRKETIA": "Stroke / TIA"
    }
    
    selected_conditions = {}
    for code, label in c_labels.items():
        selected_conditions[code] = st.checkbox(label, value=False)
        
    st.subheader("Longitudinal Claims History (Prior Year)")
    claim_count = st.slider("Prior Inpatient Claims Count", min_value=0, max_value=10, value=1)
    
    if claim_count > 0:
        total_pmt = st.number_input("Total Prior Inpatient Claim Amount ($)", min_value=0.0, max_value=150000.0, value=12000.0, step=1000.0)
        primary_payer_pmt = st.number_input("Total Primary Payer Paid Amount ($)", min_value=0.0, max_value=50000.0, value=1500.0, step=100.0)
        avg_duration = st.slider("Average Claim Duration (Days)", min_value=1, max_value=30, value=6)
    else:
        total_pmt = 0.0
        primary_payer_pmt = 0.0
        avg_duration = 0.0

# Map inputs to features
# Race conversion
race_map = {"White": 1, "Black": 2, "Hispanic": 5, "Others": 3}
race_code = race_map[race]

# Construct single row DataFrame for tabular inputs
input_dict = {
    "AGE": age,
    "IS_FEMALE": 1 if sex == "Female" else 0,
    "RACE_1": 1 if race_code == 1 else 0,
    "RACE_2": 1 if race_code == 2 else 0,
    "RACE_3": 1 if race_code == 3 else 0,
    "RACE_5": 1 if race_code == 5 else 0,
    "NUM_CHRONIC_CONDITIONS": sum(selected_conditions.values())
}

for col, val in selected_conditions.items():
    input_dict[col] = 1 if val else 0
    
input_dict.update({
    "TOTAL_CLAIM_COUNT": claim_count,
    "TOTAL_CLAIM_PMT": total_pmt,
    "AVG_CLAIM_PMT": total_pmt / claim_count if claim_count > 0 else 0.0,
    "MAX_CLAIM_PMT": total_pmt / claim_count if claim_count > 0 else 0.0,
    "AVG_CLAIM_DURATION": avg_duration,
    "TOTAL_PRIMARY_PAYER_PMT": primary_payer_pmt
})

# Make sure all features are aligned in the correct order
X_tab_single = pd.DataFrame([input_dict])[feature_names]

# Scaled inputs
X_tab_single_scaled = scaler.transform(X_tab_single)

# Prediction logic
with col_results:
    st.markdown('<div class="dark-card"><h3>Predicted Annual Insurance Cost</h3></div>', unsafe_allow_html=True)
    
    # Predict using the selected model
    if m_type == "keras":
        raw_pred = active_model.predict(X_tab_single_scaled)[0][0]
    else:
        raw_pred = active_model.predict(X_tab_single_scaled)[0]
        
    predicted_cost = max(0.0, float(raw_pred))
    
    # Display cost metric card
    st.markdown(f"""
    <div class="card" style="text-align: center; border-left: 6px solid #10B981; margin-bottom: 1rem;">
        <span class="metric-label">Active Predictor: {selected_model_name if "Best" not in selected_model_name else f"Best Model ({model_display_name})"}</span>
        <div class="metric-val">${predicted_cost:,.2f}</div>
        <span style="color: #6B7280; font-size: 0.85rem;">Estimated Total Annual Medicare Cost (Reimbursement + Copays)</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Display active model's test performance metrics
    comp_df = load_comparison_metrics()
    if comp_df is not None:
        row = comp_df[comp_df["Model"] == model_display_name]
        if not row.empty:
            mae_val = row.iloc[0]["MAE"]
            rmse_val = row.iloc[0]["RMSE"]
            r2_val = row.iloc[0]["R2 Score"]
            
            st.markdown("#### 📈 Model Test Performance Metrics")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Mean Absolute Error (MAE)", f"${mae_val:,.2f}")
            m_col2.metric("Root Mean Squared Error (RMSE)", f"${rmse_val:,.2f}")
            m_col3.metric("R² Score (Variance Explained)", f"{r2_val * 100:.2f}%")
            st.markdown("---")
            
    # Section: SHAP Explainability
    st.markdown("### 🔍 Explainable AI (SHAP Insights)")
    st.markdown("This chart displays the positive (red) and negative (blue) forces driving the patient's predicted cost relative to the baseline population average.")
    
    try:
        # We create a summarized background dataset (zeros or representative mean)
        background_data = np.zeros((5, len(feature_names)))
        explainer = get_shap_explainer(active_model, background_data, model_type=m_type)
        
        fig = explain_single_prediction(
            explainer, 
            X_tab_single_scaled[0], 
            feature_names, 
            model_type=m_type
        )
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.info("SHAP explainability plot is loading or unavailable. Using feature weights overview instead.")
        st.error(f"SHAP explanation details: {e}")

