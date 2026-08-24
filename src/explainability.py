import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def get_shap_explainer(model, X_train_summary, model_type="xgboost"):
    """
    Creates a SHAP explainer based on the model type.
    X_train_summary is a summarized/representative background dataset (e.g. 50-100 rows)
    to speed up SHAP computations.
    """
    if model_type == "xgboost":
        # TreeExplainer is fast and accurate for XGBoost
        explainer = shap.TreeExplainer(model)
    elif model_type == "sklearn":
        # LinearExplainer or KernelExplainer
        try:
            explainer = shap.LinearExplainer(model, X_train_summary)
        except Exception:
            explainer = shap.KernelExplainer(model.predict, X_train_summary)
    elif model_type == "keras":
        # KernelExplainer is more stable across TF/Keras versions
        explainer = shap.KernelExplainer(model.predict, X_train_summary)
    else:
        explainer = shap.Explainer(model, X_train_summary)
    return explainer

def explain_single_prediction(explainer, feature_vector, feature_names, model_type="xgboost"):
    """
    Generates a SHAP explanation plot for a single patient prediction.
    Saves and returns the plot figure.
    """
    # Reshape feature vector to 2D
    X_explain = feature_vector.reshape(1, -1)
    
    # Compute SHAP values
    shap_values = explainer(X_explain)
    
    # Create matplotlib figure
    fig = plt.figure(figsize=(10, 4))
    
    # Generate waterfall plot
    if hasattr(shap_values, "base_values"):
        # For shap 0.36+ Explainer objects
        # We need to set feature names
        shap_values.feature_names = feature_names
        # Handle 3D explanation arrays (e.g. from Keras multi-output or single-output with extra dim)
        if len(shap_values.shape) == 3:
            shap.plots.waterfall(shap_values[0, :, 0], show=False)
        else:
            shap.plots.waterfall(shap_values[0], show=False)
    else:
        # Fallback to bar plot or summary
        # If shap_values is a list or numpy array
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values
        shap.bar_plot(sv[0], feature_names=feature_names, show=False)
        
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Test stub
    print("SHAP explainability module loaded successfully.")
