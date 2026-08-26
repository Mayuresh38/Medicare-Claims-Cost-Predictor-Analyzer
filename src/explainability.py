import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def get_shap_explainer(model, X_train_summary, model_type="keras"):
    """
    Creates a SHAP explainer based on the model type.
    X_train_summary is summarized/sampled to speed up SHAP computations.
    """
    # Sample background data if it's large
    if hasattr(X_train_summary, "shape") and X_train_summary.shape[0] > 10:
        X_train_summary = shap.sample(X_train_summary, 10, random_state=42)
        
    if model_type == "keras":
        # Enforce KernelExplainer on predicting function
        explainer = shap.KernelExplainer(model.predict, X_train_summary)
    elif model_type == "sklearn":
        explainer = shap.KernelExplainer(model.predict, X_train_summary)
    else:
        explainer = shap.Explainer(model.predict, X_train_summary)
    return explainer

def explain_single_prediction(explainer, feature_vector, feature_names, model_type="keras"):
    """
    Generates a SHAP explanation waterfall plot for a single patient prediction.
    Flattens SHAP values to 1D array to prevent dimension mismatches.
    """
    # Reshape feature vector to 2D for prediction
    X_explain = feature_vector.reshape(1, -1)
    
    # Compute SHAP values
    shap_vals = explainer.shap_values(X_explain)
    
    # Extract first output for single-target regression
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
        
    # Flatten to 1D
    shap_vals_1d = np.array(shap_vals).flatten()
    
    # Extract base expected value
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[0]
        
    # Construct a shap.Explanation object for the waterfall plot
    explanation = shap.Explanation(
        values=shap_vals_1d,
        base_values=float(base_val),
        data=feature_vector.flatten(),
        feature_names=feature_names
    )
    
    # Create matplotlib figure
    fig = plt.figure(figsize=(10, 4))
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Test stub
    print("SHAP explainability module loaded successfully.")
