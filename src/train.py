import os
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import Ridge

from src.data_loader import load_or_create_dataset
from src.features import build_tabular_features, PipelineScaler
from src.models import (
    build_regression_model,
    build_ann_model
)

# Output directory for saved models/scalers
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_evaluate_pipeline(use_mock=True, num_bene=3000, num_claims=6000, epochs=20):
    """
    Executes the simplified end-to-end training and comparison pipeline with hyperparameter tuning.
    """
    print("--- Phase 1: Ingesting and Integrating Data ---")
    bene_df, ip_df = load_or_create_dataset(use_mock=use_mock, num_bene=num_bene, num_claims=num_claims)
    
    print("--- Phase 2 & 3: Preprocessing & Feature Engineering ---")
    # Tabular features and target
    X_tab, y, feature_cols = build_tabular_features(bene_df, ip_df)
    max_seq_len = 5 # retained for backward compatibility with metadata files
    
    # Train-test split
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_tab_train, X_tab_test = X_tab.iloc[train_idx].reset_index(drop=True), X_tab.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
    
    # Scale tabular features
    scaler = PipelineScaler()
    X_tab_train_scaled = scaler.fit_transform(X_tab_train)
    X_tab_test_scaled = scaler.transform(X_tab_test)
    
    # Save the scaler and features metadata
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, "features.json"), "w") as f:
        json.dump({"features": feature_cols, "max_seq_len": max_seq_len}, f)
        
    results = {}
    
    print("--- Phase 4: Training & Hyperparameter Tuning ---")
    
    # 1. Ridge Regression Hyperparameter Tuning
    print("Tuning Ridge Regression...")
    ridge = Ridge()
    param_grid_ridge = {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'fit_intercept': [True, False]
    }
    grid_ridge = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_ridge.fit(X_tab_train_scaled, y_train)
    best_ridge = grid_ridge.best_estimator_
    print(f"Best Ridge params: {grid_ridge.best_params_}")
    
    ridge_preds = best_ridge.predict(X_tab_test_scaled)
    results["Ridge Regression"] = {
        "model": best_ridge,
        "preds": ridge_preds,
        "type": "sklearn"
    }
    
    # 2. Keras ANN Hyperparameter Tuning
    print("Tuning Keras ANN...")
    # Split training set to get validation set for hyperparameter tuning
    X_ann_tr, X_ann_val, y_ann_tr, y_ann_val = train_test_split(
        X_tab_train_scaled, y_train, test_size=0.2, random_state=42
    )
    
    ann_params_grid = []
    for lr in [0.001, 0.01]:
        for do in [0.1, 0.3]:
            for bs in [16, 64]:
                ann_params_grid.append({
                    'learning_rate': lr,
                    'dropout_rate': do,
                    'batch_size': bs
                })
                
    best_ann_val_mae = float('inf')
    best_ann_params = None
    best_ann_model = None
    
    for idx, params in enumerate(ann_params_grid):
        print(f"Candidate {idx+1}/{len(ann_params_grid)}: {params}")
        # Build model
        model = build_ann_model(
            input_dim=X_tab_train_scaled.shape[1],
            learning_rate=params['learning_rate'],
            dropout_rate=params['dropout_rate']
        )
        
        # Train model with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        model.fit(
            X_ann_tr, y_ann_tr,
            validation_data=(X_ann_val, y_ann_val),
            epochs=epochs,
            batch_size=params['batch_size'],
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate on validation set
        val_preds = model.predict(X_ann_val, verbose=0).flatten()
        val_mae = mean_absolute_error(y_ann_val, val_preds)
        print(f"  Validation MAE: {val_mae:.2f}")
        
        if val_mae < best_ann_val_mae:
            best_ann_val_mae = val_mae
            best_ann_params = params
            best_ann_model = model
            
    print(f"Best ANN params: {best_ann_params} with Validation MAE: {best_ann_val_mae:.2f}")
    
    # Evaluate best ANN on test set
    ann_preds = best_ann_model.predict(X_tab_test_scaled).flatten()
    results["ANN"] = {
        "model": best_ann_model,
        "preds": ann_preds,
        "type": "keras"
    }
    
    print("--- Phase 5: Model Comparison ---")
    comparison_data = []
    best_r2 = -float("inf")
    best_model_name = None
    
    for name, r in results.items():
        preds = r["preds"]
        # Ensure non-negative cost predictions
        preds = np.clip(preds, a_min=0, a_max=None)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        comparison_data.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2 Score": round(r2, 4)
        })
        print(f"{name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)
    
    print(f"\nBest Model: {best_model_name} with R2: {best_r2:.4f}")
    
    # Save the models
    for name, r in results.items():
        model = r["model"]
        m_type = r["type"]
        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        
        if m_type == "sklearn":
            with open(os.path.join(MODELS_DIR, f"{safe_name}.pkl"), "wb") as f:
                pickle.dump(model, f)
        elif m_type == "keras":
            model.save(os.path.join(MODELS_DIR, f"{safe_name}.keras"))
            
    # Copy best model to a generic "best_model" target
    best_safe_name = best_model_name.lower().replace(" ", "_").replace("/", "_")
    best_info = {
        "best_model_name": best_model_name,
        "best_model_file": f"{best_safe_name}",
        "type": results[best_model_name]["type"]
    }
    with open(os.path.join(MODELS_DIR, "best_model_info.json"), "w") as f:
        json.dump(best_info, f)
        
    print(f"All models and comparison statistics saved to: {MODELS_DIR}")
    return comparison_df

if __name__ == "__main__":
    train_and_evaluate_pipeline(use_mock=True, num_bene=1000, num_claims=2000, epochs=15)
