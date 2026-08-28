import os
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_or_create_dataset
from src.features import build_tabular_features, PipelineScaler
from src.models import build_ann_model

# Output directory for saved models/scalers
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_evaluate_pipeline(use_mock=False, num_bene=3000, num_claims=6000, epochs=80):
    """
    Executes the standardized 5-Fold Cross-Validation and training pipeline.
    """
    print("--- Phase 1: Ingesting and Integrating Data ---")
    bene_df, ip_df = load_or_create_dataset(use_mock=use_mock, num_bene=num_bene, num_claims=num_claims)
    
    print("--- Phase 2 & 3: Preprocessing & Feature Engineering ---")
    X_tab, y, feature_cols = build_tabular_features(bene_df, ip_df)
    max_seq_len = 5 # retained for backward compatibility
    
    print("--- Phase 4: Standardized 5-Fold Cross-Validation ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    ann_metrics = {'MAE': [], 'RMSE': [], 'R2': [], 'RMSLE': []}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tab)):
        print(f"Running Fold {fold + 1} / 5...")
        
        # Split
        X_train, X_val = X_tab.iloc[train_idx], X_tab.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Target transformation for ANN
        y_train_trans = np.log1p(y_train)
        y_val_trans = np.log1p(y_val)
        
        # Keras ANN
        ann = build_ann_model(input_dim=X_train_scaled.shape[1], learning_rate=0.001)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        ann.fit(
            X_train_scaled, y_train_trans,
            validation_data=(X_val_scaled, y_val_trans),
            epochs=epochs,
            batch_size=64,
            callbacks=[early_stopping, lr_reducer],
            verbose=0
        )
        
        ann_pred_trans = ann.predict(X_val_scaled, verbose=0).flatten()
        ann_pred = np.maximum(0, np.expm1(ann_pred_trans))
        
        # Calculate ANN metrics
        a_mae = mean_absolute_error(y_val, ann_pred)
        a_rmse = root_mean_squared_error(y_val, ann_pred)
        a_r2 = r2_score(y_val, ann_pred)
        a_rmsle = np.sqrt(np.mean((np.log1p(y_val) - np.log1p(ann_pred)) ** 2))
        
        ann_metrics['MAE'].append(a_mae)
        ann_metrics['RMSE'].append(a_rmse)
        ann_metrics['R2'].append(a_r2)
        ann_metrics['RMSLE'].append(a_rmsle)
        
        print(f"  ANN   -> MAE: {a_mae:.2f}, R2: {a_r2:.4f}, RMSLE: {a_rmsle:.4f}")

    print("--- Phase 5: Final Evaluation & Benchmarking ---")
    avg_ann = {k: np.mean(v) for k, v in ann_metrics.items()}
    
    comparison_data = [
        {
            "Model": "ANN",
            "MAE": round(avg_ann['MAE'], 2),
            "RMSE": round(avg_ann['RMSE'], 2),
            "R2 Score": round(avg_ann['R2'], 4),
            "RMSLE": round(avg_ann['RMSLE'], 4)
        }
    ]
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)
    
    print("\nStandardized CV Results:")
    print(comparison_df)
    
    print("--- Phase 6: Training Deployed Model on All Data ---")
    # Fit final feature scaler
    scaler = PipelineScaler()
    X_tab_scaled = scaler.fit_transform(X_tab)
    
    # Save the feature scaler and feature metadata
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, "features.json"), "w") as f:
        json.dump({"features": feature_cols, "max_seq_len": max_seq_len}, f)
        
    # Save a dummy/empty target scaler to maintain backward compatibility
    dummy_target_scaler = StandardScaler()
    dummy_target_scaler.fit(y.values.reshape(-1, 1))
    with open(os.path.join(MODELS_DIR, "target_scaler.pkl"), "wb") as f:
        pickle.dump(dummy_target_scaler, f)
        
    # Train final ANN model
    y_trans = np.log1p(y.values)
    final_ann = build_ann_model(input_dim=X_tab_scaled.shape[1], learning_rate=0.001)
    
    # Split training set slightly just for validation callbacks to prevent overfitting
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_tab_scaled, y_trans, test_size=0.1, random_state=42
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    final_ann.fit(
        X_train_final, y_train_final,
        validation_data=(X_val_final, y_val_final),
        epochs=epochs,
        batch_size=64,
        callbacks=[early_stopping, lr_reducer],
        verbose=1
    )
    
    final_ann.save(os.path.join(MODELS_DIR, "ann.keras"))
    
    # Save model info
    best_info = {
        "best_model_name": "ANN",
        "best_model_file": "ann",
        "type": "keras"
    }
    with open(os.path.join(MODELS_DIR, "best_model_info.json"), "w") as f:
        json.dump(best_info, f)
        
    print("Training completed successfully!")
    return comparison_df

if __name__ == "__main__":
    train_and_evaluate_pipeline(use_mock=False, num_bene=1000, num_claims=2000, epochs=80)
