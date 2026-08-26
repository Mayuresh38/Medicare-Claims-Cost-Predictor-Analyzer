import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.preprocessing import preprocess_beneficiaries, preprocess_claims, CHRONIC_CONDITIONS

def build_tabular_features(bene_df, ip_df):
    """
    Builds tabular features for Regression and ANN models.
    Combines beneficiary demographics, chronic conditions, and aggregated claims statistics.
    """
    # Preprocess
    df_bene = preprocess_beneficiaries(bene_df)
    df_claims = preprocess_claims(ip_df)
    
    # Calculate claims aggregates per beneficiary
    claims_agg = df_claims.groupby("DESYNPUF_ID").agg(
        TOTAL_CLAIM_COUNT=("CLM_ID", "count"),
        AVG_CLAIM_DURATION=("CLAIM_DURATION", "mean")
    ).reset_index()
    
    # Merge aggregates back to beneficiary summary
    features_df = pd.merge(df_bene, claims_agg, on="DESYNPUF_ID", how="left")
    
    # Fill missing values for beneficiaries with no claims
    fill_cols = ["TOTAL_CLAIM_COUNT", "AVG_CLAIM_DURATION"]
    features_df[fill_cols] = features_df[fill_cols].fillna(0.0)
    
    # Compute Comorbidity score (number of chronic conditions)
    features_df["NUM_CHRONIC_CONDITIONS"] = features_df[CHRONIC_CONDITIONS].sum(axis=1)
    
    # Define features to use
    feature_cols = [
        "AGE", "IS_FEMALE",
        "RACE_1", "RACE_2", "RACE_3", "RACE_5",
        "NUM_CHRONIC_CONDITIONS"
    ] + CHRONIC_CONDITIONS + [
        "TOTAL_CLAIM_COUNT", "AVG_CLAIM_DURATION"
    ]
    
    X = features_df[feature_cols].copy()
    y = features_df["TOTAL_ANNUAL_COST"].copy()
    
    return X, y, feature_cols

class PipelineScaler:
    """
    Helper to scale training data and keep the scaler instance for app deployment.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit_transform(self, X):
        return self.scaler.fit_transform(X)
        
    def transform(self, X):
        return self.scaler.transform(X)

if __name__ == "__main__":
    from src.data_loader import load_or_create_dataset
    b, i = load_or_create_dataset(use_mock=True, num_bene=100, num_claims=200)
    X, y, cols = build_tabular_features(b, i)
    
    print("Tabular features columns:\n", X.columns)
    print("Tabular features shape:", X.shape)
    print("Target stats:\n", y.describe())
