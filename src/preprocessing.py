import pandas as pd
import numpy as np

CHRONIC_CONDITIONS = [
    "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD", 
    "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS", "SP_RA_OA", "SP_STRKETIA"
]

def preprocess_beneficiaries(bene_df):
    """
    Cleans demographic features and target values.
    """
    df = bene_df.copy()
    
    # Calculate Age (assuming reference year 2008)
    df["BENE_BIRTH_DT"] = pd.to_datetime(df["BENE_BIRTH_DT"], format="%Y%m%d", errors="coerce")
    df["AGE"] = 2008 - df["BENE_BIRTH_DT"].dt.year
    df["AGE"] = df["AGE"].fillna(df["AGE"].median())
    
    # Recode Sex: 1 (Male) -> 0, 2 (Female) -> 1
    df["IS_FEMALE"] = (df["BENE_SEX_IDENT_CD"] == 2).astype(int)
    
    # Recode Race: One-hot encode race codes
    # Coded as: 1=White, 2=Black, 3=Others, 5=Hispanic
    for r_code in [1, 2, 3, 5]:
        df[f"RACE_{r_code}"] = (df["BENE_RACE_CD"] == r_code).astype(int)
        
    # Recode Chronic conditions: 1 (Yes) -> 1, 2 (No) -> 0
    for col in CHRONIC_CONDITIONS:
        if col in df.columns:
            df[col] = (df[col] == 1).astype(int)
            
    # Convert dollar values to Indian Rupees (INR): 1 USD = 83 INR
    cost_cols = [
        "MEDREIMB_IP", "BENRES_IP", "PPPYMT_IP",
        "MEDREIMB_OP", "BENRES_OP", "PPPYMT_OP",
        "MEDREIMB_CAR", "BENRES_CAR", "PPPYMT_CAR"
    ]
    for col in cost_cols:
        if col in df.columns:
            df[col] = df[col] * 83.0
            
    # Calculate total annual costs (Target Variable)
    # Target = Medicare Reimbursement + Beneficiary Responsibility for Inpatient + Outpatient + Carrier
    df["TOTAL_ANNUAL_COST"] = (
        df.get("MEDREIMB_IP", 0.0) + df.get("BENRES_IP", 0.0) +
        df.get("MEDREIMB_OP", 0.0) + df.get("BENRES_OP", 0.0) +
        df.get("MEDREIMB_CAR", 0.0) + df.get("BENRES_CAR", 0.0)
    )
    
    return df

def preprocess_claims(ip_df):
    """
    Cleans inpatient claims, converts dates, and calculates durations.
    """
    df = ip_df.copy()
    df["CLM_FROM_DT"] = pd.to_datetime(df["CLM_FROM_DT"], format="%Y%m%d", errors="coerce")
    df["CLM_THRU_DT"] = pd.to_datetime(df["CLM_THRU_DT"], format="%Y%m%d", errors="coerce")
    
    # Handle missing dates
    df = df.dropna(subset=["CLM_FROM_DT"])
    
    # Calculate claim duration
    df["CLAIM_DURATION"] = (df["CLM_THRU_DT"] - df["CLM_FROM_DT"]).dt.days
    df["CLAIM_DURATION"] = df["CLAIM_DURATION"].clip(lower=0) # ensure non-negative
    
    # Default numeric fills and convert dollar values to INR
    df["CLM_PMT_AMT"] = df["CLM_PMT_AMT"].fillna(0.0) * 83.0
    df["NCH_PRMRY_PYR_CLM_PD_AMT"] = df["NCH_PRMRY_PYR_CLM_PD_AMT"].fillna(0.0) * 83.0
    
    return df

def build_longitudinal_sequences(bene_df, ip_df, max_seq_len=5):
    """
    Constructs sequential patient claim histories for sequential models (LSTM/GRU).
    For each patient, we build a history of their inpatient claims.
    Each claim in the sequence has features:
      [claim_payment, primary_payer_payment, claim_duration, start_day_of_year]
      
    Returns:
        X_seq (numpy array of shape [num_benes, max_seq_len, num_features])
    """
    processed_bene = preprocess_beneficiaries(bene_df)
    processed_claims = preprocess_claims(ip_df)
    
    # Sort claims chronologically
    processed_claims = processed_claims.sort_values(by=["DESYNPUF_ID", "CLM_FROM_DT"])
    
    num_features = 4  # [CLM_PMT_AMT, NCH_PRMRY_PYR_CLM_PD_AMT, CLAIM_DURATION, START_DAY_OF_YEAR]
    sequences = []
    
    for bene_id in processed_bene["DESYNPUF_ID"]:
        bene_claims = processed_claims[processed_claims["DESYNPUF_ID"] == bene_id]
        
        # Build features for each claim
        seq_data = []
        for _, claim in bene_claims.iterrows():
            start_day = claim["CLM_FROM_DT"].timetuple().tm_yday
            features = [
                claim["CLM_PMT_AMT"],
                claim["NCH_PRMRY_PYR_CLM_PD_AMT"],
                claim["CLAIM_DURATION"],
                start_day
            ]
            seq_data.append(features)
            
        # Pad or truncate sequence
        if len(seq_data) == 0:
            # Pad with zeros
            seq_data = np.zeros((max_seq_len, num_features))
        elif len(seq_data) < max_seq_len:
            # Pre-pad with zeros
            padding = np.zeros((max_seq_len - len(seq_data), num_features))
            seq_data = np.vstack([padding, seq_data])
        else:
            # Truncate to keep the most recent claims
            seq_data = np.array(seq_data[-max_seq_len:])
            
        sequences.append(seq_data)
        
    return np.array(sequences)

if __name__ == "__main__":
    from data_loader import load_or_create_dataset
    b, i = load_or_create_dataset(use_mock=True, num_bene=100, num_claims=200)
    
    b_proc = preprocess_beneficiaries(b)
    i_proc = preprocess_claims(i)
    seqs = build_longitudinal_sequences(b, i, max_seq_len=5)
    
    print("Preprocessed Beneficiaries columns:\n", b_proc.columns)
    print("Preprocessed Inpatient Claims columns:\n", i_proc.columns)
    print("Longitudinal sequence shape:", seqs.shape)
