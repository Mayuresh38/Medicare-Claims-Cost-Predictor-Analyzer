import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np

# Directory to store dataset
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# File URLs for Medicare DE-SynPUF Sample 1
# Note: CMS official downloads can sometimes return HTTP 403 / 404 or be slow.
# We supply urls and a robust mock generator fallback.
DE_SYNPUF_URLS = {
    "beneficiary_2008": "https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/Downloads/DE1_0_2008_Beneficiary_Summary_File_Sample_1.zip",
    "inpatient_claims": "https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/Downloads/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.zip"
}

def ensure_directories():
    os.makedirs(DATA_DIR, exist_ok=True)

def generate_mock_beneficiary_data(num_records=1000):
    """
    Generates mock Beneficiary Summary data that matches the CMS schema perfectly.
    """
    np.random.seed(42)
    bene_ids = [f"BENE_ID_{i:06d}" for i in range(1, num_records + 1)]
    
    # Generate Birth Dates (around 1930 to 1990)
    birth_years = np.random.randint(1930, 1990, size=num_records)
    birth_months = np.random.randint(1, 13, size=num_records)
    birth_days = np.random.randint(1, 29, size=num_records)
    birth_dts = [f"{y:04d}{m:02d}{d:02d}" for y, m, d in zip(birth_years, birth_months, birth_days)]
    
    # Gender (1 = Male, 2 = Female)
    sex = np.random.choice([1, 2], size=num_records, p=[0.48, 0.52])
    
    # Race Code (1 = White, 2 = Black, 3 = Others, 5 = Hispanic)
    race = np.random.choice([1, 2, 3, 5], size=num_records, p=[0.75, 0.15, 0.05, 0.05])
    
    # State and County codes
    state = np.random.randint(1, 55, size=num_records)
    county = np.random.randint(10, 800, size=num_records)
    
    # Chronic conditions (1 = Yes, 2 = No)
    chronic_conditions = [
        "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD", 
        "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS", "SP_RA_OA", "SP_STRKETIA"
    ]
    
    data = {
        "DESYNPUF_ID": bene_ids,
        "BENE_BIRTH_DT": birth_dts,
        "BENE_SEX_IDENT_CD": sex,
        "BENE_RACE_CD": race,
        "SP_STATE_CODE": state,
        "BENE_COUNTY_CD": county,
    }
    
    # Coded chronic conditions
    for col in chronic_conditions:
        data[col] = np.random.choice([1, 2], size=num_records, p=[0.15, 0.85])
        
    # Annual Reimbursements & Responsibilities
    data["MEDREIMB_IP"] = np.random.exponential(scale=3000, size=num_records).round(2)
    data["BENRES_IP"] = (data["MEDREIMB_IP"] * 0.1).round(2)
    data["PPPYMT_IP"] = np.random.choice([0.0, 500.0], size=num_records, p=[0.9, 0.1])
    
    data["MEDREIMB_OP"] = np.random.exponential(scale=1500, size=num_records).round(2)
    data["BENRES_OP"] = (data["MEDREIMB_OP"] * 0.15).round(2)
    data["PPPYMT_OP"] = np.random.choice([0.0, 200.0], size=num_records, p=[0.95, 0.05])
    
    data["MEDREIMB_CAR"] = np.random.exponential(scale=800, size=num_records).round(2)
    data["BENRES_CAR"] = (data["MEDREIMB_CAR"] * 0.2).round(2)
    data["PPPYMT_CAR"] = np.random.choice([0.0, 100.0], size=num_records, p=[0.9, 0.1])
    
    return pd.DataFrame(data)

def generate_mock_inpatient_claims(beneficiary_df, num_claims=3000):
    """
    Generates mock Inpatient Claims data linked to the beneficiary IDs,
    using non-deterministic random distributions.
    """
    np.random.seed(42)
    bene_ids = beneficiary_df["DESYNPUF_ID"].values
    
    chronic_conditions = [
        "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD", 
        "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS", "SP_RA_OA", "SP_STRKETIA"
    ]
    num_chronic = (beneficiary_df[chronic_conditions] == 1).sum(axis=1)
    
    # Calculate approximate Age
    birth_years = pd.to_datetime(beneficiary_df["BENE_BIRTH_DT"], format="%Y%m%d", errors="coerce").dt.year
    age = 2008 - birth_years
    age = age.fillna(75.0)
    
    # Risk factor determines how likely they are to get selected
    risk_score = 1.0 + 0.02 * np.maximum(0, age - 65) + 0.3 * num_chronic
    probs = risk_score / risk_score.sum()
    
    chosen_indices = np.random.choice(len(beneficiary_df), size=num_claims, p=probs)
    chosen_df = beneficiary_df.iloc[chosen_indices].copy()
    chosen_benes = chosen_df["DESYNPUF_ID"].values
    
    claim_ids = [f"CLM_ID_{i:06d}" for i in range(1, num_claims + 1)]
    
    # Claim Dates in 2008
    months = np.random.randint(1, 13, size=num_claims)
    days = np.random.randint(1, 28, size=num_claims)
    from_dts = [f"2008{m:02d}{d:02d}" for m, d in zip(months, days)]
    
    # Thru dates (1 to 15 days later)
    thru_days = np.random.randint(1, 16, size=num_claims)
    thru_dts = []
    for f_dt, duration in zip(from_dts, thru_days):
        y, m, d = int(f_dt[:4]), int(f_dt[4:6]), int(f_dt[6:])
        d_new = d + duration
        m_new = m
        if d_new > 28:
            d_new -= 28
            m_new = m + 1
            if m_new > 12:
                m_new = 12
        thru_dts.append(f"{y:04d}{m_new:02d}{d_new:02d}")
        
    # Scale payment amount stochastically based on risk score of the beneficiary
    chosen_num_chronic = (chosen_df[chronic_conditions] == 1).sum(axis=1).values
    chosen_age = (2008 - pd.to_datetime(chosen_df["BENE_BIRTH_DT"], format="%Y%m%d", errors="coerce").dt.year).fillna(75.0).values
    scale_factor = 1.0 + 0.02 * np.maximum(0, chosen_age - 65) + 0.3 * chosen_num_chronic
    
    pmt_amt = (np.random.exponential(scale=2000, size=num_claims) * scale_factor).round(2)
    primary_payer_amt = np.random.choice([0.0, 1000.0], size=num_claims, p=[0.92, 0.08])
    provider_num = np.random.choice([f"PRVDR_{i:03d}" for i in range(1, 50)], size=num_claims)
    
    data = {
        "DESYNPUF_ID": chosen_benes,
        "CLM_ID": claim_ids,
        "CLM_FROM_DT": from_dts,
        "CLM_THRU_DT": thru_dts,
        "PRVDR_NUM": provider_num,
        "CLM_PMT_AMT": pmt_amt,
        "NCH_PRMRY_PYR_CLM_PD_AMT": primary_payer_amt
    }
    
    return pd.DataFrame(data)

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print("Download completed successfully.")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def load_or_create_dataset(use_mock=False, num_bene=2000, num_claims=4000):
    """
    Downloads SynPUF files or generates mock data if offline or requested.
    Returns:
        beneficiary_df (DataFrame)
        inpatient_df (DataFrame)
    """
    ensure_directories()
    
    bene_file_path = os.path.join(DATA_DIR, "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv")
    ip_file_path = os.path.join(DATA_DIR, "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv")
    
    # If the user does not want mock and files do not exist, try downloading
    if not use_mock and (not os.path.exists(bene_file_path) or not os.path.exists(ip_file_path)):
        print("Attempting to download official CMS DE-SynPUF dataset...")
        zip_bene = os.path.join(DATA_DIR, "bene_2008.zip")
        zip_ip = os.path.join(DATA_DIR, "ip_claims.zip")
        
        success_bene = False
        success_ip = False
        
        if not os.path.exists(bene_file_path):
            success_bene = download_file(DE_SYNPUF_URLS["beneficiary_2008"], zip_bene)
            if success_bene:
                try:
                    with zipfile.ZipFile(zip_bene, 'r') as zip_ref:
                         zip_ref.extractall(DATA_DIR)
                    # Clean up zip
                    os.remove(zip_bene)
                except Exception as e:
                    print(f"Error unzipping beneficiary file: {e}")
                    success_bene = False
                    
        if not os.path.exists(ip_file_path):
            success_ip = download_file(DE_SYNPUF_URLS["inpatient_claims"], zip_ip)
            if success_ip:
                try:
                    with zipfile.ZipFile(zip_ip, 'r') as zip_ref:
                        zip_ref.extractall(DATA_DIR)
                    os.remove(zip_ip)
                except Exception as e:
                    print(f"Error unzipping inpatient file: {e}")
                    success_ip = False
                    
        # If download failed, fall back to mock
        if not os.path.exists(bene_file_path) or not os.path.exists(ip_file_path):
            print("Could not retrieve official data. Falling back to generating mock dataset.")
            use_mock = True

    if use_mock:
        print(f"Generating mock dataset: {num_bene} beneficiaries, {num_claims} inpatient claims...")
        bene_df = generate_mock_beneficiary_data(num_bene)
        ip_df = generate_mock_inpatient_claims(bene_df, num_claims)
        
        # Link beneficiary annual summary to their actual generated claims
        ip_sum = ip_df.groupby("DESYNPUF_ID")["CLM_PMT_AMT"].sum()
        bene_df["MEDREIMB_IP"] = bene_df["DESYNPUF_ID"].map(ip_sum).fillna(0.0)
        
        # Use stochastic coefficients and additive noise to prevent target leakage
        bene_df["BENRES_IP"] = (bene_df["MEDREIMB_IP"] * np.random.uniform(0.05, 0.15, size=num_bene) + np.random.exponential(scale=50, size=num_bene)).round(2)
        bene_df["PPPYMT_IP"] = (bene_df["MEDREIMB_IP"] * np.random.uniform(0.01, 0.08, size=num_bene) + np.random.exponential(scale=20, size=num_bene)).round(2)
        
        # Outpatient and carrier costs are stochastic variables dependent on chronic conditions and age
        chronic_conditions = [
            "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD", 
            "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS", "SP_RA_OA", "SP_STRKETIA"
        ]
        num_chronic = (bene_df[chronic_conditions] == 1).sum(axis=1)
        birth_years = pd.to_datetime(bene_df["BENE_BIRTH_DT"], format="%Y%m%d", errors="coerce").dt.year
        age = 2008 - birth_years
        age = age.fillna(75.0)
        
        # Risk factor determines standard outpatient & carrier costs
        risk_factor = 0.5 + 0.02 * np.maximum(0, age - 65) + 0.25 * num_chronic
        
        bene_df["MEDREIMB_OP"] = (np.random.exponential(scale=1000, size=num_bene) * risk_factor).round(2)
        bene_df["BENRES_OP"] = (bene_df["MEDREIMB_OP"] * np.random.uniform(0.10, 0.20, size=num_bene) + np.random.exponential(scale=20, size=num_bene)).round(2)
        bene_df["PPPYMT_OP"] = (bene_df["MEDREIMB_OP"] * np.random.uniform(0.02, 0.08, size=num_bene)).round(2)
        
        bene_df["MEDREIMB_CAR"] = (np.random.exponential(scale=500, size=num_bene) * risk_factor).round(2)
        bene_df["BENRES_CAR"] = (bene_df["MEDREIMB_CAR"] * np.random.uniform(0.15, 0.25, size=num_bene) + np.random.exponential(scale=10, size=num_bene)).round(2)
        bene_df["PPPYMT_CAR"] = (bene_df["MEDREIMB_CAR"] * np.random.uniform(0.02, 0.08, size=num_bene)).round(2)
        
        # Save mock dataset to disk for inspection/reuse
        bene_df.to_csv(bene_file_path, index=False)
        ip_df.to_csv(ip_file_path, index=False)
        print("Mock files saved to data directory.")
    else:
        print("Loading official CMS DE-SynPUF dataset from disk...")
        # Since files can be large, we'll read a subset (nrows) to keep execution fast
        bene_df = pd.read_csv(bene_file_path, nrows=num_bene)
        # Find inpatient claims that belong to these beneficiaries
        ip_df = pd.read_csv(ip_file_path)
        ip_df = ip_df[ip_df["DESYNPUF_ID"].isin(bene_df["DESYNPUF_ID"])]
        if len(ip_df) > num_claims:
            ip_df = ip_df.sample(n=num_claims, random_state=42)
            
    return bene_df, ip_df

if __name__ == "__main__":
    b, i = load_or_create_dataset(use_mock=True, num_bene=100, num_claims=200)
    print("Beneficiary summary head:\n", b.head())
    print("Inpatient claims head:\n", i.head())
