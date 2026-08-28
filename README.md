# Medicare Claims Cost Predictor & Analyzer

An end-to-end prospective patient-level prediction pipeline to model and forecast total annual Medicare costs using a **Keras Artificial Neural Network (ANN)** deep learning regression model, complete with an interactive Streamlit dashboard and SHAP explainability.

## 🏥 Objectives
1. **Prospective Feature Engineering**: Predicts annual expenditures using baseline demographics, comorbidities (chronic conditions), and non-monetary prior-year healthcare utilization metrics.
2. **Regression Analysis**: Trains and evaluates a deep learning regression model (Keras ANN) under 5-Fold Cross-Validation splits, reporting standard regression evaluation metrics (MAE, RMSE, R² Score, and RMSLE).
3. **Interactive Control & Explainability**: Computes predictions on Streamlit and displays demographic and clinical feature contributions in original US Dollars ($) using SHAP waterfall plots.

## 📊 Tabular Predictors & Targets
- **Predictor Matrix X**:
  - **Demographics**: `AGE`, `IS_FEMALE`, and one-hot encoded race variables (`RACE_1`, `RACE_2`, `RACE_3`, `RACE_5`).
  - **Clinical Conditions**: 11 binary chronic conditions and comorbidity score (`NUM_CHRONIC_CONDITIONS`).
  - **Prior Utilization**: `TOTAL_CLAIM_COUNT`, `AVG_CLAIM_DURATION`.
- **Target Variable y**:
  - `TOTAL_ANNUAL_COST` (native CMS Medicare DE-SynPUF dollar ($) expenditures, representing Medicare reimbursement plus beneficiary responsibility).

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute Training Pipeline**:
   Runs 5-fold cross-validation, saves metrics to `models/model_comparison.csv`, and fits the final Keras ANN model on the entire dataset.
   ```bash
   python -m src.train
   ```
3. **Launch Streamlit Dashboard**:
   Loads the saved feature scaler, final models, and launches the interactive frontend.
   ```bash
   streamlit run app.py
   ```
4. **Run Unit Tests**:
   ```bash
   python -m unittest discover -s tests -p "test_*.py"
   ```
