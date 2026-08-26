# Medicare Claims Cost Predictor & Analyzer

This document provides a comprehensive explanation of the Medicare Claims Cost Predictor project, detailing the objectives, dataset, modeling choices, hyperparameter tuning methodology, explainable AI, and final results.

---

## 1. Project Objectives
The primary objective of this project is to predict and analyze the annual healthcare costs of Medicare beneficiaries using:
1. **Regression Analysis**: A traditional statistical modeling approach (Ridge Regression).
2. **Deep Learning**: A modern feedforward Artificial Neural Network (ANN) built in Keras/TensorFlow.

By comparing a tuned linear model against a tuned deep learning model on the same feature set, we evaluate the predictive power, complexity, and explainability of both approaches for healthcare expenditure forecasting.

---

## 2. Dataset & Features
The project utilizes the **CMS Medicare DE-SynPUF (Synthetic Public Use Files)** dataset, which includes:
- **Beneficiary Summary Files**: Containing demographics (Age, Sex, Race, State, County) and chronic conditions (Alzheimer's, Heart Failure, Cancer, Diabetes, etc.).
- **Inpatient Claims Files**: Containing claims details, claim durations, and payment amounts.

### Tabular Features Constructed
For each beneficiary, the feature engineering pipeline constructs a tabular feature vector:
1. **Demographics**:
   - `AGE` (calculated from birth date relative to reference year 2008).
   - `IS_FEMALE` (binary encoding of biological sex).
   - `RACE_1`, `RACE_2`, `RACE_3`, `RACE_5` (one-hot encoded race categories).
2. **Chronic Conditions**:
   - 11 individual binary flags representing conditions such as Diabetes, COPD, Cancer, Depression, and Heart Failure.
   - `NUM_CHRONIC_CONDITIONS`: A comorbidity count summing the total active chronic conditions for the patient.
3. **Prior Utilization (Non-Monetary)**:
   - `TOTAL_CLAIM_COUNT`: Number of inpatient claims in the prior year.
   - `AVG_CLAIM_DURATION`: Average hospital stay duration in days.

### Target Variable
- `TOTAL_ANNUAL_COST`: The sum of Medicare inpatient, outpatient, and carrier reimbursement amounts plus beneficiary copays and deductibles, natively denominated in US Dollars ($).

---

## 3. Mock Data Generator Fallback
In case downloading authentic CMS data fails, we provide a high-fidelity stochastic mock data fallback in [data_loader.py](file:///c:/Users/asus/OneDrive/Desktop/new_project/src/data_loader.py) which:
1. **Demographics & Chronic Conditions**: Generates realistic patient distributions of age, gender, race, and comorbidities.
2. **Stochastic Claims**: Simulates inpatient claims count and stays with purely probabilistic metrics.
3. **No Target Leakage**: Avoids deterministic or linear relationships. Cost and reimbursement metrics are drawn from independent exponential and log-normal distributions, with realistic added noise, ensuring no mathematical target leakage exists in the features matrix.

---

## 4. Modeling Choices & Hyperparameter Tuning
To keep the codebase simple and focused, we selected exactly **one Regression Analysis model** and **one Deep Learning model**:

### A. Ridge Regression (Regression Analysis)
- **Why Ridge?**: Ridge Regression applies L2 regularization to prevent overfitting on collinear features (such as comorbidity indicators and claim amounts).
- **Hyperparameter Tuning**: We performed a grid search (`GridSearchCV` with 5-fold cross-validation) over:
  - `alpha` (regularization strength): `[0.01, 0.1, 1.0, 10.0, 100.0]`
  - `fit_intercept`: `[True, False]`
- **Best Parameters Found**: `{'alpha': 0.1, 'fit_intercept': True}`

### B. Keras Artificial Neural Network (Deep Learning)
- **Why ANN?**: A multilayer feedforward neural network can capture complex, non-linear interactions between demographic risk factors and healthcare costs.
- **Architecture**:
  - Input Layer matching feature dimensions.
  - Three dense layers with ReLU activation, Batch Normalization, and Dropout to prevent overfitting.
  - Linear single-node output layer predicting annual cost.
- **Hyperparameter Tuning**: We implemented a validation split (80-20) grid search over:
  - `learning_rate`: `[0.001, 0.01]`
  - `dropout_rate`: `[0.1, 0.3]`
  - `batch_size`: `[16, 64]`
- **Best Parameters Found**: `{'learning_rate': 0.01, 'dropout_rate': 0.3, 'batch_size': 16}`

---

## 5. Model Performance Results
After running the updated 5-Fold Cross-Validation pipeline on the prospective USD dataset, the models achieved the following cross-validated metrics:

| Model | MAE ($) | RMSE ($) | $R^2$ Score | RMSLE |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression** | **$5,349.22** | **$6,930.74** | **-0.3746** | **0.6355** |
| **Keras ANN (Deep Learning)** | **$10,872.06** | **$12,653.16** | **-3.1403** | **8.3252** |

### Analysis
- **Prospective Benchmark Bounds**: In prospective patient-level healthcare cost prediction (where concurrent claims cost leaks are fully eliminated), typical baseline $R^2$ scores range from **15% to 45%** on large-scale real-world datasets.
- **Model Comparison**: On this small dataset, Ridge Regression performs more stably than the deep neural network. The neural network achieves a cross-validated MAE of **$10,872.06** and is stabilized using a `log1p` target scaling and Huber loss to handle the highly right-skewed and heavy-tailed distribution of expenditures.

---

## 6. Explainable AI & Model Controls
To make the predictions trustworthy for clinicians, the dashboard integrates **SHAP (SHapley Additive exPlanations)** and interactive model toggles:
- **Model Selector Dropdown:** An interactive dropdown in the sidebar allows users to dynamically switch predictions and SHAP explainability between the **Best Model**, **Ridge Regression**, and **Keras ANN**.
- **Metrics Display:** When a model is selected, its corresponding evaluation metrics (MAE, RMSE, and $R^2$ Score) are displayed dynamically directly below the predicted cost.
- **Explainability Plot:**
  - **Ridge Regression**: Explained via `LinearExplainer`, which computes exact feature contributions.
  - **Keras ANN**: Explained via `KernelExplainer`, which perturbs features to compute local feature importances (squeezed to 1D to prevent dimensionality issues).
  - The dashboard displays a waterfall or bar plot showing how demographic age, sex, chronic diseases, and inpatient claim metrics push the patient's predicted cost above or below the baseline average.

---

## 7. How to Run the Project
1. **Train/Tune Models**:
   Run the training script in python to load data, run hyperparameter tuning, compare models, and save the weights:
   ```bash
   python -m src.train
   ```
2. **Launch Streamlit Dashboard**:
   Run the Streamlit application to open the interactive UI:
   ```bash
   streamlit run app.py
   ```
3. **Run Unit Tests**:
   Verify pipeline components:
   ```bash
   python -m unittest discover -s tests -p "test_*.py"
   ```
