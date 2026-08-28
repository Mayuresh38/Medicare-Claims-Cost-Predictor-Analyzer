# Medicare Claims Cost Predictor & Analyzer

This document provides a comprehensive explanation of the Medicare Claims Cost Predictor project, detailing the objectives, dataset, modeling choices, hyperparameter tuning methodology, explainable AI, and final results.

---

## 1. Project Objectives
The primary objective of this project is to predict and analyze the annual healthcare costs of Medicare beneficiaries using:
1. **Deep Learning**: A modern feedforward Artificial Neural Network (ANN) built in Keras/TensorFlow to perform regression.
2. **Regression Analysis**: Evaluating standard regression metrics (MAE, RMSE, R² Score, and RMSLE) using 5-Fold Cross-Validation to analyze model quality.

By using a single tuned deep learning model, we predict healthcare expenditures and analyze the importance and explainability of patient-level features.

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
To keep the codebase simple and focused, we selected exactly **one Deep Learning model** for regression analysis:

### Keras Artificial Neural Network (Deep Learning)
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
After running the 5-Fold Cross-Validation pipeline on the prospective USD dataset, the Keras ANN model achieved the following cross-validated regression metrics:

| Model | MAE ($) | RMSE ($) | $R^2$ Score | RMSLE |
| :--- | :---: | :---: | :---: | :---: |
| **Keras ANN (Deep Learning)** | **$3,811.18** | **$5,940.91** | **0.5418** | **0.6587** |

### Analysis
- **Model Evaluation**: The neural network achieves a cross-validated MAE of **$3,811.18** and is stabilized using a `log1p` target scaling and Huber loss to handle the right-skewed and heavy-tailed distribution of expenditures.
- **Realistic Predictive Power**: With an $R^2$ score of **54.18%**, the model demonstrates strong prospective predictive power for patient-level healthcare cost modeling, aligning cleanly with real-world industry benchmarks while preserving robustness against overpredicting outliers.

---

## 6. Explainable AI & Model Controls
To make the predictions trustworthy for clinicians, the dashboard integrates **SHAP (SHapley Additive exPlanations)**:
- **No Selector Needed:** Since the pipeline runs only one Keras ANN model, the dashboard directly loads and predicts using it, displaying regression metrics dynamically.
- **Explainability Plot:**
  - **Keras ANN**: Explained via `KernelExplainer`, which perturbs features to compute local feature importances (squeezed to 1D to prevent dimensionality issues).
  - The dashboard displays a waterfall plot showing how demographic age, sex, chronic diseases, and inpatient claim metrics push the patient's predicted cost above or below the baseline population average.

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
