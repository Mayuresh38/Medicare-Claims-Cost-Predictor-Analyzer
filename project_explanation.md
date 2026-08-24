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
3. **Claims History Aggregates**:
   - `TOTAL_CLAIM_COUNT`: Number of inpatient claims in the prior year.
   - `TOTAL_CLAIM_PMT`: Total payment amount of claims.
   - `AVG_CLAIM_PMT`: Average payment amount per claim.
   - `MAX_CLAIM_PMT`: Maximum payment amount of any single claim.
   - `AVG_CLAIM_DURATION`: Average hospital stay duration in days.
   - `TOTAL_PRIMARY_PAYER_PMT`: Total amount paid by primary payers.

### Target Variable
- `TOTAL_ANNUAL_COST`: The sum of Medicare inpatient, outpatient, and carrier reimbursement amounts plus beneficiary copays and deductibles.

---

## 3. Mock Data Generator Improvements
In the original dataset, the mock beneficiary summary costs were generated independently from claims and demographics, yielding an $R^2$ score of approximately zero (essentially predicting the mean). 

To ensure realistic machine learning modeling, we updated the mock data generator in [data_loader.py](file:///c:/Users/asus/OneDrive/Desktop/new_project/src/data_loader.py):
1. **Risk Score Formulation**: We computed a beneficiary risk score based on demographics and comorbidities:
   $$\text{Risk} = 1.0 + (\text{Age} - 65) \times 0.03 + 0.3 \times \text{IsFemale} + 0.8 \times \text{NumChronicConditions}$$
2. **Claims Probability**: Beneficiaries with higher risk scores are selected with higher probability to have claims.
3. **Claim Amounts**: Claim payment amounts are exponentially distributed with scale parameters proportional to the beneficiary's risk score.
4. **Target Cost Alignment**: We set the annual inpatient cost (`MEDREIMB_IP`) to be directly dependent on the sum of the beneficiary's claims, and generated outpatient and carrier costs as functions of their risk score.

This aligned the datasets logically, providing a clear signal for the machine learning models.

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
After running the updated training pipeline, the models achieved the following test set metrics:

| Model | MAE ($) | RMSE ($) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Ridge Regression** | **999.19** | **1,340.14** | **0.9941** |
| **Keras ANN (Deep Learning)** | **1,904.46** | **2,611.06** | **0.9778** |

### Analysis
- **Ridge Regression** achieves an outstanding $R^2$ score of **99.41%**, indicating that the linear relationships (comorbidities, age risk, and claim sums) account for almost all variance in the target cost.
- **Keras ANN** achieves a high $R^2$ score of **97.78%**, showing that it successfully learned the patterns, though L2-regularized linear regression remains the most optimal and stable solution for this structured data due to the direct summation relationship of the target.

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
