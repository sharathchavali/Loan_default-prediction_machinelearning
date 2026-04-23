# Loan Default Prediction — Credit Risk Assessment

Ensemble machine learning pipeline for credit default prediction with a focus on regulatory interpretability and class imbalance handling. Built with financial services compliance frameworks (ECOA / FCRA) in mind.

## Tech Stack

- **Language:** Python
- **ML Libraries:** Scikit-learn, XGBoost, imbalanced-learn
- **Interpretability:** SHAP
- **Data Handling:** Pandas, NumPy
- **Workflow:** Jupyter Notebook, Git

## Models Built

| Model | Purpose |
|-------|---------|
| Logistic Regression | Baseline, interpretable |
| Random Forest | Non-linear relationships |
| XGBoost | Final production model |

## Results

- **Accuracy:** 92.2%
- **AUC-ROC:** 0.965
- **Minority class recall:** +18% improvement after SMOTE

## Key Techniques

- **SMOTE** for class imbalance — minority class recall improved by 18%
- **SHAP values** for model interpretability, aligned with ECOA/FCRA adverse-action-notice requirements
- **Stratified k-fold cross-validation** for robust performance estimation
- **Feature importance analysis** to identify the strongest drivers of default risk

## Why Interpretability Matters Here

In credit decisioning, regulators (ECOA, FCRA) require that applicants who are denied credit receive specific reasons. SHAP values make this possible for tree-based ensembles — each prediction can be decomposed into per-feature contributions, producing audit-ready reason codes.

## Author

Sharath Chandra Chavali — Data Analyst
