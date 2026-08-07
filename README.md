# 🩺 Disease Prediction from Medical Data

**Objective:** Predict the possibility of disease based on structured patient data.
**Approach:** Classification models trained on medical datasets, with full evaluation and an interactive demo.

## Project Structure

```
disease_prediction/
├── main.py                  # Run this: trains & evaluates everything
├── app.py                   # Streamlit interactive prediction demo
├── requirements.txt
├── data/                    # Put real UCI CSVs here (optional, see below)
├── src/
│   ├── data_loader.py       # Loads Breast Cancer / Heart / Diabetes data
│   ├── preprocess.py        # Cleaning, scaling, train/test split
│   ├── models.py            # Logistic Regression, SVM, Random Forest, XGBoost
│   └── evaluate.py          # Metrics, confusion matrix, ROC curve, feature importance
└── outputs/
    ├── models/               # Trained model .pkl files + scalers
    ├── plots/                # Confusion matrices, ROC curves, feature importance
    └── reports/              # Per-model classification reports + summary CSVs
```

## Datasets

| Dataset | Features | Target | Status in this project |
|---|---|---|---|
| **Breast Cancer (Wisconsin)** | 30 numeric features (cell nuclei measurements) | malignant / benign | ✅ Real data, loaded automatically via `sklearn.datasets` |
| **Heart Disease (UCI Cleveland)** | 13 features (age, sex, chest pain type, cholesterol, etc.) | disease / no disease | ⚠️ Synthetic placeholder (see below) |
| **Diabetes (Pima Indians)** | 8 features (glucose, BMI, blood pressure, etc.) | diabetic / not diabetic | ⚠️ Synthetic placeholder (see below) |

### Using the real Heart Disease / Diabetes datasets

This environment doesn't have internet access, so `heart.csv` and `diabetes.csv`
couldn't be downloaded automatically. The code **generates realistic synthetic
data with the same columns** so the whole pipeline runs today. To switch to the
real data (recommended for anything beyond a demo):

1. Download from the UCI Machine Learning Repository:
   - Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease
   - Diabetes (Pima Indians): https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Save them as `data/heart.csv` (must have a `target` column, 1 = disease) and
   `data/diabetes.csv` (must have an `Outcome` column, 1 = diabetic).
3. Re-run `python main.py` — the loader automatically detects and uses the real files.

## Models

- **Logistic Regression** — fast, interpretable baseline
- **SVM (RBF Kernel)** — good for non-linear boundaries
- **Random Forest** — ensemble of decision trees, handles non-linearity + feature importance
- **XGBoost** — gradient boosting, usually the strongest performer on tabular data
  (falls back to scikit-learn's `GradientBoostingClassifier` automatically if
  `xgboost` isn't installed)

All models are wrapped identically so they're evaluated on the exact same
train/test split and metrics for a fair comparison.

## How to Run

```bash
pip install -r requirements.txt

# Train + evaluate all models on all 3 datasets
python main.py

# Or just one dataset
python main.py --dataset breast_cancer
python main.py --dataset heart
python main.py --dataset diabetes

# Launch the interactive prediction demo (after running main.py at least once)
streamlit run app.py
```

## What gets produced

For every (dataset × model) combination:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC → `outputs/reports/<dataset>_summary.csv`
- Confusion matrix heatmap → `outputs/plots/<dataset>_<model>_confusion.png`
- Feature importance chart → `outputs/plots/<dataset>_<model>_importance.png`
- Full classification report (text) → `outputs/reports/<dataset>_<model>_report.txt`
- Trained model → `outputs/models/<dataset>_<model>.pkl`

Plus, per dataset:
- Combined ROC curve comparing all 4 models → `outputs/plots/<dataset>_roc_curves.png`
- A best-model recommendation (by F1-Score), printed to console

And overall:
- `outputs/reports/all_datasets_summary.csv` — every model/dataset combination in one table

## Example Results (this run)

Breast Cancer, being real, high-quality, well-separated data, scores highest
(~97% accuracy). Heart/Diabetes numbers are lower and more "realistic" because
they're synthetic data without a truly learnable signal — swap in the real
UCI files to get meaningful clinical results there.

See `outputs/reports/all_datasets_summary.csv` for exact numbers from this run.

## Notes on Methodology

- **Preprocessing:** duplicate rows dropped, missing values median-imputed,
  features standardized (zero mean, unit variance) before SVM/Logistic Regression
  (tree-based models don't strictly need scaling but it doesn't hurt them here).
- **Train/test split:** 80/20, stratified by target class, `random_state=42` for reproducibility.
- **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC — precision/recall
  are reported because in medical diagnosis, false negatives (missed disease)
  and false positives (unnecessary alarm) matter differently than raw accuracy.
  
## 🎓 Internship Information

This project was developed as part of the **CodeAlpha Internship Program**. It focuses on applying machine learning techniques to predict diseases using structured medical datasets while following software engineering best practices for project organization, documentation, and model evaluation.
## Disclaimer

This is an educational project, not a medical device. Predictions should never
be used for actual clinical decision-making.
