"""
data_loader.py
---------------
Loads the three medical datasets used in this project:

1. Breast Cancer (Wisconsin) - loaded directly from scikit-learn (real UCI data,
   no download required).
2. Heart Disease (UCI Cleveland) - loaded from data/heart.csv if present,
   otherwise a realistic synthetic dataset with the same schema is generated
   so the pipeline can run end-to-end without internet access.
3. Diabetes (Pima Indians) - loaded from data/diabetes.csv if present,
   otherwise a realistic synthetic dataset with the same schema is generated.

To use the REAL UCI datasets:
- Download "heart.csv" (UCI Heart Disease / Cleveland, target column named
  'target', 1 = disease) and place it in data/heart.csv
- Download "diabetes.csv" (Pima Indians Diabetes Database, target column
  named 'Outcome') and place it in data/diabetes.csv
The loader functions below automatically prefer the real file if it exists.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

RANDOM_STATE = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_breast_cancer_data():
    """Real UCI Breast Cancer Wisconsin dataset via sklearn."""
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()
    df.rename(columns={"target": "target"}, inplace=True)
    # In sklearn's version, target=0 means malignant, 1 means benign.
    # Flip so 1 = disease present (malignant), consistent with the other datasets.
    df["target"] = 1 - df["target"]
    return df, "Breast Cancer (Wisconsin)"


def _generate_synthetic_heart(n=400, seed=RANDOM_STATE):
    """Synthetic stand-in for the UCI Heart Disease dataset (same 13 features)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(29, 78, n)
    sex = rng.integers(0, 2, n)
    cp = rng.integers(0, 4, n)                       # chest pain type
    trestbps = rng.integers(94, 201, n)               # resting blood pressure
    chol = rng.integers(126, 565, n)                  # serum cholesterol
    fbs = rng.integers(0, 2, n)                        # fasting blood sugar > 120
    restecg = rng.integers(0, 3, n)
    thalach = rng.integers(71, 203, n)                # max heart rate achieved
    exang = rng.integers(0, 2, n)                     # exercise induced angina
    oldpeak = np.round(rng.uniform(0, 6.2, n), 1)
    slope = rng.integers(0, 3, n)
    ca = rng.integers(0, 4, n)
    thal = rng.integers(0, 3, n)

    # Build target with a realistic logistic relationship to risk factors
    risk = (
        0.035 * (age - 50)
        + 0.6 * sex
        + 0.5 * cp
        + 0.015 * (trestbps - 130)
        + 0.006 * (chol - 240)
        + 0.4 * fbs
        - 0.02 * (thalach - 150)
        + 0.9 * exang
        + 0.35 * oldpeak
        + 0.4 * ca
        + rng.normal(0, 1.0, n)
    )
    risk_z = (risk - risk.mean()) / risk.std()  # center so classes end up ~balanced
    prob = 1 / (1 + np.exp(-risk_z))
    target = (rng.uniform(0, 1, n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal, "target": target,
    })
    return df


def _generate_synthetic_diabetes(n=500, seed=RANDOM_STATE):
    """Synthetic stand-in for the Pima Indians Diabetes dataset (same 8 features)."""
    rng = np.random.default_rng(seed)
    pregnancies = rng.integers(0, 15, n)
    glucose = rng.integers(60, 200, n)
    blood_pressure = rng.integers(40, 122, n)
    skin_thickness = rng.integers(0, 60, n)
    insulin = rng.integers(0, 300, n)
    bmi = np.round(rng.uniform(15, 55, n), 1)
    dpf = np.round(rng.uniform(0.05, 2.5, n), 3)      # diabetes pedigree function
    age = rng.integers(21, 81, n)

    risk = (
        0.04 * (glucose - 120)
        + 0.03 * (bmi - 30)
        + 0.02 * (age - 40)
        + 0.4 * dpf
        + 0.15 * pregnancies
        + 0.01 * (blood_pressure - 70)
        + rng.normal(0, 1.2, n)
    )
    risk_z = (risk - risk.mean()) / risk.std()  # center so classes end up ~balanced
    prob = 1 / (1 + np.exp(-risk_z))
    outcome = (rng.uniform(0, 1, n) < prob).astype(int)

    df = pd.DataFrame({
        "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age, "Outcome": outcome,
    })
    return df


def load_heart_data():
    path = os.path.join(DATA_DIR, "heart.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "target" not in df.columns:
            raise ValueError("heart.csv must contain a 'target' column (1=disease, 0=no disease)")
        source = "Heart Disease (UCI Cleveland - real data)"
    else:
        df = _generate_synthetic_heart()
        source = "Heart Disease (SYNTHETIC placeholder - replace data/heart.csv with the real UCI file)"
    return df, source


def load_diabetes_data():
    path = os.path.join(DATA_DIR, "diabetes.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "Outcome" not in df.columns:
            raise ValueError("diabetes.csv must contain an 'Outcome' column (1=diabetic, 0=not diabetic)")
        df = df.rename(columns={"Outcome": "target"})
        source = "Diabetes (Pima Indians - real data)"
    else:
        df = _generate_synthetic_diabetes()
        df = df.rename(columns={"Outcome": "target"})
        source = "Diabetes (SYNTHETIC placeholder - replace data/diabetes.csv with the real Pima Indians file)"
    return df, source


DATASETS = {
    "breast_cancer": load_breast_cancer_data,
    "heart": load_heart_data,
    "diabetes": load_diabetes_data,
}


if __name__ == "__main__":
    for key, loader in DATASETS.items():
        df, source = loader()
        print(f"[{key}] {source} -> shape={df.shape}, "
              f"positive_rate={df['target'].mean():.2f}")
