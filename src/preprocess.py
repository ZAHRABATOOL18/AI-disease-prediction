"""
preprocess.py
-------------
Cleaning, outlier handling, scaling, feature selection, and train/test
splitting for the medical datasets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

RANDOM_STATE = 42


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop exact duplicate rows, impute missing values."""
    df = df.drop_duplicates().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != "target"]
    if df[feature_cols].isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy="median")
        df[feature_cols] = imputer.fit_transform(df[feature_cols])
    return df


def detect_and_handle_outliers(df: pd.DataFrame, method: str = "iqr",
                                action: str = "cap", factor: float = 1.5):
    """
    Detects outliers per numeric feature using the IQR rule
    (Q1 - factor*IQR, Q3 + factor*IQR) and either caps (winsorizes) or
    removes rows containing them.

    Returns (cleaned_df, report_df) where report_df has one row per feature
    showing how many outliers were found and what bounds were used.
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c != "target"]

    report_rows = []
    outlier_mask_any = pd.Series(False, index=df.index)

    for col in feature_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        col_mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = int(col_mask.sum())
        report_rows.append({
            "feature": col, "lower_bound": round(lower, 3),
            "upper_bound": round(upper, 3), "n_outliers": n_outliers,
            "pct_outliers": round(100 * n_outliers / len(df), 2),
        })

        if action == "cap":
            df[col] = df[col].clip(lower=lower, upper=upper)
        elif action == "remove":
            outlier_mask_any |= col_mask

    if action == "remove":
        df = df[~outlier_mask_any].reset_index(drop=True)

    report_df = pd.DataFrame(report_rows).sort_values("n_outliers", ascending=False)
    return df, report_df


def select_features(X_train, y_train, feature_names, k="auto"):
    """
    Univariate feature selection with ANOVA F-value (SelectKBest, f_classif).
    k='auto' keeps the top half of features (min 5). Returns
    (selected_indices, selected_names, scores_df) where scores_df ranks
    every feature by its F-score (for a report/plot even if not dropped).
    """
    n_features = X_train.shape[1]
    if k == "auto":
        k = max(5, n_features // 2)
    k = min(k, n_features)

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)

    scores_df = pd.DataFrame({
        "feature": feature_names,
        "f_score": np.round(selector.scores_, 3),
        "p_value": np.round(selector.pvalues_, 5),
        "selected": selector.get_support(),
    }).sort_values("f_score", ascending=False).reset_index(drop=True)

    selected_indices = np.where(selector.get_support())[0]
    selected_names = [feature_names[i] for i in selected_indices]

    return selected_indices, selected_names, scores_df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2):
    """Train/test split + standard scaling of features. Returns arrays + scaler."""
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()
