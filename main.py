"""
main.py
-------
Disease Prediction from Medical Data — end-to-end pipeline.

Trains Logistic Regression, SVM, Random Forest, and XGBoost on three
structured medical datasets (Breast Cancer, Heart Disease, Diabetes),
evaluates each model, and saves plots + a summary report.

Usage:
    python main.py                 # run all datasets
    python main.py --dataset heart # run a single dataset
"""

import argparse
import json
import os
import pickle
import time

import pandas as pd

from src.data_loader import DATASETS
from src.preprocess import clean, detect_and_handle_outliers, select_features, split_and_scale
from src.models import get_models
from src.tuning import tune_all_models
from src.pca_viz import plot_pca_scatter, plot_pca_explained_variance
from src.explain import compute_shap_values, plot_shap_summary, plot_shap_waterfall, top_reasons, format_reasons_text
from src.evaluate import (
    compute_metrics, plot_confusion_matrix, plot_roc_curves,
    plot_feature_importance, save_classification_report,
    save_confusion_matrix_data, save_roc_data
)
from src.utils import safe_model_filename

DISEASE_LABELS = {
    "breast_cancer": "Malignant Tumor",
    "heart": "Heart Disease",
    "diabetes": "Diabetes",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")


def run_dataset(dataset_key: str):
    loader = DATASETS[dataset_key]
    df, source_label = loader()
    df = clean(df)

    df, outlier_report = detect_and_handle_outliers(df, method="iqr", action="cap")
    outlier_report.to_csv(os.path.join(REPORTS_DIR, f"{dataset_key}_outlier_report.csv"), index=False)
    n_flagged = int((outlier_report["n_outliers"] > 0).sum())
    print(f"\nOutlier check ({dataset_key}): {n_flagged}/{len(outlier_report)} features had "
          f"IQR outliers -> capped (winsorized) to the IQR bounds.")

    print(f"\n{'='*70}\nDATASET: {source_label}\nShape: {df.shape} | "
          f"Positive (disease) rate: {df['target'].mean():.2%}\n{'='*70}")

    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)

    # --- PCA visualization (before feature selection, on full scaled feature set) ---
    n_for_95 = plot_pca_explained_variance(
        X_train, f"PCA Explained Variance — {dataset_key}",
        os.path.join(PLOTS_DIR, f"{dataset_key}_pca_variance.png")
    )
    plot_pca_scatter(
        X_train, y_train.values, f"PCA Projection (PC1 vs PC2) — {dataset_key}",
        os.path.join(PLOTS_DIR, f"{dataset_key}_pca_scatter.png")
    )
    print(f"PCA: {n_for_95}/{len(feature_names)} components needed to reach 95% variance.")

    # --- Feature selection (SelectKBest, ANOVA F-value) ---
    selected_idx, selected_names, fs_scores = select_features(X_train, y_train, feature_names)
    fs_scores.to_csv(os.path.join(REPORTS_DIR, f"{dataset_key}_feature_scores.csv"), index=False)
    print(f"Feature selection: keeping top {len(selected_names)}/{len(feature_names)} features "
          f"by ANOVA F-score -> {selected_names}")

    with open(os.path.join(MODELS_DIR, f"{dataset_key}_feature_map.json"), "w") as f:
        json.dump({"all_features": feature_names, "selected_features": selected_names}, f, indent=2)

    X_train = X_train[:, selected_idx]
    X_test = X_test[:, selected_idx]
    feature_names = selected_names

    models = get_models()
    print(f"\nTuning {len(models)} models with RandomizedSearchCV + StratifiedKFold (5 folds)...")
    tuned = tune_all_models(models, X_train, y_train)

    all_results = {}
    roc_input = {}
    best_params_log = {}
    fitted_models = {}

    for name, info in tuned.items():
        model = info["model"]
        t0 = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        elapsed = time.time() - t0

        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["CV F1 (5-fold)"] = round(info["cv_f1"], 4)
        metrics["Train Time (s)"] = round(elapsed, 3)
        all_results[name] = metrics
        roc_input[name] = (y_test, y_proba)
        best_params_log[name] = info["params"]

        print(f"\n-- {name} --")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

        safe_name = safe_model_filename(name)
        plot_confusion_matrix(
            y_test, y_pred, f"{name} — {dataset_key}",
            os.path.join(PLOTS_DIR, f"{dataset_key}_{safe_name}_confusion.png")
        )
        save_classification_report(
            y_test, y_pred,
            os.path.join(REPORTS_DIR, f"{dataset_key}_{safe_name}_report.txt")
        )
        save_confusion_matrix_data(
            y_test, y_pred,
            os.path.join(REPORTS_DIR, f"{dataset_key}_{safe_name}_confusion.json")
        )
        save_roc_data(
            y_test, y_proba,
            os.path.join(REPORTS_DIR, f"{dataset_key}_{safe_name}_roc.json")
        )
        plot_feature_importance(
            model, feature_names, f"Feature Importance — {name} ({dataset_key})",
            os.path.join(PLOTS_DIR, f"{dataset_key}_{safe_name}_importance.png")
        )

        with open(os.path.join(MODELS_DIR, f"{dataset_key}_{safe_name}.pkl"), "wb") as f:
            pickle.dump(model, f)

        fitted_models[name] = model

    plot_roc_curves(roc_input, dataset_key, os.path.join(PLOTS_DIR, f"{dataset_key}_roc_curves.png"))

    with open(os.path.join(MODELS_DIR, f"{dataset_key}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    results_df = pd.DataFrame(all_results).T
    results_df.index.name = "Model"
    results_df.to_csv(os.path.join(REPORTS_DIR, f"{dataset_key}_summary.csv"))

    with open(os.path.join(REPORTS_DIR, f"{dataset_key}_best_params.json"), "w") as f:
        json.dump(best_params_log, f, indent=2, default=str)

    best_model = results_df["F1-Score"].astype(float).idxmax()
    print(f"\nBest model for {dataset_key} (by F1-Score): {best_model}")
    print(f"\nComparison table — {dataset_key}:")
    print(results_df[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "CV F1 (5-fold)"]]
          .astype(float).round(4).to_markdown())

    # --- Explainable AI: SHAP on the best model ---
    print(f"\nComputing SHAP explanations for the best model ({best_model})...")
    best_fitted = fitted_models[best_model]
    shap_values = compute_shap_values(best_fitted, X_train, X_test, feature_names)

    plot_shap_summary(
        shap_values, f"SHAP Feature Importance — {best_model} ({dataset_key})",
        os.path.join(PLOTS_DIR, f"{dataset_key}_shap_summary.png")
    )
    plot_shap_waterfall(
        shap_values, 0, f"SHAP Explanation — Example Prediction ({dataset_key})",
        os.path.join(PLOTS_DIR, f"{dataset_key}_shap_waterfall_example.png")
    )

    example_proba = best_fitted.predict_proba(X_test[:1])[0, 1]
    example_pred_label = DISEASE_LABELS.get(dataset_key, "Disease") if example_proba >= 0.5 else "No Disease"
    reasons = top_reasons(shap_values, 0)
    reasons_text = format_reasons_text(example_pred_label, example_proba, reasons)
    print(f"\nExample SHAP explanation ({dataset_key}, test row 0):\n{reasons_text}")

    with open(os.path.join(REPORTS_DIR, f"{dataset_key}_shap_example_explanation.txt"), "w") as f:
        f.write(reasons_text)

    return results_df, source_label


def main():
    parser = argparse.ArgumentParser(description="Disease Prediction from Medical Data")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    overall = {}
    sources = {}
    for key in targets:
        results_df, source_label = run_dataset(key)
        overall[key] = results_df
        sources[key] = source_label

    # Combined summary across all datasets
    combined_rows = []
    for key, df in overall.items():
        for model_name, row in df.iterrows():
            combined_rows.append({"Dataset": key, "Model": model_name, **row.to_dict()})
    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(os.path.join(REPORTS_DIR, "all_datasets_summary.csv"), index=False)

    with open(os.path.join(REPORTS_DIR, "dataset_sources.json"), "w") as f:
        json.dump(sources, f, indent=2)

    print(f"\n{'='*70}\nALL DONE. See outputs/reports and outputs/plots.\n{'='*70}")


if __name__ == "__main__":
    main()
