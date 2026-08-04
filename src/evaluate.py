"""
evaluate.py
-----------
Evaluation utilities: metric computation and plot generation
(confusion matrix heatmaps, ROC curves, feature importance).
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

sns.set_theme(style="whitegrid")


def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba)
    return metrics


def plot_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(title, fontsize=11)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curves(results_dict, dataset_name, out_path):
    """results_dict: {model_name: (y_true, y_proba)}"""
    plt.figure(figsize=(6, 5))
    for model_name, (y_true, y_proba) in results_dict.items():
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves — {dataset_name}")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, title, out_path, top_n=12):
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is None:
        return  # model type doesn't expose importances (e.g. SVM w/ RBF kernel)

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(6, 5))
    sns.barplot(data=imp_df, x="importance", y="feature", color="#4C72B0")
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_classification_report(y_true, y_pred, out_path):
    report = classification_report(y_true, y_pred, target_names=["No Disease", "Disease"])
    with open(out_path, "w") as f:
        f.write(report)


def save_confusion_matrix_data(y_true, y_pred, out_path):
    """Saves the raw confusion matrix as JSON so a dashboard can re-render
    it interactively (e.g. with Plotly) instead of loading a static PNG."""
    cm = confusion_matrix(y_true, y_pred)
    data = {
        "labels": ["No Disease", "Disease"],
        "matrix": cm.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def save_roc_data(y_true, y_proba, out_path):
    """Saves fpr/tpr/auc as JSON so a dashboard can re-render the ROC curve
    interactively instead of loading a static PNG."""
    if y_proba is None:
        data = {"fpr": [], "tpr": [], "auc": None}
    else:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(float(auc), 4)}
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return data
