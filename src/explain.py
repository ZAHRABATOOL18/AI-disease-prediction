"""
explain.py
----------
Explainable AI (Level 3): SHAP-based explanations for the best model on
each dataset — global feature importance (summary/bar plot), a waterfall
plot for one example prediction, and a plain-text "reasons" breakdown like:

    Prediction: Disease
    Confidence: 87%
    Reasons:
      + Glucose: +32%
      + BMI: +18%
      + Age: +12%

Works for any model with predict_proba (tree, linear, or kernel-based)
via shap.Explainer's model-agnostic Permutation backend, so it doesn't
matter which model won the comparison.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap


def _binary_class_explanation(shap_values, class_idx=1):
    """
    shap.Explainer(model.predict_proba, ...) on a binary classifier returns
    values shaped (n_samples, n_features, 2) — one SHAP value set per class.
    We only care about the "disease present" class (index 1).
    """
    if shap_values.values.ndim == 3:
        return shap.Explanation(
            values=shap_values.values[..., class_idx],
            base_values=shap_values.base_values[..., class_idx],
            data=shap_values.data,
            feature_names=shap_values.feature_names,
        )
    return shap_values


def compute_shap_values(model, X_train, X_eval, feature_names,
                         max_background=50, max_eval=25):
    """Builds a model-agnostic SHAP explainer and computes values for a
    (capped, for speed) slice of the evaluation set."""
    background = X_train[:max_background]
    eval_data = X_eval[:max_eval]

    explainer = shap.Explainer(model.predict_proba, background, feature_names=feature_names)
    raw_values = explainer(eval_data)
    return _binary_class_explanation(raw_values, class_idx=1)


def plot_shap_summary(shap_values, title, out_path):
    """Global feature importance: mean |SHAP value| per feature."""
    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=12)
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_waterfall(shap_values, index, title, out_path):
    """Per-prediction breakdown: how each feature pushed the prediction
    away from the average, for one specific example."""
    plt.figure()
    shap.plots.waterfall(shap_values[index], show=False, max_display=10)
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def top_reasons(shap_values, index, top_n=5):
    """
    Returns a plain-text-friendly breakdown of the top contributing
    features for one prediction, as a list of dicts:
    [{"feature": "Glucose", "direction": "+", "pct": 32.1}, ...]
    Percentages are each feature's share of total |SHAP value| for
    that prediction (so they roughly sum to 100%).
    """
    row = shap_values[index]
    values = row.values
    names = row.feature_names

    total_abs = np.sum(np.abs(values))
    if total_abs == 0:
        return []

    order = np.argsort(-np.abs(values))[:top_n]
    reasons = []
    for i in order:
        pct = 100 * abs(values[i]) / total_abs
        reasons.append({
            "feature": names[i],
            "direction": "+" if values[i] > 0 else "-",
            "pct": round(float(pct), 1),
        })
    return reasons


def format_reasons_text(disease_label, confidence, reasons):
    """Formats top_reasons() output into the human-readable block used
    in reports/README examples."""
    lines = [f"Prediction: {disease_label}", f"Confidence: {confidence:.0%}", "Reasons:"]
    for r in reasons:
        lines.append(f"  {r['direction']} {r['feature']}: {r['direction']}{r['pct']}%")
    return "\n".join(lines)
