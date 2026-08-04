"""
utils.py
--------
Small shared helpers used by both main.py (training) and app.py (dashboard)
so filenames stay consistent between the two.
"""


def safe_model_filename(name: str) -> str:
    """Converts a model display name (e.g. 'XGBoost (fallback: GradientBoosting)')
    into a filesystem-safe stub used for saved models/plots/reports."""
    return (name.replace(" ", "_").replace("(", "").replace(")", "")
            .replace(":", "").replace("__", "_"))
