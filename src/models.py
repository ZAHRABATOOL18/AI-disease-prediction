"""
models.py
---------
Defines the classification models used across all datasets:
Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, and CatBoost.

Each gradient-boosting library falls back to scikit-learn's
GradientBoostingClassifier if it isn't installed, so the project still runs
end-to-end. Install the real libraries for the real thing:
    pip install xgboost lightgbm catboost
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

RANDOM_STATE = 42

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def get_models():
    """Returns a dict of {model_name: sklearn-compatible estimator}."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE,
        )
    else:
        models["XGBoost (fallback: GradientBoosting)"] = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE
        )

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=-1,
        )
    else:
        models["LightGBM (fallback: GradientBoosting)"] = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE
        )

    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=False,
        )
    else:
        models["CatBoost (fallback: GradientBoosting)"] = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE
        )

    return models


# Hyperparameter search spaces for RandomizedSearchCV / GridSearchCV, keyed
# by the same names get_models() produces. Only real (non-fallback) models
# get tuned with their native params; fallback models share a GB grid.
def get_param_grids():
    grids = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        "SVM (RBF Kernel)": {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.1],
        },
        "Random Forest": {
            "n_estimators": [200, 300, 400, 500],
            "max_depth": [None, 6, 10, 16],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    }

    gb_grid = {
        "n_estimators": [150, 250, 350],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
    }

    if XGBOOST_AVAILABLE:
        grids["XGBoost"] = {
            "n_estimators": [150, 250, 350],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
        }
    else:
        grids["XGBoost (fallback: GradientBoosting)"] = gb_grid

    if LIGHTGBM_AVAILABLE:
        grids["LightGBM"] = {
            "n_estimators": [150, 250, 350],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.85, 1.0],
        }
    else:
        grids["LightGBM (fallback: GradientBoosting)"] = gb_grid

    if CATBOOST_AVAILABLE:
        grids["CatBoost"] = {
            "iterations": [150, 250, 350],
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
        }
    else:
        grids["CatBoost (fallback: GradientBoosting)"] = gb_grid

    return grids
