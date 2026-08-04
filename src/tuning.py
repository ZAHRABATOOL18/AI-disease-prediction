"""
tuning.py
---------
Hyperparameter tuning with RandomizedSearchCV (fast, good enough for these
grid sizes) on top of Stratified K-Fold cross-validation, so every model is
tuned and scored on the same, class-balanced folds.
"""

import warnings

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score

from src.models import get_param_grids

RANDOM_STATE = 42
N_SPLITS = 5
N_ITER = 15  # how many random param combos to try per model


def tune_model(name, model, X_train, y_train, n_iter=N_ITER, n_splits=N_SPLITS, verbose=False):
    """
    Runs RandomizedSearchCV with Stratified K-Fold CV for one model.
    Returns (best_estimator, best_cv_f1_score, best_params).
    If the model has no param grid registered, it's just cross-validated as-is.
    """
    grids = get_param_grids()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    if name not in grids:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        model.fit(X_train, y_train)
        return model, scores.mean(), {}

    param_grid = grids[name]
    # Don't ask for more combos than actually exist in a tiny grid
    n_candidates = 1
    for v in param_grid.values():
        n_candidates *= len(v)
    n_iter_eff = min(n_iter, n_candidates)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter_eff,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=1 if verbose else 0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_train, y_train)

    return search.best_estimator_, search.best_score_, search.best_params_


def tune_all_models(models_dict, X_train, y_train, verbose=False):
    """
    Tunes every model in models_dict (name -> estimator).
    Returns dict: name -> {"model": best_estimator, "cv_f1": float, "params": dict}
    """
    results = {}
    for name, model in models_dict.items():
        print(f"  Tuning {name} (StratifiedKFold={N_SPLITS}, RandomizedSearchCV)...")
        best_model, cv_f1, best_params = tune_model(
            name, model, X_train, y_train, verbose=verbose
        )
        results[name] = {"model": best_model, "cv_f1": cv_f1, "params": best_params}
        print(f"    -> best CV F1: {cv_f1:.4f}" + (f" | params: {best_params}" if best_params else ""))
    return results
