from __future__ import annotations

"""Stacking Ensemble utility for highly imbalanced binary-classification tasks.
The stacking ensemble uses base models consisting of Logistic Regression (LR) + XGBoost (XGB) + LightGBM (LGB),
with another Logistic Regression as the meta-model (to blend the base predictions).

Usage (inside a notebook or another script)
------------------------------------------
>>> from stacking_ensemble import run_stacking_ensemble
>>> stack_model, results = run_stacking_ensemble(X_train_tfbert, y_train_tfbert, dataset_name="tfbert", best_lr_params=best_lr_params, best_xgb_params=best_xgb_params, best_lgb_params=best_lgb_params)

All artefacts – model pickle, metrics CSV and PNG plots – are
saved under ``<project-root>/saved/<dataset_name>/``.
"""

from pathlib import Path
import json
from typing import Tuple, Optional, Dict
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# ────────────────────────────────────────────────────────────────────────────────
# GLOBALS (tweak if you want different defaults)
# ------------------------------------------------------------------------------
CV_SPLITS = 3
RANDOM_STATE = 42
INTERNAL_CV_SPLITS = 3  # For stacking internal CV

# ────────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ------------------------------------------------------------------------------

def run_stacking_ensemble(
    X_train: pd.DataFrame | "np.ndarray",
    y_train: pd.Series | "np.ndarray",
    *,
    dataset_name: str,
    best_lr_params: Dict,
    best_xgb_params: Dict,
    best_lgb_params: Dict,
    saved_root: str | Path | None = None,
    cv: StratifiedKFold | None = None,
    sampler: str | None = None,  # "ros", "smote", or None
) -> Tuple[StackingClassifier, pd.DataFrame]:
    """Train a stacking ensemble and save all artefacts.

    Parameters
    ----------
    X_train, y_train
        Training data and labels.
    dataset_name
        Used to create a sub-folder under ``saved_root``.
    best_lr_params
        Best parameters for Logistic Regression from screening (e.g., with L1 penalty).
    best_xgb_params
        Best parameters for XGBoost from Optuna.
    best_lgb_params
        Best parameters for LightGBM from Optuna.
    saved_root
        Where to create the ``saved/`` folder. Defaults to the *parent* of the
        current working directory.
    cv
        Optional CV splitter.

    Returns
    -------
    stack_model : StackingClassifier
        The fitted stacking model.
    results : pd.DataFrame
        Metrics from CV.
    """

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    root_dir = Path(saved_root) if saved_root else Path.cwd().parent
    saved_dir = root_dir / "saved" / dataset_name
    saved_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CV splitter
    # ------------------------------------------------------------------
    if cv is None:
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # ------------------------------------------------------------------
    # Define base models
    # ------------------------------------------------------------------
    lr = LogisticRegression(**best_lr_params)
    xgb_model = xgb.XGBClassifier(**best_xgb_params)
    lgb_model = lgb.LGBMClassifier(**best_lgb_params)
    estimators = [('lr', lr), ('xgb', xgb_model), ('lgb', lgb_model)]

    # Stacking classifier (meta is LR, using similar params as base LR but fixed solver if needed)
    meta_params = best_lr_params.copy()
    if 'solver' not in meta_params:
        meta_params['solver'] = 'liblinear'  # Suitable for L1
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(**meta_params),
        cv=INTERNAL_CV_SPLITS,
        n_jobs=-1,
        verbose=0
    )

    # ------------------------------------------------------------------
    # Choose sampler
    # ------------------------------------------------------------------
    if sampler == "ros":
        sampler_obj = RandomOverSampler(sampling_strategy=0.5, random_state=RANDOM_STATE)
    elif sampler == "smote":
        sampler_obj = SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=RANDOM_STATE)
    else:
        sampler_obj = None

    # ------------------------------------------------------------------
    # Evaluate with CV loop
    # ------------------------------------------------------------------
    ap_scores = []
    precision_scores = []
    recall_scores = []
    roc_auc_scores = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
        # Handle indexing for both pandas and numpy
        X_tr = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
        y_tr = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
        X_val = X_train.iloc[valid_idx] if hasattr(X_train, 'iloc') else X_train[valid_idx]
        y_val = y_train.iloc[valid_idx] if hasattr(y_train, 'iloc') else y_train[valid_idx]

        # Apply sampler to train only
        if sampler_obj:
            X_tr, y_tr = sampler_obj.fit_resample(X_tr, y_tr)

        # Fit stacking on train
        stack.fit(X_tr, y_tr)

        # Predict on valid
        y_prob = stack.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Skip if no positives in valid
        if y_val.sum() == 0:
            continue  # Or handle appropriately

        ap = average_precision_score(y_val, y_prob, pos_label=1)
        prec = precision_score(y_val, y_pred, zero_division=0, pos_label=1)
        rec = recall_score(y_val, y_pred, zero_division=0, pos_label=1)
        roc = roc_auc_score(y_val, y_prob)

        ap_scores.append(ap)
        precision_scores.append(prec)
        recall_scores.append(rec)
        roc_auc_scores.append(roc)

    # Compute means (ignore NaNs if any)
    mean_ap = np.nanmean(ap_scores)
    mean_precision = np.nanmean(precision_scores)
    mean_recall = np.nanmean(recall_scores)
    mean_roc_auc = np.nanmean(roc_auc_scores)

    results = pd.DataFrame({
        "ap": [mean_ap],
        "precision": [mean_precision],
        "recall": [mean_recall],
        "roc_auc": [mean_roc_auc],
    })

    # ------------------------------------------------------------------
    # Fit on full data
    # ------------------------------------------------------------------
    X_full = X_train
    y_full = y_train
    if sampler_obj:
        X_full, y_full = sampler_obj.fit_resample(X_full, y_full)
    stack.fit(X_full, y_full)

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    ts = pd.Timestamp.now().strftime("%Y%m%d")
    joblib.dump(stack, saved_dir / f"stacking_ensemble_model_{ts}.pkl")
    results.to_csv(saved_dir / f"stacking_ensemble_metrics_{ts}.csv", index=False)

    # plots
    _plot_metric_bars(results, saved_dir, ts)

    return stack, results


# ────────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ------------------------------------------------------------------------------

def _plot_metric_bars(results: pd.DataFrame, out_dir: Path, ts: str) -> None:
    """Create two bar charts for the stacking ensemble metrics."""

    # Compute means (single values)
    means = results.mean()

    # First plot: Avg Precision, Precision, Recall
    pr_data = pd.Series({
        "Avg Precision": means["ap"],
        "Precision": means["precision"],
        "Recall": means["recall"]
    })
    ax1 = pr_data.plot(kind="bar", figsize=(8, 6))
    ax1.set_title("Stacking Ensemble - Average Precision, Precision, Recall")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1)
    ax1.figure.tight_layout()
    fig_path1 = out_dir / f"stacking_pr_metrics_{ts}.png"
    ax1.figure.savefig(fig_path1, dpi=300)
    plt.show()
    # plt.close(ax1.figure)

    # Second plot: ROC AUC
    roc_data = pd.Series({"ROC AUC": means["roc_auc"]})
    ax2 = roc_data.plot(kind="bar", figsize=(6, 6))
    ax2.set_title("Stacking Ensemble - ROC AUC")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1)
    ax2.figure.tight_layout()
    fig_path2 = out_dir / f"stacking_roc_auc_{ts}.png"
    ax2.figure.savefig(fig_path2, dpi=300)
    plt.show()
    # plt.close(ax2.figure)


# ────────────────────────────────────────────────────────────────────────────────
# CLI example (optional)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification

    print("⚠️  Demo mode – synthetic data (not for production)")

    X_demo, y_demo = make_classification(
        n_samples=5000,
        n_features=40,
        n_informative=10,
        weights=[0.99, 0.01],
        random_state=RANDOM_STATE,
    )
    X_demo = pd.DataFrame(X_demo)
    y_demo = pd.Series(y_demo)

    # Dummy best params for demo
    best_lr_params = {"penalty": "l1", "class_weight": "balanced", "max_iter": 1000, "random_state": RANDOM_STATE, "solver": "liblinear"}
    best_xgb_params = {"eta": 0.1, "max_depth": 6, "scale_pos_weight": 99}
    best_lgb_params = {"learning_rate": 0.1, "max_depth": 6, "scale_pos_weight": 99}

    run_stacking_ensemble(X_demo, y_demo, dataset_name="demo", best_lr_params=best_lr_params, best_xgb_params=best_xgb_params, best_lgb_params=best_lgb_params)