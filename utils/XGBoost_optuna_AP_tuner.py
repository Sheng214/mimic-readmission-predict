from __future__ import annotations

"""XGBoost + Optuna tuning utility for highly imbalanced binary-classification tasks.

Usage (inside a notebook or another script)
------------------------------------------
>>> from xgb_optuna_tuner import run_xgb_optuna
>>> study, results = run_xgb_optuna(X_train_tfidf, y_train_tfidf, dataset_name="tfidf")

All artefacts – study pickle, best-param JSON, metrics CSV and PNG plots – are
saved under ``<project-root>/saved/<dataset_name>/``.

Modified based on lgb_optuna_tuner.py, which has "Metric Mismatch: XGBoost's internal
metric='aucpr' optimizes average precision directly during training."
"""

from pathlib import Path
import json
import multiprocessing as mp
from datetime import datetime
from typing import Tuple, Optional
import numpy as np
import joblib
import optuna
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
from optuna.trial import TrialState
from imblearn.over_sampling import RandomOverSampler, SMOTE

# ────────────────────────────────────────────────────────────────────────────────
# GLOBALS (tweak if you want different defaults)
# ------------------------------------------------------------------------------
CV_SPLITS = 3
RANDOM_STATE = 42
MAX_PARALLEL_TRIALS = 4                      # cap Optuna jobs (sane on laptops)

# ────────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ------------------------------------------------------------------------------

def run_xgb_optuna(
    X_train: pd.DataFrame | "np.ndarray",
    y_train: pd.Series | "np.ndarray",
    *,
    dataset_name: str,
    n_trials: int = 100,
    saved_root: str | Path | None = None,
    n_optuna_jobs: int | None = None,
    cv: StratifiedKFold | None = None,
    sampler: str | None = None,                 # NEW ("ros", "smote", or None)
) -> Tuple[optuna.Study, pd.DataFrame]:
    """Tune an XGBoost model with Optuna and save all artefacts.

    Parameters
    ----------
    X_train, y_train
        Training data and labels.
    dataset_name
        Used to create a sub-folder under ``saved_root``.
    n_trials
        Number of Optuna trials.
    saved_root
        Where to create the ``saved/`` folder. Defaults to the *parent* of the
        current working directory (works well when you launch notebooks from a
        ``notebooks`` folder).
    n_optuna_jobs
        Parallel Optuna trials. Defaults to ``min(MAX_PARALLEL_TRIALS, cpu_count)``.

    Returns
    -------
    study : optuna.Study
        The completed Optuna study (already pickled to disk).
    results : pd.DataFrame
        Metrics per trial.
    """

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    root_dir = Path(saved_root) if saved_root else Path.cwd().parent
    saved_dir = root_dir / "saved" / dataset_name
    saved_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CV splitter (fixed so every trial sees same folds)
    # ------------------------------------------------------------------
    if cv is None:
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # ------------------------------------------------------------------
    # Objective function (closure captures data & cv)
    # ------------------------------------------------------------------
    def _objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",  # Prioritize AP
            "verbosity": 0,
            "booster": "gbtree",
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,  # let XGBoost use all cores inside a fold
        }

        pos_weight = (y_train.shape[0] - y_train.sum()) / y_train.sum()   # neg / pos
        params.update({
            "scale_pos_weight": pos_weight,   # Handle imbalance
        })

        # choose sampler
        if sampler == "ros":
            sampler_obj = RandomOverSampler(
                sampling_strategy=0.5, random_state=RANDOM_STATE
            )
        elif sampler == "smote":
            sampler_obj = SMOTE(
                sampling_strategy=0.5, k_neighbors=5, random_state=RANDOM_STATE
            )
        else:
            sampler_obj = None

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

            # Create datasets
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dvalid = xgb.DMatrix(X_val, label=y_val)

            # Train with early stopping
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dvalid, 'valid')],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            # Predict on valid
            y_prob = bst.predict(dvalid)
            y_pred = (y_prob >= 0.5).astype(int)

            # Skip if no positives in valid
            if y_val.sum() == 0:
                raise optuna.exceptions.TrialPruned("No positives in valid fold")

            ap = average_precision_score(y_val, y_prob, pos_label=1)
            prec = precision_score(y_val, y_pred, zero_division=0, pos_label=1)
            rec = recall_score(y_val, y_pred, zero_division=0, pos_label=1)
            roc = roc_auc_score(y_val, y_prob)

            ap_scores.append(ap)
            precision_scores.append(prec)
            recall_scores.append(rec)
            roc_auc_scores.append(roc)

        mean_ap = np.mean(ap_scores)

        # Store extra info
        trial.set_user_attr("ap", mean_ap)
        trial.set_user_attr("precision", np.mean(precision_scores))
        trial.set_user_attr("recall", np.mean(recall_scores))
        trial.set_user_attr("roc_auc", np.mean(roc_auc_scores))
        trial.set_user_attr("reg_alpha", params["reg_alpha"])
        trial.set_user_attr("reg_lambda", params["reg_lambda"])

        return mean_ap  # Maximize mean AP

    # ------------------------------------------------------------------
    # Run study
    # ------------------------------------------------------------------
    if n_optuna_jobs is None:
        n_optuna_jobs = min(MAX_PARALLEL_TRIALS, mp.cpu_count())

    study = optuna.create_study(direction="maximize", study_name=f"XGB_AP_{dataset_name}")
    study.optimize(
        _objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=n_optuna_jobs,
    )

    # ------------------------------------------------------------------
    # Collect results
    # ------------------------------------------------------------------
    def _penalty(l1: float, l2: float, tol: float = 1e-12) -> str:
        if l1 > tol and l2 < tol:
            return "L1"
        if l2 > tol and l1 < tol:
            return "L2"
        if l1 > tol and l2 > tol:
            return "ElasticNet"
        return "None"
    
    # ------------ Aggregate only * Successfully completed* trials ----------
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if not completed:
        raise ValueError(
            "No completed trials with valid scores - "
            "reduce CV splits or up-sample so each fold has positives."
        )

    results = pd.DataFrame(
        {
            "trial": [t.number for t in completed],
            "ap": [t.user_attrs["ap"] for t in completed],
            "precision": [t.user_attrs["precision"] for t in completed],
            "recall": [t.user_attrs["recall"] for t in completed],
            "roc_auc": [t.user_attrs["roc_auc"] for t in completed],
            "reg_alpha": [t.user_attrs["reg_alpha"] for t in completed],
            "reg_lambda": [t.user_attrs["reg_lambda"] for t in completed],
        }
    )
    results["penalty"] = results.apply(lambda r: _penalty(r.reg_alpha, r.reg_lambda), axis=1)

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    ts = pd.Timestamp.now().strftime("%Y%m%d")
    joblib.dump(study, saved_dir / f"xgb_optuna_study_{ts}.pkl")
    with (saved_dir / f"xgb_optuna_best_params_{ts}.json").open("w") as fp:
        json.dump(study.best_params, fp, indent=2)
    results.to_csv(saved_dir / f"xgb_optuna_metrics_{ts}.csv", index=False)

    # plots
    _plot_metric_bars(results, saved_dir, ts)

    return study, results


# ────────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ------------------------------------------------------------------------------

def _plot_metric_bars(results: pd.DataFrame, out_dir: Path, ts: str) -> None:
    """Create two bar charts: one with grouped PR metrics per penalty, and one for ROC AUC."""

    # First plot: Avg Precision, Precision, Recall
    metrics1 = ["ap", "precision", "recall"]
    grouped1 = results.groupby("penalty")[metrics1].mean()
    grouped1 = grouped1.rename(columns={
        "ap": "Avg Precision",
        "precision": "Precision",
        "recall": "Recall"
    })
    # Sort by Avg Precision descending
    grouped1 = grouped1.sort_values("Avg Precision", ascending=False)
    # Plot grouped bars
    ax1 = grouped1.plot(kind="bar", figsize=(10, 6))
    ax1.set_title("Penalties - Average Precision, Precision, Recall")
    ax1.set_ylabel("Score")
    ax1.figure.tight_layout()
    # Save to file
    fig_path1 = out_dir / f"pr_metrics_by_penalty_{ts}.png"
    ax1.figure.savefig(fig_path1, dpi=300)
    plt.show()
    # plt.close(ax1.figure)

    # Second plot: ROC AUC only
    metrics2 = ["roc_auc"]
    grouped2 = results.groupby("penalty")[metrics2].mean()
    grouped2 = grouped2.rename(columns={"roc_auc": "ROC AUC"})
    # Sort by ROC AUC descending
    grouped2 = grouped2.sort_values("ROC AUC", ascending=False)
    # Plot single bars
    ax2 = grouped2.plot(kind="bar", figsize=(10, 6), legend=True)
    ax2.set_title("Penalties - ROC AUC")
    ax2.set_ylabel("Score")
    ax2.figure.tight_layout()
    # Save to file
    fig_path2 = out_dir / f"roc_auc_by_penalty_{ts}.png"
    ax2.figure.savefig(fig_path2, dpi=300)
    plt.show()
    # plt.close(ax2.figure)


# ────────────────────────────────────────────────────────────────────────────────
# CLI example (optional)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
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

    run_xgb_optuna(X_demo, y_demo, dataset_name="demo", n_trials=10)