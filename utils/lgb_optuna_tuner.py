from __future__ import annotations

"""LightGBM + Optuna tuning utility for highly imbalanced binary‑classification tasks.

Usage (inside a notebook or another script)
------------------------------------------
>>> from lgb_optuna_tuner import run_lgb_optuna
>>> study, results = run_lgb_optuna(X_train_tfidf, y_train_tfidf, dataset_name="tfidf")

All artefacts – study pickle, best‑param JSON, metrics CSV and PNG plots – are
saved under ``<project‑root>/saved/<dataset_name>/``.
"""

from pathlib import Path
import json
import multiprocessing as mp
from datetime import datetime
from typing import Tuple, Optional
import numpy as np
import joblib
import optuna
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
from optuna.trial import TrialState
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline

# ────────────────────────────────────────────────────────────────────────────────
# GLOBALS (tweak if you want different defaults)
# ------------------------------------------------------------------------------
CV_SPLITS = 3
RANDOM_STATE = 42
MAX_PARALLEL_TRIALS = 4                      # cap Optuna jobs (sane on laptops)
# METRICS = {
#     "ap": make_scorer(average_precision_score, needs_threshold=True, pos_label=1),
#     "precision": make_scorer(precision_score, needs_threshold=False, zero_division=0, pos_label=1),
#     "recall": make_scorer(recall_score, needs_threshold=False, zero_division=0, pos_label=1),
#     "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
# }

METRICS = {
    "ap": make_scorer(average_precision_score, response_method="predict_proba", pos_label=1),
    "precision": make_scorer(precision_score, zero_division=0, pos_label=1),  # Defaults to response_method="predict"
    "recall": make_scorer(recall_score, zero_division=0, pos_label=1),        # Defaults to response_method="predict"
    "roc_auc": "roc_auc",
}

# ────────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ------------------------------------------------------------------------------

def run_lgb_optuna(
    X_train: pd.DataFrame | "np.ndarray",
    y_train: pd.Series | "np.ndarray",
    *,
    dataset_name: str,
    n_trials: int = 100,
    saved_root: str | Path | None = None,
    n_optuna_jobs: int | None = None,
    cv: StratifiedKFold | None = None,
    sampler: str | None = None,                 #  NEW  ("ros", "smote", or None)
) -> Tuple[optuna.Study, pd.DataFrame]:
    """Tune a LightGBM model with Optuna and save all artefacts.

    Parameters
    ----------
    X_train, y_train
        Training data and labels.
    dataset_name
        Used to create a sub‑folder under ``saved_root``.
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
            "objective": "binary",
            "metric": "aucpr", # Changed to prioritize AP
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 30),
            "min_gain_to_split":trial.suggest_float("min_gain_to_split", 0.0, 0.1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.01, 1.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 1.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,  # let LightGBM use all cores inside a fold
            # 'class_weight': 'balanced',  # Handle imbalance, # redundant or Conflicting weighting with "is_unbalance":
        }

        pos_weight = (y_train.shape[0] - y_train.sum()) / y_train.sum()   # neg / pos
        params.update({
            # "is_unbalance": False,            # use one or the other
            "scale_pos_weight": pos_weight,   # comment this line if using is_unbalance=True
        })

        # choose sampler
        if sampler == "ros":
            sampler_obj = RandomOverSampler(
                sampling_strategy=0.3, random_state=RANDOM_STATE
            )
        elif sampler == "smote":
            sampler_obj = SMOTE(
                sampling_strategy=0.1, k_neighbors=5, random_state=RANDOM_STATE
            )
        else:
            sampler_obj = None

        model = lgb.LGBMClassifier(**params)

        if sampler_obj is None:
            estimator = model
        else:
            estimator = Pipeline(steps=[("sampler", sampler_obj), ("lgb", model)])

        cv_results = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=METRICS,
            n_jobs=1,  # avoid nested parallelism with LightGBM
            error_score='raise', # or np.nan,   # ← let folds fail softly
            return_train_score=False,
        )
        for fold in range(CV_SPLITS):                                                                # temporary
            if np.isnan(cv_results['test_ap'][fold]):
                # Get split
                train_idx, test_idx = list(cv.split(X_train, y_train))[fold]
                estimator.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                y_prob = estimator.predict_proba(X_train.iloc[test_idx])[:, 1]
                print(f"Fold {fold}: unique probs = {len(np.unique(y_prob))}, min/max = {y_prob.min()}/{y_prob.max()}")

        print(cv_results)  # Inspect per-fold scores
        # print(cross_validate.__code__.co_varnames)  # Inspect the parameters, for debugging
        # print(scoring)  # If scoring is a variable, print its value, for debugging

        # --- skip trials where any fold produced nan (often happens with ~0 pos in that fold)
        # if np.isnan(cv_results['test_ap']).any():
        #     raise optuna.exceptions.TrialPruned("nan score - likely no positives in a fold")

        mean_scores = {k: v.mean() for k, v in cv_results.items() if k.startswith("test_")}
        ap = mean_scores["test_ap"]

        # store extra info
        trial.set_user_attr("ap", ap)
        trial.set_user_attr("precision", mean_scores["test_precision"])
        trial.set_user_attr("recall", mean_scores["test_recall"])
        trial.set_user_attr("roc_auc", mean_scores["test_roc_auc"])
        trial.set_user_attr("lambda_l1", params["lambda_l1"])
        trial.set_user_attr("lambda_l2", params["lambda_l2"])

        return ap  # we optimise *average‑precision*

    # ------------------------------------------------------------------
    # Run study
    # ------------------------------------------------------------------
    if n_optuna_jobs is None:
        n_optuna_jobs = min(MAX_PARALLEL_TRIALS, mp.cpu_count())

    study = optuna.create_study(direction="maximize", study_name=f"LGB_AP_{dataset_name}")
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
    
    # ------------ Aggregate only * Succesfully completed* trials ----------
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if not completed:
        raise ValueError(
            "No completed trials with valid scores - "
            "reduce CV splits or up-sample so each fold has positives."
        )


    results = pd.DataFrame(
        {
            "trial": [t.number for t in study.trials],
            "ap": [t.user_attrs["ap"] for t in study.trials],
            "precision": [t.user_attrs["precision"] for t in study.trials],
            "recall": [t.user_attrs["recall"] for t in study.trials],
            "roc_auc": [t.user_attrs["roc_auc"] for t in study.trials],
            "lambda_l1": [t.user_attrs["lambda_l1"] for t in study.trials],
            "lambda_l2": [t.user_attrs["lambda_l2"] for t in study.trials],
        }
    )
    results["penalty"] = results.apply(lambda r: _penalty(r.lambda_l1, r.lambda_l2), axis=1)

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    ts = pd.Timestamp.now().strftime("%Y%m%d")
    joblib.dump(study, saved_dir / f"lgb_optuna_study_{ts}.pkl")
    with (saved_dir / f"lgb_optuna_best_params_{ts}.json").open("w") as fp:
        json.dump(study.best_params, fp, indent=2)
    results.to_csv(saved_dir / f"lgb_optuna_metrics_{ts}.csv", index=False)

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

    print("⚠️  Demo mode - synthetic data (not for production)")

    X_demo, y_demo = make_classification(
        n_samples=5000,
        n_features=40,
        n_informative=10,
        weights=[0.99, 0.01],
        random_state=RANDOM_STATE,
    )
    X_demo = pd.DataFrame(X_demo)
    y_demo = pd.Series(y_demo)

    run_lgb_optuna(X_demo, y_demo, dataset_name="demo", n_trials=10)