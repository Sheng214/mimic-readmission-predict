from __future__ import annotations

from pathlib import Path
import json, multiprocessing as mp
from typing import Tuple, Optional
import numpy as np
import joblib, optuna, xgboost as xgb, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score, roc_auc_score
)
import matplotlib.pyplot as plt
from optuna.trial import TrialState
from imblearn.over_sampling import RandomOverSampler, SMOTE
from scipy.sparse import issparse

# ────────────────────────────────────────────────────────────────────────────────
CV_SPLITS = 3
RANDOM_STATE = 42
MAX_PARALLEL_TRIALS = 4

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def _safe_resample(
    X, y, method: str = "smote", target_ratio: float = 0.5, random_state: int = 42
):
    """
    Resample (X, y) to the desired positive ratio.
    - If X is sparse, uses RandomOverSampler (SMOTE doesn't support sparse).
    - If very few minority samples, reduces k_neighbors or falls back to ROS.
    """
    if method not in {"smote", "ros"}:
        return X, y

    # Sparse → use ROS
    if issparse(X):
        ros = RandomOverSampler(sampling_strategy=target_ratio, random_state=random_state)
        return ros.fit_resample(X, y)

    if method == "ros":
        ros = RandomOverSampler(sampling_strategy=target_ratio, random_state=random_state)
        return ros.fit_resample(X, y)

    # SMOTE on dense
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    n_minority = int(y_arr.sum())
    if n_minority < 2:
        ros = RandomOverSampler(sampling_strategy=target_ratio, random_state=random_state)
        return ros.fit_resample(X, y)
    k = max(1, min(5, n_minority - 1))  # SMOTE requires k < n_minority
    sm = SMOTE(sampling_strategy=target_ratio, k_neighbors=k, random_state=random_state)
    return sm.fit_resample(X, y)


def _plot_metric_bars(results: pd.DataFrame, out_dir: Path, ts: str) -> None:
    """Two bar charts: (1) Avg Precision/Precision/Recall, (2) ROC AUC."""
    # If you later add a "penalty" column, this will still work.
    label = "Avg Precision" if "ap" in results.columns else "AP"

    pr_df = results[["ap", "precision", "recall"]].copy()
    pr_df = pr_df.rename(columns={"ap": label}).mean(numeric_only=True).to_frame("Score")
    ax1 = pr_df.plot(kind="bar", figsize=(8, 6), legend=False)
    ax1.set_ylim(0, 1)
    ax1.set_title("XGBoost – Average Precision, Precision, Recall")
    ax1.set_ylabel("Score")
    ax1.figure.tight_layout()
    ax1.figure.savefig(out_dir / f"xgb_pr_metrics_{ts}.png", dpi=300)
    plt.show()

    roc_df = results[["roc_auc"]].mean(numeric_only=True).to_frame("Score")
    ax2 = roc_df.plot(kind="bar", figsize=(6, 6), legend=False)
    ax2.set_ylim(0, 1)
    ax2.set_title("XGBoost – ROC AUC")
    ax2.set_ylabel("Score")
    ax2.figure.tight_layout()
    ax2.figure.savefig(out_dir / f"xgb_roc_auc_{ts}.png", dpi=300)
    plt.show()


# ────────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────────
def run_xgb_optuna(
    X_train: pd.DataFrame | "np.ndarray",
    y_train: pd.Series | "np.ndarray",
    *,
    dataset_name: str,
    n_trials: int = 100,
    saved_root: str | Path | None = None,
    n_optuna_jobs: int | None = None,
    cv: StratifiedKFold | None = None,
    sampler: str | None = "smote",   # "smote", "ros", or None  (applied to BOTH train & valid)
    make_plots: bool = True,
) -> Tuple[optuna.Study, pd.DataFrame]:
    """Tune an XGBoost model with Optuna (CV with SMOTE/ROS on both train & valid)."""

    # Paths
    root_dir = Path(saved_root) if saved_root else Path.cwd().parent
    saved_dir = root_dir / "saved" / dataset_name
    saved_dir.mkdir(parents=True, exist_ok=True)

    # CV splitter
    if cv is None:
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Objective
    def _objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",  # XGB optimizes AP directly
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
            "n_jobs": -1,
        }

        # If oversampling BOTH train & valid, avoid extra weighting
        if sampler in {"smote", "ros"}:
            params["scale_pos_weight"] = 1.0
        else:
            pos_weight = (y_train.shape[0] - y_train.sum()) / y_train.sum()
            params["scale_pos_weight"] = float(pos_weight)

        ap_scores, precision_scores, recall_scores, roc_auc_scores = [], [], [], []

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
            # Indexing
            X_tr = X_train.iloc[train_idx] if hasattr(X_train, "iloc") else X_train[train_idx]
            y_tr = y_train.iloc[train_idx] if hasattr(y_train, "iloc") else y_train[train_idx]
            X_va = X_train.iloc[valid_idx] if hasattr(X_train, "iloc") else X_train[valid_idx]
            y_va = y_train.iloc[valid_idx] if hasattr(y_train, "iloc") else y_train[valid_idx]

            # ── OVERSAMPLE BOTH TRAIN AND VALIDATION (nonstandard) ─────
            if sampler in {"smote", "ros"}:
                X_tr, y_tr = _safe_resample(X_tr, y_tr, method=sampler, target_ratio=0.5, random_state=RANDOM_STATE)
                X_va, y_va = _safe_resample(X_va, y_va, method=sampler, target_ratio=0.5, random_state=RANDOM_STATE)

            # DMatrix
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dvalid = xgb.DMatrix(X_va, label=y_va)

            # Train with early stopping on the oversampled valid set
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            # Predict on the oversampled valid set
            # Use best_iteration for a stable prediction length
            best_iter = getattr(bst, "best_iteration", None)
            if best_iter is not None:
                y_prob = bst.predict(dvalid, iteration_range=(0, best_iter + 1))
            else:
                y_prob = bst.predict(dvalid)

            y_pred = (y_prob >= 0.5).astype(int)

            # Safety (should not occur after oversampling)
            if (np.array(y_va) == 1).sum() == 0:
                raise optuna.exceptions.TrialPruned("No positives in (oversampled) valid fold")

            ap   = average_precision_score(y_va, y_prob)
            prec = precision_score(y_va, y_pred, zero_division=0)
            rec  = recall_score(y_va, y_pred, zero_division=0)
            roc  = roc_auc_score(y_va, y_prob)

            ap_scores.append(ap)
            precision_scores.append(prec)
            recall_scores.append(rec)
            roc_auc_scores.append(roc)

        mean_ap = float(np.mean(ap_scores))
        trial.set_user_attr("ap", mean_ap)
        trial.set_user_attr("precision", float(np.mean(precision_scores)))
        trial.set_user_attr("recall", float(np.mean(recall_scores)))
        trial.set_user_attr("roc_auc", float(np.mean(roc_auc_scores)))
        trial.set_user_attr("reg_alpha", params["reg_alpha"])
        trial.set_user_attr("reg_lambda", params["reg_lambda"])
        return mean_ap

    # Run study
    if n_optuna_jobs is None:
        n_optuna_jobs = min(MAX_PARALLEL_TRIALS, mp.cpu_count())
    study = optuna.create_study(direction="maximize", study_name=f"XGB_AP_%s" % dataset_name)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True, n_jobs=n_optuna_jobs)

    # Collect results (completed only)
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        raise ValueError(
            "No completed trials with valid scores - reduce CV splits or up-sample so each fold has positives."
        )

    results = pd.DataFrame({
        "trial":     [t.number for t in completed],
        "ap":        [t.user_attrs["ap"] for t in completed],
        "precision": [t.user_attrs["precision"] for t in completed],
        "recall":    [t.user_attrs["recall"] for t in completed],
        "roc_auc":   [t.user_attrs["roc_auc"] for t in completed],
        "reg_alpha": [t.user_attrs["reg_alpha"] for t in completed],
        "reg_lambda":[t.user_attrs["reg_lambda"] for t in completed],
    })

    # Save artefacts
    ts = pd.Timestamp.now().strftime("%Y%m%d")
    joblib.dump(study, saved_dir / f"xgb_optuna_study_{ts}.pkl")
    with (saved_dir / f"xgb_optuna_best_params_{ts}.json").open("w") as fp:
        json.dump(study.best_params, fp, indent=2)
    results.to_csv(saved_dir / f"xgb_optuna_metrics_{ts}.csv", index=False)

    if make_plots:
        _plot_metric_bars(results, saved_dir, ts)

    return study, results
