from __future__ import annotations
from pathlib import Path
import json, multiprocessing as mp
from typing import Tuple, Optional
import numpy as np
import joblib, optuna, lightgbm as lgb, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from optuna.trial import TrialState
from imblearn.over_sampling import RandomOverSampler, SMOTE
from scipy.sparse import issparse  # safe handling of TF‑IDF matrices

# ────────────────────────────────────────────────────────────────────────────────
CV_SPLITS = 3
RANDOM_STATE = 42
MAX_PARALLEL_TRIALS = 4

# ────────────────────────────────────────────────────────────────────────────────
# Plot helper  ✅ (previously missing)
# ────────────────────────────────────────────────────────────────────────────────
def _plot_metric_bars(results: pd.DataFrame, out_dir: Path, ts: str) -> None:
    """
    Save two bar charts:
      (1) Avg Precision / Precision / Recall (grouped by penalty if present)
      (2) ROC AUC
    """
    if "penalty" in results.columns:
        grouped1 = (results.groupby("penalty")[["ap","precision","recall"]]
                    .mean()
                    .rename(columns={"ap":"Avg Precision"})
                    .sort_values("Avg Precision", ascending=False))
        ax1 = grouped1.plot(kind="bar", figsize=(10, 6))
        ax1.set_title("Penalties – Average Precision, Precision, Recall")
        ax1.set_ylabel("Score"); ax1.set_ylim(0, 1); ax1.figure.tight_layout()
        ax1.figure.savefig(out_dir / f"pr_metrics_by_penalty_{ts}.png", dpi=300)
        plt.show()

        grouped2 = (results.groupby("penalty")[["roc_auc"]]
                    .mean()
                    .rename(columns={"roc_auc":"ROC AUC"})
                    .sort_values("ROC AUC", ascending=False))
        ax2 = grouped2.plot(kind="bar", figsize=(10, 6), legend=True)
        ax2.set_title("Penalties – ROC AUC")
        ax2.set_ylabel("Score"); ax2.set_ylim(0, 1); ax2.figure.tight_layout()
        ax2.figure.savefig(out_dir / f"roc_auc_by_penalty_{ts}.png", dpi=300)
        plt.show()
    else:
        pr = results[["ap","precision","recall"]].rename(columns={"ap":"Avg Precision"}).iloc[0]
        ax1 = pr.plot(kind="bar", figsize=(8, 6))
        ax1.set_title("Average Precision, Precision, Recall")
        ax1.set_ylabel("Score"); ax1.set_ylim(0, 1); ax1.figure.tight_layout()
        ax1.figure.savefig(out_dir / f"pr_metrics_{ts}.png", dpi=300)
        plt.show()

        roc = results[["roc_auc"]].rename(columns={"roc_auc":"ROC AUC"}).iloc[0]
        ax2 = roc.plot(kind="bar", figsize=(6, 6))
        ax2.set_title("ROC AUC")
        ax2.set_ylabel("Score"); ax2.set_ylim(0, 1); ax2.figure.tight_layout()
        ax2.figure.savefig(out_dir / f"roc_auc_{ts}.png", dpi=300)
        plt.show()

# ────────────────────────────────────────────────────────────────────────────────
# Resampling helper
# ────────────────────────────────────────────────────────────────────────────────
def _safe_resample(
    X, y, method: str = "smote", target_ratio: float = 0.5, random_state: int = 42
):
    """
    Resample (X, y) using SMOTE if possible; otherwise fall back to ROS.
    - Adapts k_neighbors for tiny minority counts.
    - Falls back to RandomOverSampler for sparse matrices.
    """
    if method == "smote" and issparse(X):
        ros = RandomOverSampler(sampling_strategy=target_ratio, random_state=random_state)
        return ros.fit_resample(X, y)

    if method == "smote":
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        n_minority = int(y_arr.sum())
        if n_minority < 2:
            ros = RandomOverSampler(sampling_strategy=target_ratio, random_state=random_state)
            return ros.fit_resample(X, y)
        k = max(1, min(5, n_minority - 1))  # SMOTE requires k < n_minority
        sm = SMOTE(sampling_strategy=target_ratio, k_neighbors=k, random_state=random_state)
        return sm.fit_resample(X, y)

    elif method == "ros":
        ros = RandomOverSampler(sampling_strategy=target_ratio, random_state=random_state)
        return ros.fit_resample(X, y)

    return X, y  # no resampling

# ────────────────────────────────────────────────────────────────────────────────
def run_lgb_optuna(
    X_train: pd.DataFrame | "np.ndarray",
    y_train: pd.Series | "np.ndarray",
    *,
    dataset_name: str,
    n_trials: int = 100,
    saved_root: str | Path | None = None,
    n_optuna_jobs: int | None = None,
    cv: StratifiedKFold | None = None,
    sampler: str | None = "smote",  # "smote", "ros", or None
    make_plots: bool = True,        # ✅ new toggle
) -> Tuple[optuna.Study, pd.DataFrame]:

    root_dir = Path(saved_root) if saved_root else Path.cwd().parent
    saved_dir = root_dir / "saved" / dataset_name
    saved_dir.mkdir(parents=True, exist_ok=True)

    if cv is None:
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def _objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "average_precision",   # LGB reports AP on valid set
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 30),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }

        # If we oversample BOTH train & valid, avoid double compensation
        if sampler in {"smote", "ros"}:
            params["scale_pos_weight"] = 1.0
        else:
            pos_weight = (y_train.shape[0] - y_train.sum()) / y_train.sum()
            params["scale_pos_weight"] = float(pos_weight)

        ap_scores, precision_scores, recall_scores, roc_auc_scores = [], [], [], []

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr = X_train.iloc[train_idx] if hasattr(X_train, "iloc") else X_train[train_idx]
            y_tr = y_train.iloc[train_idx] if hasattr(y_train, "iloc") else y_train[train_idx]
            X_va = X_train.iloc[valid_idx] if hasattr(X_train, "iloc") else X_train[valid_idx]
            y_va = y_train.iloc[valid_idx] if hasattr(y_train, "iloc") else y_train[valid_idx]

            # ── OVERSAMPLE BOTH TRAIN AND VALIDATION (nonstandard) ─────
            if sampler in {"smote", "ros"}:
                X_tr, y_tr = _safe_resample(X_tr, y_tr, method=sampler, target_ratio=0.5, random_state=RANDOM_STATE)
                X_va, y_va = _safe_resample(X_va, y_va, method=sampler, target_ratio=0.5, random_state=RANDOM_STATE)

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)

            bst = lgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                valid_sets=[dvalid],
                valid_names=["valid"],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )

            y_prob = bst.predict(X_va, num_iteration=bst.best_iteration)
            y_hat  = (y_prob >= 0.5).astype(int)

            if (np.array(y_va) == 1).sum() == 0:
                raise optuna.exceptions.TrialPruned("No positives in (oversampled) valid fold")

            ap   = average_precision_score(y_va, y_prob)
            prec = precision_score(y_va, y_hat, zero_division=0)
            rec  = recall_score(y_va, y_hat, zero_division=0)
            roc  = roc_auc_score(y_va, y_prob)

            ap_scores.append(ap); precision_scores.append(prec); recall_scores.append(rec); roc_auc_scores.append(roc)

        mean_ap = float(np.mean(ap_scores))
        trial.set_user_attr("ap", mean_ap)
        trial.set_user_attr("precision", float(np.mean(precision_scores)))
        trial.set_user_attr("recall", float(np.mean(recall_scores)))
        trial.set_user_attr("roc_auc", float(np.mean(roc_auc_scores)))
        trial.set_user_attr("lambda_l1", params["lambda_l1"])
        trial.set_user_attr("lambda_l2", params["lambda_l2"])
        return mean_ap

    if n_optuna_jobs is None:
        n_optuna_jobs = min(MAX_PARALLEL_TRIALS, mp.cpu_count())

    study = optuna.create_study(direction="maximize", study_name=f"LGB_AP_{dataset_name}")
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True, n_jobs=n_optuna_jobs)

    # ── Collect results (completed trials only)
    def _penalty(l1: float, l2: float, tol: float = 1e-12) -> str:
        if l1 > tol and l2 < tol: return "L1"
        if l2 > tol and l1 < tol: return "L2"
        if l1 > tol and l2 > tol: return "ElasticNet"
        return "None"

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        raise ValueError("No completed trials with valid scores - reduce CV splits or up-sample so each fold has positives.")

    results = pd.DataFrame({
        "trial":     [t.number for t in completed],
        "ap":        [t.user_attrs["ap"] for t in completed],
        "precision": [t.user_attrs["precision"] for t in completed],
        "recall":    [t.user_attrs["recall"] for t in completed],
        "roc_auc":   [t.user_attrs["roc_auc"] for t in completed],
        "lambda_l1": [t.user_attrs["lambda_l1"] for t in completed],
        "lambda_l2": [t.user_attrs["lambda_l2"] for t in completed],
    })
    results["penalty"] = results.apply(lambda r: _penalty(r.lambda_l1, r.lambda_l2), axis=1)

    # Save artefacts
    ts = pd.Timestamp.now().strftime("%Y%m%d")
    joblib.dump(study, saved_dir / f"lgb_optuna_study_{ts}.pkl")
    with (saved_dir / f"lgb_optuna_best_params_{ts}.json").open("w") as fp:
        json.dump(study.best_params, fp, indent=2)
    results.to_csv(saved_dir / f"lgb_optuna_metrics_{ts}.csv", index=False)

    if make_plots:  # ✅ avoid NameError and let you toggle plotting
        _plot_metric_bars(results, saved_dir, ts)

    return study, results
