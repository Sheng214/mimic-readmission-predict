# xgb_optuna_tuner.py
from __future__ import annotations

"""
XGBoost + Optuna tuning utility for *moderately* imbalanced binary classification.

Changes & Capabilities based on XGBoost_optuna_AP_tuner.py

1) Adapted for less imbalance (~14% positives, not <1%)
   - Keeps XGBoost’s AP-centric training (`eval_metric="aucpr"`).
   - Uses a conservative `scale_pos_weight` (capped when 5–20% positive rate).
   - Optional resampling (`ros`/`smote`) available.

2) Speed optimizations
   - `tree_method="hist"`, `max_bin=256`, tighter per-dataset search spaces.
   - Early stopping (`early_stopping_rounds=30`) and fewer rounds (`num_boost_round=500`).
   - Parallel Optuna trials with a sane cap.

3) Progress bar
   - `tqdm` integrated via an Optuna callback for a single, clean trial progress bar.

4) Quiet console, rotated logs
   - No prints / `plt.show()`; Optuna set to WARNING.
   - Python logging → rotating file handlers only.

5) File-first outputs (no console dumps)
   - Study pickle, best-params JSON, per-trial metrics CSV, full trials CSV.
   - PNG plots saved to disk; summary CSVs for quick audits.

6) Explicit, dataset-aware tuning
   - Pass `dataset_name="tfidf" | "clinbert" | "tfidfbert"` when calling.
   - Search spaces adapt to the dataset; all folders and filenames are tagged with the dataset id.

7) Threshold tuning (default: maximize F1 on OOF)
   - Recomputes out-of-fold probs under best params; picks threshold (F1 by default; supports Fβ / constrained strategies).
   - Saves threshold JSON + CSV and plots (PR curve, F1-vs-threshold).

8) Final model & serving config
   - Fits a final booster on the full training set with best params (optional resampling).
   - Saves the model (`.json`/`.ubj`) and a `serving_config` JSON containing best params,
     tuned threshold, dataset id, positive rate, and (when available) feature names.

9) Dataset-tagged outputs & folders:
   - All artefacts live under `saved/<dataset_id>/` and filenames include the dataset id
     (`tfidf`, `clinbert`, or `tfidfbert`).

Key features:
- tqdm progress bar integrated with Optuna; minimal console noise otherwise.
- Rotating file logs only (no console spam).
- Save outputs (CSV + PNG) instead of printing.
- Threshold tuning on OOF predictions for best params; saves threshold + plots.
- Fit final model on full training set; save Booster + serving_config.

# CHANGED / NEW (per request 1/2):
- Require explicit dataset id via `dataset_name` ("tfidf" | "clinbert" | "tfidfbert").
- All saved file **names** and subfolders include the dataset id.
- Removed auto-detection logic and related helpers.
"""

from pathlib import Path
import json
import multiprocessing as mp
from typing import Tuple, Optional, Iterable, Dict
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

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
    precision_recall_curve,
    f1_score,
)
import matplotlib.pyplot as plt
from optuna.trial import TrialState
from imblearn.over_sampling import RandomOverSampler, SMOTE

# ────────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ------------------------------------------------------------------------------
CV_SPLITS = 5
RANDOM_STATE = 42
MAX_PARALLEL_TRIALS = 4
EARLY_STOP_ROUNDS = 30
NUM_BOOST_ROUND = 500
_ALLOWED_DATASET_IDS = {"tfidf", "clinbert", "tfidfbert"}  # NEW (per request 2)

# ────────────────────────────────────────────────────────────────────────────────
# Logging (file-only, rotated)
# ------------------------------------------------------------------------------
def _init_logger(log_dir: Path, dataset_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"xgb_optuna.{dataset_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = RotatingFileHandler(
        log_dir / f"xgb_optuna_{dataset_name}_{ts}.log",  # CHANGED: dataset id in filename
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

# ────────────────────────────────────────────────────────────────────────────────
# Param space (keyed by dataset_name explicitly)                                  # CHANGED
# ------------------------------------------------------------------------------
def _param_space_for(dataset_name: str, trial: optuna.Trial) -> dict:
    common = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "verbosity": 0,
        "booster": "gbtree",
        "tree_method": "hist",
        "max_bin": 256,
        "random_state": RANDOM_STATE,
        "nthread": max(1, mp.cpu_count() - 0),
    }
    if dataset_name == "tfidf":
        specific = {
            "eta": trial.suggest_float("eta", 0.03, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "gamma": trial.suggest_float("gamma", 0.0, 0.4),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 2.0, log=True),
        }
    elif dataset_name == "clinbert":
        specific = {
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 3.0, log=True),
        }
    elif dataset_name == "tfidfbert":
        specific = {
            "eta": trial.suggest_float("eta", 0.02, 0.25, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 3.0, log=True),
        }
    else:  # shouldn't happen with validation
        specific = {
            "eta": trial.suggest_float("eta", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 3.0, log=True),
        }
    common.update(specific)
    return common

# ────────────────────────────────────────────────────────────────────────────────
# Threshold selection utilities (unchanged)
# ------------------------------------------------------------------------------
def _select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "f1",
    beta: float = 1.0,
    min_precision: Optional[float] = None,
    min_recall: Optional[float] = None,
) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thr = np.r_[0.0, thresholds]  # align

    if strategy == "f1":
        f1_vals = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        idx = int(np.nanargmax(f1_vals))
        return {
            "threshold": float(thr[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1_vals[idx]),
            "strategy": "f1",
        }

    if strategy == "fbeta":
        beta2 = beta ** 2
        fbeta_vals = (1 + beta2) * precision * recall / np.clip(beta2 * precision + recall, 1e-12, None)
        idx = int(np.nanargmax(fbeta_vals))
        return {
            "threshold": float(thr[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "fbeta": float(fbeta_vals[idx]),
            "beta": float(beta),
            "strategy": "fbeta",
        }

    if strategy == "precision_at_recall":
        if min_recall is None:
            raise ValueError("min_recall must be set for strategy='precision_at_recall'")
        mask = recall >= min_recall
        if not np.any(mask):
            idx = int(np.nanargmax(recall))
        else:
            idx = np.arange(len(precision))[mask][int(np.nanargmax(precision[mask]))]
        f1_val = 2 * precision[idx] * recall[idx] / np.clip(precision[idx] + recall[idx], 1e-12, None)
        return {
            "threshold": float(thr[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1_val),
            "min_recall": float(min_recall),
            "strategy": "precision_at_recall",
        }

    if strategy == "recall_at_precision":
        if min_precision is None:
            raise ValueError("min_precision must be set for strategy='recall_at_precision'")
        mask = precision >= min_precision
        if not np.any(mask):
            idx = int(np.nanargmax(precision))
        else:
            idx = np.arange(len(recall))[mask][int(np.nanargmax(recall[mask]))]
        f1_val = 2 * precision[idx] * recall[idx] / np.clip(precision[idx] + recall[idx], 1e-12, None)
        return {
            "threshold": float(thr[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1_val),
            "min_precision": float(min_precision),
            "strategy": "recall_at_precision",
        }

    raise ValueError(f"Unknown threshold strategy: {strategy}")

def _plot_pr_and_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: Path,
    ts: str,
    chosen_threshold: float,
    dataset_name: str,  # NEW (per request 1): tag plots with dataset id
) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thr = np.r_[0.0, thresholds]
    # PR curve
    fig1 = plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (OOF) — {dataset_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(out_dir / f"oof_pr_curve_{dataset_name}_{ts}.png", dpi=300)  # CHANGED
    plt.close(fig1)
    # F1 vs threshold
    f1_vals = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    fig2 = plt.figure(figsize=(8, 6))
    plt.plot(thr, f1_vals, marker=".", linewidth=1)
    plt.axvline(chosen_threshold, linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("F1 (OOF)")
    plt.title(f"F1 vs Threshold (OOF) — {dataset_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(out_dir / f"oof_f1_vs_threshold_{dataset_name}_{ts}.png", dpi=300)  # CHANGED
    plt.close(fig2)

# ────────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ------------------------------------------------------------------------------
def run_xgb_optuna(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    *,
    dataset_name: Optional[str] = None,        # CHANGED: now REQUIRED and validated
    n_trials: int = 100,
    saved_root: str | Path | None = None,
    n_optuna_jobs: int | None = None,
    cv: StratifiedKFold | None = None,
    sampler: str | None = None,  # "ros", "smote", or None
    use_tqdm: bool = True,
    clinbert_cols: Optional[Iterable[str]] = None,  # kept for BC, ignored now  # CHANGED
    # Threshold tuning params
    tune_threshold: bool = True,
    threshold_strategy: str = "f1",
    fbeta_beta: float = 1.0,
    min_precision: Optional[float] = None,
    min_recall: Optional[float] = None,
    # Final model params
    fit_final_model: bool = True,
    final_fit_sampler: Optional[str] = None,
    final_model_format: str = "json",
) -> Tuple[optuna.Study, pd.DataFrame]:
    """Tune XGBoost with Optuna, save artefacts, tune decision threshold, and (optionally) fit final model.

    Parameters
    ----------
    dataset_name : {"tfidf","clinbert","tfidfbert"}
        REQUIRED. Dataset id used to (a) choose search space and (b) tag folders/files.
    """

    # ── Validate dataset id (per request 2) ─────────────────────────────────────
    if dataset_name is None or dataset_name not in _ALLOWED_DATASET_IDS:
        raise ValueError(
            f"`dataset_name` must be one of {_ALLOWED_DATASET_IDS}. "
            f"Received: {dataset_name!r}"
        )

    # Paths
    root_dir = Path(saved_root) if saved_root else Path.cwd().parent
    saved_dir = root_dir / "saved"
    saved_dir.mkdir(parents=True, exist_ok=True)

    # Folder per dataset id (per request 1)
    ds_dir = saved_dir / dataset_name
    plots_dir = ds_dir / "plots"
    logs_dir = ds_dir / "logs"
    ds_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger = _init_logger(logs_dir, dataset_name)
    logger.info(f"Run start: dataset_name={dataset_name}, trials={n_trials}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # CV
    if cv is None:
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Imbalance handling
    pos_rate = float(np.mean(y_train))
    if pos_rate < 0.05:
        scale_pos_weight_default = (1.0 - pos_rate) / max(1e-12, pos_rate)
    elif pos_rate < 0.20:
        scale_pos_weight_default = min((1.0 - pos_rate) / max(1e-12, pos_rate), 3.0)
    else:
        scale_pos_weight_default = 1.0
    logger.info(f"Estimated positive rate={pos_rate:.4f}; scale_pos_weight={scale_pos_weight_default:.3f}")

    # Sparse detection to guard samplers
    try:
        import scipy.sparse as sp  # type: ignore
        is_sparse = sp.issparse(X_train)
    except Exception:
        is_sparse = False

    if sampler in {"smote"} and (is_sparse or not hasattr(X_train, "iloc")):
        logger.info("SMOTE disabled: unsupported for sparse arrays or non-DataFrame inputs.")
        _sampler_obj = None
    elif sampler == "smote":
        _sampler_obj = SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=RANDOM_STATE)
    elif sampler == "ros":
        _sampler_obj = RandomOverSampler(sampling_strategy=0.5, random_state=RANDOM_STATE)
    else:
        _sampler_obj = None

    # Objective
    def _objective(trial: optuna.Trial) -> float:
        params = _param_space_for(dataset_name, trial)  # CHANGED: keyed by dataset_name
        params.update({"scale_pos_weight": scale_pos_weight_default})

        ap_scores, precision_scores, recall_scores, roc_auc_scores = [], [], [], []

        for _, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr = X_train.iloc[train_idx] if hasattr(X_train, "iloc") else X_train[train_idx]
            y_tr = y_train.iloc[train_idx] if hasattr(y_train, "iloc") else y_train[train_idx]
            X_val = X_train.iloc[valid_idx] if hasattr(X_train, "iloc") else X_train[valid_idx]
            y_val = y_train.iloc[valid_idx] if hasattr(y_train, "iloc") else y_train[valid_idx]

            if _sampler_obj is not None:
                try:
                    X_tr, y_tr = _sampler_obj.fit_resample(X_tr, y_tr)
                except Exception as e:
                    logger.info(f"Sampler disabled due to error: {type(e).__name__}: {e}")
                    pass

            dtrain = xgb.DMatrix(X_tr, label=y_tr, nthread=params.get("nthread", 0))
            dvalid = xgb.DMatrix(X_val, label=y_val, nthread=params.get("nthread", 0))

            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=NUM_BOOST_ROUND,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=EARLY_STOP_ROUNDS,
                verbose_eval=False,
            )

            y_prob = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))
            y_pred = (y_prob >= 0.5).astype(int)

            if float(np.sum(y_val)) == 0.0:
                raise optuna.exceptions.TrialPruned("No positives in valid fold")

            ap = average_precision_score(y_val, y_prob, pos_label=1)
            prec = precision_score(y_val, y_pred, zero_division=0, pos_label=1)
            rec = recall_score(y_val, y_pred, zero_division=0, pos_label=1)
            roc = roc_auc_score(y_val, y_prob)

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

    # Run study (with optional tqdm)
    if n_optuna_jobs is None:
        n_optuna_jobs = min(MAX_PARALLEL_TRIALS, mp.cpu_count())
    study = optuna.create_study(direction="maximize", study_name=f"XGB_AP_{dataset_name}")  # CHANGED
    if use_tqdm:
        from tqdm.auto import tqdm
        pbar = tqdm(total=n_trials, desc=f"Optuna Trials ({dataset_name})", miniters=1)
        def _tqdm_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            pbar.update(1)
        study.optimize(_objective, n_trials=n_trials, n_jobs=n_optuna_jobs, callbacks=[_tqdm_cb], show_progress_bar=False)
        pbar.close()
    else:
        study.optimize(_objective, n_trials=n_trials, n_jobs=n_optuna_jobs, show_progress_bar=False)

    logger.info(f"Best value (AP): {study.best_value:.6f}")
    logger.info(f"Best params: {study.best_params}")

    # Collect results (completed trials only)
    def _penalty(l1: float, l2: float, tol: float = 1e-12) -> str:
        if l1 > tol and l2 < tol:
            return "L1"
        if l2 > tol and l1 < tol:
            return "L2"
        if l1 > tol and l2 > tol:
            return "ElasticNet"
        return "None"

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        raise ValueError("No completed trials with valid scores - adjust CV or resampling.")

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

    # Save artefacts (filenames include dataset id)                                 # CHANGED
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(study, ds_dir / f"xgb_optuna_study_{dataset_name}_{ts}.pkl")
    with (ds_dir / f"xgb_optuna_best_params_{dataset_name}_{ts}.json").open("w") as fp:
        json.dump(study.best_params, fp, indent=2)
    results.to_csv(ds_dir / f"xgb_optuna_metrics_{dataset_name}_{ts}.csv", index=False)

    try:
        trials_df = study.trials_dataframe()
        trials_df.to_csv(ds_dir / f"xgb_optuna_trials_full_{dataset_name}_{ts}.csv", index=False)
    except Exception as e:
        logger.info(f"Failed to export trials_dataframe: {type(e).__name__}: {e}")

    _plot_metric_bars(results, plots_dir, ts, dataset_name)  # CHANGED: pass dataset_name

    # Summary CSV
    best_summary = {
        "timestamp": ts,
        "dataset_name": dataset_name,
        "pos_rate": pos_rate,
        "best_ap": study.best_value,
        "best_trial": study.best_trial.number,
        **{f"param_{k}": v for k, v in study.best_params.items()},
    }
    pd.DataFrame([best_summary]).to_csv(ds_dir / f"xgb_optuna_best_summary_{dataset_name}_{ts}.csv", index=False)

    # ────────────────────────────────────────────────────────────────────────────
    # Threshold tuning on OOF predictions for **best params**
    # ────────────────────────────────────────────────────────────────────────────
    chosen_threshold_info = None
    if tune_threshold:
        logger.info("Starting threshold tuning on OOF predictions (best params).")
        params_best = _param_space_for(dataset_name, study.best_trial)
        params_best.update(study.best_params)
        params_best.update({"scale_pos_weight": scale_pos_weight_default})

        oof_prob = np.zeros_like(np.asarray(y_train, dtype=float))
        oof_true = np.asarray(y_train, dtype=int)

        for _, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr = X_train.iloc[train_idx] if hasattr(X_train, "iloc") else X_train[train_idx]
            y_tr = y_train.iloc[train_idx] if hasattr(y_train, "iloc") else y_train[train_idx]
            X_val = X_train.iloc[valid_idx] if hasattr(X_train, "iloc") else X_train[valid_idx]
            y_val = y_train.iloc[valid_idx] if hasattr(y_train, "iloc") else y_train[valid_idx]

            dtrain = xgb.DMatrix(X_tr, label=y_tr, nthread=params_best.get("nthread", 0))
            dvalid = xgb.DMatrix(X_val, label=y_val, nthread=params_best.get("nthread", 0))

            bst = xgb.train(
                params_best,
                dtrain,
                num_boost_round=NUM_BOOST_ROUND,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=EARLY_STOP_ROUNDS,
                verbose_eval=False,
            )
            oof_prob[valid_idx] = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))

        # Choose threshold
        if threshold_strategy == "fbeta":
            chosen_threshold_info = _select_threshold(oof_true, oof_prob, strategy="fbeta", beta=fbeta_beta)
        elif threshold_strategy == "precision_at_recall":
            chosen_threshold_info = _select_threshold(oof_true, oof_prob, strategy="precision_at_recall", min_recall=min_recall)
        elif threshold_strategy == "recall_at_precision":
            chosen_threshold_info = _select_threshold(oof_true, oof_prob, strategy="recall_at_precision", min_precision=min_precision)
        else:
            chosen_threshold_info = _select_threshold(oof_true, oof_prob, strategy="f1")

        thr = chosen_threshold_info["threshold"]
        oof_pred = (oof_prob >= thr).astype(int)
        chosen_threshold_info.update({
            "oof_precision": float(precision_score(oof_true, oof_pred, zero_division=0)),
            "oof_recall": float(recall_score(oof_true, oof_pred, zero_division=0)),
            "oof_f1": float(f1_score(oof_true, oof_pred, zero_division=0)),
            "oof_ap": float(average_precision_score(oof_true, oof_prob)),
            "oof_roc_auc": float(roc_auc_score(oof_true, oof_prob)),
        })

        # Save threshold (dataset id in names)                                      # CHANGED
        thr_json = ds_dir / f"xgb_optuna_best_threshold_{dataset_name}_{ts}.json"
        with thr_json.open("w") as fp:
            json.dump(chosen_threshold_info, fp, indent=2)
        pd.DataFrame([{
            "timestamp": ts,
            "dataset_name": dataset_name,
            "strategy": chosen_threshold_info.get("strategy"),
            "threshold": chosen_threshold_info.get("threshold"),
            "precision": chosen_threshold_info.get("oof_precision"),
            "recall": chosen_threshold_info.get("oof_recall"),
            "f1": chosen_threshold_info.get("oof_f1"),
            "ap": chosen_threshold_info.get("oof_ap"),
            "roc_auc": chosen_threshold_info.get("oof_roc_auc"),
        }]).to_csv(ds_dir / f"xgb_optuna_best_threshold_{dataset_name}_{ts}.csv", index=False)

        _plot_pr_and_f1(oof_true, oof_prob, plots_dir, ts, chosen_threshold_info["threshold"], dataset_name)  # CHANGED
        logger.info(f"Chosen threshold={chosen_threshold_info['threshold']:.6f} via {chosen_threshold_info['strategy']}")

    # ────────────────────────────────────────────────────────────────────────────
    # Fit FINAL model on full training data (best params)
    # ────────────────────────────────────────────────────────────────────────────
    if fit_final_model:
        logger.info("Fitting final model on full training set with best params.")
        params_final = _param_space_for(dataset_name, study.best_trial)
        params_final.update(study.best_params)
        params_final.update({"scale_pos_weight": scale_pos_weight_default})

        X_fit, y_fit = X_train, y_train

        # Optional resampling for final fit (default None to avoid distribution shift)
        final_sampler_obj = None
        if final_fit_sampler in {"ros", "smote"}:
            if final_fit_sampler == "smote":
                if is_sparse or not hasattr(X_train, "iloc"):
                    logger.info("Final SMOTE disabled (sparse or non-DataFrame input).")
                else:
                    final_sampler_obj = SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=RANDOM_STATE)
            elif final_fit_sampler == "ros":
                final_sampler_obj = RandomOverSampler(sampling_strategy=0.5, random_state=RANDOM_STATE)

        if final_sampler_obj is not None:
            try:
                X_fit, y_fit = final_sampler_obj.fit_resample(X_fit, y_fit)
            except Exception as e:
                logger.info(f"Final-fit sampler disabled due to error: {type(e).__name__}: {e}")
                X_fit, y_fit = X_train, y_train

        dtrain_full = xgb.DMatrix(X_fit, label=y_fit, nthread=params_final.get("nthread", 0))
        final_bst = xgb.train(
            params_final,
            dtrain_full,
            num_boost_round=NUM_BOOST_ROUND,
            verbose_eval=False,
        )

        # Save model (JSON or UBJ) with dataset id                                  # CHANGED
        model_suffix = "json" if final_model_format.lower() == "json" else "ubj"
        model_path = ds_dir / f"final_xgb_model_{dataset_name}_{ts}.{model_suffix}"
        final_bst.save_model(str(model_path))

        # Save serving config: params + tuned threshold + dataset meta              # CHANGED
        serving_config = {
            "timestamp": ts,
            "dataset_name": dataset_name,
            "pos_rate": pos_rate,
            "best_params": study.best_params,
            "threshold": (chosen_threshold_info or {}).get("threshold", 0.5),
            "threshold_strategy": (chosen_threshold_info or {}).get("strategy", "fixed_0.5"),
            "model_file": model_path.name,
            "model_format": model_suffix,
            "feature_names": list(map(str, X_train.columns)) if hasattr(X_train, "columns") else None,
        }
        with (ds_dir / f"serving_config_{dataset_name}_{ts}.json").open("w") as fp:
            json.dump(serving_config, fp, indent=2)

        logger.info(f"Saved final model -> {model_path.name}")
        logger.info(f"Saved serving_config_{dataset_name}_{ts}.json")

    logger.info(f"Saved metrics and plots under {ds_dir}")
    logger.info("Run finished.")

    return study, results

# ────────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ------------------------------------------------------------------------------
def _plot_metric_bars(results: pd.DataFrame, out_dir: Path, ts: str, dataset_name: str) -> None:
    """Create two bar charts: one with grouped PR metrics per penalty, and one for ROC AUC (saved to files)."""
    metrics1 = ["ap", "precision", "recall"]
    grouped1 = results.groupby("penalty", dropna=False)[metrics1].mean()
    grouped1 = grouped1.rename(columns={"ap": "Avg Precision", "precision": "Precision", "recall": "Recall"})
    grouped1 = grouped1.sort_values("Avg Precision", ascending=False)
    ax1 = grouped1.plot(kind="bar", figsize=(10, 6))
    ax1.set_title(f"Penalties - Average Precision, Precision, Recall — {dataset_name}")
    ax1.set_ylabel("Score")
    ax1.figure.tight_layout()
    ax1.figure.savefig(out_dir / f"pr_metrics_by_penalty_{dataset_name}_{ts}.png", dpi=300)  # CHANGED
    plt.close(ax1.figure)

    grouped2 = results.groupby("penalty", dropna=False)[["roc_auc"]].mean().rename(columns={"roc_auc": "ROC AUC"})
    grouped2 = grouped2.sort_values("ROC AUC", ascending=False)
    ax2 = grouped2.plot(kind="bar", figsize=(10, 6), legend=True)
    ax2.set_title(f"Penalties - ROC AUC — {dataset_name}")
    ax2.set_ylabel("Score")
    ax2.figure.tight_layout()
    ax2.figure.savefig(out_dir / f"roc_auc_by_penalty_{dataset_name}_{ts}.png", dpi=300)  # CHANGED
    plt.close(ax2.figure)

# ────────────────────────────────────────────────────────────────────────────────
# CLI example (optional; quiet)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from sklearn.datasets import make_classification

    parser = argparse.ArgumentParser(description="Demo run with synthetic data (for smoke test).")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--out", type=str, default=None, help="Root to save artefacts; default = cwd/..")
    parser.add_argument("--dataset-name", type=str, default="tfidf", choices=list(_ALLOWED_DATASET_IDS))  # CHANGED
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--fit-final", action="store_true")
    parser.add_argument("--final-sampler", type=str, default=None, choices=[None, "ros", "smote"])
    parser.add_argument("--final-format", type=str, default="json", choices=["json", "ubj"])
    args = parser.parse_args()

    X_demo, y_demo = make_classification(
        n_samples=5000,
        n_features=200,
        n_informative=20,
        weights=[0.86, 0.14],
        random_state=RANDOM_STATE,
    )
    X_demo = pd.DataFrame(X_demo)
    y_demo = pd.Series(y_demo)

    run_xgb_optuna(
        X_demo,
        y_demo,
        dataset_name=args.dataset_name,  # CHANGED
        n_trials=args.trials,
        saved_root=args.out,
        sampler=None,
        use_tqdm=True,
        tune_threshold=bool(args.tune_threshold),
        fit_final_model=bool(args.fit_final),
        final_fit_sampler=args.final_sampler if args.final_sampler not in ("None", "none") else None,
        final_model_format=args.final_format,
    )
