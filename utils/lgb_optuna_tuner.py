# lgb_optuna_tuner.py
from __future__ import annotations
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

"""
LightGBM + Optuna tuner for moderately imbalanced binary classification.

Highlights
- tuned for moderately imbalanced data (~14%) (still safe for rarer cases) 
- Faster search space + early stopping
- Progress bars (Optuna + tqdm)
- Quiet console (RotatingFileHandler logs only)
- all artefacts saved to disk (trials.csv, best_params.json, study.pkl, PR curve & F1-vs-threshold plots PNGs, final_model.pkl, tuned_threshold.json)
- Dataset-aware defaults (tfidf | clinbert | tfidfbert), The dataset_name drives sensible defaults for TF-IDF vs. CLINBERT vs. mixed features.
- OOF threshold tuning (maximize F1)
- paths & filenames include dataset id

Usage (in notebook)
-------------------
from lgb_optuna_tuner import run_lgb_optuna  # alias: run_xgb_optuna
study, results, artefacts = run_lgb_optuna(
    X_train, y_train, dataset_name="tfidf", n_trials=60
)
"""

from pathlib import Path
import json
import multiprocessing as mp
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib
import optuna
import lightgbm as lgb
from optuna.trial import TrialState
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    f1_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

# ───────────────────────── logging: file only, rotated ─────────────────────────
import logging
from logging.handlers import RotatingFileHandler

def _setup_logger(log_dir: Path, dataset_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"lgb_optuna_{dataset_name}")
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers if function is called multiple times
    if not logger.handlers:
        fh = RotatingFileHandler(
            filename=log_dir / f"lgb_optuna_{dataset_name}.log",
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        # Do NOT add a StreamHandler -> keep console clean
        logger.propagate = False
    return logger

# ─────────────────────────────── helpers ───────────────────────────────

def _dataset_defaults(name: str) -> Dict[str, Any]:
    """Return dataset-aware defaults & search ranges."""
    name = name.lower()
    if name == "tfidf":
        # very high-dim sparse
        return dict(
            learning_rate=(0.03, 0.2),
            num_leaves=(16, 128),
            max_depth=(3, 10),
            min_data_in_leaf=(20, 200),
            feature_fraction=(0.2, 0.7),
            bagging_fraction=(0.6, 1.0),
            lambda_l1=(1e-3, 2.0),
            lambda_l2=(1e-3, 2.0),
            min_gain_to_split=(0.0, 0.2),
            n_estimators=(200, 1500),
        )
    elif name == "clinbert":
        # lower-dim dense embeddings
        return dict(
            learning_rate=(0.02, 0.2),
            num_leaves=(31, 512),
            max_depth=(3, 16),
            min_data_in_leaf=(10, 100),
            feature_fraction=(0.6, 1.0),
            bagging_fraction=(0.6, 1.0),
            lambda_l1=(1e-4, 1.0),
            lambda_l2=(1e-4, 1.0),
            min_gain_to_split=(0.0, 0.2),
            n_estimators=(200, 2000),
        )
    elif name == "tfidfbert":
        # mixed: sparse + dense
        return dict(
            learning_rate=(0.02, 0.2),
            num_leaves=(24, 256),
            max_depth=(3, 14),
            min_data_in_leaf=(20, 150),
            feature_fraction=(0.35, 0.9),
            bagging_fraction=(0.6, 1.0),
            lambda_l1=(1e-3, 1.5),
            lambda_l2=(1e-3, 1.5),
            min_gain_to_split=(0.0, 0.2),
            n_estimators=(200, 1800),
        )
    else:
        # sensible generic
        return dict(
            learning_rate=(0.03, 0.2),
            num_leaves=(31, 256),
            max_depth=(3, 12),
            min_data_in_leaf=(20, 150),
            feature_fraction=(0.4, 0.9),
            bagging_fraction=(0.6, 1.0),
            lambda_l1=(1e-3, 1.0),
            lambda_l2=(1e-3, 1.0),
            min_gain_to_split=(0.0, 0.2),
            n_estimators=(200, 1500),
        )

def _safe_scale_pos_weight(y: np.ndarray) -> Optional[float]:
    """Use scale_pos_weight only for very rare positives (<5%)."""
    pos = y.sum()
    n = len(y)
    rate = pos / max(n, 1)
    if rate < 0.05:
        neg = n - pos
        return float(neg / max(pos, 1))
    return None

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _fit_with_safe_early_stopping(
    model, X_tr, y_tr, X_va, y_va, early_stopping_rounds: int, metric: str = "average_precision"
):
    """
    Fit LightGBM with early stopping. If validation set is single-class (metric can't be computed),
    fall back to using the training slice for early stopping (to avoid callback ValueError).
    """
    import numpy as np

    # Check if y_va has both classes
    has_both = np.unique(y_va).size >= 2

    if has_both:
        eval_set = [(X_va, y_va)]
    else:
        # Fallback: use training split for early stopping to keep callback happy
        eval_set = [(X_tr, y_tr)]

    model.fit(
        X_tr, y_tr,
        eval_set=eval_set,
        eval_metric=metric,             # "average_precision" is broadly supported
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
    )
    return model


# ─────────────────────────────── main API ──────────────────────────────

def run_lgb_optuna(
    X_train: "np.ndarray | pd.DataFrame",
    y_train: "np.ndarray | pd.Series",
    *,
    dataset_name: str,
    n_trials: int = 80,
    saved_root: str | Path | None = None,
    n_optuna_jobs: int | None = None,
    cv_splits: int = 5,
    random_state: int = 42,
    early_stopping_rounds: int = 50,
) -> Tuple[optuna.Study, pd.DataFrame, Dict[str, Path]]:
    """
    Tune LightGBM with Optuna; perform OOF threshold tuning (max F1); fit final model.

    Returns
    -------
    study : optuna.Study
    results : pd.DataFrame      # per-trial metrics
    artefacts : Dict[str, Path] # paths to saved files
    """
    # Paths
    root_dir = Path(saved_root) if saved_root else Path.cwd().parent
    saved_dir = root_dir / "saved" / dataset_name
    plots_dir = saved_dir / "plots"
    logs_dir = saved_dir / "logs"
    saved_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(logs_dir, dataset_name)
    logger.info(f"Start run | dataset={dataset_name} | trials={n_trials} | cv={cv_splits}")

    # CV splitter
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # dataset-aware ranges
    space = _dataset_defaults(dataset_name)

    # scale_pos_weight if truly rare
    y_np = np.asarray(y_train).astype(int)
    spw = _safe_scale_pos_weight(y_np)
    if spw is not None:
        logger.info(f"Detected rare positives; using scale_pos_weight={spw:.2f}")
    else:
        logger.info("Moderate imbalance detected; not using scale_pos_weight.")

    # parallel Optuna trials
    if n_optuna_jobs is None:
        n_optuna_jobs = min(4, mp.cpu_count())

    # Objective
    def _objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "average_precision",  # optimize AP
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": random_state,
            "n_jobs": -1,
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "num_leaves": trial.suggest_int("num_leaves", *space["num_leaves"]),
            "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", *space["min_data_in_leaf"]),
            "feature_fraction": trial.suggest_float("feature_fraction", *space["feature_fraction"]),
            "bagging_fraction": trial.suggest_float("bagging_fraction", *space["bagging_fraction"]),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", *space["lambda_l1"], log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", *space["lambda_l2"], log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", *space["min_gain_to_split"]),
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
        }
        if spw is not None:
            params["scale_pos_weight"] = spw

        # fold loop (so we can do early stopping & progress)
        ap_scores = []
        roc_scores = []
        for fold_idx, (trn_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, X_va = X_train.iloc[trn_idx] if hasattr(X_train, "iloc") else X_train[trn_idx], \
                         X_train.iloc[val_idx] if hasattr(X_train, "iloc") else X_train[val_idx]
            y_tr, y_va = y_train.iloc[trn_idx] if hasattr(y_train, "iloc") else y_train[trn_idx], \
                         y_train.iloc[val_idx] if hasattr(y_train, "iloc") else y_train[val_idx]

            model = lgb.LGBMClassifier(**params)
            model = _fit_with_safe_early_stopping(
                model, X_tr, y_tr, X_va, y_va, early_stopping_rounds, metric="average_precision"
                )
            y_prob = model.predict_proba(X_va)[:, 1]
            ap_scores.append(average_precision_score(y_va, y_prob))
            roc_scores.append(roc_auc_score(y_va, y_prob) if np.unique(y_va).size >= 2 else np.nan)

        ap_mean = float(np.mean(ap_scores))
        roc_mean = float(np.mean(roc_scores))

        trial.set_user_attr("ap", ap_mean)
        trial.set_user_attr("roc_auc", roc_mean)
        return ap_mean  # Optimize AP

    study = optuna.create_study(direction="maximize", study_name=f"LGB_AP_{dataset_name}")
    study.optimize(_objective, n_trials=n_trials, n_jobs=n_optuna_jobs, show_progress_bar=True)
    logger.info(f"Study complete | best_value(AP)={study.best_value:.5f}")

    # Aggregate results (completed trials only)
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("No completed trials. Consider fewer CV splits or adjust search space.")

    results = pd.DataFrame(
        {
            "trial": [t.number for t in completed],
            "ap": [t.user_attrs.get("ap", np.nan) for t in completed],
            "roc_auc": [t.user_attrs.get("roc_auc", np.nan) for t in completed],
            **{k: [t.params.get(k, np.nan) for t in completed] for k in study.best_params.keys()},
        }
    ).sort_values("ap", ascending=False)

    # Save trials and best params
    ts = _timestamp()
    best_params_path = saved_dir / f"best_params_{dataset_name}_{ts}.json"
    trials_csv_path = saved_dir / f"trials_{dataset_name}_{ts}.csv"
    study_pkl_path = saved_dir / f"study_{dataset_name}_{ts}.pkl"
    results.to_csv(trials_csv_path, index=False)
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    joblib.dump(study, study_pkl_path)
    logger.info(f"Saved trials -> {trials_csv_path.name}, best_params -> {best_params_path.name}")

    # ── OOF predictions with best params for threshold tuning ──
    best_params = dict(study.best_params)
    # keep core params consistent
    best_params.update(dict(
        objective="binary",
        metric="average_precision",
        boosting_type="gbdt",
        verbosity=-1,
        random_state=random_state,
        n_jobs=1,
    ))
    if spw is not None:
        best_params["scale_pos_weight"] = spw

    oof_prob = np.zeros(len(y_train), dtype=float)
    oof_true = np.asarray(y_train).astype(int)
    # per-fold fit with early stopping
    for fold_idx, (trn_idx, val_idx) in enumerate(tqdm(cv.split(X_train, y_train), total=cv_splits, desc="OOF")):
        X_tr, X_va = X_train.iloc[trn_idx] if hasattr(X_train, "iloc") else X_train[trn_idx], \
                     X_train.iloc[val_idx] if hasattr(X_train, "iloc") else X_train[val_idx]
        y_tr, y_va = y_train.iloc[trn_idx] if hasattr(y_train, "iloc") else y_train[trn_idx], \
                     y_train.iloc[val_idx] if hasattr(y_train, "iloc") else y_train[val_idx]

        model = lgb.LGBMClassifier(**best_params)
        model = _fit_with_safe_early_stopping(
            model, X_tr, y_tr, X_va, y_va, early_stopping_rounds, metric="average_precision"
        )
        oof_prob[val_idx] = model.predict_proba(X_va)[:, 1]


    # Threshold tuning: maximize F1 on OOF
    prec, rec, thr = precision_recall_curve(oof_true, oof_prob)
    # precision_recall_curve returns threshold array len = len(prec)-1
    f1_vals = (2 * prec[:-1] * rec[:-1]) / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    best_idx = int(np.nanargmax(f1_vals))
    tuned_threshold = float(thr[best_idx])
    tuned_f1 = float(f1_vals[best_idx])
    tuned_ap = float(average_precision_score(oof_true, oof_prob))
    tuned_roc = float(roc_auc_score(oof_true, oof_prob))
    logger.info(f"Tuned threshold={tuned_threshold:.6f} | OOF F1={tuned_f1:.4f} AP={tuned_ap:.4f} ROC_AUC={tuned_roc:.4f}")

    # Save threshold & OOF metrics
    threshold_json_path = saved_dir / f"threshold_{dataset_name}_{ts}.json"
    with open(threshold_json_path, "w") as f:
        json.dump(
            dict(threshold=tuned_threshold, oof_f1=tuned_f1, oof_ap=tuned_ap, oof_roc_auc=tuned_roc),
            f, indent=2
        )

    # ── Plots: PR curve + F1 vs threshold ──
    import matplotlib.pyplot as plt

    # PR curve
    plt.figure(figsize=(7, 6))
    plt.plot(rec, prec, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({dataset_name}) | AP={tuned_ap:.3f}")
    pr_png = plots_dir / f"pr_curve_{dataset_name}_{ts}.png"
    plt.tight_layout()
    plt.savefig(pr_png, dpi=300)
    plt.close()

    # F1 vs threshold
    plt.figure(figsize=(7, 6))
    plt.plot(thr, f1_vals, linewidth=2)
    plt.axvline(tuned_threshold, linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.title(f"F1 vs Threshold ({dataset_name}) | Best={tuned_f1:.3f} @ {tuned_threshold:.3f}")
    f1_png = plots_dir / f"f1_vs_threshold_{dataset_name}_{ts}.png"
    plt.tight_layout()
    plt.savefig(f1_png, dpi=300)
    plt.close()

    # ── Fit final model on full training set ──
    final_model = lgb.LGBMClassifier(**best_params)

    # use small validation split for early stopping even on full fit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=random_state)
    (tr_idx, va_idx), = sss.split(
        np.arange(len(y_train)),
        np.asarray(y_train).astype(int)
    )


    X_tr_full = X_train.iloc[tr_idx] if hasattr(X_train, "iloc") else X_train[tr_idx]
    y_tr_full = y_train.iloc[tr_idx] if hasattr(y_train, "iloc") else y_train[tr_idx]
    X_va_full = X_train.iloc[va_idx] if hasattr(X_train, "iloc") else X_train[va_idx]
    y_va_full = y_train.iloc[va_idx] if hasattr(y_train, "iloc") else y_train[va_idx]

   # Use the robust fit wrapper you defined earlier:
    final_model = _fit_with_safe_early_stopping(
        final_model, X_tr_full, y_tr_full, X_va_full, y_va_full,
        early_stopping_rounds,
        metric="average_precision"   # <— key change from "aucpr"
    )

    final_model_path = saved_dir / f"final_model_{dataset_name}_{ts}.pkl"
    joblib.dump(dict(model=final_model, threshold=tuned_threshold, params=best_params), final_model_path)
    logger.info(f"Saved final_model -> {final_model_path.name}")

    artefacts = {
        "saved_dir": saved_dir,
        "trials_csv": trials_csv_path,
        "best_params_json": best_params_path,
        "study_pkl": study_pkl_path,
        "threshold_json": threshold_json_path,
        "pr_png": pr_png,
        "f1_png": f1_png,
        "final_model_pkl": final_model_path,
        "log_file": logs_dir / f"lgb_optuna_{dataset_name}.log",
    }
    return study, results, artefacts

# Convenience alias (in case the notebook accidentally calls xgb name)
def run_xgb_optuna(*args, **kwargs):
    return run_lgb_optuna(*args, **kwargs)

# CLI demo
if __name__ == "__main__":
    print("Demo on synthetic data (not for production).")
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=5000,
        n_features=60,
        n_informative=15,
        weights=[0.86, 0.14],  # ~14% positives
        random_state=42,
    )
    X = pd.DataFrame(X)
    y = pd.Series(y)
    run_lgb_optuna(X, y, dataset_name="demo", n_trials=20)
