from scipy.sparse import issparse
from imblearn.over_sampling import RandomOverSampler, SMOTE

def _safe_resample(X, y, method: str = "smote", target_ratio: float = 0.5, random_state: int = 42):
    """
    Resample (X, y) to the given positive-class ratio.
    - If X is sparse, uses RandomOverSampler (SMOTE doesn't support sparse).
    - If very few minority samples, reduces k_neighbors or falls back to ROS.
    """
    if method not in {"smote", "ros"}:
        return X, y  # no resampling

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
    sampler: str | None = "smote",   # "smote", "ros", or None  ⟵ default oversamples both sides
) -> Tuple[StackingClassifier, pd.DataFrame]:
    """Train a stacking ensemble and save artefacts (CV with SMOTE/ROS on train & valid)."""

    # Paths
    root_dir = Path(saved_root) if saved_root else Path.cwd().parent
    saved_dir = root_dir / "saved" / dataset_name
    saved_dir.mkdir(parents=True, exist_ok=True)

    # CV splitter
    if cv is None:
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # ---- Base model params (avoid double-compensation when oversampling both sides)
    lr_params  = best_lr_params.copy()
    xgb_params = best_xgb_params.copy()
    lgb_params = best_lgb_params.copy()

    if sampler in {"smote", "ros"}:
        # If provided, neutralize weighting because we oversample both train and valid
        if lr_params.get("class_weight") == "balanced":
            lr_params["class_weight"] = None
        if "scale_pos_weight" in xgb_params:  # XGBoost
            xgb_params["scale_pos_weight"] = 1.0
        if "scale_pos_weight" in lgb_params:  # LightGBM
            lgb_params["scale_pos_weight"] = 1.0

    # Define base models
    lr        = LogisticRegression(**lr_params)
    xgb_model = xgb.XGBClassifier(**xgb_params)
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    estimators = [('lr', lr), ('xgb', xgb_model), ('lgb', lgb_model)]

    # Stacking classifier (meta = LR). Ensure solver is set.
    meta_params = lr_params.copy()
    if 'solver' not in meta_params:
        meta_params['solver'] = 'liblinear'
    final_lr = LogisticRegression(**meta_params)

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_lr,
        cv=INTERNAL_CV_SPLITS,
        n_jobs=-1,
        verbose=0
    )

    # ---- CV evaluation
    ap_scores, precision_scores, recall_scores, roc_auc_scores = [], [], [], []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
        # Indexing for pandas/ndarray
        X_tr = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
        y_tr = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
        X_va = X_train.iloc[valid_idx] if hasattr(X_train, 'iloc') else X_train[valid_idx]
        y_va = y_train.iloc[valid_idx] if hasattr(y_train, 'iloc') else y_train[valid_idx]

        # ── OVERSAMPLE BOTH TRAIN AND VALIDATION ──────────────────────
        if sampler in {"smote", "ros"}:
            X_tr, y_tr = _safe_resample(X_tr, y_tr, method=sampler, target_ratio=0.5, random_state=RANDOM_STATE)
            X_va, y_va = _safe_resample(X_va, y_va, method=sampler, target_ratio=0.5, random_state=RANDOM_STATE)

        # Fit and evaluate
        stack.fit(X_tr, y_tr)
        y_prob = stack.predict_proba(X_va)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        if (np.array(y_va) == 1).sum() == 0:
            # Should not happen after oversampling, but guard anyway
            continue

        ap   = average_precision_score(y_va, y_prob)
        prec = precision_score(y_va, y_pred, zero_division=0)
        rec  = recall_score(y_va, y_pred, zero_division=0)
        roc  = roc_auc_score(y_va, y_prob)

        ap_scores.append(ap); precision_scores.append(prec); recall_scores.append(rec); roc_auc_scores.append(roc)

    # Means
    mean_ap, mean_precision = (np.nanmean(ap_scores), np.nanmean(precision_scores))
    mean_recall, mean_roc_auc = (np.nanmean(recall_scores), np.nanmean(roc_auc_scores))

    results = pd.DataFrame({
        "ap": [mean_ap],
        "precision": [mean_precision],
        "recall": [mean_recall],
        "roc_auc": [mean_roc_auc],
    })

    # ---- Fit on full (optionally oversampled) data
    X_full, y_full = X_train, y_train
    if sampler in {"smote", "ros"}:
        X_full, y_full = _safe_resample(X_full, y_full, method=sampler, target_ratio=0.5, random_state=RANDOM_STATE)
    stack.fit(X_full, y_full)

    # Save artefacts
    ts = pd.Timestamp.now().strftime("%Y%m%d")
    joblib.dump(stack, saved_dir / f"stacking_ensemble_model_{ts}.pkl")
    results.to_csv(saved_dir / f"stacking_ensemble_metrics_{ts}.csv", index=False)

    _plot_metric_bars(results, saved_dir, ts)
    return stack, results
