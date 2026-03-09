"""
src/model.py
------------
Model training and evaluation for the trio-aware variant pathogenicity pipeline.

Public API
----------
    from src.model import train, evaluate, DEFAULT_CONFIG, PARAM_GRIDS

    pipeline = train(X_train, y_train, groups_train, config)
    metrics  = evaluate(pipeline, X_test, y_test)

Design notes
------------
* GroupKFold (groups = trio_id) prevents any two members of the same family
  appearing in different folds — the key anti-leakage constraint.
* Hyperparameter search uses RandomizedSearchCV by default (fast) or
  GridSearchCV when config["search_strategy"] == "grid".
* MLflow logging is best-effort: if mlflow is unavailable or no run is active
  both train() and evaluate() still work normally.
* Calibration (Platt/isotonic) is applied as a post-search step so that
  GridSearchCV evaluates the *base* classifier uniformly across all param
  combinations, then the winning parameters are refitted with calibration.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional heavy dependencies — graceful degradation
try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    warnings.warn(
        "xgboost not installed; 'xgboost' classifier is unavailable.",
        ImportWarning, stacklevel=1,
    )

try:
    from lightgbm import LGBMClassifier
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    warnings.warn(
        "lightgbm not installed; 'lightgbm' classifier is unavailable.",
        ImportWarning, stacklevel=1,
    )

try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

from src.features import feature_groups


# ---------------------------------------------------------------------------
# Defaults & hyperparameter grids
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    # Which base classifier to use
    # Options: "random_forest" | "xgboost" | "lightgbm" | "logistic"
    "classifier": "random_forest",

    # GroupKFold splits (number of folds; must be <= number of unique groups)
    "cv_folds": 5,

    # "random" (RandomizedSearchCV) or "grid" (GridSearchCV)
    "search_strategy": "random",

    # Number of parameter combinations to try (random search only)
    "n_iter": 20,

    # sklearn scoring metric for cross-validation
    "scoring": "f1_macro",

    # Wrap the winning classifier with CalibratedClassifierCV(isotonic)
    "calibrate": True,

    # MLflow experiment name (created if it doesn't exist)
    "mlflow_experiment": "variant_pathogenicity",

    # Optional MLflow run display name (None → auto-generated)
    "mlflow_run_name": None,

    # Local directory where the final pipeline .joblib is saved
    "models_dir": "models",

    # Reproducibility seed (passed to classifiers + RandomizedSearchCV)
    "random_state": 42,

    # Parallelism for search and tree classifiers (-1 = all cores).
    # Set to 1 in environments where multiprocessing fork causes deadlocks
    # (e.g. WSL with multi-threaded parent processes).
    "n_jobs": -1,
}

# Param-grid keys must use the step name "classifier" as prefix so they work
# with the sklearn Pipeline.  Values are lists usable by both GridSearchCV and
# RandomizedSearchCV.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    # -----------------------------------------------------------------------
    # Random Forest
    # -----------------------------------------------------------------------
    "random_forest": {
        "classifier__n_estimators":    [100, 300, 500],
        "classifier__max_depth":       [None, 10, 20],
        "classifier__min_samples_leaf":[1, 5, 10],
        "classifier__max_features":    ["sqrt", "log2"],
        "classifier__class_weight":    ["balanced", "balanced_subsample"],
    },

    # -----------------------------------------------------------------------
    # XGBoost
    # -----------------------------------------------------------------------
    "xgboost": {
        "classifier__n_estimators":     [100, 300, 500],
        "classifier__max_depth":        [3, 6, 9],
        "classifier__learning_rate":    [0.01, 0.05, 0.1, 0.3],
        "classifier__subsample":        [0.7, 0.85, 1.0],
        "classifier__colsample_bytree": [0.7, 0.85, 1.0],
        "classifier__min_child_weight": [1, 5, 10],
        # Handles class imbalance: ratio of neg/pos samples
        "classifier__scale_pos_weight": [1, 5, 10],
    },

    # -----------------------------------------------------------------------
    # LightGBM
    # -----------------------------------------------------------------------
    "lightgbm": {
        "classifier__n_estimators":      [100, 300, 500],
        "classifier__max_depth":         [-1, 6, 10],
        "classifier__learning_rate":     [0.01, 0.05, 0.1],
        "classifier__num_leaves":        [15, 31, 63, 127],
        "classifier__min_child_samples": [5, 10, 20, 50],
        "classifier__reg_alpha":         [0.0, 0.1, 1.0],
        "classifier__reg_lambda":        [0.0, 0.1, 1.0],
        "classifier__class_weight":      ["balanced"],
    },

    # -----------------------------------------------------------------------
    # Logistic Regression (fast baseline)
    # -----------------------------------------------------------------------
    "logistic": {
        "classifier__C":       [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "classifier__penalty": ["l1", "l2"],
        "classifier__solver":  ["liblinear"],
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_classifier(name: str, config: dict[str, Any]):
    """Instantiate a base (uncalibrated) classifier from config."""
    rs = config.get("random_state", 42)
    n_jobs = config.get("n_jobs", -1)
    if name == "random_forest":
        return RandomForestClassifier(random_state=rs, n_jobs=n_jobs)
    if name == "xgboost":
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is not installed.")
        return XGBClassifier(
            random_state=rs,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        )
    if name == "lightgbm":
        if not _LGB_AVAILABLE:
            raise ImportError("lightgbm is not installed.")
        return LGBMClassifier(random_state=rs, verbosity=-1)
    if name == "logistic":
        return LogisticRegression(random_state=rs, max_iter=1000)
    raise ValueError(
        f"Unknown classifier {name!r}. "
        "Choose from: 'random_forest', 'xgboost', 'lightgbm', 'logistic'."
    )


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer that scales numeric features and passes
    already-encoded one-hot columns through unchanged."""
    groups = feature_groups(X)
    numeric_cols = [c for c in groups.get("numeric", []) if c in X.columns]
    onehot_cols  = [c for c in groups.get("onehot",  []) if c in X.columns]

    transformers: list[tuple] = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]),
            numeric_cols,
        ))
    if onehot_cols:
        # Already binary 0/1; just impute any stray NaN with 0
        transformers.append((
            "ohe",
            SimpleImputer(strategy="constant", fill_value=0),
            onehot_cols,
        ))

    if not transformers:
        raise ValueError(
            "feature_groups(X) returned no columns. "
            "Ensure X was produced by build_features()."
        )

    return ColumnTransformer(transformers, remainder="drop")


def _log_mlflow_safe(**kwargs) -> None:
    """Call mlflow.log_* helpers only when mlflow is available and a run is
    active (or mlflow auto-creates one).  Failures are silently ignored so
    the rest of the pipeline is unaffected."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        params  = kwargs.get("params",  {})
        metrics = kwargs.get("metrics", {})
        tags    = kwargs.get("tags",    {})
        artifact = kwargs.get("artifact_path")

        if params:
            mlflow.log_params(params)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        if tags:
            mlflow.set_tags(tags)
        if artifact and Path(artifact).exists():
            mlflow.log_artifact(artifact)
    except Exception as exc:
        warnings.warn(f"MLflow logging failed (non-fatal): {exc}", RuntimeWarning)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(
    X: pd.DataFrame,
    y: pd.Series,
    groups: "pd.Series | np.ndarray",
    config: dict[str, Any] | None = None,
) -> Pipeline:
    """Train a variant pathogenicity classifier using group-safe cross-validation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix from ``build_features()``.
    y : pd.Series
        Binary target (1 = pathogenic / likely pathogenic).
    groups : array-like of shape (n_samples,)
        Trio IDs used for GroupKFold splitting.  All three family members from
        the same trio share the same group label to prevent data leakage.
    config : dict, optional
        Override any key in ``DEFAULT_CONFIG``.

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline (preprocessor → classifier).
        If ``config["calibrate"]`` is True the classifier step is wrapped in
        ``CalibratedClassifierCV`` and refitted on the full training set.

    Side effects
    ------------
    * Saves the pipeline to ``{config["models_dir"]}/{classifier}_pipeline.joblib``.
    * Logs parameters and CV metrics to MLflow (best-effort).
    """
    cfg: dict[str, Any] = {**DEFAULT_CONFIG, **(config or {})}
    clf_name: str = cfg["classifier"]

    # --- 1. Preprocessor & base pipeline ---------------------------------
    preprocessor = _build_preprocessor(X)
    base_clf      = _build_classifier(clf_name, cfg)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   base_clf),
    ])

    # --- 2. Cross-validation splitter ------------------------------------
    cv = GroupKFold(n_splits=cfg["cv_folds"])

    # --- 4. Hyperparameter search ----------------------------------------
    param_grid = PARAM_GRIDS.get(clf_name, {})
    n_jobs = cfg.get("n_jobs", DEFAULT_CONFIG["n_jobs"])

    if cfg["search_strategy"] == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=cfg["scoring"],
            refit=True,
            n_jobs=n_jobs,
            verbose=0,
        )
    else:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            cv=cv,
            scoring=cfg["scoring"],
            refit=True,
            n_jobs=n_jobs,
            verbose=0,
            n_iter=cfg.get("n_iter", DEFAULT_CONFIG["n_iter"]),
            random_state=cfg.get("random_state", DEFAULT_CONFIG["random_state"]),
        )

    # --- 4. Fit -----------------------------------------------------------
    if _MLFLOW_AVAILABLE:
        mlflow.set_experiment(cfg.get("mlflow_experiment", "variant_pathogenicity"))

    search.fit(X, y, groups=groups)

    best_pipeline: Pipeline = search.best_estimator_
    best_params:   dict     = search.best_params_
    best_cv_score: float    = float(search.best_score_)

    # --- 5. Optional calibration (refit on full training data) -----------
    if cfg.get("calibrate", True):
        best_clf_step = best_pipeline.named_steps["classifier"]
        calibrated    = CalibratedClassifierCV(
            best_clf_step, cv=3, method="isotonic"
        )
        final_pipeline = Pipeline([
            ("preprocessor", best_pipeline.named_steps["preprocessor"]),
            ("classifier",   calibrated),
        ])
        final_pipeline.fit(X, y)
    else:
        final_pipeline = best_pipeline

    # --- 6. Persist -------------------------------------------------------
    models_dir = Path(cfg.get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{clf_name}_pipeline.joblib"
    joblib.dump(final_pipeline, model_path)

    # --- 7. MLflow logging ------------------------------------------------
    # Create a fresh run each time so repeated calls don't collide.
    if _MLFLOW_AVAILABLE:
        run_name = cfg.get("mlflow_run_name") or f"train_{clf_name}"
        with mlflow.start_run(run_name=run_name, nested=True):
            _log_mlflow_safe(
                params={
                    "classifier":      clf_name,
                    "cv_folds":        cfg["cv_folds"],
                    "search_strategy": cfg["search_strategy"],
                    "calibrate":       cfg.get("calibrate", True),
                    "scoring":         cfg["scoring"],
                    **best_params,
                },
                metrics={
                    f"cv_{cfg['scoring']}": best_cv_score,
                },
                tags={"stage": "train"},
                artifact_path=str(model_path),
            )

    return final_pipeline


def evaluate(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Evaluate a fitted pipeline on a held-out test set.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted sklearn Pipeline (from ``train()`` or loaded from disk).
    X_test : pd.DataFrame
        Feature matrix with the same column layout as the training set.
    y_test : pd.Series
        True binary labels.
    log_to_mlflow : bool
        Whether to log metrics to MLflow (best-effort).

    Returns
    -------
    dict with keys:
        accuracy, f1_macro, f1_weighted, roc_auc, average_precision,
        confusion_matrix (list[list[int]]), classification_report (str),
        n_test_samples (int)
    """
    y_pred  = pipeline.predict(X_test)
    y_proba = None
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    except AttributeError:
        pass  # calibrated pipeline always has predict_proba; handle edge case

    metrics: dict[str, Any] = {
        "accuracy":              float(accuracy_score(y_test, y_pred)),
        "f1_macro":              float(f1_score(y_test, y_pred, average="macro",    zero_division=0)),
        "f1_weighted":           float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "roc_auc":               float(roc_auc_score(y_test, y_proba))
                                    if y_proba is not None else float("nan"),
        "average_precision":     float(average_precision_score(y_test, y_proba))
                                    if y_proba is not None else float("nan"),
        "confusion_matrix":      confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "n_test_samples":        int(len(y_test)),
    }

    if log_to_mlflow:
        scalar_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, float) and not isinstance(v, str)
        }
        if _MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name="evaluate", nested=True):
                _log_mlflow_safe(
                    metrics=scalar_metrics,
                    tags={"stage": "evaluate"},
                )

    return metrics
