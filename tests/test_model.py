"""
tests/test_model.py
-------------------
Unit and integration tests for src/model.py.

All tests use a small in-memory synthetic dataset (no file I/O) and write
MLflow runs to a temporary directory so they leave no side effects.
"""
from __future__ import annotations

import os
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.model import (
    DEFAULT_CONFIG,
    PARAM_GRIDS,
    _build_classifier,
    _build_preprocessor,
    evaluate,
    train,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_fake_X_y_groups(
    n_trios: int = 8,
    variants_per_trio: int = 10,
    n_positive: int = 3,
    rng_seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, groups) with the same column layout as build_features()."""
    rng = np.random.default_rng(rng_seed)
    n = n_trios * variants_per_trio

    groups = pd.Series(
        np.repeat([f"T{i:02d}" for i in range(1, n_trios + 1)], variants_per_trio),
        name="trio_id",
    )

    X = pd.DataFrame({
        # numeric features
        "cadd_score":            rng.uniform(0, 60, n),
        "revel_score":           rng.uniform(0, 1,  n),
        "papi_score":            rng.uniform(0, 1,  n),
        "polyphen2_score":       rng.uniform(0, 1,  n),
        "sift_score":            rng.uniform(0, 1,  n),
        "gnomad_af":             rng.uniform(0, 0.05, n),
        "pathogenicity_score":   rng.uniform(0, 1,  n),
        "sample_coverage":       rng.uniform(30, 200, n),
        "alt_depth":             rng.uniform(5, 80,  n),
        "sample_coverage_reference": rng.uniform(5, 80, n),
        "vaf":                   rng.uniform(0.3, 0.7, n),
        "criteria_score":        rng.integers(0, 5, n).astype(float),
        "hpo_overlap_score":     rng.uniform(0, 1, n),
        "father_has_variant":    rng.integers(0, 2, n).astype(float),
        "mother_has_variant":    rng.integers(0, 2, n).astype(float),
        # ordinal
        "effect_severity":       rng.integers(-1, 14, n).astype(float),
        "zygosity_code":         rng.choice([-1, 0, 1], n).astype(float),
        # one-hot (inheritance modes)
        "inh_de_novo":           rng.integers(0, 2, n).astype(float),
        "inh_paternal":          rng.integers(0, 2, n).astype(float),
        "inh_maternal":          rng.integers(0, 2, n).astype(float),
        # one-hot (condition inheritance)
        "ci_Autosomal_recessive": rng.integers(0, 2, n).astype(float),
        "ci_Autosomal_dominant":  rng.integers(0, 2, n).astype(float),
    })

    y = pd.Series(np.zeros(n, dtype=int), name="is_pathogenic")
    y.iloc[:n_positive] = 1

    return X, y, groups


@pytest.fixture
def fake_data():
    return _make_fake_X_y_groups(n_trios=8, variants_per_trio=10)


@pytest.fixture
def tmp_models(tmp_path):
    return tmp_path / "models"


@pytest.fixture
def base_config(tmp_models):
    """Config that suppresses MLflow I/O and uses tiny search space.
    n_jobs=1 avoids WSL fork/deadlock when the parent process is multi-threaded.
    """
    return {
        "classifier":        "random_forest",
        "cv_folds":          2,
        "search_strategy":   "random",
        "n_iter":            2,
        "calibrate":         False,   # faster tests
        "models_dir":        str(tmp_models),
        "mlflow_experiment": "test_variant",
        "random_state":      42,
        "n_jobs":            1,
    }


# ---------------------------------------------------------------------------
# DEFAULT_CONFIG
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    def test_required_keys_present(self):
        required = {
            "classifier", "cv_folds", "search_strategy", "n_iter",
            "scoring", "calibrate", "models_dir", "random_state",
        }
        assert required.issubset(DEFAULT_CONFIG.keys())

    def test_default_classifier_is_valid(self):
        assert DEFAULT_CONFIG["classifier"] in PARAM_GRIDS

    def test_cv_folds_positive_int(self):
        assert isinstance(DEFAULT_CONFIG["cv_folds"], int)
        assert DEFAULT_CONFIG["cv_folds"] > 1


# ---------------------------------------------------------------------------
# PARAM_GRIDS
# ---------------------------------------------------------------------------

class TestParamGrids:
    def test_all_classifiers_have_grids(self):
        assert set(PARAM_GRIDS.keys()) >= {"random_forest", "xgboost", "lightgbm", "logistic"}

    def test_all_keys_prefixed_with_classifier(self):
        for clf_name, grid in PARAM_GRIDS.items():
            for key in grid.keys():
                assert key.startswith("classifier__"), (
                    f"{clf_name}: key {key!r} must start with 'classifier__'"
                )

    def test_all_values_are_lists(self):
        for clf_name, grid in PARAM_GRIDS.items():
            for key, vals in grid.items():
                assert isinstance(vals, list) and len(vals) >= 1, (
                    f"{clf_name} → {key}: expected non-empty list"
                )


# ---------------------------------------------------------------------------
# _build_classifier
# ---------------------------------------------------------------------------

class TestBuildClassifier:
    def test_random_forest(self):
        from sklearn.ensemble import RandomForestClassifier
        clf = _build_classifier("random_forest", {})
        assert isinstance(clf, RandomForestClassifier)

    def test_logistic(self):
        from sklearn.linear_model import LogisticRegression
        clf = _build_classifier("logistic", {})
        assert isinstance(clf, LogisticRegression)

    def test_random_state_propagated(self):
        clf = _build_classifier("random_forest", {"random_state": 99})
        assert clf.random_state == 99

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            _build_classifier("magic_tree", {})

    def test_xgboost(self):
        pytest.importorskip("xgboost")
        from xgboost import XGBClassifier
        clf = _build_classifier("xgboost", {})
        assert isinstance(clf, XGBClassifier)

    def test_lightgbm(self):
        pytest.importorskip("lightgbm")
        from lightgbm import LGBMClassifier
        clf = _build_classifier("lightgbm", {})
        assert isinstance(clf, LGBMClassifier)


# ---------------------------------------------------------------------------
# _build_preprocessor
# ---------------------------------------------------------------------------

class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        from sklearn.compose import ColumnTransformer
        X, _, _ = _make_fake_X_y_groups()
        pp = _build_preprocessor(X)
        assert isinstance(pp, ColumnTransformer)

    def test_transform_shape(self):
        X, _, _ = _make_fake_X_y_groups()
        pp = _build_preprocessor(X)
        pp.fit(X)
        out = pp.transform(X)
        assert out.shape[0] == len(X)
        assert out.shape[1] > 0

    def test_no_nans_in_output(self):
        X, _, _ = _make_fake_X_y_groups()
        # Introduce NaN
        X_nan = X.copy()
        X_nan.iloc[0, 0] = float("nan")
        pp = _build_preprocessor(X_nan)
        pp.fit(X_nan)
        out = pp.transform(X_nan)
        assert not np.isnan(out).any()

    def test_empty_X_raises(self):
        with pytest.raises(Exception):
            _build_preprocessor(pd.DataFrame())


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------

class TestTrain:
    def test_returns_pipeline(self, fake_data, base_config):
        X, y, groups = fake_data
        pipeline = train(X, y, groups, base_config)
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_can_predict(self, fake_data, base_config):
        X, y, groups = fake_data
        pipeline = train(X, y, groups, base_config)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_pipeline_can_predict_proba(self, fake_data, base_config):
        X, y, groups = fake_data
        pipeline = train(X, y, groups, base_config)
        proba = pipeline.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_model_file_saved(self, fake_data, base_config, tmp_models):
        X, y, groups = fake_data
        train(X, y, groups, base_config)
        expected_path = tmp_models / "random_forest_pipeline.joblib"
        assert expected_path.exists()

    def test_calibrated_pipeline_saved(self, fake_data, base_config, tmp_models):
        X, y, groups = fake_data
        cfg = {**base_config, "calibrate": True}
        pipeline = train(X, y, groups, cfg)
        # The classifier step should be a CalibratedClassifierCV
        from sklearn.calibration import CalibratedClassifierCV
        assert isinstance(pipeline.named_steps["classifier"], CalibratedClassifierCV)

    def test_default_config_merged(self, fake_data, tmp_models):
        """train() must work with only a few keys overridden."""
        X, y, groups = fake_data
        pipeline = train(
            X, y, groups,
            {
                "models_dir":      str(tmp_models),
                "cv_folds":        2,
                "n_iter":          1,
                "calibrate":       False,
                "n_jobs":          1,
            },
        )
        assert isinstance(pipeline, Pipeline)

    def test_logistic_classifier(self, fake_data, base_config, tmp_models):
        X, y, groups = fake_data
        cfg = {**base_config, "classifier": "logistic"}
        pipeline = train(X, y, groups, cfg)
        assert isinstance(pipeline, Pipeline)

    def test_grid_search_strategy(self, fake_data, tmp_models):
        X, y, groups = fake_data
        cfg = {
            "classifier":      "logistic",
            "cv_folds":        2,
            "search_strategy": "grid",
            "calibrate":       False,
            "models_dir":      str(tmp_models),
            "random_state":    42,
            "n_jobs":          1,
        }
        # Use a tiny param grid to keep the test fast
        tiny_grid_patch = {
            "logistic": {
                "classifier__C":       [0.1, 1.0],
                "classifier__penalty": ["l2"],
                "classifier__solver":  ["liblinear"],
            }
        }
        with patch("src.model.PARAM_GRIDS", tiny_grid_patch):
            pipeline = train(X, y, groups, cfg)
        assert isinstance(pipeline, Pipeline)

    def test_xgboost_classifier(self, fake_data, base_config, tmp_models):
        pytest.importorskip("xgboost")
        X, y, groups = fake_data
        cfg = {**base_config, "classifier": "xgboost", "n_jobs": 1}
        pipeline = train(X, y, groups, cfg)
        assert isinstance(pipeline, Pipeline)

    def test_lightgbm_classifier(self, fake_data, base_config, tmp_models):
        pytest.importorskip("lightgbm")
        X, y, groups = fake_data
        cfg = {**base_config, "classifier": "lightgbm", "n_jobs": 1}
        pipeline = train(X, y, groups, cfg)
        assert isinstance(pipeline, Pipeline)


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

class TestEvaluate:
    def _trained_pipeline(self, tmp_models):
        X, y, groups = _make_fake_X_y_groups()
        cfg = {
            "classifier":   "random_forest",
            "cv_folds":     2,
            "n_iter":       1,
            "calibrate":    False,
            "models_dir":   str(tmp_models),
            "random_state": 42,
            "n_jobs":       1,
        }
        return train(X, y, groups, cfg), X, y

    def test_returns_dict(self, tmp_models):
        pipeline, X, y = self._trained_pipeline(tmp_models)
        metrics = evaluate(pipeline, X, y, log_to_mlflow=False)
        assert isinstance(metrics, dict)

    def test_required_keys_present(self, tmp_models):
        pipeline, X, y = self._trained_pipeline(tmp_models)
        metrics = evaluate(pipeline, X, y, log_to_mlflow=False)
        required = {
            "accuracy", "f1_macro", "f1_weighted", "roc_auc",
            "average_precision", "confusion_matrix",
            "classification_report", "n_test_samples",
        }
        assert required.issubset(metrics.keys())

    def test_accuracy_in_range(self, tmp_models):
        pipeline, X, y = self._trained_pipeline(tmp_models)
        metrics = evaluate(pipeline, X, y, log_to_mlflow=False)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_f1_macro_in_range(self, tmp_models):
        pipeline, X, y = self._trained_pipeline(tmp_models)
        metrics = evaluate(pipeline, X, y, log_to_mlflow=False)
        assert 0.0 <= metrics["f1_macro"] <= 1.0

    def test_roc_auc_in_range(self, tmp_models):
        pipeline, X, y = self._trained_pipeline(tmp_models)
        metrics = evaluate(pipeline, X, y, log_to_mlflow=False)
        v = metrics["roc_auc"]
        assert math.isnan(v) or 0.0 <= v <= 1.0

    def test_confusion_matrix_is_list_of_lists(self, tmp_models):
        pipeline, X, y = self._trained_pipeline(tmp_models)
        metrics = evaluate(pipeline, X, y, log_to_mlflow=False)
        cm = metrics["confusion_matrix"]
        assert isinstance(cm, list)
        assert all(isinstance(row, list) for row in cm)

    def test_n_test_samples_correct(self, tmp_models):
        pipeline, X, y = self._trained_pipeline(tmp_models)
        metrics = evaluate(pipeline, X, y, log_to_mlflow=False)
        assert metrics["n_test_samples"] == len(y)

    def test_classification_report_is_string(self, tmp_models):
        pipeline, X, y = self._trained_pipeline(tmp_models)
        metrics = evaluate(pipeline, X, y, log_to_mlflow=False)
        assert isinstance(metrics["classification_report"], str)
        assert "precision" in metrics["classification_report"]
