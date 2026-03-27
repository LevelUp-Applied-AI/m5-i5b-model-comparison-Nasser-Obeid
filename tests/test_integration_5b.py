"""Autograder tests for Integration 5B — Model Comparison & Decision Memo."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "starter"))

from model_comparison import (load_and_prepare, build_preprocessor,
                               define_models, evaluate_all)


def test_data_loaded():
    result = load_and_prepare(
        os.path.join(os.path.dirname(__file__), "..", "starter", "data", "telecom_churn.csv")
    )
    assert result is not None, "load_and_prepare returned None"
    X, y = result
    assert X.shape[0] > 1000
    assert "churned" not in X.columns


def test_preprocessor():
    prep = build_preprocessor()
    assert prep is not None
    assert hasattr(prep, "fit_transform")


def test_models_defined():
    models = define_models()
    assert models is not None
    assert len(models) >= 6, f"Expected 6 models, got {len(models)}"
    for name, pipe in models.items():
        assert hasattr(pipe, "fit"), f"'{name}' must have fit method"


def test_evaluation_runs():
    result = load_and_prepare(
        os.path.join(os.path.dirname(__file__), "..", "starter", "data", "telecom_churn.csv")
    )
    assert result is not None
    X, y = result
    models = define_models()
    assert models is not None
    results = evaluate_all(models, X, y)
    assert results is not None, "evaluate_all returned None"
    assert len(results) >= 6, f"Expected 6 rows, got {len(results)}"
    for col in ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean"]:
        assert col in results.columns, f"Missing column: {col}"
