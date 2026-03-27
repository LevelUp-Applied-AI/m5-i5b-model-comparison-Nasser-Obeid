"""
Module 5 Week B — Integration: Model Comparison & Decision Memo

Module 5 culminating deliverable. Compare 6 model configurations,
produce PR curves, calibration plots, an experiment log, and a
decision memo.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (average_precision_score, PrecisionRecallDisplay,
                             make_scorer)
from sklearn.calibration import CalibrationDisplay
from joblib import dump
import matplotlib.pyplot as plt


NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents"]

CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service",
                        "payment_method"]


def load_and_prepare(filepath="data/telecom_churn.csv"):
    """Load data and separate features from target.

    Returns:
        Tuple of (X, y).
    """
    # TODO: Load CSV, drop customer_id, separate X and y
    pass


def build_preprocessor():
    """Build a ColumnTransformer for mixed feature types.

    Returns:
        ColumnTransformer.
    """
    # TODO: StandardScaler for numeric, OneHotEncoder for categorical
    pass


def define_models():
    """Define 6 model configurations as Pipelines.

    Returns:
        Dictionary of {name: Pipeline}.
    """
    # TODO: Create 6 Pipelines:
    #   1. "LogReg_default"
    #   2. "LogReg_L1"
    #   3. "DecisionTree"
    #   4. "RandomForest_default"
    #   5. "RandomForest_balanced"
    #   6. "Dummy_baseline"
    pass


def evaluate_all(models, X, y, cv=5, random_state=42):
    """Cross-validate all models and return results DataFrame.

    Returns:
        DataFrame with: model, accuracy_mean, accuracy_std,
        precision_mean, recall_mean, f1_mean, pr_auc_mean.
    """
    # TODO: Loop over models, run cross_validate with multiple scoring metrics
    pass


def save_results(results_df, output_dir="results"):
    """Save comparison table to CSV.

    Args:
        results_df: Results DataFrame.
        output_dir: Directory for output files.
    """
    # TODO: Save results_df to comparison_table.csv
    pass


def plot_pr_curves(models, X, y, top_n=3, output_dir="results"):
    """Plot PR curves for the top N models and save.

    Args:
        models: Dict of {name: Pipeline}.
        X, y: Full dataset (uses train/test split internally).
        top_n: Number of top models to plot.
        output_dir: Directory for output files.
    """
    # TODO: Train models, plot PR curves, save to pr_curves.png
    pass


def plot_calibration(models, X, y, top_n=3, output_dir="results"):
    """Plot calibration diagram for top N models and save.

    Args:
        models: Dict of {name: Pipeline}.
        X, y: Full dataset.
        top_n: Number of top models to plot.
        output_dir: Directory for output files.
    """
    # TODO: Train models, plot calibration curves, save to calibration.png
    pass


def save_best_model(models, results_df, X, y, output_dir="results"):
    """Save the best model using joblib.

    Args:
        models: Dict of {name: Pipeline}.
        results_df: Results with model rankings.
        X, y: Full dataset for final training.
        output_dir: Directory for output files.
    """
    # TODO: Identify best model from results, train on full data, save with joblib
    pass


def log_experiment(results_df, output_dir="results"):
    """Save experiment log to CSV with timestamps.

    Args:
        results_df: Results DataFrame.
        output_dir: Directory for output files.
    """
    # TODO: Add timestamp column and save to experiment_log.csv
    pass


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    data = load_and_prepare()
    if data:
        X, y = data
        print(f"Data: {X.shape[0]} rows, churn rate: {y.mean():.2%}")

        models = define_models()
        if models:
            results = evaluate_all(models, X, y)
            if results is not None:
                print("\n=== Model Comparison Table ===")
                print(results.to_string(index=False))

                save_results(results)
                plot_pr_curves(models, X, y)
                plot_calibration(models, X, y)
                save_best_model(models, results, X, y)
                log_experiment(results)

                print("\nResults saved to results/")
                print("Write your decision memo in the PR description.")
