"""
Module 5 Week B — Integration: Model Comparison & Decision Memo

Module 5 culminating deliverable. Compare 6 model configurations,
produce PR curves, calibration plots, an experiment log, and a
decision memo.
"""

import pandas as pd
import numpy as np
import os
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
    df = pd.read_csv(filepath)
    df = df.drop(columns=["customer_id"])
    X = df.drop(columns=["churned"])
    y = df["churned"]
    return X, y


def build_preprocessor():
    """Build a ColumnTransformer for mixed feature types.

    Returns:
        ColumnTransformer.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def define_models():
    """Define 6 model configurations as Pipelines.

    Returns:
        Dictionary of {name: Pipeline}.
    """
    return {
        "LogReg_default": Pipeline([("pre", build_preprocessor()), ("clf", LogisticRegression(l1_ratio=0, random_state=42, max_iter=1000))]),
        "LogReg_L1": Pipeline([("pre", build_preprocessor()), ("clf", LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", random_state=42, max_iter=1000))]),
        "DecisionTree": Pipeline([("pre", build_preprocessor()), ("clf", DecisionTreeClassifier(max_depth=5, random_state=42))]),
        "RandomForest_default": Pipeline([("pre", build_preprocessor()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
        "RandomForest_balanced": Pipeline([("pre", build_preprocessor()), ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))]),
        "Dummy_baseline": Pipeline([("pre", build_preprocessor()), ("clf", DummyClassifier(strategy="most_frequent", random_state=42))]),
    }


def evaluate_all(models, X, y, cv=5, random_state=42):
    """Cross-validate all models and return results DataFrame.

    Returns:
        DataFrame with: model, accuracy_mean, accuracy_std,
        precision_mean, recall_mean, f1_mean, pr_auc_mean.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {"accuracy": "accuracy", "precision": "precision",
               "recall": "recall", "f1": "f1", "pr_auc": "average_precision"}
    rows = []
    for name, pipe in models.items():
        cv_res = cross_validate(pipe, X, y, cv=skf, scoring=scoring)
        rows.append({
            "model": name,
            "accuracy_mean": cv_res["test_accuracy"].mean(),
            "accuracy_std": cv_res["test_accuracy"].std(),
            "precision_mean": cv_res["test_precision"].mean(),
            "precision_std": cv_res["test_precision"].std(),
            "recall_mean": cv_res["test_recall"].mean(),
            "recall_std": cv_res["test_recall"].std(),
            "f1_mean": cv_res["test_f1"].mean(),
            "f1_std": cv_res["test_f1"].std(),
            "pr_auc_mean": cv_res["test_pr_auc"].mean(),
            "pr_auc_std": cv_res["test_pr_auc"].std(),
        })
    return pd.DataFrame(rows)


def save_results(results_df, output_dir="results"):
    """Save comparison table to CSV.

    Args:
        results_df: Results DataFrame.
        output_dir: Directory for output files.
    """
    results_df.to_csv(f"{output_dir}/comparison_table.csv", index=False)


def plot_pr_curves(models, X, y, top_n=3, output_dir="results"):
    """Plot PR curves for the top N models and save.

    Args:
        models: Dict of {name: Pipeline}.
        X, y: Full dataset (uses train/test split internally).
        top_n: Number of top models to plot.
        output_dir: Directory for output files.
    """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # Get top models by pr_auc from evaluate_all (re-evaluate quickly via score)
    scores = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        if hasattr(pipe, "predict_proba"):
            scores[name] = average_precision_score(y_test, pipe.predict_proba(X_test)[:, 1])
    top_names = sorted(scores, key=scores.get, reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in top_names:
        PrecisionRecallDisplay.from_estimator(models[name], X_test, y_test, ax=ax, name=f"{name} (AP={scores[name]:.3f})")
    ax.set_title("Precision-Recall Curves (Top 3)")
    fig.savefig(f"{output_dir}/pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(models, X, y, top_n=3, output_dir="results"):
    """Plot calibration diagram for top N models and save.

    Args:
        models: Dict of {name: Pipeline}.
        X, y: Full dataset.
        top_n: Number of top models to plot.
        output_dir: Directory for output files.
    """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scores = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        if hasattr(pipe, "predict_proba"):
            scores[name] = average_precision_score(y_test, pipe.predict_proba(X_test)[:, 1])
    top_names = sorted(scores, key=scores.get, reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in top_names:
        CalibrationDisplay.from_estimator(models[name], X_test, y_test, n_bins=10, ax=ax, name=name)
    ax.set_title("Calibration Curves (Top 3)")
    fig.savefig(f"{output_dir}/calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_best_model(models, results_df, X, y, output_dir="results"):
    """Save the best model using joblib.

    Args:
        models: Dict of {name: Pipeline}.
        results_df: Results with model rankings.
        X, y: Full dataset for final training.
        output_dir: Directory for output files.
    """
    best_name = results_df.loc[results_df["pr_auc_mean"].idxmax(), "model"]
    best_pipe = models[best_name]
    best_pipe.fit(X, y)
    dump(best_pipe, f"{output_dir}/best_model.joblib")
    print(f"Best model: {best_name}")


def log_experiment(results_df, output_dir="results"):
    """Save experiment log to CSV with timestamps.

    Args:
        results_df: Results DataFrame.
        output_dir: Directory for output files.
    """
    log = results_df[["model"]].copy()
    log = log.rename(columns={"model": "model_name"})
    log["hyperparams"] = ""
    for col in ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "pr_auc_mean"]:
        log[col.replace("_mean", "")] = results_df[col]
    log["timestamp"] = datetime.now().isoformat()
    path = f"{output_dir}/experiment_log.csv"
    log.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


if __name__ == "__main__":
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