"""
model_trainer.py
----------------
Trains a baseline DecisionTreeClassifier and an improved RandomForestClassifier.
Evaluates both models and returns the best one along with metrics.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def _evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Compute classification metrics for a fitted model.

    Returns
    -------
    dict with accuracy, precision, recall, f1
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
    }


def _print_metrics(name: str, metrics: Dict[str, float]) -> None:
    print(f"\n  [{name}]")
    print(f"    Accuracy  : {metrics['accuracy']:.4f}")
    print(f"    Precision : {metrics['precision']:.4f}")
    print(f"    Recall    : {metrics['recall']:.4f}")
    print(f"    F1 Score  : {metrics['f1_score']:.4f}")


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train baseline (DecisionTree) and improved (RandomForest) models.
    Select the best model based on F1 score.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target vector.
    test_size : float
        Proportion of data for the test split.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    Tuple[model, results_dict]
        Best fitted model and a dict containing metrics for both models.
    """
    print("[Model Training] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ------------------------------------------------------------------ #
    # Baseline: Decision Tree
    # ------------------------------------------------------------------ #
    print("\n[Model Training] Training Baseline — DecisionTreeClassifier...")
    baseline = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    baseline.fit(X_train, y_train)
    baseline_metrics = _evaluate_model(baseline, X_test, y_test)
    _print_metrics("DecisionTreeClassifier (Baseline)", baseline_metrics)

    # ------------------------------------------------------------------ #
    # Improved: Random Forest
    # ------------------------------------------------------------------ #
    print("\n[Model Training] Training Improved — RandomForestClassifier...")
    improved = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    improved.fit(X_train, y_train)
    improved_metrics = _evaluate_model(improved, X_test, y_test)
    _print_metrics("RandomForestClassifier (Improved)", improved_metrics)

    # ------------------------------------------------------------------ #
    # Model selection
    # ------------------------------------------------------------------ #
    if improved_metrics["f1_score"] >= baseline_metrics["f1_score"]:
        best_model = improved
        best_name = "RandomForestClassifier"
        best_metrics = improved_metrics
    else:
        best_model = baseline
        best_name = "DecisionTreeClassifier"
        best_metrics = baseline_metrics

    print(f"\n[Model Selection] Best model: {best_name} (F1={best_metrics['f1_score']:.4f})")

    results = {
        "baseline": {
            "name": "DecisionTreeClassifier",
            "model": baseline,
            "metrics": baseline_metrics,
        },
        "improved": {
            "name": "RandomForestClassifier",
            "model": improved,
            "metrics": improved_metrics,
        },
        "best": {
            "name": best_name,
            "model": best_model,
            "metrics": best_metrics,
        },
    }

    return best_model, results
