"""Model training, evaluation, and artifact persistence helpers."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier


DEFAULT_RANDOM_STATE = 42


def build_models(
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, ClassifierMixin]:
    """Create the classifiers used in the training workflow."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def cross_validate_models(
    models: dict[str, ClassifierMixin],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Evaluate models with stratified cross-validation."""
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }
    results = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )
        results.append(
            {
                "Model": name,
                "Accuracy": scores["test_accuracy"].mean(),
                "Precision": scores["test_precision_macro"].mean(),
                "Recall": scores["test_recall_macro"].mean(),
                "F1-score": scores["test_f1_macro"].mean(),
            }
        )
    return pd.DataFrame(results)


def train_models(
    models: dict[str, ClassifierMixin],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, ClassifierMixin]:
    """Fit each classifier on the training data."""
    trained_models: dict[str, ClassifierMixin] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models


def evaluate_models(
    models: dict[str, ClassifierMixin],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Evaluate fitted classifiers and return metrics, matrices, and reports."""
    class_labels = sorted(y_test.unique())
    results = []
    confusion_matrices: dict[str, pd.DataFrame] = {}
    reports: dict[str, dict[str, Any]] = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred, labels=class_labels)
        confusion_matrices[name] = pd.DataFrame(
            matrix,
            index=[f"Actual {label}" for label in class_labels],
            columns=[f"Predicted {label}" for label in class_labels],
        )
        reports[name] = classification_report(
            y_test,
            y_pred,
            zero_division=0,
            output_dict=True,
        )
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "F1-score": f1_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
            }
        )
    return pd.DataFrame(results), confusion_matrices, reports


def save_training_artifacts(
    models: dict[str, ClassifierMixin],
    confusion_matrices: dict[str, pd.DataFrame],
    cross_validation_results: pd.DataFrame,
    test_results: pd.DataFrame,
    models_dir: str | Path = "models",
    results_dir: str | Path = "results",
) -> tuple[Path, Path]:
    """Persist fitted models, confusion matrices, and evaluation tables."""
    models_path = Path(models_dir)
    results_path = Path(results_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        filename = name.lower().replace(" ", "_")
        joblib.dump(model, models_path / f"{filename}.joblib")
        confusion_matrices[name].to_csv(
            results_path / f"{filename}_confusion_matrix.csv"
        )

    cross_validation_results.to_csv(
        results_path / "cross_validation_results.csv",
        index=False,
    )
    test_results.to_csv(
        results_path / "model_comparison_results.csv",
        index=False,
    )
    return models_path, results_path
