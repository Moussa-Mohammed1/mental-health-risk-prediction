"""Airflow DAG for the student mental-health risk project."""

import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset, load_processed_splits
from src.model_training import (
    build_models,
    cross_validate_models,
    evaluate_models,
    save_training_artifacts,
    train_models,
)
from src.preprocessing import preprocess_dataset, save_preprocessing_result


def prepare_data():
    """Check that the dataset can be loaded."""
    load_dataset()


def preprocess_data():
    """Prepare and save the train and test data."""
    data = load_dataset()
    result = preprocess_dataset(data)
    save_preprocessing_result(result)


def train_and_evaluate():
    """Train the models, evaluate them, and save the results."""
    X_train, X_test, y_train, y_test = load_processed_splits()
    models = build_models()
    cross_validation_results = cross_validate_models(models, X_train, y_train)
    trained_models = train_models(models, X_train, y_train)
    test_results, confusion_matrices, _ = evaluate_models(
        trained_models,
        X_test,
        y_test,
    )
    save_training_artifacts(
        trained_models,
        confusion_matrices,
        cross_validation_results,
        test_results,
    )


with DAG(
    dag_id="mental_health_risk_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
) as dag:
    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
    )
    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )
    train = PythonOperator(
        task_id="train_and_evaluate",
        python_callable=train_and_evaluate,
    )

    prepare >> preprocess >> train
