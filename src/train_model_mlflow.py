"""
Containerized training script with MLflow tracking.
Logs dataset fingerprint, all metrics, and registers model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from datetime import datetime
import os
import sys

from src.utils.dataset_fingerprint import get_dataset_metadata
from src.features.schema import FEATURE_COLUMNS, TARGET_COLUMN, extract_features, validate_features

# Add utils to path
sys.path.append("/app")
# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "credit-risk-prediction"
MODEL_NAME = "credit-risk-model"


def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Prepare data for training using canonical feature schema.
    Validates dataset schema and extracts canonical features in exact order.
    """
    df_clean = df.fillna(df.median(numeric_only=True))

    is_valid, issues = validate_features(df_clean)
    if not is_valid:
        raise ValueError(f"Training dataset schema validation failed: {issues}")

    X = extract_features(df_clean)
    y = df_clean[TARGET_COLUMN]

    return X, y


def compute_metrics(y_true, y_pred, y_pred_proba, prefix: str = "") -> dict:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        prefix: Metric name prefix (e.g., 'train_', 'test_')
    """
    return {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        f"{prefix}brier_score": float(brier_score_loss(y_true, y_pred_proba)),
    }


def train_and_log(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_metadata: dict,
    model_params: dict,
) -> str:
    """
    Train model and log everything to MLflow.

    Returns:
        run_id: MLflow run ID
    """
    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        run_id = run.info.run_id
        print(f"\n🚀 MLflow Run ID: {run_id}")

        # --- LOG DATASET FINGERPRINT (CRITICAL) ---
        mlflow.log_params(dataset_metadata)
        print(f"📊 Dataset hash: {dataset_metadata['dataset_hash'][:16]}...")

        # --- LOG MODEL PARAMETERS ---
        mlflow.log_params(model_params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # --- TRAIN MODEL ---
        print("🎯 Training model...")
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        # --- COMPUTE PREDICTIONS ---
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_test = model.predict_proba(X_test)[:, 1]

        # --- COMPUTE METRICS ---
        train_metrics = compute_metrics(y_train, y_pred_train, y_proba_train, "train_")
        test_metrics = compute_metrics(y_test, y_pred_test, y_proba_test, "test_")

        all_metrics = {**train_metrics, **test_metrics}
        mlflow.log_metrics(all_metrics)

        # --- PRINT KEY METRICS ---
        print("\n📊 Model Performance:")
        print(f"  Test ROC AUC:     {test_metrics['test_roc_auc']:.4f}")
        print(f"  Test F1:          {test_metrics['test_f1']:.4f}")
        print(f"  Test Brier Score: {test_metrics['test_brier_score']:.4f}")

        # --- LOG FEATURE IMPORTANCE ---
        feature_importance = pd.DataFrame(
            {"feature": X_train.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        # Save as artifact
        import tempfile

        importance_file = os.path.join(tempfile.gettempdir(), "feature_importance.csv")
        feature_importance.to_csv(importance_file, index=False)
        mlflow.log_artifact(importance_file, "feature_importance")

        print("\n🔍 Top 5 Features:")
        for idx, row in feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # --- LOG MODEL WITH SIGNATURE ---
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
            input_example=X_train.iloc[:1],
        )

        print(f"\n💾 Model registered: {MODEL_NAME}")
        print(
            f"🔗 View run: {MLFLOW_TRACKING_URI}/#/experiments/{mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id}/runs/{run_id}"
        )

        return run_id


def promote_to_production(model_name: str, run_id: str):
    """
    Promote the latest model version to Production stage.

    This is a manual step in Phase 2.
    In Phase 4, this will be automated with evaluation gates.
    """
    client = mlflow.tracking.MlflowClient()

    # Get model version for this run
    versions = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")

    if not versions:
        print("⚠️  Model not found in registry")
        return

    version = versions[0].version

    # Transition to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,  # Archive old production models
    )

    print(f"\n✅ Model version {version} promoted to Production")
    print("   Previous Production models archived")


def main():
    """Main training pipeline."""

    print("=" * 60)
    print("🚀 CONTAINERIZED TRAINING PIPELINE")
    print("=" * 60)

    # --- LOAD DATA ---
    data_path = os.getenv(
        "DATA_PATH",
        "/app/data/cs-training.csv"
        if os.path.exists("/app/data/cs-training.csv")
        else "data/cs-training.csv",
    )
    print(f"\n📁 Loading data from: {data_path}")

    df = pd.read_csv(data_path)
    print(f"   Dataset shape: {df.shape}")

    # --- COMPUTE DATASET FINGERPRINT ---
    dataset_metadata = get_dataset_metadata(df)
    print("\n🔍 Dataset Fingerprint:")
    print(f"   Hash: {dataset_metadata['dataset_hash'][:16]}...")
    print(f"   Rows: {dataset_metadata['dataset_rows']:,}")
    print(f"   Columns: {dataset_metadata['dataset_columns']}")

    # --- PREPARE DATA ---
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n📊 Data Split:")
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test:  {len(X_test):,} samples")

    # --- MODEL PARAMETERS ---
    model_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    }

    # --- TRAIN AND LOG ---
    run_id = train_and_log(X_train, y_train, X_test, y_test, dataset_metadata, model_params)

    # --- PROMOTE TO PRODUCTION ---
    print("\n🎯 Promoting model to Production stage...")
    promote_to_production(MODEL_NAME, run_id)

    print("\n" + "=" * 60)
    print("✨ TRAINING COMPLETE")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"MLflow UI: {MLFLOW_TRACKING_URI}")
    print(f"Model: {MODEL_NAME} (Production)")
    print("=" * 60)
    print("Training features:", list(X_train.columns))
    print("Number of features:", X_train.shape[1])


if __name__ == "__main__":
    main()
