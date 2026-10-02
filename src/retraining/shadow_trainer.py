"""
Shadow model trainer with TEMPORAL WINDOWS.

CRITICAL CHANGE: No random splits.
Use time-based train/eval windows.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from datetime import datetime
from typing import Dict, Tuple, Optional, Any, List
import logging
import sys
from src.utils.temporal_utils import TemporalWindows
from src.utils.dataset_fingerprint import get_dataset_metadata

sys.path.append("/app")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShadowModelTrainer:
    """
    Train shadow models using temporal windows.

    NO random splits. Time-based splits only.
    """

    from src.features.schema import FEATURE_COLUMNS

    def __init__(
        self, model_name: str = "credit-risk-model", mlflow_tracking_uri: str = "http://mlflow:5000"
    ):
        self.model_name = model_name
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    def prepare_training_data_temporal(
        self,
        predictions_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        train_end_date: str,
        eval_start_date: str,
        eval_end_date: str,
        min_eval_samples: int = 30,
    ) -> Tuple[
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Optional[pd.Series],
        Optional[pd.Series],
        Dict[str, Any],
    ]:
        """
        Prepare temporal train/eval split with deduplication and validation.

        ENFORCED: No temporal overlap, no duplicate prediction_ids.

        Args:
            predictions_df: Predictions with features
            labels_df: Ground truth labels
            train_end_date: End of training window
            eval_start_date: Start of eval window
            eval_end_date: End of eval window
            min_eval_samples: Minimum samples required in eval set

        Returns:
            (X_train, X_eval, y_train, y_eval, validation_status)
            validation_status: dict with 'valid', 'message', and optional 'issues'
        """
        logger.info("=" * 80)
        logger.info("TEMPORAL DATA PREPARATION")
        logger.info("=" * 80)

        validation_status: Dict[str, Any] = {"valid": True, "message": "OK", "issues": []}

        # Join predictions with labels
        merged = predictions_df.merge(
            labels_df[["prediction_id", "true_label"]], on="prediction_id", how="inner"
        )

        if len(merged) == 0:
            validation_status["valid"] = False
            validation_status["message"] = "No labeled data available"
            validation_status["issues"].append("no_labeled_data")
            return None, None, None, None, validation_status

        logger.info(f"Total labeled samples after merge: {len(merged)}")

        # Check for duplicate prediction_ids (data quality issue)
        duplicates_count = merged["prediction_id"].duplicated().sum()
        if duplicates_count > 0:
            logger.warning(
                f"⚠️  Found {duplicates_count} duplicate prediction_ids. Dropping duplicates (keeping last)."
            )
            merged = merged.drop_duplicates(subset=["prediction_id"], keep="last")
            validation_status["issues"].append(f"dropped_{duplicates_count}_duplicates")

        logger.info(f"Total labeled samples after deduplication: {len(merged)}")

        # Temporal split
        temporal = TemporalWindows()

        train_df, eval_df = temporal.create_temporal_split(
            merged,
            train_end_date=train_end_date,
            eval_start_date=eval_start_date,
            eval_end_date=eval_end_date,
            date_column="application_date",
        )

        # Extract features and labels
        X_train = train_df[self.FEATURE_COLUMNS]
        y_train = train_df["true_label"]

        X_eval = eval_df[self.FEATURE_COLUMNS]
        y_eval = eval_df["true_label"]

        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Eval samples: {len(X_eval)}")

        # Validate eval set
        if len(X_eval) < min_eval_samples:
            validation_status["valid"] = False
            validation_status["message"] = f"Eval set too small: {len(X_eval)} < {min_eval_samples}"
            validation_status["issues"].append("eval_too_small")
            logger.warning(f"🚨 {validation_status['message']}")

        num_classes = y_eval.nunique()
        if num_classes < 2:
            validation_status["valid"] = False
            validation_status[
                "message"
            ] = f"Eval set is single-class: {num_classes} class(es) found"
            validation_status["issues"].append("eval_single_class")
            logger.warning(f"🚨 {validation_status['message']}")

        if validation_status["valid"]:
            logger.info(f"✅ Eval set validation passed (n={len(X_eval)}, classes={num_classes})")

        logger.info("=" * 80)

        return X_train, X_eval, y_train, y_eval, validation_status

    def train_shadow_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        model_params: Dict = None,
        trigger_reason: str = "manual",
        validation_status: Dict = None,
    ) -> Tuple[object, str, Dict]:
        """
        Train shadow model and log to MLflow.

        Args:
            X_train, y_train: Training data
            X_eval, y_eval: Evaluation data
            model_params: Model hyperparameters
            trigger_reason: Why retraining triggered
            validation_status: Data validation result (from prepare_training_data_temporal)

        Returns:
            (model, run_id, train_result_status)
        """
        if model_params is None:
            model_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1,
            }

        if validation_status is None:
            validation_status = {"valid": True, "message": "OK", "issues": []}

        with mlflow.start_run(run_name=f"shadow_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id

            logger.info("=" * 80)
            logger.info(f"TRAINING SHADOW MODEL: {run_id}")
            logger.info("=" * 80)

            # Log validation status first
            mlflow.log_param(
                "data_validation_status", "valid" if validation_status["valid"] else "invalid"
            )
            mlflow.log_param("validation_message", validation_status.get("message", ""))
            if validation_status.get("issues"):
                mlflow.log_param("validation_issues", ",".join(validation_status["issues"]))

            # ABORT if validation failed
            if not validation_status["valid"]:
                logger.error(f"🚨 Data validation failed: {validation_status['message']}")
                mlflow.log_param("model_type", "shadow")
                mlflow.log_param("trigger_reason", trigger_reason)
                mlflow.log_param("training_status", "aborted_validation_failure")
                logger.info("=" * 80)
                return (
                    None,
                    run_id,
                    {"status": "aborted", "reason": validation_status["message"], "run_id": run_id},
                )

            # Log metadata
            mlflow.log_param("model_type", "shadow")
            mlflow.log_param("trigger_reason", trigger_reason)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("eval_size", len(X_eval))
            mlflow.log_params(model_params)

            # Dataset fingerprint
            dataset_metadata = get_dataset_metadata(pd.concat([X_train, X_eval], axis=0))
            mlflow.log_params(dataset_metadata)

            # Train
            logger.info("Training model...")
            model = RandomForestClassifier(**model_params)
            model.fit(X_train, y_train)

            # Evaluate on eval set
            from src.analytics.model_evaluator import ModelEvaluator

            evaluator = ModelEvaluator()

            y_pred = model.predict(X_eval)
            y_prob = model.predict_proba(X_eval)[:, 1]

            metrics = evaluator.evaluate_predictions(y_eval, pd.Series(y_pred), pd.Series(y_prob))

            # Log metrics (DIAGNOSTIC ONLY - NOT used for promotion decisions)
            # Promotion decisions use ONLY replay-based evaluation metrics

            if "primary_metrics" in metrics:
                for key, value in metrics["primary_metrics"].items():
                    mlflow.log_metric(f"diagnostic_{key}", value)  # ✅ Renamed
                    logger.info(f"  [DIAGNOSTIC] {key}: {value:.4f}")

            if "calibration_metrics" in metrics:
                for key, value in metrics["calibration_metrics"].items():
                    if value is not None:
                        mlflow.log_metric(f"diagnostic_{key}", value)  # ✅ Renamed

            # Log confusion matrix (CRITICAL: validates no perfect metrics from duplication)
            if "confusion_matrix" in metrics:
                cm = metrics["confusion_matrix"]
                mlflow.log_metric("confusion_matrix_tn", cm.get("true_negatives", 0))
                mlflow.log_metric("confusion_matrix_fp", cm.get("false_positives", 0))
                mlflow.log_metric("confusion_matrix_fn", cm.get("false_negatives", 0))
                mlflow.log_metric("confusion_matrix_tp", cm.get("true_positives", 0))

                tn = cm.get("true_negatives", 0)
                fp = cm.get("false_positives", 0)
                fn = cm.get("false_negatives", 0)
                tp = cm.get("true_positives", 0)

                logger.info("Confusion Matrix:")
                logger.info(f"  TN={tn}, FP={fp}")
                logger.info(f"  FN={fn}, TP={tp}")

                # Warn if perfect predictions (suspicious)
                if fp == 0 and fn == 0:
                    logger.warning(
                        "⚠️  Perfect predictions (no FP/FN). Verify no data leakage or duplication."
                    )

            # Log model
            signature = infer_signature(X_train, model.predict(X_train))

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                registered_model_name=self.model_name,
            )

            # Transition to Staging
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(
                f"name='{self.model_name}' and run_id='{run_id}'"
            )

            if versions:
                version = versions[0].version
                client.transition_model_version_stage(
                    name=self.model_name, version=version, stage="Staging"
                )
                logger.info(f"✅ Model v{version} → Staging")

            logger.info("=" * 80)

            return model, run_id, {"status": "success", "run_id": run_id}
