"""
Model evaluator with REPLAY-BASED evaluation.

CRITICAL CHANGE: Compare models on IDENTICAL samples.

Old way:
- Train shadow on new split
- Compare shadow test metrics to production test metrics
- Problem: Different data! Not fair comparison.

New way:
- Get historical predictions with features
- Re-score with production model
- Re-score with shadow model
- Compare on SAME samples
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple
import logging
import mlflow

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate models with ground truth labels.

    Supports:
    - Standard evaluation (y_true vs y_pred)
    - Replay-based evaluation (re-score with both models)
    """

    def __init__(
        self, segment_features: List[str] = None, mlflow_tracking_uri: str = "http://mlflow:5000"
    ):
        """
        Initialize evaluator.

        Args:
            segment_features: Features for segment analysis
            mlflow_tracking_uri: MLflow server
        """
        self.segment_features = segment_features or []
        mlflow.set_tracking_uri(mlflow_tracking_uri)  # type: ignore
        self.client = mlflow.tracking.MlflowClient()

    def evaluate_predictions(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_prob: pd.Series,
        segment_data: pd.DataFrame = None,
    ) -> Dict:
        """
        Standard evaluation (already predicted).

        Args:
            y_true: Ground truth
            y_pred: Predicted classes
            y_prob: Predicted probabilities
            segment_data: Optional segment features

        Returns:
            Metrics dictionary
        """
        if len(y_true) == 0:
            return {"status": "no_labels"}

        results = {
            "num_samples": len(y_true),
            "primary_metrics": self._compute_primary_metrics(y_true, y_pred, y_prob),
            "calibration_metrics": self._compute_calibration_metrics(y_true, y_prob),
            "confusion_matrix": self._compute_confusion_matrix(y_true, y_pred),
        }

        if segment_data is not None and len(self.segment_features) > 0:
            results["segment_performance"] = self._compute_segment_metrics(
                y_true, y_pred, y_prob, segment_data
            )

        return results

    def replay_evaluation(
        self,
        eval_df: pd.DataFrame,
        production_model_version: str,
        shadow_model_version: str,
        feature_columns: List[str],
    ) -> Tuple[Dict, Dict]:
        """
        REPLAY-BASED EVALUATION (identical samples).

        Steps:
        1. Load production and shadow models from MLflow
        2. Re-score eval_df with BOTH models
        3. Compare metrics on SAME data

        Args:
            eval_df: DataFrame with features and true_label
            production_model_version: Production model version
            shadow_model_version: Shadow model version
            feature_columns: Feature column names

        Returns:
            (production_metrics, shadow_metrics)
        """
        logger.info("=" * 80)
        logger.info("REPLAY-BASED EVALUATION (PROMOTION AUTHORITY)")  # ✅ Clarified
        logger.info("⚠️  These metrics determine promotion decisions")
        logger.info("=" * 80)

        # Validate input
        if "true_label" not in eval_df.columns:
            raise ValueError("eval_df must have 'true_label' column")

        missing_features = set(feature_columns) - set(eval_df.columns)
        if missing_features:
            raise ValueError(f"Missing features in eval_df: {missing_features}")

        X_eval = eval_df[feature_columns]
        y_true = eval_df["true_label"]

        logger.info(f"Evaluation samples: {len(eval_df)}")
        logger.info(f"Features: {len(feature_columns)}")

        # Load production model
        logger.info(f"\nLoading production model (v{production_model_version})...")
        prod_model = self._load_model_from_registry(
            model_name="credit-risk-model", version=production_model_version
        )

        # Load shadow model
        logger.info(f"Loading shadow model (v{shadow_model_version})...")
        shadow_model = self._load_model_from_registry(
            model_name="credit-risk-model", version=shadow_model_version
        )

        # Re-score with production model
        logger.info("\n[1/2] Re-scoring with PRODUCTION model...")
        prod_pred = prod_model.predict(X_eval)
        prod_prob = prod_model.predict_proba(X_eval)[:, 1]

        prod_metrics = self.evaluate_predictions(y_true, pd.Series(prod_pred), pd.Series(prod_prob))

        logger.info(f"  Production F1: {prod_metrics['primary_metrics']['f1_score']:.4f}")
        logger.info(f"  Production Brier: {prod_metrics['calibration_metrics']['brier_score']:.4f}")

        # Re-score with shadow model
        logger.info("\n[2/2] Re-scoring with SHADOW model...")
        shadow_pred = shadow_model.predict(X_eval)
        shadow_prob = shadow_model.predict_proba(X_eval)[:, 1]

        shadow_metrics = self.evaluate_predictions(
            y_true, pd.Series(shadow_pred), pd.Series(shadow_prob)
        )

        logger.info(f"  Shadow F1: {shadow_metrics['primary_metrics']['f1_score']:.4f}")
        logger.info(f"  Shadow Brier: {shadow_metrics['calibration_metrics']['brier_score']:.4f}")

        # Log comparison
        f1_diff = (
            shadow_metrics["primary_metrics"]["f1_score"]
            - prod_metrics["primary_metrics"]["f1_score"]
        )
        logger.info(f"\n✅ Replay complete. F1 difference: {f1_diff:+.4f}")

        return prod_metrics, shadow_metrics

    def _load_model_from_registry(self, model_name: str, version: str):
        """
        Load model from MLflow registry.

        Args:
            model_name: Registered model name
            version: Model version

        Returns:
            Loaded sklearn model
        """
        import os
        import tempfile

        model_uri = f"models:/{model_name}/{version}"

        # Use a temporary directory for the downloaded model to avoid permission issues
        # with the shared MLflow artifacts volume
        with tempfile.TemporaryDirectory() as tmpdir:
            model = mlflow.sklearn.load_model(model_uri, dst_path=tmpdir)
            logger.info(f"  Loaded: {model_uri}")
            return model

    def compare_models(self, production_metrics: Dict, shadow_metrics: Dict) -> Dict:
        """
        Compare shadow vs production.

        Args:
            production_metrics: Production model metrics
            shadow_metrics: Shadow model metrics

        Returns:
            Comparison dictionary
        """
        if not production_metrics or not shadow_metrics:
            return {"status": "insufficient_data"}

        prod_primary = production_metrics.get("primary_metrics", {})
        shadow_primary = shadow_metrics.get("primary_metrics", {})

        prod_calib = production_metrics.get("calibration_metrics", {})
        shadow_calib = shadow_metrics.get("calibration_metrics", {})

        prod_f1 = prod_primary.get("f1_score", 0)
        shadow_f1 = shadow_primary.get("f1_score", 0)

        comparison = {
            "f1_improvement": shadow_f1 - prod_f1,
            "f1_improvement_pct": (((shadow_f1 - prod_f1) / prod_f1 * 100) if prod_f1 > 0 else 0.0),
            "roc_auc_improvement": (
                shadow_primary.get("roc_auc", 0) - prod_primary.get("roc_auc", 0)
            ),
            "brier_change": (shadow_calib.get("brier_score", 1) - prod_calib.get("brier_score", 1)),
            "calibration_degraded": (
                shadow_calib.get("brier_score", 1) > prod_calib.get("brier_score", 0)
            ),
        }

        return comparison

    def _compute_primary_metrics(
        self, y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series
    ) -> Dict:
        """Core classification metrics."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
        }

    def _compute_calibration_metrics(self, y_true: pd.Series, y_prob: pd.Series) -> Dict:
        """Calibration metrics."""
        brier = float(brier_score_loss(y_true, y_prob))

        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10, strategy="uniform"
            )
            ece = float(np.mean(np.abs(fraction_of_positives - mean_predicted_value)))
        except Exception as e:
            logger.warning(f"Could not compute ECE: {e}")
            ece = None

        return {"brier_score": brier, "expected_calibration_error": ece}

    def _compute_confusion_matrix(self, y_true: pd.Series, y_pred: pd.Series) -> Dict:
        """Confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        return {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        }

    def _compute_segment_metrics(
        self, y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series, segment_data: pd.DataFrame
    ) -> Dict:
        """Per-segment performance."""
        segment_results = {}

        for feature in self.segment_features:
            if feature not in segment_data.columns:
                continue

            segments = segment_data[feature].unique()
            feature_results = {}

            for segment in segments:
                mask = segment_data[feature] == segment

                if mask.sum() < 30:
                    continue

                segment_metrics = self._compute_primary_metrics(
                    y_true[mask], y_pred[mask], y_prob[mask]
                )

                feature_results[str(segment)] = segment_metrics

            segment_results[feature] = feature_results

        return segment_results
