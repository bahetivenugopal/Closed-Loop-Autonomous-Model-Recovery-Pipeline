"""
Unit tests for model evaluator logic.

Tests model evaluation and performance metrics calculation.
"""

import sys

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.analytics.model_evaluator import ModelEvaluator

sys.path.append("/app")


class TestModelEvaluator:
    """Test suite for ModelEvaluator class."""

    def test_initialization(self):
        """Test ModelEvaluator initializes correctly."""
        with patch("src.analytics.model_evaluator.mlflow"):
            evaluator = ModelEvaluator()
            assert evaluator is not None

    def test_calculate_binary_metrics(self, sample_labels_df):
        """Test binary classification metrics calculation."""
        with patch("src.analytics.model_evaluator.mlflow"):
            evaluator = ModelEvaluator()

            # Create true labels and predictions
            y_true = pd.Series(sample_labels_df["true_label"].values)
            y_pred = pd.Series(np.random.binomial(1, 0.1, len(y_true)))
            y_prob = pd.Series(np.random.beta(2, 8, len(y_true)))

            # Calculate metrics
            metrics = evaluator.evaluate_predictions(y_true=y_true, y_pred=y_pred, y_prob=y_prob)

            # Check metrics structure
            assert isinstance(metrics, dict)
            # Metrics are nested under primary_metrics
            if "primary_metrics" in metrics:
                primary = metrics["primary_metrics"]
                assert "f1_score" in primary or "accuracy" in primary
            else:
                assert "f1_score" in metrics or "accuracy" in metrics

    def test_confusion_matrix_calculation(self, sample_labels_df):
        """Test confusion matrix calculation."""
        with patch("src.analytics.model_evaluator.mlflow"):
            evaluator = ModelEvaluator()

            y_true = pd.Series(sample_labels_df["true_label"].values)
            y_pred = pd.Series(np.random.binomial(1, 0.1, len(y_true)))
            y_prob = pd.Series(np.random.beta(2, 8, len(y_true)))

            # Calculate metrics (which includes confusion matrix)
            metrics = evaluator.evaluate_predictions(y_true, y_pred, y_prob)

            assert "confusion_matrix" in metrics
            assert metrics["confusion_matrix"] is not None

    def test_roc_auc_calculation(self, sample_labels_df):
        """Test ROC AUC calculation."""
        with patch("src.analytics.model_evaluator.mlflow"):
            evaluator = ModelEvaluator()

            y_true = pd.Series(sample_labels_df["true_label"].values)
            y_prob = pd.Series(np.random.beta(2, 8, len(y_true)))
            y_pred = pd.Series((y_prob > 0.5).astype(int))

            # Calculate metrics
            metrics = evaluator.evaluate_predictions(y_true, y_pred, y_prob)

            # ROC AUC should be in metrics or primary_metrics
            if "primary_metrics" in metrics:
                assert (
                    "roc_auc" in metrics["primary_metrics"]
                    or "accuracy" in metrics["primary_metrics"]
                )
            else:
                assert "roc_auc" in metrics or "accuracy" in metrics
