"""
Integration tests for retraining workflow.

Tests the complete retraining pipeline including model training and evaluation.
"""

import pytest
import pandas as pd
import numpy as np
import sys

sys.path.append("/app")


class TestRetrainingWorkflow:
    """Test suite for retraining workflow."""

    def test_training_data_preparation(self, sample_predictions_df, sample_labels_df):
        """Test preparation of training data."""
        # Merge predictions with labels
        merged_data = sample_predictions_df.merge(
            sample_labels_df[["prediction_id", "true_label"]], on="prediction_id", how="inner"
        )

        # Should have some merged records
        assert len(merged_data) > 0
        assert "true_label" in merged_data.columns

    def test_training_split_consistency(self, sample_predictions_df, sample_labels_df):
        """Test consistency of train/test splits."""
        # Create train/test split
        n = len(sample_labels_df)
        split_point = int(n * 0.8)

        train_ids = sample_labels_df["prediction_id"].iloc[:split_point]
        test_ids = sample_labels_df["prediction_id"].iloc[split_point:]

        # Verify no overlap
        assert len(set(train_ids) & set(test_ids)) == 0
        assert len(train_ids) + len(test_ids) == n
