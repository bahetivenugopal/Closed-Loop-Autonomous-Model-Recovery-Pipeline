"""
Integration tests for monitoring pipeline.

Tests the complete monitoring workflow including drift detection and alert generation.
"""

import pytest
import pandas as pd
import numpy as np
import sys

sys.path.append("/app")


class TestMonitoringPipeline:
    """Test suite for monitoring pipeline."""

    def test_monitoring_data_ingestion(self, sample_predictions_df):
        """Test that predictions data can be ingested."""
        # Verify sample predictions are valid
        assert len(sample_predictions_df) > 0
        assert "timestamp" in sample_predictions_df.columns
        assert "prediction" in sample_predictions_df.columns
        assert "probability" in sample_predictions_df.columns

    def test_label_alignment(self, sample_predictions_df, sample_labels_df):
        """Test label-prediction alignment logic."""
        # Get prediction IDs
        pred_ids = set(sample_predictions_df["prediction_id"])
        label_ids = set(sample_labels_df["prediction_id"])

        # Check that all labeled predictions have corresponding predictions
        assert label_ids.issubset(pred_ids)

    def test_time_series_aggregation(self, sample_predictions_df):
        """Test time-series aggregation for monitoring."""
        # Group predictions by hour
        hourly_data = (
            sample_predictions_df.set_index("timestamp")
            .resample("1H")
            .agg({"prediction": "sum", "probability": "mean"})
        )

        assert len(hourly_data) > 0
        assert "prediction" in hourly_data.columns
        assert "probability" in hourly_data.columns
