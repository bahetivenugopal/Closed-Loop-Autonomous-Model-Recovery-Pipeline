"""
Unit tests for proxy metrics calculation.

Tests the calculation of proxy metrics for model monitoring.
"""

import sys

import pytest
import pandas as pd
import numpy as np
from src.analytics.proxy_metrics import (
    compute_probability_entropy,
    compute_prediction_distribution_stats,
)

sys.path.append("/app")


class TestProxyMetrics:
    """Test suite for proxy metrics functions."""

    def test_calculate_prediction_entropy(self, sample_predictions_df):
        """Test entropy calculation from predictions."""
        # Calculate entropy
        entropy = compute_probability_entropy(sample_predictions_df)
        # Entropy should be between 0 and inf
        assert entropy >= 0

    def test_compute_distribution_stats(self, sample_predictions_df):
        """Test prediction distribution statistics calculation."""
        stats = compute_prediction_distribution_stats(sample_predictions_df)

        # Check stats structure
        assert "num_predictions" in stats
        assert "positive_rate" in stats
        assert "probability_mean" in stats
        assert stats["num_predictions"] > 0

    def test_compute_distribution_stats_empty(self):
        """Test that empty data returns status."""
        empty_df = pd.DataFrame({"prediction": [], "probability": []})
        result = compute_prediction_distribution_stats(empty_df)

        assert result.get("status") == "no_data"
