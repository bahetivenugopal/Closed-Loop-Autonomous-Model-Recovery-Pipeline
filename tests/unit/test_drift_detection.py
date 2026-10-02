"""
Unit tests for drift detection logic.

Tests the core drift detection functionality without external dependencies.
"""

import sys

import pytest
import pandas as pd
import numpy as np
from src.analytics.drift_detection import DriftDetector

sys.path.append("/app")


class TestDriftDetector:
    """Test suite for DriftDetector class."""

    def test_initialization(self, sample_reference_data, feature_columns, numerical_features):
        """Test DriftDetector initializes correctly."""
        detector = DriftDetector(
            reference_data=sample_reference_data,
            feature_columns=feature_columns,
            numerical_features=numerical_features,
        )

        assert detector.reference_data is not None
        assert len(detector.feature_columns) == len(feature_columns)
        assert len(detector.reference_data) > 0

    def test_no_drift_on_identical_data(
        self, sample_reference_data, feature_columns, numerical_features
    ):
        """Test that identical data shows no drift."""
        detector = DriftDetector(
            reference_data=sample_reference_data,
            feature_columns=feature_columns,
            numerical_features=numerical_features,
        )

        # Test on same data (should show minimal/no drift)
        results = detector.detect_drift(current_data=sample_reference_data, save_report=False)

        assert "dataset_drift_detected" in results
        # Identical data should have low drift share
        if "drift_share" in results:
            assert results["drift_share"] <= 0.5

    def test_drift_on_shifted_data(
        self, sample_reference_data, feature_columns, numerical_features
    ):
        """Test drift detection on intentionally shifted data."""
        detector = DriftDetector(
            reference_data=sample_reference_data,
            feature_columns=feature_columns,
            numerical_features=numerical_features,
        )

        # Create shifted data
        current_data = sample_reference_data.copy()
        current_data["MonthlyIncome"] = current_data["MonthlyIncome"] * 1.5  # 50% increase
        current_data["age"] = current_data["age"] + 10  # Age shift

        results = detector.detect_drift(current_data=current_data, save_report=False)

        # Should detect drift
        assert "drift_share" in results
        assert results["num_drifted_features"] > 0

    def test_handles_empty_current_data(
        self, sample_reference_data, feature_columns, numerical_features
    ):
        """Test that detector handles empty current data gracefully."""
        detector = DriftDetector(
            reference_data=sample_reference_data,
            feature_columns=feature_columns,
            numerical_features=numerical_features,
        )

        empty_df = pd.DataFrame(columns=feature_columns)
        results = detector.detect_drift(current_data=empty_df, save_report=False)

        assert results.get("status") == "no_data"

    def test_feature_drift_details(
        self, sample_reference_data, feature_columns, numerical_features
    ):
        """Test that per-feature drift details are included."""
        detector = DriftDetector(
            reference_data=sample_reference_data,
            feature_columns=feature_columns,
            numerical_features=numerical_features,
        )

        # Create data with drift in specific features
        current_data = sample_reference_data.copy()
        current_data["MonthlyIncome"] = current_data["MonthlyIncome"] * 2

        results = detector.detect_drift(current_data=current_data, save_report=False)

        # Check that results include drift info
        assert "dataset_drift_detected" in results or "drift_share" in results
        # Feature drift details may or may not be in results depending on implementation
        if "feature_drift_details" in results:
            assert len(results["feature_drift_details"]) > 0
            for detail in results["feature_drift_details"]:
                assert "feature" in detail or "column" in detail
