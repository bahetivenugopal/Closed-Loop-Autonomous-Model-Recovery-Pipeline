"""
Data validation tests using Pandera.

Ensures data schemas are enforced and breaking changes are caught.
"""

import pytest
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import sys

sys.path.append("/app")


# Define schemas
class PredictionSchema:
    """Schema for prediction data."""

    schema = DataFrameSchema(
        {
            "timestamp": Column(pa.DateTime, nullable=False),
            "prediction_id": Column(pa.String, nullable=False, unique=True),
            "prediction": Column(pa.Int, pa.Check.isin([0, 1]), nullable=False),
            "probability": Column(pa.Float, pa.Check.in_range(0, 1), nullable=False),
            "model_version": Column(pa.String, nullable=False),
            # Features
            "RevolvingUtilizationOfUnsecuredLines": Column(
                pa.Float, pa.Check.greater_than_or_equal_to(0)
            ),
            "age": Column(pa.Int, pa.Check.in_range(18, 120)),
            "NumberOfTime30_59DaysPastDueNotWorse": Column(
                pa.Int, pa.Check.greater_than_or_equal_to(0)
            ),
            "DebtRatio": Column(pa.Float, pa.Check.greater_than_or_equal_to(0)),
            "MonthlyIncome": Column(pa.Float, pa.Check.greater_than_or_equal_to(0)),
            "NumberOfOpenCreditLinesAndLoans": Column(pa.Int, pa.Check.greater_than_or_equal_to(0)),
            "NumberOfTimes90DaysLate": Column(pa.Int, pa.Check.greater_than_or_equal_to(0)),
            "NumberRealEstateLoansOrLines": Column(pa.Int, pa.Check.greater_than_or_equal_to(0)),
            "NumberOfTime60_89DaysPastDueNotWorse": Column(
                pa.Int, pa.Check.greater_than_or_equal_to(0)
            ),
            "NumberOfDependents": Column(pa.Int, pa.Check.greater_than_or_equal_to(0)),
        }
    )


class LabelSchema:
    """Schema for label data."""

    schema = DataFrameSchema(
        {
            "prediction_id": Column(pa.String, nullable=False),
            "true_label": Column(pa.Int, pa.Check.isin([0, 1]), nullable=False),
            "label_timestamp": Column(pa.DateTime, nullable=False),
            "label_source": Column(pa.String, nullable=False),
        }
    )


class TestDataValidation:
    """Test data validation with Pandera schemas."""

    def test_valid_predictions_pass_schema(self, sample_predictions_df):
        """Test that valid prediction data passes schema validation."""
        try:
            PredictionSchema.schema.validate(sample_predictions_df)
            assert True
        except pa.errors.SchemaError as e:
            pytest.fail(f"Valid data failed schema validation: {e}")

    def test_invalid_prediction_values_fail(self, sample_predictions_df):
        """Test that invalid prediction values are caught."""
        # Create invalid data (prediction not in [0, 1])
        invalid_df = sample_predictions_df.copy()
        invalid_df.loc[0, "prediction"] = 2

        with pytest.raises(pa.errors.SchemaError):
            PredictionSchema.schema.validate(invalid_df)

    def test_invalid_probability_range_fails(self, sample_predictions_df):
        """Test that probabilities outside [0, 1] are caught."""
        invalid_df = sample_predictions_df.copy()
        invalid_df.loc[0, "probability"] = 1.5

        with pytest.raises(pa.errors.SchemaError):
            PredictionSchema.schema.validate(invalid_df)

    def test_negative_age_fails(self, sample_predictions_df):
        """Test that invalid age values are caught."""
        invalid_df = sample_predictions_df.copy()
        invalid_df.loc[0, "age"] = -5

        with pytest.raises(pa.errors.SchemaError):
            PredictionSchema.schema.validate(invalid_df)

    def test_missing_required_columns_fails(self, sample_predictions_df):
        """Test that missing required columns are caught."""
        invalid_df = sample_predictions_df.drop(columns=["prediction"])

        with pytest.raises(pa.errors.SchemaError):
            PredictionSchema.schema.validate(invalid_df)

    def test_duplicate_prediction_ids_fail(self, sample_predictions_df):
        """Test that duplicate prediction IDs are caught."""
        invalid_df = sample_predictions_df.copy()
        invalid_df.loc[1, "prediction_id"] = invalid_df.loc[0, "prediction_id"]

        with pytest.raises(pa.errors.SchemaError):
            PredictionSchema.schema.validate(invalid_df)

    def test_valid_labels_pass_schema(self, sample_labels_df):
        """Test that valid label data passes schema validation."""
        try:
            LabelSchema.schema.validate(sample_labels_df)
            assert True
        except pa.errors.SchemaError as e:
            pytest.fail(f"Valid labels failed schema validation: {e}")
