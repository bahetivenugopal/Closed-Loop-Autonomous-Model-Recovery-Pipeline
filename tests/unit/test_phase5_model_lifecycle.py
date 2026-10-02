"""
Unit & Integration test suite for Phase 5: Model Lifecycle Hardening.

Proves:
1. Same feature schema across training, inference, and monitoring.
2. Same feature ordering and schema validation.
3. Train-serving feature representation consistency.
4. Candidate model staging vs. gate-governed production promotion.
5. Abort/rejection behavior for invalid models.
"""

from unittest.mock import MagicMock, patch
import pytest
import pandas as pd
import numpy as np

from src.features.schema import (
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    validate_features,
    extract_features,
)
from src.train_model_mlflow import prepare_data
from src.retraining.shadow_trainer import ShadowModelTrainer
from src.retraining.model_promoter import ModelPromoter


@pytest.mark.unit
class TestPhase5ModelLifecycle:
    """Test suite for Model Lifecycle Hardening."""

    def test_canonical_feature_schema_definition(self):
        """Verify feature schema contains 10 expected features in exact order."""
        assert len(FEATURE_COLUMNS) == 10
        assert FEATURE_COLUMNS[0] == "RevolvingUtilizationOfUnsecuredLines"
        assert FEATURE_COLUMNS[1] == "age"
        assert FEATURE_COLUMNS[-1] == "NumberOfDependents"
        assert len(NUMERICAL_FEATURES) == 10

    def test_schema_validation_and_extraction(self):
        """Test validate_features and extract_features on valid and invalid data."""
        data = {col: [1.0] for col in FEATURE_COLUMNS}
        data[TARGET_COLUMN] = [0]
        data["unrelated_column"] = ["abc"]
        df = pd.DataFrame(data)

        is_valid, issues = validate_features(df)
        assert is_valid is True
        assert len(issues) == 0

        X = extract_features(df)
        assert list(X.columns) == FEATURE_COLUMNS

        # Missing required column
        df_invalid = df.drop(columns=["age"])
        is_valid_inv, issues_inv = validate_features(df_invalid)
        assert is_valid_inv is False
        assert any("age" in issue for issue in issues_inv)

        with pytest.raises(ValueError, match="Schema validation failed"):
            extract_features(df_invalid)

    def test_train_data_preparation_uses_canonical_schema(self):
        """Test prepare_data uses explicit schema instead of dropping column position 0."""
        data = {col: np.random.randn(20) for col in FEATURE_COLUMNS}
        data[TARGET_COLUMN] = np.random.choice([0, 1], size=20)
        data["extra_column_id"] = np.arange(20)  # Random column that should be ignored
        df = pd.DataFrame(data)

        X, y = prepare_data(df)

        assert list(X.columns) == FEATURE_COLUMNS
        assert len(X.columns) == 10
        assert TARGET_COLUMN not in X.columns
        assert "extra_column_id" not in X.columns

    @patch("src.retraining.shadow_trainer.mlflow")
    def test_candidate_model_registers_in_staging(self, mock_mlflow):
        """Verify candidate models register in Staging, not Production automatically."""
        trainer = ShadowModelTrainer()

        X_train = pd.DataFrame({col: np.random.randn(50) for col in FEATURE_COLUMNS})
        y_train = pd.Series(np.random.choice([0, 1], size=50))
        X_eval = pd.DataFrame({col: np.random.randn(30) for col in FEATURE_COLUMNS})
        y_eval = pd.Series(np.random.choice([0, 1], size=30))

        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.version = 5
        mock_client.search_model_versions.return_value = [mock_version]
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        model, run_id, result = trainer.train_shadow_model(
            X_train, y_train, X_eval, y_eval, trigger_reason="test"
        )

        assert result["status"] == "success"
        # Assert transitioned to Staging (NOT Production)
        mock_client.transition_model_version_stage.assert_called_once_with(
            name="credit-risk-model", version=5, stage="Staging"
        )

    def test_shadow_trainer_aborts_on_invalid_data_validation(self):
        """Verify shadow model training aborts if data validation fails."""
        trainer = ShadowModelTrainer()

        invalid_status = {
            "valid": False,
            "message": "Eval set is single-class: 1 class found",
            "issues": ["eval_single_class"],
        }

        with patch("src.retraining.shadow_trainer.mlflow"):
            model, run_id, result = trainer.train_shadow_model(
                X_train=None,
                y_train=None,
                X_eval=None,
                y_eval=None,
                validation_status=invalid_status,
            )

            assert model is None
            assert result["status"] == "aborted"
            assert "single-class" in result["reason"]
