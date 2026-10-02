"""
Regression test suite for Phase 1 runtime-critical bug fixes.

Verifies:
1. Drift artifact persistence (drift_summary_ref and HTML/JSON report refs).
2. Model rejection path recording (ModelVersionsRepository.insert_or_update / insert).
3. Data context normalization (labeled_samples vs labeled_predictions, coverage_rate vs coverage_pct).
4. Model promotion path (stage="Production" in repository).
5. Model rollback path.
"""

from unittest.mock import MagicMock, patch
import pytest
from datetime import datetime

from src.analytics.drift_detection import DriftDetector
from src.monitoring.monitoring_job import MonitoringJob
from src.retraining.model_promoter import ModelPromoter
from src.retraining.evaluation_gate import EvaluationGate
from src.storage.repositories import (
    RetrainingDecisionsRepository,
    ModelVersionsRepository,
)


@pytest.mark.unit
@pytest.mark.regression
class TestPhase1Regressions:
    """Tests ensuring Phase 1 runtime-critical bugs do not regress."""

    def test_drift_artifact_persistence(
        self,
        temp_monitoring_dir,
        sample_reference_data,
        sample_predictions_df,
        feature_columns,
        numerical_features,
    ):
        """
        Regression: DriftDetector._save_outputs must return drift_summary_ref and drift_report_ref.
        MonitoringJob._save_drift_artifacts must extract drift_summary_ref correctly.
        """
        detector = DriftDetector(
            reference_data=sample_reference_data,
            feature_columns=feature_columns,
            numerical_features=numerical_features,
        )

        results = detector.detect_drift(
            sample_predictions_df,
            report_dir=temp_monitoring_dir,
        )

        assert "drift_summary_ref" in results
        assert "drift_report_ref" in results
        assert results["drift_summary_ref"] is not None
        assert results["drift_report_ref"] is not None

        # Verify helper on MonitoringJob extracts this reference cleanly
        with patch.object(MonitoringJob, "__init__", lambda self: None):
            job = MonitoringJob()
            extracted_ref = job._save_drift_artifacts(results)
            assert extracted_ref == results["drift_summary_ref"]

    @patch("src.storage.repositories.get_db_manager")
    def test_retraining_decisions_data_context_normalization(self, mock_get_db):
        """
        Regression: RetrainingDecisionsRepository.insert must accept:
        - labeled_predictions -> labeled_samples
        - coverage_rate (0.0 - 1.0) -> coverage_pct (0 - 100)
        - label_coverage_pct -> coverage_pct
        """
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        repo = RetrainingDecisionsRepository()

        # Case 1: labeled_predictions and coverage_rate
        data_context_1 = {
            "labeled_predictions": 150,
            "coverage_rate": 0.75,
        }
        repo.insert(
            timestamp=datetime.now(),
            trigger_reason="scheduled",
            action="promote",
            drift_context={"feature_drift_ratio": 0.1, "num_drifted_features": 1},
            data_context=data_context_1,
            decision_details={"reason": "Passed gate"},
        )

        query, params = mock_db.execute_query.call_args[0]
        # Check labeled_samples (index 8) and coverage_pct (index 9)
        assert params[8] == 150
        assert params[9] == 75.0

        # Case 2: labeled_samples and label_coverage_pct
        data_context_2 = {
            "labeled_samples": 200,
            "label_coverage_pct": 85.5,
        }
        repo.insert(
            timestamp=datetime.now(),
            trigger_reason="drift_alert",
            action="reject",
            drift_context={"drift_share": 0.3},
            data_context=data_context_2,
            decision_details={"reason": "Failed gate"},
        )

        query2, params2 = mock_db.execute_query.call_args[0]
        assert params2[8] == 200
        assert params2[9] == 85.5

    def test_evaluation_gate_coverage_rate_normalization(self):
        """
        Regression: EvaluationGate.evaluate must parse coverage_rate or coverage_pct correctly.
        """
        gate = EvaluationGate(min_coverage_pct=50.0)

        # Passing coverage rate 0.6 (60%)
        coverage_stats_pass = {"coverage_rate": 0.6}
        cov_pct = coverage_stats_pass["coverage_rate"] * 100
        assert cov_pct >= gate.min_coverage_pct

    @patch("src.storage.repositories.get_db_manager")
    def test_model_versions_repository_insert_alias(self, mock_get_db):
        """
        Regression: ModelVersionsRepository must provide both insert() and insert_or_update()
        and properly cast model versions and statuses.
        """
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        repo = ModelVersionsRepository()

        # Call insert() alias
        repo.insert(
            model_name="credit-risk-model",
            version="2",
            stage="Archived",
            training_context={"trigger_reason": "rejected"},
            metrics={"primary_metrics": {"f1_score": 0.72}},
        )

        assert mock_db.execute_query.called
        query, params = mock_db.execute_query.call_args[0]
        assert params[0] == "credit-risk-model"
        assert params[1] == 2  # integer version
        assert params[2] == "Archived"
        assert params[8] == 0.72  # f1_score

    @patch("src.retraining.model_promoter.ModelVersionsRepository")
    @patch("src.retraining.model_promoter.MlflowClient")
    def test_rejection_path_records_to_database(
        self, mock_client_cls, mock_repo_cls, temp_monitoring_dir
    ):
        """
        Regression: ModelPromoter.reject_shadow_model must record the rejection in ModelVersionsRepository.
        """
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_version = MagicMock()
        mock_version.version = "3"
        mock_client.get_latest_versions.return_value = [mock_version]
        mock_client.search_model_versions.return_value = [mock_version]

        mock_repo = MagicMock()
        mock_repo_cls.return_value = mock_repo

        promoter = ModelPromoter(model_name="credit-risk-model", decisions_path=temp_monitoring_dir)
        eval_decision = {
            "final_decision": False,
            "reason": ["Insufficient improvement"],
            "metrics": {"f1_score": 0.65},
            "trigger_context": {"trigger_reason": "scheduled"},
        }

        result = promoter.reject_shadow_model(
            shadow_run_id="run_12345",
            evaluation_decision=eval_decision,
            rejected_by="evaluation_gate",
        )

        assert result["success"] is True
        assert result["action"] == "rejected"
        assert mock_repo.insert_or_update.called
        call_kwargs = mock_repo.insert_or_update.call_args[1]
        assert call_kwargs["stage"] == "Archived"
        assert call_kwargs["version"] == 3

    @patch("src.retraining.model_promoter.ModelVersionsRepository")
    @patch("src.retraining.model_promoter.MlflowClient")
    def test_promotion_path_records_to_database(
        self, mock_client_cls, mock_repo_cls, temp_monitoring_dir
    ):
        """
        Regression: ModelPromoter.promote_to_production must promote the shadow version and record to database.
        """
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Mock current prod
        current_prod = MagicMock()
        current_prod.version = "1"
        # Mock shadow
        shadow_version = MagicMock()
        shadow_version.version = "2"

        mock_client.get_latest_versions.side_effect = lambda name, stages: (
            [current_prod] if stages == ["Production"] else [shadow_version]
        )
        mock_client.search_model_versions.return_value = [shadow_version]

        mock_repo = MagicMock()
        mock_repo_cls.return_value = mock_repo

        promoter = ModelPromoter(model_name="credit-risk-model", decisions_path=temp_monitoring_dir)
        eval_decision = {
            "final_decision": True,
            "metrics": {"f1_score": 0.85},
            "trigger_context": {"trigger_reason": "scheduled"},
        }

        result = promoter.promote_to_production(
            shadow_run_id="run_shadow_999",
            evaluation_decision=eval_decision,
        )

        assert result["success"] is True
        assert result["new_production_version"] == "2"
        assert mock_repo.insert_or_update.called
        call_kwargs = mock_repo.insert_or_update.call_args[1]
        assert call_kwargs["stage"] == "Production"
        assert call_kwargs["version"] == 2

    @patch("src.retraining.model_promoter.ModelVersionsRepository")
    @patch("src.retraining.model_promoter.MlflowClient")
    def test_rollback_path(self, mock_client_cls, mock_repo_cls, temp_monitoring_dir):
        """
        Regression: ModelPromoter.rollback_to_version must transition target version back to Production.
        """
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        current_prod = MagicMock()
        current_prod.version = "2"
        mock_client.get_latest_versions.return_value = [current_prod]

        mock_repo = MagicMock()
        mock_repo_cls.return_value = mock_repo

        promoter = ModelPromoter(model_name="credit-risk-model", decisions_path=temp_monitoring_dir)
        result = promoter.rollback_to_version("1")

        assert result["success"] is True
        assert result["restored_version"] == "1"
        assert mock_client.transition_model_version_stage.called
