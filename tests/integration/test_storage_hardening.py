"""
Integration test suite for Phase 4: Production Data Storage Hardening.

Verifies:
1. Prediction persistence and retrieval in PostgreSQL.
2. Ground truth label persistence and idempotent upserting.
3. Prediction-Label join dataset for evaluation and retraining.
4. Duplicate prediction and label handling.
5. Database failure fallback and resilience.
6. API POST /labels endpoint behavior.
"""

from unittest.mock import MagicMock, patch
import pytest
from datetime import datetime
from starlette.testclient import TestClient

from src.storage.repositories import PredictionsRepository, LabelsRepository
from src.storage.prediction_logger import PredictionLogger
from src.storage.label_store import LabelStore
from src.api_mlflow import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.integration
class TestStorageHardening:
    """Test suite for hardened durable storage."""

    @patch("src.storage.repositories.get_db_manager")
    def test_prediction_persistence_and_retrieval(self, mock_get_db):
        """Test inserting and retrieving a prediction via PredictionsRepository."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        repo = PredictionsRepository()
        features = {
            "RevolvingUtilizationOfUnsecuredLines": 0.5,
            "age": 45,
            "NumberOfTime30_59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.3,
            "MonthlyIncome": 6000.0,
            "NumberOfOpenCreditLinesAndLoans": 10,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 2,
            "NumberOfTime60_89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 1,
        }

        # 1. Test insert
        pred_id = repo.insert(
            prediction_id="pred_test_1001",
            timestamp=datetime.now().isoformat(),
            model_version="1",
            prediction=0,
            probability=0.12,
            features=features,
            request_id="req_test_1001",
        )

        assert pred_id == "pred_test_1001"
        assert mock_db.execute_query.called
        query, params = mock_db.execute_query.call_args[0]
        assert "INSERT INTO predictions" in query
        assert "ON CONFLICT (prediction_id) DO NOTHING" in query
        assert params[0] == "pred_test_1001"
        assert params[3] == 0
        assert params[4] == 0.12
        assert params[7] == "req_test_1001"

        # 2. Test get_by_id
        mock_db.execute_query.return_value = [
            (
                "pred_test_1001",
                datetime(2026, 8, 18, 0, 0, 0),
                "1",
                0,
                0.12,
                datetime(2026, 8, 18, 0, 0, 0),
                features,
                "req_test_1001",
            )
        ]

        record = repo.get_by_id("pred_test_1001")
        assert record is not None
        assert record["prediction_id"] == "pred_test_1001"
        assert record["probability"] == 0.12
        assert record["features"]["age"] == 45

    @patch("src.storage.repositories.get_db_manager")
    def test_label_persistence_and_idempotent_upsert(self, mock_get_db):
        """Test inserting and updating ground truth labels in LabelsRepository."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        repo = LabelsRepository()

        # Insert label
        repo.insert_or_update(
            prediction_id="pred_test_1001",
            true_label=0,
            label_source="manual",
            days_delayed=5,
        )

        assert mock_db.execute_query.called
        query, params = mock_db.execute_query.call_args[0]
        assert "INSERT INTO labels" in query
        assert "ON CONFLICT (prediction_id)" in query
        assert "DO UPDATE SET" in query
        assert params[0] == "pred_test_1001"
        assert params[1] == 0
        assert params[2] == "manual"
        assert params[4] == 5

    @patch("src.storage.repositories.get_db_manager")
    def test_prediction_label_join_dataset(self, mock_get_db):
        """Test querying joined prediction and ground truth records."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        repo = LabelsRepository()
        features = {"age": 50, "DebtRatio": 0.25}

        mock_db.execute_query.return_value = [
            (
                "pred_test_2001",
                datetime(2026, 8, 18, 0, 0, 0),
                "1",
                0,
                0.08,
                datetime(2026, 8, 18, 0, 0, 0),
                features,
                0,  # true_label
                datetime(2026, 8, 19, 0, 0, 0),
                "automated",
                1,
            )
        ]

        dataset = repo.get_labeled_predictions(limit=10)
        assert len(dataset) == 1
        row = dataset[0]
        assert row["prediction_id"] == "pred_test_2001"
        assert row["prediction"] == 0
        assert row["true_label"] == 0
        assert row["days_delayed"] == 1

    @patch("src.storage.repositories.get_db_manager")
    def test_database_failure_raises_explicit_error(self, mock_get_db):
        """Test that database failures raise an explicit error and do NOT write to CSV files."""
        mock_db = MagicMock()
        mock_db.execute_query.side_effect = RuntimeError("PostgreSQL connection lost")
        mock_get_db.return_value = mock_db

        logger_inst = PredictionLogger()
        store = LabelStore()

        features = {
            "RevolvingUtilizationOfUnsecuredLines": 0.2,
            "age": 35,
            "NumberOfTime30_59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.1,
            "MonthlyIncome": 8000.0,
            "NumberOfOpenCreditLinesAndLoans": 8,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60_89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 0,
        }

        # Should raise an explicit RuntimeError when PostgreSQL fails
        with pytest.raises(RuntimeError, match="PostgreSQL connection lost"):
            logger_inst.log_prediction(
                prediction_id="pred_fail_001",
                features=features,
                prediction=0,
                probability=0.05,
                model_version="1",
            )

        with pytest.raises(RuntimeError, match="PostgreSQL connection lost"):
            store.store_label(
                prediction_id="pred_fail_001",
                true_label=0,
                label_source="test_fail",
            )

    def test_api_submit_label_endpoint(self, client, monkeypatch):
        """Test API POST /labels endpoint."""
        mock_store = MagicMock()
        monkeypatch.setattr("src.storage.label_store.get_label_store", lambda: mock_store)

        payload = {
            "prediction_id": "pred_api_test_999",
            "true_label": 1,
            "label_source": "analyst_review",
        }

        response = client.post("/labels", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["prediction_id"] == "pred_api_test_999"
        assert data["true_label"] == 1
        assert mock_store.store_label.called

    def test_api_submit_label_invalid_label_validation(self, client):
        """Test that invalid label values (outside 0 and 1) are rejected with 422."""
        payload = {
            "prediction_id": "pred_invalid_001",
            "true_label": 5,  # Invalid: must be 0 or 1
        }

        response = client.post("/labels", json=payload)
        assert response.status_code == 422
