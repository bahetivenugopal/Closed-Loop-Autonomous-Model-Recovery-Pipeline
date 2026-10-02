"""
Integration tests for API endpoints.

Tests the API as a whole, including request/response flow, reliability, readiness, and error safety.
"""

import sys
from unittest.mock import MagicMock
import pytest
from starlette.testclient import TestClient
import src.api_mlflow as api_module
from src.api_mlflow import app

sys.path.append("/app")


@pytest.fixture
def client():
    """Fixture to provide a TestClient instance for ASGI app testing."""
    return TestClient(app)


class TestAPIEndpoints:
    """Test suite for API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Credit Risk Prediction API"
        assert data["status"] == "healthy"

    def test_health_liveness_endpoint(self, client):
        """Test lightweight health/liveness check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_readiness_endpoint_unready_without_model(self, client, monkeypatch):
        """Test readiness check returns 503 when model is not loaded."""
        monkeypatch.setattr(api_module, "model", None)
        monkeypatch.setattr(api_module, "model_version", None)

        response = client.get("/ready")
        assert response.status_code == 503
        assert "Service not ready" in response.json()["detail"]

    def test_readiness_endpoint_ready_with_model(self, client, monkeypatch):
        """Test readiness check returns 200 when model is loaded and storage accessible."""
        mock_model = MagicMock()
        monkeypatch.setattr(api_module, "model", mock_model)
        monkeypatch.setattr(api_module, "model_version", "1")

        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["ready"] is True
        assert data["model_loaded"] is True
        assert data["storage_accessible"] is True

    def test_model_info_endpoint(self, client, monkeypatch):
        """Test model info endpoint."""
        mock_model = MagicMock()
        monkeypatch.setattr(api_module, "model", mock_model)
        monkeypatch.setattr(api_module, "model_version", "1")

        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "credit-risk-model"
        assert data["model_version"] == "1"

    def test_request_correlation_headers(self, client):
        """Test request correlation ID generation and propagation."""
        custom_request_id = "req_custom_test_12345"
        response = client.get("/health", headers={"X-Request-ID": custom_request_id})
        assert response.status_code == 200
        assert response.headers.get("X-Request-ID") == custom_request_id

        # Auto-generated request ID when not provided
        response_auto = client.get("/health")
        assert response_auto.status_code == 200
        assert "X-Request-ID" in response_auto.headers
        assert response_auto.headers["X-Request-ID"].startswith("req_")

    def test_predict_endpoint_validation(self, client):
        """Test that predict endpoint validates input schema."""
        invalid_input = {
            "age": 45
            # Missing other required fields
        }

        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422  # Pydantic validation error

    @pytest.mark.timeout(5)
    def test_predict_endpoint_valid_input(self, client, monkeypatch):
        """Test prediction with valid input and mock model."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.85, 0.15]]

        monkeypatch.setattr(api_module, "model", mock_model)
        monkeypatch.setattr(api_module, "model_version", "1")

        valid_input = {
            "RevolvingUtilizationOfUnsecuredLines": 0.766127,
            "age": 45,
            "NumberOfTime30_59DaysPastDueNotWorse": 2,
            "DebtRatio": 0.802982,
            "MonthlyIncome": 9120.0,
            "NumberOfOpenCreditLinesAndLoans": 13,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 6,
            "NumberOfTime60_89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 2,
        }

        response = client.post(
            "/predict",
            json=valid_input,
            headers={"X-Request-ID": "req_test_predict_001"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 0
        assert data["probability"] == 0.15
        assert data["model_version"] == "1"
        assert "prediction_id" in data
        assert data["request_id"] == "req_test_predict_001"
        assert response.headers.get("X-Request-ID") == "req_test_predict_001"

    def test_predict_model_unavailable_returns_503(self, client, monkeypatch):
        """Test that 503 is returned when model is not available."""
        monkeypatch.setattr(api_module, "model", None)

        valid_input = {
            "RevolvingUtilizationOfUnsecuredLines": 0.5,
            "age": 30,
            "NumberOfTime30_59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.3,
            "MonthlyIncome": 5000.0,
            "NumberOfOpenCreditLinesAndLoans": 5,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60_89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 0,
        }

        response = client.post("/predict", json=valid_input)
        assert response.status_code == 503
        assert response.json()["detail"] == "Model not loaded"

    def test_predict_safe_error_response_no_leakage(self, client, monkeypatch):
        """Test that unexpected internal exceptions do not leak stack traces or secret strings to clients."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError(
            "Secret database connection failed at internal_host:5432/pass=secret123"
        )

        monkeypatch.setattr(api_module, "model", mock_model)
        monkeypatch.setattr(api_module, "model_version", "1")

        valid_input = {
            "RevolvingUtilizationOfUnsecuredLines": 0.5,
            "age": 30,
            "NumberOfTime30_59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.3,
            "MonthlyIncome": 5000.0,
            "NumberOfOpenCreditLinesAndLoans": 5,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60_89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 0,
        }

        response = client.post("/predict", json=valid_input)
        assert response.status_code == 500
        # Safe generic response returned
        assert response.json()["detail"] == "Prediction failed"
        # Verify internal secret error is NOT in response body
        assert "secret123" not in response.text
        assert "RuntimeError" not in response.text

    def test_predict_endpoint_invalid_values(self, client):
        """Test that invalid feature values are rejected with 422."""
        invalid_input = {
            "RevolvingUtilizationOfUnsecuredLines": 0.5,
            "age": -5,  # Invalid age
            "NumberOfTime30_59DaysPastDueNotWorse": 2,
            "DebtRatio": 0.8,
            "MonthlyIncome": 9120.0,
            "NumberOfOpenCreditLinesAndLoans": 13,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 6,
            "NumberOfTime60_89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 2,
        }

        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_monitoring_stats_endpoint(self, client):
        """Test monitoring stats endpoint."""
        response = client.get("/monitoring/stats")
        assert response.status_code in [200, 500]
