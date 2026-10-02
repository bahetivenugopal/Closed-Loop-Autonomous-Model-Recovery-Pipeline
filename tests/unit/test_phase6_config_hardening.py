"""
Unit test suite for Phase 6: Configuration and Secrets Hardening.

Verifies:
1. Centralized Settings configuration module functionality.
2. Absence of plaintext production passwords in .env.example.
3. Disabled exposure of Airflow configuration in docker-compose.
"""

from pathlib import Path
import os
import pytest

from src.config import Settings


@pytest.mark.unit
class TestPhase6ConfigHardening:
    """Test suite for configuration hardening."""

    def test_settings_module_environment_overrides(self, monkeypatch):
        """Test that settings reads environment variable overrides."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://custom-mlflow:5000")
        monkeypatch.setenv("POSTGRES_USER", "custom_user")
        monkeypatch.setenv("POSTGRES_PASSWORD", "custom_pass")

        custom_settings = Settings()
        assert custom_settings.MLFLOW_TRACKING_URI == "http://custom-mlflow:5000"
        assert custom_settings.POSTGRES_USER == "custom_user"
        assert custom_settings.POSTGRES_PASSWORD == "custom_pass"

    def test_env_example_contains_placeholders_only(self):
        """Verify .env.example contains only placeholders, no hardcoded production passwords."""
        env_example_path = Path(".env.example")
        assert env_example_path.exists()

        content = env_example_path.read_text(encoding="utf-8")
        assert "POSTGRES_PASSWORD=your_mlops_db_password" in content
        assert "AIRFLOW_DB_PASSWORD=your_airflow_db_password" in content
        assert "SECRET_KEY=your-production-secret-key-here" in content
        assert "AIRFLOW_PASSWORD=your_airflow_password" in content

    def test_airflow_expose_config_disabled(self):
        """Verify AIRFLOW__WEBSERVER__EXPOSE_CONFIG is set to False in docker-compose."""
        docker_compose_path = Path("docker-compose.yml")
        assert docker_compose_path.exists()

        content = docker_compose_path.read_text(encoding="utf-8")
        assert "AIRFLOW__WEBSERVER__EXPOSE_CONFIG=False" in content
