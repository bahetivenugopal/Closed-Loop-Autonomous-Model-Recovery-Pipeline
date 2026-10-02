"""
Centralized Configuration Settings for Self-Healing MLOps.

All environment-specific configuration, service URIs, and credentials
are managed dynamically via environment variables.
"""

import os


class Settings:
    """Dynamic configuration settings loading directly from environment variables."""

    @property
    def MLFLOW_TRACKING_URI(self) -> str:
        return os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    @property
    def MLFLOW_EXPERIMENT_NAME(self) -> str:
        return os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-risk-prediction")

    @property
    def MODEL_NAME(self) -> str:
        return os.getenv("MODEL_NAME", "credit-risk-model")

    @property
    def POSTGRES_HOST(self) -> str:
        return os.getenv("POSTGRES_HOST", "postgres")

    @property
    def POSTGRES_PORT(self) -> int:
        return int(os.getenv("POSTGRES_PORT", "5432"))

    @property
    def MLOPS_DB_NAME(self) -> str:
        return os.getenv("MLOPS_DB_NAME", "mlops")

    @property
    def POSTGRES_USER(self) -> str:
        return os.getenv("POSTGRES_USER", "airflow")

    @property
    def POSTGRES_PASSWORD(self) -> str:
        return os.getenv("POSTGRES_PASSWORD", "airflow")

    @property
    def AIRFLOW_USERNAME(self) -> str:
        return os.getenv("AIRFLOW_USERNAME", "admin")

    @property
    def AIRFLOW_PASSWORD(self) -> str:
        return os.getenv("AIRFLOW_PASSWORD", "admin")

    @property
    def MONITORING_LOOKBACK_HOURS(self) -> int:
        return int(os.getenv("MONITORING_LOOKBACK", "24"))

    @property
    def SECRET_KEY(self) -> str:
        return os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")


settings = Settings()
