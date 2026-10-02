"""
Prediction logger with FULL feature storage in PostgreSQL (Single Source of Truth).

Stores raw features for replay-based evaluation and drift analysis in PostgreSQL JSONB.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from src.storage.repositories import PredictionsRepository

logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Log predictions WITH full feature vectors to PostgreSQL.
    """

    def __init__(self, repo: Optional[PredictionsRepository] = None):
        self.feature_columns = [
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30_59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60_89DaysPastDueNotWorse",
            "NumberOfDependents",
        ]
        self.repo = repo or PredictionsRepository()

    def log_prediction(
        self,
        features: dict,
        prediction: int,
        probability: float,
        model_version: str,
        application_date: str = None,
        prediction_id: str = None,
        request_id: str = None,
    ) -> str:
        """
        Log prediction with FULL features to PostgreSQL.

        Raises exception if PostgreSQL operation fails.
        """
        if prediction_id is None:
            prediction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Validate all features present
        missing = set(self.feature_columns) - set(features.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")

        return self.repo.insert(
            prediction_id=prediction_id,
            timestamp=timestamp,
            model_version=str(model_version),
            prediction=prediction,
            probability=probability,
            features=features,
            application_date=application_date or timestamp,
            request_id=request_id,
        )

    def get_predictions_with_features(self, prediction_ids: Optional[list] = None) -> pd.DataFrame:
        """
        Get predictions WITH features for replay evaluation from PostgreSQL.
        """
        records = self.repo.get_recent_predictions(days=365)
        if not records:
            return pd.DataFrame()

        rows = []
        for r in records:
            row = {
                "prediction_id": r["prediction_id"],
                "timestamp": r["timestamp"],
                "model_version": r["model_version"],
                "prediction": r["prediction"],
                "probability": r["probability"],
                "application_date": r["application_date"],
            }
            feats = r.get("features", {})
            if isinstance(feats, dict):
                row.update(feats)
            rows.append(row)

        df = pd.DataFrame(rows)
        if prediction_ids is not None and not df.empty:
            df = df[df["prediction_id"].isin(prediction_ids)]

        logger.info(f"Retrieved {len(df)} predictions with features from PostgreSQL")
        return df

    def get_recent_predictions(
        self, days: int = 30, date_column: str = "application_date"
    ) -> pd.DataFrame:
        """
        Get recent predictions from PostgreSQL.
        """
        records = self.repo.get_recent_predictions(days=days)
        if not records:
            return pd.DataFrame()

        rows = []
        for r in records:
            row = {
                "prediction_id": r["prediction_id"],
                "timestamp": r["timestamp"],
                "model_version": r["model_version"],
                "prediction": r["prediction"],
                "probability": r["probability"],
                "application_date": r["application_date"],
            }
            feats = r.get("features", {})
            if isinstance(feats, dict):
                row.update(feats)
            rows.append(row)

        df = pd.DataFrame(rows)
        if date_column in df.columns and not df.empty:
            df[date_column] = pd.to_datetime(df[date_column])

        logger.info(f"Retrieved {len(df)} predictions from last {days} days from PostgreSQL")
        return df


# Singleton instance
_logger_instance = None


def get_prediction_logger() -> PredictionLogger:
    """Get or create singleton prediction logger."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = PredictionLogger()
    return _logger_instance
