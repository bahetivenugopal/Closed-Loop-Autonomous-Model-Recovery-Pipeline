"""
Label store with IDEMPOTENT label updates in PostgreSQL (Single Source of Truth).
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd

from src.storage.repositories import LabelsRepository

logger = logging.getLogger(__name__)


class LabelStore:
    """
    Store ground truth labels with idempotency guarantee in PostgreSQL.

    ENFORCED: One label per prediction_id.
    If label exists -> update, don't duplicate.
    """

    def __init__(self, repo: Optional[LabelsRepository] = None):
        self.repo = repo or LabelsRepository()

    def store_label(
        self,
        prediction_id: str,
        true_label: int,
        label_source: str = "manual",
        prediction_timestamp: Optional[str] = None,
    ):
        """
        Store label with idempotency in PostgreSQL.
        """
        label_timestamp = datetime.now().isoformat()

        days_delayed = None
        if prediction_timestamp:
            try:
                pred_time = pd.to_datetime(prediction_timestamp)
                label_time = pd.to_datetime(label_timestamp)
                days_delayed = (label_time - pred_time).days
            except Exception as e:
                logger.warning(f"Could not calculate delay: {e}")

        self.repo.insert_or_update(
            prediction_id=prediction_id,
            true_label=int(true_label),
            label_source=label_source,
            label_timestamp=label_timestamp,
            days_delayed=days_delayed,
        )
        logger.info(f"Stored label for {prediction_id} in PostgreSQL: {true_label}")

    def get_labeled_predictions(
        self, predictions_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Get predictions joined with ground truth labels from PostgreSQL v_labeled_predictions.
        """
        records = self.repo.get_labeled_predictions()
        if not records:
            return pd.DataFrame()

        rows = []
        for r in records:
            row = {
                "prediction_id": r["prediction_id"],
                "prediction_timestamp": r.get("prediction_timestamp"),
                "model_version": r.get("model_version"),
                "prediction": r.get("prediction"),
                "probability": r.get("probability"),
                "application_date": r.get("application_date"),
                "true_label": r.get("true_label"),
                "label_timestamp": r.get("label_timestamp"),
                "label_source": r.get("label_source"),
                "days_delayed": r.get("days_delayed"),
            }
            feats = r.get("features", {})
            if isinstance(feats, dict):
                row.update(feats)
            rows.append(row)

        df = pd.DataFrame(rows)

        if predictions_df is not None and not predictions_df.empty and not df.empty:
            if "prediction_id" in predictions_df.columns:
                df = df[df["prediction_id"].isin(predictions_df["prediction_id"])]

        logger.info(f"Retrieved {len(df)} labeled predictions from PostgreSQL")
        return df

    def get_label_coverage(self, predictions_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate label coverage statistics directly from PostgreSQL.
        """
        return self.repo.get_coverage_stats()


# Singleton instance
_label_store_instance = None


def get_label_store() -> LabelStore:
    """Get or create singleton label store."""
    global _label_store_instance
    if _label_store_instance is None:
        _label_store_instance = LabelStore()
    return _label_store_instance
