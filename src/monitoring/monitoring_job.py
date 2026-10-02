from typing import Dict, Any
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import logging
import sys
from src.analytics.proxy_metrics import analyze_proxy_metrics
from src.analytics.drift_detection import DriftDetector, load_reference_data
from src.storage.repositories import MonitoringMetricsRepository
from scripts.bootstrap_reference import verify_reference_integrity
from src.features.schema import (
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)

sys.path.append("/app")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MIN_SAMPLES_FOR_ANALYSIS = 200


class MonitoringJob:
    def __init__(
        self,
        reference_dir: str = "/app/monitoring/reference",
        output_dir: str = "/app/monitoring/metrics/monitoring_results",
    ):
        self.reference_dir = reference_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_repo = MonitoringMetricsRepository()

        logger.info("Verifying reference data integrity...")
        verify_reference_integrity(reference_dir)

        self.reference_data, self.reference_metadata = load_reference_data(reference_dir)
        logger.info("Monitoring job initialized with PostgreSQL storage")

    def load_predictions(self, lookback_hours: int = 24) -> pd.DataFrame:
        from src.storage.prediction_logger import get_prediction_logger

        logger_inst = get_prediction_logger()
        days = max(1, lookback_hours // 24)
        df = logger_inst.get_recent_predictions(days=days)

        if len(df) == 0:
            return df

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=lookback_hours)
            df = df[df["timestamp"] > cutoff]
        return df

    def run_monitoring_job(self, lookback_hours: int = 24) -> Dict[str, Any]:
        run_timestamp = datetime.now()
        predictions = self.load_predictions(lookback_hours)

        if len(predictions) < MIN_SAMPLES_FOR_ANALYSIS:
            return {"status": "insufficient_data"}

        results: Dict[str, Any] = {"timestamp": run_timestamp.isoformat()}

        # --- PROXY METRICS ---
        proxy_results: Dict[str, Any]
        try:
            proxy_results = analyze_proxy_metrics(predictions)
            results["proxy_metrics"] = proxy_results
        except Exception as e:
            proxy_results = {"status": "error", "error": str(e)}
            results["proxy_metrics"] = proxy_results

        # --- DRIFT DETECTION ---
        drift_summary: Dict[str, Any] = {}
        drift_summary_ref = None
        try:
            drift_detector = DriftDetector(
                reference_data=self.reference_data,
                feature_columns=FEATURE_COLUMNS,
                numerical_features=NUMERICAL_FEATURES,
                categorical_features=CATEGORICAL_FEATURES,
                reference_metadata=self.reference_metadata,
            )
            drift_results = drift_detector.detect_drift(predictions)

            drift_summary = {
                "dataset_drift_detected": drift_results.get("dataset_drift_detected", False),
                "feature_drift_ratio": drift_results.get("drift_share", 0.0),
                "num_drifted_features": drift_results.get("num_drifted_features", 0),
            }

            drift_summary_ref = self._save_drift_artifacts(drift_results, run_timestamp)
            results["drift_metrics"] = drift_summary
            results["drift_summary_ref"] = drift_summary_ref

        except Exception as e:
            logger.exception(f"Drift detection failed in monitoring job: {e}")
            drift_summary = {"status": "error", "error": str(e)}
            results["drift_metrics"] = drift_summary

        # --- WRITE TO DATABASE ---
        self._write_to_database(
            timestamp=run_timestamp,
            lookback_hours=lookback_hours,
            num_predictions=len(predictions),
            proxy_metrics=proxy_results,
            drift_summary=drift_summary,
            drift_summary_ref=drift_summary_ref,
        )

        return results

    def _save_drift_artifacts(
        self, drift_results: Dict[str, Any], timestamp: datetime = None
    ) -> Any:
        """
        Extract or ensure drift artifact reference from drift detection results.
        """
        return drift_results.get("drift_summary_ref")

    def _write_to_database(
        self,
        timestamp: datetime,
        lookback_hours: int,
        num_predictions: int,
        proxy_metrics: Dict[str, Any],
        drift_summary: Dict[str, Any],
        drift_summary_ref: Any = None,
    ):
        overall_stats = proxy_metrics.get("overall_stats", {})

        self.metrics_repo.insert(
            timestamp=timestamp,
            lookback_hours=lookback_hours,
            num_predictions=num_predictions,
            proxy_metrics={
                "positive_rate": overall_stats.get("positive_rate"),
                "probability_mean": overall_stats.get("probability_mean"),
                "probability_std": overall_stats.get("probability_std"),
                "entropy": proxy_metrics.get("entropy"),
            },
            drift_summary={
                "dataset_drift_detected": drift_summary.get("dataset_drift_detected", False),
                "feature_drift_ratio": drift_summary.get("feature_drift_ratio", 0.0),
                "num_drifted_features": drift_summary.get("num_drifted_features", 0),
            },
            drift_summary_ref=drift_summary_ref,
        )


_monitoring_job_instance: MonitoringJob | None = None


def run_monitoring_job(lookback_hours: int = 24) -> Dict[str, Any]:
    """Module-level entrypoint used by the scheduler service."""
    global _monitoring_job_instance
    if _monitoring_job_instance is None:
        _monitoring_job_instance = MonitoringJob()
    return _monitoring_job_instance.run_monitoring_job(lookback_hours=lookback_hours)
