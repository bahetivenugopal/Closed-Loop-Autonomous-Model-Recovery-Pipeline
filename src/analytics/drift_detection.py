"""
Drift detection using Evidently AI (v0.4.15).

CRITICAL FRAMING:
Evidently tells us IF distributions changed.
Evidently does NOT tell us:
- Whether change is bad
- Whether we should retrain
- Whether accuracy degraded (no labels here)

DRIFT ≠ MODEL FAILURE

The decision of "what to do about drift" belongs to Phase 4.
This module is Phase 3 ONLY: observation + statistics.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
from evidently.pipeline.column_mapping import ColumnMapping

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Evidently 0.4.15–compatible drift detector.

    DESIGN PRINCIPLES:
    - Explicit over implicit
    - Stable schema over convenience
    - Statistics only (no decisions)
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_columns: List[str],
        numerical_features: List[str],
        categorical_features: Optional[List[str]] = None,
        reference_metadata: Optional[Dict] = None,
    ):
        """
        Initialize with FROZEN reference data.

        Args:
            reference_data: Immutable reference dataset
            feature_columns: All monitored feature names
            numerical_features: Explicit numeric features
            categorical_features: Explicit categorical features
            reference_metadata: Optional metadata (creation time, source, etc.)
        """

        self.feature_columns = feature_columns
        self.reference_data = reference_data[feature_columns].copy()
        self.reference_metadata = reference_metadata or {}

        self.column_mapping = ColumnMapping(
            numerical_features=numerical_features,
            categorical_features=categorical_features or [],
        )

        logger.info(
            f"DriftDetector initialized | "
            f"samples={len(self.reference_data)} | "
            f"features={len(self.feature_columns)}"
        )

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        save_report: bool = True,
        report_dir: str = "/app/monitoring/reports/drift_reports",
    ) -> Dict:
        """
        Run drift detection.

        Returns:
            Structured statistical drift summary (machine-readable).
        """

        if current_data.empty:
            logger.warning("No current data provided for drift detection")
            return {"status": "no_data"}

        # Filter both reference and current to ONLY feature columns
        # Skip: Unnamed: 0 (index), SeriousDlqin2yrs (target/label)
        # Keep: Only the 10 input features
        reference_features = self.reference_data[self.feature_columns].copy()
        current_features = current_data[self.feature_columns].copy()

        report = Report(
            metrics=[
                DatasetDriftMetric(),
                *[ColumnDriftMetric(column_name=col) for col in self.feature_columns],
            ]
        )

        try:
            report.run(
                reference_data=reference_features,
                current_data=current_features,
                column_mapping=self.column_mapping,
            )
        except Exception as e:
            logger.exception("Evidently drift detection failed")
            return {"status": "error", "error": str(e)}

        report_dict = report.as_dict()
        drift_summary = self._parse_dataset_drift(report_dict)

        if save_report:
            self._save_outputs(report, drift_summary, report_dir)

        return drift_summary

    def _parse_dataset_drift(self, report_dict: Dict) -> Dict:
        """
        Parse DatasetDriftMetric and ColumnDriftMetric output for Evidently v0.4.15.

        CRITICAL: Ensure we extract ALL required fields from Evidently output.
        """

        logger.info("=" * 80)
        logger.info("PARSING DATASET DRIFT METRIC")
        logger.info("=" * 80)

        metrics = report_dict.get("metrics", [])
        dataset_metric = None
        column_metrics = {}

        # Extract DatasetDriftMetric
        for metric in metrics:
            if metric.get("metric") == "DatasetDriftMetric":
                dataset_metric = metric.get("result")
                break

        if dataset_metric is None:
            logger.error("DatasetDriftMetric not found in Evidently output!")
            logger.error(f"Available metrics: {[m.get('metric') for m in metrics]}")
            return {
                "status": "parse_error",
                "reason": "DatasetDriftMetric not found in Evidently output",
            }

        logger.info("Found DatasetDriftMetric")
        logger.info(f"Keys in dataset_metric: {list(dataset_metric.keys())}")

        # Extract ColumnDriftMetric for each feature
        for metric in metrics:
            if metric.get("metric") == "ColumnDriftMetric":
                result = metric.get("result", {})
                column_name = result.get("column_name")
                if column_name:
                    column_metrics[column_name] = result

        logger.info(f"Found {len(column_metrics)} ColumnDriftMetric results")

        # Determine drifted features
        drifted_features = [
            col for col, info in column_metrics.items() if info.get("drift_detected", False)
        ]

        # Calculate excluded features (in total but not in evaluation)
        total_features = len(self.feature_columns)
        evaluated_features = len(column_metrics)
        excluded_features = [f for f in self.feature_columns if f not in column_metrics]
        num_excluded = len(excluded_features)

        logger.info("Feature analysis:")
        logger.info(f"  Total features: {total_features}")
        logger.info(f"  Evaluated: {evaluated_features}")
        logger.info(f"  Excluded: {num_excluded}")
        logger.info(f"  Drifted: {len(drifted_features)}")

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "reference_metadata": self.reference_metadata,
            # Dataset-level statistics
            "dataset_drift_detected": dataset_metric.get("dataset_drift", False),
            "drift_share": dataset_metric.get("drift_share", 0.0),
            "num_drifted_features": len(drifted_features),
            "num_features_total": total_features,
            "num_features_evaluated": evaluated_features,
            "num_features_excluded": num_excluded,
            "excluded_features": excluded_features,
            # Feature-level statistics
            "features": [],
        }

        # Extract feature-level drift details
        for feature in self.feature_columns:
            if feature in column_metrics:
                col_result = column_metrics[feature]
                summary["features"].append(
                    {
                        "feature": feature,
                        "drift_detected": col_result.get("drift_detected", False),
                        "stat_test": col_result.get("stattest_name", "unknown"),
                        "p_value": col_result.get("p_value", None),
                        "threshold": col_result.get("stattest_threshold", None),
                        "drift_score": col_result.get("drift_score", None),
                    }
                )

        logger.info(f"Summary structure keys: {list(summary.keys())}")
        logger.info(f"Features array length: {len(summary['features'])}")

        logger.info(
            f"Drift summary | "
            f"dataset_drift={summary['dataset_drift_detected']} | "
            f"drift_share={summary['drift_share']:.2%} | "
            f"drifted={summary['num_drifted_features']}/"
            f"{summary['num_features_evaluated']} evaluated "
            f"({summary['num_features_excluded']} excluded)"
        )

        logger.info("=" * 80)

        return summary

    def _save_outputs(self, report: Report, summary: Dict, report_dir: str) -> None:
        """
        Save Evidently HTML report and JSON summary.
        """

        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        html_path = report_path / f"drift_report_{timestamp}.html"
        report.save_html(str(html_path))

        json_path = report_path / f"drift_summary_{timestamp}.json"
        summary["drift_summary_ref"] = str(json_path)
        summary["drift_report_ref"] = str(html_path)

        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved drift report: {html_path}")
        logger.info(f"Saved drift summary: {json_path}")


def load_reference_data(
    reference_dir: str = "/app/monitoring/reference",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load frozen reference data and metadata.

    Raises if reference data does not exist.
    """

    reference_path = Path(reference_dir)
    reference_file = reference_path / "reference_data.csv"
    metadata_file = reference_path / "reference_metadata.json"

    if not reference_file.exists():
        raise FileNotFoundError(
            f"Reference data not found at {reference_file}. " f"Bootstrap reference data first."
        )

    df = pd.read_csv(reference_file)

    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)

    logger.info(f"Loaded reference data | samples={len(df)}")
    return df, metadata
