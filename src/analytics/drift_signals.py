"""
Drift signal checker for retraining triggers.

Reads drift reports from Phase 3 monitoring and determines if retraining is needed.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Any
import logging

logger = logging.getLogger(__name__)


class DriftSignalChecker:
    """
    Check if drift signals warrant retraining.
    """

    def __init__(
        self,
        reports_path: str = "/app/monitoring/reports/drift_reports",
        drift_threshold: float = 0.3,
        lookback_hours: int = 24,
    ):
        self.reports_path = Path(reports_path)
        self.drift_threshold = drift_threshold
        self.lookback_hours = lookback_hours

    def check_drift_signals(self) -> Tuple[bool, Dict[str, Any]]:
        logger.info("=" * 80)
        logger.info("CHECKING DRIFT SIGNALS")
        logger.info("=" * 80)

        if not self.reports_path.exists():
            return False, {"status": "no_reports"}

        recent_reports = self._get_recent_reports()

        if len(recent_reports) == 0:
            return False, {"status": "no_recent_reports"}

        drift_detected = False
        drift_details: Dict[str, Any] = {
            "num_reports_checked": len(recent_reports),
            "drift_threshold": self.drift_threshold,
            "drift_share": 0.0,
            "drifted_feature_names": [],
            "num_drifted_features": 0,
            "drift_timestamps": [],
        }

        for report in recent_reports:
            drift_score, drifted_features = self._parse_drift_report(report)

            if drift_score > drift_details["drift_share"]:
                drift_details["drift_share"] = drift_score

            if drift_score >= self.drift_threshold and len(drifted_features) > 0:
                drift_detected = True
                drift_details["drifted_feature_names"].extend(drifted_features)
                drift_details["drift_timestamps"].append(report["timestamp"])

        drift_details["drifted_feature_names"] = list(set(drift_details["drifted_feature_names"]))
        drift_details["num_drifted_features"] = len(drift_details["drifted_feature_names"])

        return drift_detected, drift_details

    def _get_recent_reports(self) -> List[Dict[str, Any]]:
        cutoff_time = datetime.now() - timedelta(hours=self.lookback_hours)
        recent_reports: List[Dict[str, Any]] = []

        all_files = list(self.reports_path.glob("drift_summary_*.json"))

        for report_file in all_files:
            try:
                timestamp_str = report_file.stem.replace("drift_summary_", "")
                date_part = timestamp_str[:8]
                time_part = timestamp_str[9:]
                formatted_timestamp = (
                    f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} "
                    f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                )
                report_time = datetime.strptime(formatted_timestamp, "%Y-%m-%d %H:%M:%S")

                if report_time < cutoff_time:
                    continue

                with open(report_file, "r") as f:
                    report_data = json.load(f)

                recent_reports.append(
                    {
                        "timestamp": timestamp_str,
                        "report_time": report_time,
                        "filepath": str(report_file),
                        "data": report_data,
                    }
                )

            except Exception as e:
                logger.warning(f"Could not parse {report_file.name}: {e}")

        recent_reports = sorted(recent_reports, key=lambda x: x["report_time"], reverse=True)
        return recent_reports

    def _parse_drift_report(self, report: Dict[str, Any]) -> Tuple[float, List[str]]:
        try:
            report_data = report["data"]

            drift_score = 0.0
            drifted_features: List[str] = []

            if "drift_share" in report_data:
                drift_score = float(report_data.get("drift_share", 0.0))

                if "features" in report_data:
                    features_array = report_data["features"]
                    if isinstance(features_array, list):
                        drifted_features = [
                            feat["feature"]
                            for feat in features_array
                            if isinstance(feat, dict) and feat.get("drift_detected", False)
                        ]

            # BACKWARD COMPATIBILITY
            if len(drifted_features) == 0 and "feature_drift_details" in report_data:
                old_details = report_data["feature_drift_details"]
                if isinstance(old_details, list):
                    drifted_features = [
                        str(f.get("feature", ""))
                        for f in old_details
                        if isinstance(f, dict) and f.get("drift_detected", False)
                    ]

            # SAFE CASTING: prevents mypy set() errors
            if len(drifted_features) == 0 and "drifted_features" in report_data:
                raw = report_data["drifted_features"]
                if isinstance(raw, list):
                    drifted_features = [str(x) for x in raw]

            return drift_score, drifted_features

        except Exception:
            return 0.0, []

    def get_drift_summary(self) -> Dict[str, Any]:
        should_retrain, details = self.check_drift_signals()
        return {
            "retraining_recommended": should_retrain,
            "drift_details": details,
            "threshold": self.drift_threshold,
            "lookback_hours": self.lookback_hours,
        }
