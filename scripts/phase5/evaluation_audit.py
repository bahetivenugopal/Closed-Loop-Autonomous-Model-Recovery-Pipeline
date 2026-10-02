"""
Phase 5 evaluation audit.

Collects interview-ready evidence after demo runs:
- prediction and label volumes
- label coverage
- latest drift summary details
- drift injection event counts
- retraining decision counts
- current production model and recent metrics from MLflow
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any

import mlflow
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> None:
    logger.info("=" * 80)
    logger.info("PHASE 5 EVALUATION AUDIT")
    logger.info("=" * 80)

    from src.storage.repositories import PredictionsRepository, LabelsRepository

    pred_repo = PredictionsRepository()
    label_repo = LabelsRepository()

    total_predictions = pred_repo.count()
    coverage_stats = label_repo.get_coverage_stats()
    total_labels = coverage_stats.get("labeled_predictions", 0)
    coverage_pct = coverage_stats.get("coverage_pct", 0.0)

    logger.info("\n[1/6] Data volumes")
    logger.info("Predictions: %s", total_predictions)
    logger.info("Labels: %s", total_labels)
    logger.info("Coverage: %.1f%%", coverage_pct)

    logger.info("\n[2/6] Drift summaries")
    drift_summaries = sorted(summaries_dir.glob("drift_summary_*.json"))
    logger.info("Drift summary files: %s", len(drift_summaries))

    latest_summary = _safe_read_json(drift_summaries[-1]) if drift_summaries else {}
    if latest_summary:
        logger.info(
            "Latest drift_share: %.3f | num_drifted_features: %s/%s",
            float(latest_summary.get("drift_share", 0.0)),
            latest_summary.get("num_drifted_features", 0),
            latest_summary.get("num_features_evaluated", "?"),
        )

    logger.info("\n[3/6] Drift injections")
    drift_log = _safe_read_json(drift_log_path)
    events = drift_log.get("drift_events", []) if isinstance(drift_log, dict) else []
    logger.info("Drift injection events logged: %s", len(events))

    logger.info("\n[4/6] Retraining decisions")
    decision_files = sorted(decisions_dir.glob("decision_*.json"))
    promotions = 0
    rejections = 0
    failures = 0

    for decision_file in decision_files:
        decision = _safe_read_json(decision_file)
        action = decision.get("action")
        if action == "promote":
            promotions += 1
        elif action == "reject":
            rejections += 1
        elif action == "promote_failed":
            failures += 1

    logger.info("Decision files: %s", len(decision_files))
    logger.info("Promotions: %s | Rejections: %s | Promotion failures: %s", promotions, rejections, failures)

    logger.info("\n[5/6] Current production model")
    import os

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("credit-risk-model", stages=["Production"])

    if versions:
        production = versions[0]
        logger.info("Production version: v%s", production.version)
        logger.info("Production run_id: %s", production.run_id)

        run = client.get_run(production.run_id)
        metrics = run.data.metrics
        interesting_keys = [
            "diagnostic_accuracy",
            "diagnostic_precision",
            "diagnostic_recall",
            "diagnostic_f1_score",
            "diagnostic_roc_auc",
            "diagnostic_brier_score",
            "test_accuracy",
            "test_f1",
            "test_roc_auc",
            "test_brier_score",
        ]

        for key in interesting_keys:
            if key in metrics:
                logger.info("%s: %.6f", key, float(metrics[key]))
    else:
        logger.warning("No model in Production stage.")

    logger.info("\n[6/6] Interview summary")
    logger.info(
        "Simulation evidence ready: coverage %.1f%%, drift files %s, decision files %s.",
        coverage_pct,
        len(drift_summaries),
        len(decision_files),
    )

    logger.info("=" * 80)
    logger.info("EVALUATION AUDIT COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
