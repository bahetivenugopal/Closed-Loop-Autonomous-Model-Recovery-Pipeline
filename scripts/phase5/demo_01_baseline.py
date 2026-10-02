"""
Demo 1: Establish Baseline Performance

Purpose:
- Feed clean data to the system
- Establish baseline metrics
- Verify system is working normally before drift injection

Expected outcome:
- Predictions logged successfully
- No drift detected
- Monitoring shows stable performance
"""

import sys
from pathlib import Path

sys.path.append("/app")

import pandas as pd
from src.simulation.data_simulator import DataSimulator
from src.storage.prediction_logger import get_prediction_logger
from src.storage.label_store import get_label_store
import time
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("DEMO 1: BASELINE ESTABLISHMENT")
    logger.info("=" * 80)

    # Step 1: Load clean data
    logger.info("\n[Step 1/5] Loading clean baseline data...")
    df = pd.read_csv("/app/data/processed/cs-training-temporal.csv")

    # Use recent data (simulated as "current" traffic)
    baseline_data = df.sample(n=500, random_state=42)

    logger.info(f"Loaded {len(baseline_data)} baseline samples")
    logger.info(
        f"Target distribution: {baseline_data['SeriousDlqin2yrs'].value_counts().to_dict()}"
    )

    # Step 2: Simulate traffic
    logger.info("\n[Step 2/5] Simulating normal API traffic...")

    simulator = DataSimulator(
        api_url="http://localhost:8000/predict", delay_seconds=0.05  # 20 requests/second
    )

    stats = simulator.simulate_traffic(data=baseline_data, num_samples=500, log_progress_every=100)

    logger.info(f"Traffic simulation complete: {stats['successful_predictions']} successful")

    # Step 3: Add labels (simulate delayed arrival - 60% coverage)
    logger.info("\n[Step 3/5] Simulating delayed label arrival (60% coverage)...")

    label_store = get_label_store()

    # Randomly select 60% of predictions to label
    import random

    labeled_count = 0

    for pred_id, true_label in zip(
        stats["prediction_ids"][:300], baseline_data["SeriousDlqin2yrs"][:300]
    ):
        label_store.store_label(
            prediction_id=pred_id, true_label=int(true_label), label_source="demo_baseline"
        )
        labeled_count += 1

    logger.info(f"Added {labeled_count} labels (60% coverage)")

    # Step 4: Wait for monitoring to run
    logger.info("\n[Step 4/5] Waiting for monitoring to process...")
    logger.info("Monitoring runs every 5 minutes. Waiting 6 minutes...")

    for i in range(6):
        time.sleep(60)
        logger.info(f"  {i+1}/6 minutes elapsed...")

    # Step 5: Verify baseline metrics
    logger.info("\n[Step 5/5] Verifying baseline metrics...")

    pred_logger = get_prediction_logger()
    predictions_df = pred_logger.get_predictions_with_features()
    coverage = label_store.get_label_coverage(predictions_df)

    logger.info(f"Total predictions in system: {coverage['total_predictions']}")
    logger.info(f"Labeled predictions: {coverage['labeled_predictions']}")
    logger.info(f"Coverage: {coverage['coverage_rate']*100:.1f}%")

    # Check drift reports
    import os

    drift_reports = list(Path("/app/monitoring/drift/summaries").glob("drift_summary_*.json"))
    logger.info(f"Drift reports generated: {len(drift_reports)}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ BASELINE ESTABLISHED")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Check monitoring reports: ls -lh /app/monitoring/reports/drift_reports/")
    logger.info(
        "2. Verify no drift detected: cat /app/monitoring/reports/drift_reports/drift_summary_*.json | tail -1"
    )
    logger.info("3. Run Demo 2 to inject covariate shift")
    logger.info("")


if __name__ == "__main__":
    main()
