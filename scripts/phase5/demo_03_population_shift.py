"""
Demo 3: Population Shift Injection

Purpose:
- Inject class balance change
- Simulate recession scenario (more defaults)
- Observe model performance degradation

Drift type: Population Shift
- Default rate increases from ~7% to 20%

Expected outcome:
- Class imbalance changes
- Model recalibration needed
- Retraining triggered
"""

import sys

sys.path.append("/app")

import pandas as pd
from src.simulation.data_simulator import DataSimulator
from src.simulation.drift_injector import DriftInjector
from src.storage.label_store import get_label_store
from pathlib import Path
import time
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("DEMO 3: POPULATION SHIFT INJECTION")
    logger.info("=" * 80)

    # Step 1: Load data
    logger.info("\n[Step 1/6] Loading data...")
    df = pd.read_csv("/app/data/processed/cs-training-temporal.csv")

    drift_data = df.sample(n=700, random_state=46)

    original_default_rate = drift_data["SeriousDlqin2yrs"].mean()
    logger.info(f"Loaded {len(drift_data)} samples")
    logger.info(f"Original default rate: {original_default_rate*100:.1f}%")

    # Step 2: Inject population shift
    logger.info("\n[Step 2/6] Injecting POPULATION SHIFT...")
    logger.info("Scenario: Economic recession - default rate increases")

    injector = DriftInjector(random_seed=42)

    drift_data = injector.inject_population_shift(
        data=drift_data,
        target_column="SeriousDlqin2yrs",
        new_positive_ratio=0.20,  # Increase to 20%
        reason="Simulated economic recession - default rate increases from 7% to 20%",
    )

    new_default_rate = drift_data["SeriousDlqin2yrs"].mean()
    logger.info(
        f"✅ Default rate changed: {original_default_rate*100:.1f}% → {new_default_rate*100:.1f}%"
    )

    # Step 3: Simulate traffic
    logger.info("\n[Step 3/6] Simulating traffic with SHIFTED population...")

    simulator = DataSimulator(api_url="http://localhost:8000/predict", delay_seconds=0.05)

    stats = simulator.simulate_traffic(data=drift_data, num_samples=700, log_progress_every=150)

    logger.info(f"Traffic sent: {stats['successful_predictions']} predictions")

    # Step 4: Add labels
    logger.info("\n[Step 4/6] Adding labels (55% coverage)...")

    label_store = get_label_store()

    for pred_id, true_label in zip(
        stats["prediction_ids"][:385], drift_data["SeriousDlqin2yrs"][:385]
    ):
        label_store.store_label(
            prediction_id=pred_id, true_label=int(true_label), label_source="demo_population_shift"
        )

    logger.info("Added 385 labels")

    # Step 5: Wait for monitoring
    logger.info("\n[Step 5/6] Waiting for drift detection...")
    logger.info("Population shift should be detected...")
    logger.info("Waiting 6 minutes...")

    for i in range(6):
        time.sleep(60)
        logger.info(f"  {i+1}/6 minutes elapsed...")

    # Step 6: Analyze results
    logger.info("\n[Step 6/6] Analyzing population shift impact...")

    # Check if model performance degraded
    from src.storage.prediction_logger import get_prediction_logger

    pred_logger = get_prediction_logger()

    predictions_df = pred_logger.get_predictions_with_features()
    labeled_df = label_store.get_labeled_predictions(predictions_df)

    if len(labeled_df) > 0:
        # Compare prediction distribution vs true labels
        pred_default_rate = labeled_df["prediction"].mean()
        true_default_rate = labeled_df["true_label"].mean()

        logger.info(f"Model predicted default rate: {pred_default_rate*100:.1f}%")
        logger.info(f"Actual default rate: {true_default_rate*100:.1f}%")
        logger.info(f"Calibration gap: {abs(pred_default_rate - true_default_rate)*100:.1f}%")

    logger.info("\n" + "=" * 80)
    logger.info("✅ POPULATION SHIFT DEMO COMPLETE")
    logger.info("=" * 80)
    logger.info("\nExpected results:")
    logger.info("1. ✅ Class balance changed (7% → 20% defaults)")
    logger.info("2. ✅ Model calibration degraded")
    logger.info("3. ⏳ Retraining triggered with new class distribution")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Check model calibration metrics")
    logger.info("2. Verify retraining was triggered")
    logger.info("3. Run Demo 4 for concept drift scenario")
    logger.info("")


if __name__ == "__main__":
    main()
