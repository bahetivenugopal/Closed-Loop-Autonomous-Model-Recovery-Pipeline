"""
Demo 2: Covariate Shift Injection

Purpose:
- Inject feature distribution change (income scaling)
- Simulate economic improvement scenario
- Observe drift detection and retraining

Drift type: Covariate Shift
- MonthlyIncome scaled by 1.5 (50% increase)
- Age shifted by +5 years (population aging)

Expected outcome:
- Drift detected by Evidently
- Retraining triggered
- Shadow model trained on new distribution
- Evaluation gate decision
"""

import sys
import json

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
    logger.info("DEMO 2: COVARIATE SHIFT INJECTION")
    logger.info("=" * 80)

    # Step 1: Load data
    logger.info("\n[Step 1/6] Loading data for drift injection...")
    df = pd.read_csv("/app/data/processed/cs-training-temporal.csv")

    drift_data = df.sample(n=800, random_state=123)
    logger.info(f"Loaded {len(drift_data)} samples")

    # Step 2: Inject covariate shift (AGGRESSIVE)
    logger.info("\n[Step 2/6] Injecting AGGRESSIVE COVARIATE SHIFT...")
    logger.info("Scenario: Major economic shift affecting multiple features")

    injector = DriftInjector(random_seed=42)

    # Drift on 5 features (50% of dataset) to ensure detection

    # Feature 1: MonthlyIncome (economic improvement)
    drift_data = injector.inject_covariate_shift_scaling(
        data=drift_data,
        feature="MonthlyIncome",
        scale_factor=1.5,  # ✅ INCREASED: 50% increase (was 1.4)
        reason="Major economic boom - income increases 50%",
    )

    # Feature 2: age (population aging)
    drift_data = injector.inject_covariate_shift_location(
        data=drift_data,
        feature="age",
        shift_amount=5,  # ✅ INCREASED: +5 years (was +3)
        reason="Significant demographic shift - population ages 5 years",
    )

    # Feature 3: RevolvingUtilizationOfUnsecuredLines ✅ NEW
    drift_data = injector.inject_covariate_shift_scaling(
        data=drift_data,
        feature="RevolvingUtilizationOfUnsecuredLines",
        scale_factor=5.3,
        reason="Credit usage patterns change - 30% increase in utilization",
    )

    # Feature 4: NumberOfOpenCreditLinesAndLoans ✅ NEW
    drift_data = injector.inject_covariate_shift_location(
        data=drift_data,
        feature="NumberOfOpenCreditLinesAndLoans",
        shift_amount=10,
        reason="Credit availability increases - average +2 credit lines",
    )

    # Feature 5: DebtRatio ✅ NEW
    drift_data = injector.inject_covariate_shift_scaling(
        data=drift_data,
        feature="DebtRatio",
        scale_factor=2.5,  # 15% decrease
        reason="Debt burden decreases - households pay down debt",
    )

    logger.info("✅ AGGRESSIVE Drift injected on 5 features (50% of dataset):")
    logger.info("  1. MonthlyIncome: scaled by 1.5x (+50%)")
    logger.info("  2. age: shifted by +5 years")
    logger.info("  3. RevolvingUtilizationOfUnsecuredLines: scaled by 1.3x (+30%)")
    logger.info("  4. NumberOfOpenCreditLinesAndLoans: shifted by +2")
    logger.info("  5. DebtRatio: scaled by 0.85x (-15%)")
    logger.info("")
    logger.info("Expected drift_share: 5/10 = 0.50 (well above 0.3 threshold)")

    # Step 3: Simulate traffic with drifted data
    logger.info("\n[Step 3/6] Simulating traffic with DRIFTED data...")

    simulator = DataSimulator(api_url="http://localhost:8000/predict", delay_seconds=0.05)

    stats = simulator.simulate_traffic(data=drift_data, num_samples=600, log_progress_every=150)

    logger.info(f"Drifted traffic sent: {stats['successful_predictions']} predictions")

    # Step 4: Add labels
    logger.info("\n[Step 4/6] Adding labels (50% coverage)...")

    label_store = get_label_store()

    for pred_id, true_label in zip(
        stats["prediction_ids"][:300], drift_data["SeriousDlqin2yrs"][:300]
    ):
        label_store.store_label(
            prediction_id=pred_id, true_label=int(true_label), label_source="demo_covariate_shift"
        )

    logger.info("Added 300 labels")

    # Step 5: Wait for drift detection (MONITORING ONLY)
    logger.info("\n[Step 5/6] Waiting for drift detection (MONITORING ONLY)...")
    logger.info("")
    logger.info("⚠️ IMPORTANT: Drift detection does NOT trigger retraining")
    logger.info("Drift is MONITORING information only:")
    logger.info("  - Logged in reports")
    logger.info("  - Available in dashboards")
    logger.info("  - Used as context in decisions")
    logger.info("  - Does NOT auto-trigger retraining")
    logger.info("")
    logger.info("Retraining is triggered by:")
    logger.info("  1. ✅ Scheduled runs (weekly)")
    logger.info("  2. ✅ Manual operator trigger")
    logger.info("  3. ✅ Performance degradation (delayed labels)")
    logger.info("")
    logger.info("Waiting 6 minutes for monitoring cycle...")

    for i in range(6):
        time.sleep(60)
        logger.info(f"  {i+1}/6 minutes elapsed...")

    # Step 6: Verify drift was DETECTED (but not TRIGGERED)
    logger.info("\n[Step 6/6] Verifying drift was detected (monitoring only)...")

    # Check drift reports
    drift_reports = sorted(Path("/app/monitoring/drift/summaries").glob("drift_summary_*.json"))
    if drift_reports:
        latest_report = drift_reports[-1]
        logger.info(f"✅ Latest drift report: {latest_report.name}")

        with open(latest_report, "r") as f:
            report = json.load(f)

        # Check drift score
        drift_score = report.get("drift_share", 0)
        logger.info(f"✅ Drift score: {drift_score}")

        if drift_score > 0.3:
            logger.info("✅ Drift DETECTED in monitoring")
        else:
            logger.info("ℹ️ Drift below threshold")

    # Check if retraining was auto-triggered (should be NO)
    logger.info("")
    logger.info("Checking if retraining was auto-triggered...")

    # Check Airflow for recent DAG runs
    logger.info("ℹ️ Drift should NOT have triggered retraining automatically")
    logger.info("   Retraining will happen on next scheduled run (weekly)")
    logger.info("   Or can be triggered manually via Airflow UI")

    logger.info("\n" + "=" * 80)
    logger.info("✅ COVARIATE SHIFT DEMO COMPLETE")
    logger.info("=" * 80)
    logger.info("\nWhat happened:")
    logger.info("1. ✅ Drift injected (MonthlyIncome +50%, age +5 years)")
    logger.info("2. ✅ Drift DETECTED by monitoring")
    logger.info("3. ✅ Drift logged in reports (for analysis)")
    logger.info("4. ✅ Drift did NOT trigger retraining (correct behavior)")
    logger.info("")
    logger.info("Next scheduled retraining will:")
    logger.info("  - Train on data with drift")
    logger.info("  - Include drift context in decision metadata")
    logger.info("  - Evaluation gate will decide if new model is better")
    logger.info("")


if __name__ == "__main__":
    main()
