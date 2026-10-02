"""
Demo 4: Manual Retraining Trigger

Purpose:
- Demonstrate human-in-the-loop retraining
- Show operator can review drift and decide to retrain
- Test manual trigger workflow

Expected outcome:
- Operator reviews drift monitoring
- Decides to trigger retraining manually
- Retraining proceeds with drift context
"""

import sys

sys.path.append("/app")

from pathlib import Path
import logging
import json
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
        airflow_port = os.getenv("AIRFLOW_WEBSERVER_HOST_PORT", "8081")

    logger.info("=" * 80)
    logger.info("DEMO 4: MANUAL RETRAINING TRIGGER")
    logger.info("=" * 80)

    # Step 1: Review drift monitoring
    logger.info("\n[Step 1/4] Reviewing drift monitoring reports...")

    drift_reports = sorted(Path("/app/monitoring/reports/drift_reports").glob("*.json"))

    if len(drift_reports) > 0:
        logger.info(f"Found {len(drift_reports)} drift reports")

        # Show last 3 reports
        for report_file in drift_reports[-3:]:
            with open(report_file, "r") as f:
                report = json.load(f)

            drift_score = report.get("drift_share", 0)
            timestamp = report_file.stem.replace("drift_summary_", "")

            logger.info(f"  {timestamp}: drift_score = {drift_score:.3f}")

    # Step 2: Review drift injection log
    logger.info("\n[Step 2/4] Reviewing drift injection events...")

    drift_log_path = Path("/app/monitoring/drift_injections/drift_log.json")
    if drift_log_path.exists():
        with open(drift_log_path, "r") as f:
            drift_log = json.load(f)

        events = drift_log.get("drift_events", [])
        logger.info(f"Found {len(events)} drift injection events")

        # Show recent events
        for event in events[-3:]:
            logger.info(f"  {event['drift_type']}: {event['affected_features']}")
            logger.info(f"    Reason: {event['reason']}")

    # Step 3: Operator decision
    logger.info("\n[Step 3/4] Operator Decision Process...")
    logger.info("")
    logger.info("Based on drift monitoring, operator can decide:")
    logger.info("")
    logger.info("Option 1: WAIT (drift is temporary/seasonal)")
    logger.info("  - Continue monitoring")
    logger.info("  - Let scheduled retraining handle it")
    logger.info("")
    logger.info("Option 2: TRIGGER RETRAINING (drift is significant)")
    logger.info("  - Manually trigger via Airflow UI")
    logger.info("  - OR run manual workflow script")
    logger.info("")

    # Step 4: Show manual trigger methods
    logger.info("\n[Step 4/4] Manual Trigger Methods...")
    logger.info("")
    logger.info("Method 1: Airflow UI")
    logger.info(f"  1. Open http://localhost:{airflow_port}")
    logger.info("  2. Navigate to 'retraining_pipeline' DAG")
    logger.info("  3. Click 'Trigger DAG' (play button)")
    logger.info("")
    logger.info("Method 2: CLI")
    logger.info("  docker-compose exec airflow-scheduler airflow dags trigger retraining_pipeline")
    logger.info("")
    logger.info("Method 3: Airflow REST API (optional advanced)")
    logger.info("  Trigger via Airflow API endpoint with auth if your team standard uses API-driven ops")
    logger.info("")

    logger.info("\n" + "=" * 80)
    logger.info("✅ MANUAL TRIGGER DEMO COMPLETE")
    logger.info("=" * 80)
    logger.info("\nKey Takeaways:")
    logger.info("1. ✅ Drift is monitoring information (not automatic trigger)")
    logger.info("2. ✅ Operator reviews drift reports and decides")
    logger.info("3. ✅ Multiple ways to trigger retraining manually")
    logger.info("4. ✅ Human judgment in the loop (prevents over-retraining)")
    logger.info("")


if __name__ == "__main__":
    main()
