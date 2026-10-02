"""
Simple scheduler for monitoring jobs.

MINIMAL CHANGES: Already a dumb scheduler, just verify DB connection at startup.
"""

import time
import logging
from datetime import datetime
from typing import Callable
import sys
import signal
from src.monitoring.monitoring_job import run_monitoring_job
from src.storage.db_manager import get_db_manager  # ✅ NEW

sys.path.append("/app")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleScheduler:
    """
    Minimal scheduler for running monitoring jobs at fixed intervals.

    ✅ NEW: Verifies database connection at startup
    """

    def __init__(self, interval_seconds: int, job_function: Callable, lookback_hours: int = 24):
        self.interval_seconds = interval_seconds
        self.job_function = job_function
        self.lookback_hours = lookback_hours
        self.running = False

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # ✅ NEW: Verify database connection
        self._verify_database()

        logger.info(f"Scheduler initialized: interval={interval_seconds}s")

    def _verify_database(self):
        """Verify database connection at startup."""
        try:
            db = get_db_manager()
            db.execute_query("SELECT 1")
            logger.info("✅ Database connection verified")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            logger.warning("Monitoring will continue but database writes may fail")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def run_once(self) -> dict:
        """Execute job once."""
        logger.info(f"Executing scheduled job: {datetime.now().isoformat()}")

        try:
            results = self.job_function(lookback_hours=self.lookback_hours)

            status = results.get("status", "unknown")
            logger.info(f"Job completed with status: {status}")

            return results

        except Exception as e:
            logger.error(f"Job execution failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    def run_forever(self):
        """Run job in an infinite loop at fixed intervals."""
        logger.info("=" * 70)
        logger.info("SCHEDULER STARTED")
        logger.info(f"Interval: {self.interval_seconds}s")
        logger.info(f"Lookback window: {self.lookback_hours}h")
        logger.info("=" * 70)

        self.running = True
        iteration = 0

        while self.running:
            iteration += 1
            logger.info(f"\n--- Iteration {iteration} ---")

            # Run job
            self.run_once()

            if not self.running:
                break

            # Sleep until next execution
            logger.info(f"Sleeping for {self.interval_seconds}s until next run...")

            # Sleep in small chunks to allow responsive shutdown
            elapsed = 0
            sleep_chunk = 10  # seconds
            while elapsed < self.interval_seconds and self.running:
                time.sleep(min(sleep_chunk, self.interval_seconds - elapsed))
                elapsed += sleep_chunk

        logger.info("\n" + "=" * 70)
        logger.info("SCHEDULER STOPPED")
        logger.info("=" * 70)


def main():
    """Main entry point for monitoring scheduler."""
    import os

    # Read configuration
    interval = int(os.getenv("MONITORING_INTERVAL", "300"))  # Default: 5 minutes
    lookback = int(os.getenv("MONITORING_LOOKBACK", "24"))  # Default: 24 hours

    # Validate configuration
    if interval < 60:
        logger.warning(
            f"Monitoring interval ({interval}s) is very short. "
            f"Consider at least 60s to avoid excessive load."
        )

    if lookback < 1:
        logger.error(f"Invalid lookback hours: {lookback}")
        return

    # Create and run scheduler
    scheduler = SimpleScheduler(
        interval_seconds=interval, job_function=run_monitoring_job, lookback_hours=lookback
    )

    try:
        scheduler.run_forever()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Scheduler failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
