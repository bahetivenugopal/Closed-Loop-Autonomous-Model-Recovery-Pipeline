"""
Drift injection event logger.

Tracks all intentional drift injections for reproducibility and analysis.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class DriftLogger:
    """
    Log all drift injection events.

    Enables:
    - Reproducibility (know exactly what drift was injected when)
    - Analysis (correlate drift with system response)
    - Debugging (understand why retraining was triggered)
    """

    def __init__(self, log_path: str = "/app/monitoring/drift_injections/drift_log.json"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.log_path.exists():
            self._initialize_log()

    def _initialize_log(self):
        """Create empty drift log."""
        with open(self.log_path, "w") as f:
            json.dump({"drift_events": []}, f, indent=2)
        logger.info(f"Initialized drift log: {self.log_path}")

    def log_drift_event(
        self,
        drift_type: str,
        affected_features: List[str],
        magnitude: float,
        reason: str,
        metadata: Dict = None,
    ):
        """
        Log a drift injection event.

        Args:
            drift_type: Type of drift (covariate_shift, population_shift, concept_drift)
            affected_features: Features modified
            magnitude: Strength of drift (e.g., 1.3 for 30% increase)
            reason: Why this drift was injected (e.g., "Simulated economic improvement")
            metadata: Additional context
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "drift_type": drift_type,
            "affected_features": affected_features,
            "magnitude": magnitude,
            "reason": reason,
            "metadata": metadata or {},
        }

        # Load existing log
        with open(self.log_path, "r") as f:
            log_data = json.load(f)

        # Append event
        log_data["drift_events"].append(event)

        # Save
        with open(self.log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Logged drift event: {drift_type} on {affected_features}")

    def get_recent_events(self, hours: int = 24) -> List[Dict]:
        """Get drift events from last N hours."""
        if not self.log_path.exists():
            return []

        with open(self.log_path, "r") as f:
            log_data = json.load(f)

        events = log_data.get("drift_events", [])

        # Filter by time
        cutoff = datetime.now().timestamp() - (hours * 3600)
        recent = [e for e in events if datetime.fromisoformat(e["timestamp"]).timestamp() >= cutoff]

        return recent

    def get_all_events(self) -> List[Dict]:
        """Get all drift events."""
        if not self.log_path.exists():
            return []

        with open(self.log_path, "r") as f:
            log_data = json.load(f)

        return log_data.get("drift_events", [])


# Singleton
_drift_logger_instance = None


def get_drift_logger() -> DriftLogger:
    """Get or create singleton drift logger."""
    global _drift_logger_instance
    if _drift_logger_instance is None:
        _drift_logger_instance = DriftLogger()
    return _drift_logger_instance
