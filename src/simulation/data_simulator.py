"""
Data simulation service.

Feeds data row-by-row to the API, simulating real-time traffic.
Supports controlled drift injection at scheduled times.
"""

import pandas as pd
import requests
import time
from typing import Dict, Optional, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataSimulator:
    """
    Simulate real-time data stream to API.

    Features:
    - Row-by-row prediction requests
    - Configurable delay between requests
    - Drift injection at scheduled points
    - Progress tracking
    """

    def __init__(self, api_url: str = "http://localhost:8000/predict", delay_seconds: float = 0.1):
        """
        Initialize data simulator.

        Args:
            api_url: API endpoint
            delay_seconds: Delay between requests (simulates traffic rate)
        """
        self.api_url = api_url
        self.delay_seconds = delay_seconds

    def simulate_traffic(
        self, data: pd.DataFrame, num_samples: int = None, log_progress_every: int = 50
    ) -> Dict:
        """
        Send data row-by-row to API.

        Args:
            data: DataFrame to simulate
            num_samples: Number of samples to send (None = all)
            log_progress_every: Log progress every N samples

        Returns:
            Simulation statistics
        """
        if num_samples is None:
            num_samples = len(data)

        sample_data = data.head(num_samples)

        stats: Dict[str, Any] = {
            "total_samples": num_samples,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "start_time": datetime.now().isoformat(),
            "prediction_ids": [],
        }

        logger.info("=" * 80)
        logger.info(f"SIMULATING TRAFFIC: {num_samples} samples")
        logger.info(f"API: {self.api_url}")
        logger.info(f"Delay: {self.delay_seconds}s between requests")
        logger.info("=" * 80)

        for idx, (_, row) in enumerate(sample_data.iterrows()):
            # Create payload
            payload = self._row_to_payload(row)

            # Make prediction
            try:
                response = requests.post(self.api_url, json=payload, timeout=5)

                if response.status_code == 200:
                    result = response.json()
                    stats["successful_predictions"] += 1
                    stats["prediction_ids"].append(result.get("prediction_id"))
                else:
                    stats["failed_predictions"] += 1
                    logger.warning(f"Request failed: HTTP {response.status_code}")

            except Exception as e:
                stats["failed_predictions"] += 1
                logger.error(f"Request error: {e}")

            # Progress logging
            if (idx + 1) % log_progress_every == 0:
                logger.info(
                    f"Progress: {idx + 1}/{num_samples} ({(idx + 1) / num_samples * 100:.1f}%)"
                )

            # Rate limiting
            time.sleep(self.delay_seconds)

        stats["end_time"] = datetime.now().isoformat()

        logger.info("=" * 80)
        logger.info("SIMULATION COMPLETE")
        logger.info(f"  Successful: {stats['successful_predictions']}")
        logger.info(f"  Failed: {stats['failed_predictions']}")
        logger.info(f"  Success rate: {stats['successful_predictions'] / num_samples * 100:.1f}%")
        logger.info("=" * 80)

        return stats

    def _row_to_payload(self, row: pd.Series) -> Dict:
        """
        Convert dataframe row to API payload.
        """
        row_dict = row.to_dict()

        def safe_float(val):
            if pd.isna(val):
                return 0.0
            return float(val)

        def safe_int(val):
            if pd.isna(val):
                return 0
            return int(val)

        payload = {
            "RevolvingUtilizationOfUnsecuredLines": safe_float(
                row_dict.get("RevolvingUtilizationOfUnsecuredLines", 0.0)
            ),
            "age": safe_int(row_dict.get("age", 0)),
            "NumberOfTime30_59DaysPastDueNotWorse": safe_int(
                row_dict.get("NumberOfTime30_59DaysPastDueNotWorse")
                or row_dict.get("NumberOfTime30-59DaysPastDueNotWorse", 0)
            ),
            "DebtRatio": safe_float(row_dict.get("DebtRatio", 0.0)),
            "MonthlyIncome": safe_float(row_dict.get("MonthlyIncome", 0.0)),
            "NumberOfOpenCreditLinesAndLoans": safe_int(
                row_dict.get("NumberOfOpenCreditLinesAndLoans", 0)
            ),
            "NumberOfTimes90DaysLate": safe_int(row_dict.get("NumberOfTimes90DaysLate", 0)),
            "NumberRealEstateLoansOrLines": safe_int(
                row_dict.get("NumberRealEstateLoansOrLines", 0)
            ),
            "NumberOfTime60_89DaysPastDueNotWorse": safe_int(
                row_dict.get("NumberOfTime60_89DaysPastDueNotWorse")
                or row_dict.get("NumberOfTime60-89DaysPastDueNotWorse", 0)
            ),
            "NumberOfDependents": safe_int(row_dict.get("NumberOfDependents", 0)),
        }

        return payload
