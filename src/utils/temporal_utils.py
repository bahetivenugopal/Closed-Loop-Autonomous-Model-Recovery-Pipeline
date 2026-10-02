"""
Temporal window utilities.

Enforces time-based train/eval splits to prevent data leakage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class TemporalWindows:
    """
    Manage temporal windows for training and evaluation.

    CRITICAL: Evaluation window must be AFTER training window.
    No overlap allowed.
    """

    @staticmethod
    def add_simulated_timestamps(
        df: pd.DataFrame,
        start_date: str = "2020-01-01",
        end_date: str = "2024-01-01",
        date_column: str = "application_date",
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Add simulated timestamps to dataset.

        Distributes rows uniformly across time period.

        Args:
            df: Input dataframe
            start_date: Start of time period
            end_date: End of time period
            date_column: Name of timestamp column

        Returns:
            DataFrame with timestamp column added
        """
        np.random.seed(random_seed)

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Generate random timestamps
        timestamps = pd.to_datetime(np.random.randint(start.value, end.value, size=len(df)))

        df = df.copy()
        df[date_column] = timestamps
        df = df.sort_values(date_column).reset_index(drop=True)

        logger.info(
            f"Added timestamps (seed={random_seed}): "  # ✅ Log seed
            f"{df[date_column].min()} → {df[date_column].max()}"
        )

        return df

    @staticmethod
    def create_temporal_split(
        df: pd.DataFrame,
        train_end_date: str,
        eval_start_date: str,
        eval_end_date: str,
        date_column: str = "application_date",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/eval split.

        Train window: [earliest → train_end_date)
        Eval window:  [eval_start_date → eval_end_date)

        ENFORCED: eval_start_date >= train_end_date (no overlap)

        Args:
            df: Input dataframe with timestamps
            train_end_date: End of training window (exclusive)
            eval_start_date: Start of evaluation window (inclusive)
            eval_end_date: End of evaluation window (exclusive)
            date_column: Timestamp column name

        Returns:
            (train_df, eval_df)
        """
        df[date_column] = pd.to_datetime(df[date_column])

        train_end = pd.to_datetime(train_end_date)
        eval_start = pd.to_datetime(eval_start_date)
        eval_end = pd.to_datetime(eval_end_date)

        # Validate no overlap
        if eval_start < train_end:
            raise ValueError(
                f"Temporal leak detected: eval_start ({eval_start}) < train_end ({train_end})"
            )

        # Split
        train_df = df[df[date_column] < train_end].copy()
        eval_df = df[(df[date_column] >= eval_start) & (df[date_column] < eval_end)].copy()

        logger.info(f"Train window: {df[date_column].min()} → {train_end}")
        logger.info(f"  Samples: {len(train_df)}")
        logger.info(f"Eval window: {eval_start} → {eval_end}")
        logger.info(f"  Samples: {len(eval_df)}")

        if len(train_df) == 0:
            raise ValueError("Empty training window")
        if len(eval_df) == 0:
            raise ValueError("Empty evaluation window")

        return train_df, eval_df

    @staticmethod
    def get_recent_window(
        df: pd.DataFrame,
        days: int = 30,
        date_column: str = "application_date",
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Get most recent N days of data.

        Args:
            df: Input dataframe
            days: Number of days to look back
            date_column: Timestamp column
            end_date: Reference end date (default: max date in df)

        Returns:
            Filtered dataframe
        """
        df[date_column] = pd.to_datetime(df[date_column])

        if end_date is None:
            end = df[date_column].max()
        else:
            end = pd.to_datetime(end_date)

        start = end - timedelta(days=days)

        filtered = df[(df[date_column] >= start) & (df[date_column] <= end)].copy()

        logger.info(f"Recent window: {start} → {end} ({len(filtered)} samples)")

        return filtered
