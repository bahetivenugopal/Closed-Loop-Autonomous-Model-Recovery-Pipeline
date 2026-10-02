"""
Generate realistic fake predictions via API with controlled drift.

What this does (correctly):
1. Loads temporal dataset
2. Injects escalating, bounded drift into features (3 phases of increasing severity)
3. Sends requests ONLY via prediction API
4. Logs predictions exactly like production traffic with precise timestamps
5. Stores delayed labels (partial coverage)
6. Produces clean, JSON-safe data for drift detection

DRIFT PHASES:
- Phase 0: No drift (baseline)
- Phase 1: Financial stress (debt ↑, income ↓, utilization ↑)
- Phase 2: Delinquency behavior (late payments ↑, credit lines ↓)
- Phase 3: SEVERE economic downturn (extreme values on all metrics)

Usage:
    docker-compose exec api python scripts/generate_fake_predictions.py \
        --num-samples 3000 \
        --coverage 50 \
        --drift-phase 2

    # For severe drift simulation:
    docker-compose exec api python scripts/generate_fake_predictions.py \
        --num-samples 3000 \
        --coverage 50 \
        --drift-phase 3
"""

import sys
import os

sys.path.append("/app")

import time
import random
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Safe conversion helpers (CRITICAL)
# ------------------------------------------------------------------


def safe_float(val, min_val=None, max_val=None, default=0.0):
    try:
        val = float(val)
        if np.isnan(val) or np.isinf(val):
            return default
        if min_val is not None:
            val = max(val, min_val)
        if max_val is not None:
            val = min(val, max_val)
        return val
    except Exception:
        return default


def safe_int(val, min_val=None, max_val=None, default=0):
    try:
        val = int(val)
        if min_val is not None:
            val = max(val, min_val)
        if max_val is not None:
            val = min(val, max_val)
        return val
    except Exception:
        return default


# ------------------------------------------------------------------
# Load temporal dataset
# ------------------------------------------------------------------


def load_temporal_data():
    path = "/app/data/processed/cs-training-temporal.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Temporal dataset not found: {path}\n" "Run bootstrap step first.")

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from temporal dataset")
    return df


# ------------------------------------------------------------------
# Drift injection (REALISTIC + AGGRESSIVE)
# ------------------------------------------------------------------


def apply_drift(
    row: pd.Series, phase: int, sample_idx: int = 0, total_samples: int = 1
) -> pd.Series:
    r = row.copy()

    # Calculate progression ratio for gradual drift over time
    progression = sample_idx / max(total_samples, 1)

    # Phase 1: financial stress drift (MODERATE)
    if phase >= 1:
        # Progressive debt increase (1.2x to 2.0x over phase)
        debt_multiplier = 1.2 + (0.8 * progression) + np.random.uniform(0, 0.3)
        r["DebtRatio"] = safe_float(
            r["DebtRatio"] * debt_multiplier,
            0.0,
            5.0,
        )

        # Progressive income decrease (0.7x to 0.5x over phase)
        income_multiplier = 0.7 + (0.2 * (1 - progression)) + np.random.uniform(-0.1, 0.1)
        r["MonthlyIncome"] = safe_float(
            r["MonthlyIncome"] * income_multiplier,
            0.0,
            1_000_000,
        )

        # Revolving utilization increases (1.1x to 1.5x over phase)
        revolving_multiplier = 1.1 + (0.4 * progression) + np.random.uniform(0, 0.2)
        r["RevolvingUtilizationOfUnsecuredLines"] = safe_float(
            r["RevolvingUtilizationOfUnsecuredLines"] * revolving_multiplier,
            0.0,
            1.0,
        )

    # Phase 2: delinquency behavior drift (AGGRESSIVE)
    if phase >= 2:
        # Significantly increased late payments
        r["NumberOfTimes90DaysLate"] = safe_int(
            r["NumberOfTimes90DaysLate"] + np.random.choice([0, 1, 2, 3, 4]),
            0,
            10,
        )

        r["NumberOfTime60_89DaysPastDueNotWorse"] = safe_int(
            r["NumberOfTime60_89DaysPastDueNotWorse"] + np.random.choice([0, 1, 2, 3]),
            0,
            10,
        )

        r["NumberOfTime30_59DaysPastDueNotWorse"] = safe_int(
            r["NumberOfTime30_59DaysPastDueNotWorse"] + np.random.choice([0, 1, 2]),
            0,
            10,
        )

        # Credit lines reduction (stress on credit)
        r["NumberOfOpenCreditLinesAndLoans"] = safe_int(
            r["NumberOfOpenCreditLinesAndLoans"] * np.random.uniform(0.5, 0.8),
            0,
            50,
        )

    # Phase 3: SEVERE SYSTEMIC DRIFT (economic downturn simulation)
    if phase >= 3:
        # Extreme debt ratio increase
        r["DebtRatio"] = safe_float(
            r["DebtRatio"] * np.random.uniform(2.5, 4.0),
            0.0,
            5.0,
        )

        # Dramatic income collapse
        r["MonthlyIncome"] = safe_float(
            r["MonthlyIncome"] * np.random.uniform(0.2, 0.4),
            0.0,
            1_000_000,
        )

        # Maxed-out revolving credit
        r["RevolvingUtilizationOfUnsecuredLines"] = safe_float(
            min(r["RevolvingUtilizationOfUnsecuredLines"] + np.random.uniform(0.3, 0.7), 1.0),
            0.0,
            1.0,
        )

        # Multiple severe delinquencies
        r["NumberOfTimes90DaysLate"] = safe_int(
            r["NumberOfTimes90DaysLate"] + np.random.choice([2, 3, 4, 5, 6]),
            0,
            10,
        )

        r["NumberOfTime60_89DaysPastDueNotWorse"] = safe_int(
            r["NumberOfTime60_89DaysPastDueNotWorse"] + np.random.choice([1, 2, 3, 4]),
            0,
            10,
        )

        r["NumberOfTime30_59DaysPastDueNotWorse"] = safe_int(
            r["NumberOfTime30_59DaysPastDueNotWorse"] + np.random.choice([1, 2, 3, 4]),
            0,
            10,
        )

        # Collapsed credit profile
        r["NumberOfOpenCreditLinesAndLoans"] = safe_int(
            r["NumberOfOpenCreditLinesAndLoans"] * np.random.uniform(0.2, 0.5),
            0,
            50,
        )

    return r


# ------------------------------------------------------------------
# Convert row to API payload
# ------------------------------------------------------------------


def to_payload(row: pd.Series) -> dict:
    return {
        "RevolvingUtilizationOfUnsecuredLines": safe_float(
            row["RevolvingUtilizationOfUnsecuredLines"], 0.0, 1.0
        ),
        "age": safe_int(row["age"], 18, 100),
        "NumberOfTime30_59DaysPastDueNotWorse": safe_int(
            row["NumberOfTime30_59DaysPastDueNotWorse"], 0, 10
        ),
        "DebtRatio": safe_float(row["DebtRatio"], 0.0, 5.0),
        "MonthlyIncome": safe_float(row["MonthlyIncome"], 0.0, 1_000_000),
        "NumberOfOpenCreditLinesAndLoans": safe_int(row["NumberOfOpenCreditLinesAndLoans"], 0, 50),
        "NumberOfTimes90DaysLate": safe_int(row["NumberOfTimes90DaysLate"], 0, 10),
        "NumberRealEstateLoansOrLines": safe_int(row["NumberRealEstateLoansOrLines"], 0, 10),
        "NumberOfTime60_89DaysPastDueNotWorse": safe_int(
            row["NumberOfTime60_89DaysPastDueNotWorse"], 0, 10
        ),
        "NumberOfDependents": safe_int(row["NumberOfDependents"], 0, 10),
    }


# ------------------------------------------------------------------
# API call
# ------------------------------------------------------------------


def call_api(payload: dict, api_url: str):
    try:
        resp = requests.post(api_url, json=payload, timeout=5)
        if resp.status_code == 200:
            return True, resp.json()
        return False, resp.text
    except Exception as e:
        return False, str(e)


# ------------------------------------------------------------------
# Label storage (delayed)
# ------------------------------------------------------------------


def store_labels(prediction_records, coverage_pct):
    from src.storage.label_store import get_label_store

    store = get_label_store()

    k = int(len(prediction_records) * coverage_pct / 100)
    selected = random.sample(prediction_records, k)

    logger.info(f"Storing {k} delayed labels ({coverage_pct}% coverage)")

    for pred_id, label, ts in selected:
        store.store_label(
            prediction_id=pred_id,
            true_label=int(label),
            label_source="simulation",
            prediction_timestamp=ts,
        )


# ------------------------------------------------------------------
# Main generation logic
# ------------------------------------------------------------------


def generate(num_samples, coverage, drift_phase, api_url, delay):
    df = load_temporal_data()

    df = df.sample(n=min(num_samples, len(df)), random_state=42)
    logger.info(f"Generating {len(df)} predictions with drift_phase={drift_phase}")
    logger.info(f"Expected drift pattern: Phase {drift_phase} (escalating financial stress)")

    prediction_records = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        # Pass sample index for progressive drift
        drifted = apply_drift(row, drift_phase, i - 1, len(df))
        payload = to_payload(drifted)

        ok, result = call_api(payload, api_url)

        if ok:
            # Store: (prediction_id, label, timestamp from result)
            pred_timestamp = result.get("prediction_timestamp", datetime.now().isoformat())
            prediction_records.append(
                (result["prediction_id"], row["SeriousDlqin2yrs"], pred_timestamp)
            )

            if i % 100 == 0:
                logger.info(
                    f"Progress: {i}/{len(df)} | Recent drift metrics - "
                    f"Debt: {drifted['DebtRatio']:.2f}, "
                    f"Income: {drifted['MonthlyIncome']:.0f}, "
                    f"Late90: {drifted['NumberOfTimes90DaysLate']:.0f}"
                )
        else:
            logger.error(f"Prediction {i} failed: {result}")

        time.sleep(delay)

    logger.info(f"✅ Generated {len(prediction_records)} predictions with timestamps")

    if prediction_records:
        store_labels(prediction_records, coverage)
        logger.info(
            f"📊 Data drift injected at Phase {drift_phase}. "
            f"Check drift_detection results in monitoring/ directory"
        )


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--coverage", type=int, default=50)
    parser.add_argument(
        "--drift-phase",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="0=baseline, 1=financial stress, 2=delinquency, 3=severe downturn",
    )
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/predict")
    parser.add_argument("--delay", type=float, default=0.03)

    args = parser.parse_args()

    generate(
        num_samples=args.num_samples,
        coverage=args.coverage,
        drift_phase=args.drift_phase,
        api_url=args.api_url,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
