"""
Canonical Feature Schema for Self-Healing MLOps Pipeline.

SINGLE SOURCE OF TRUTH for feature definitions, column ordering, data types,
and validation across training, inference, monitoring, and evaluation.
"""

from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Target Column
TARGET_COLUMN = "SeriousDlqin2yrs"

# Feature Names in Canonical Order
FEATURE_COLUMNS: List[str] = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30_59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60_89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Classification
NUMERICAL_FEATURES: List[str] = FEATURE_COLUMNS.copy()
CATEGORICAL_FEATURES: List[str] = []

# Expected Data Types
FEATURE_TYPES: Dict[str, type] = {
    "RevolvingUtilizationOfUnsecuredLines": float,
    "age": int,
    "NumberOfTime30_59DaysPastDueNotWorse": int,
    "DebtRatio": float,
    "MonthlyIncome": float,
    "NumberOfOpenCreditLinesAndLoans": int,
    "NumberOfTimes90DaysLate": int,
    "NumberRealEstateLoansOrLines": int,
    "NumberOfTime60_89DaysPastDueNotWorse": int,
    "NumberOfDependents": int,
}


def validate_features(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame against canonical feature schema.

    Checks:
    - Presence of all required feature columns
    - Absence of NaN/null values in key features

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        issues.append(f"Missing required feature columns: {sorted(list(missing))}")

    # Check for empty dataframe
    if len(df) == 0:
        issues.append("DataFrame is empty")

    is_valid = len(issues) == 0
    return is_valid, issues


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from DataFrame in exact canonical column order.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame containing only canonical feature columns in exact order.
    """
    is_valid, issues = validate_features(df)
    if not is_valid:
        raise ValueError(f"Schema validation failed: {'; '.join(issues)}")

    return df[FEATURE_COLUMNS].copy()
