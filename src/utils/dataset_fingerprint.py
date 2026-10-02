"""
Dataset fingerprinting for reproducibility.
Compute unique hash of dataset to track data→model lineage.
"""

import hashlib
import pandas as pd


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Compute deterministic hash of DataFrame content.

    Args:
        df: Input DataFrame

    Returns:
        SHA256 hash as hex string
    """
    # Convert to bytes in a deterministic way
    # Sort columns to ensure consistency
    df_sorted = df.sort_index(axis=1)

    # Convert to CSV string (deterministic representation)
    csv_string = df_sorted.to_csv(index=False)

    # Compute hash
    hash_object = hashlib.sha256(csv_string.encode("utf-8"))
    return hash_object.hexdigest()


def get_dataset_metadata(df: pd.DataFrame) -> dict:
    """
    Extract dataset metadata for MLflow logging.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with dataset fingerprint and stats
    """
    return {
        "dataset_hash": compute_dataframe_hash(df),
        "dataset_rows": len(df),
        "dataset_columns": len(df.columns),
        "dataset_memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


if __name__ == "__main__":
    # Test
    df = pd.read_csv("data/cs-training.csv")
    metadata = get_dataset_metadata(df)
    print("Dataset Fingerprint:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
