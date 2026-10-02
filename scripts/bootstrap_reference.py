"""
Bootstrap frozen reference data for monitoring.

THIS SCRIPT RUNS ONCE to establish the baseline.
Reference data represents a period when the model was known to perform well.

Sources (choose one):
1. Test set from training (what we'll use)
2. Early production data from first N days
3. Hand-selected "golden" production window

Once created, reference data is IMMUTABLE.
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import hashlib


def compute_reference_hash(df: pd.DataFrame) -> str:
    """Compute hash to detect if reference is accidentally modified."""
    csv_string = df.to_csv(index=False)
    return hashlib.sha256(csv_string.encode()).hexdigest()


def bootstrap_from_training_data(
    training_data_path: str = "/app/data/cs-training.csv",
    output_dir: str = "/app/monitoring/reference",
):
    """
    Create frozen reference from training test split.

    Why this approach:
    - Training data is clean and representative
    - We know the model was trained on this distribution
    - It's available before production deployment

    Alternative: Use first N days of production (requires waiting).
    """
    print("=" * 60)
    print("BOOTSTRAPPING REFERENCE DATA")
    print("=" * 60)

    # Load training data
    df = pd.read_csv(training_data_path)
    print(f"\n1. Loaded training data: {len(df)} rows")

    # Use same preprocessing as training
    df_clean = df.fillna(df.median(numeric_only=True))

    # Use last 20% as reference (mimics test set)
    split_idx = int(len(df_clean) * 0.8)
    reference_data = df_clean.iloc[split_idx:].copy()

    print(f"2. Created reference data: {len(reference_data)} rows")
    print(f"   (Last 20% of training data)")

    # Compute fingerprint
    reference_hash = compute_reference_hash(reference_data)
    print(f"3. Reference hash: {reference_hash[:16]}...")

    # Create metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "source": "training_test_split",
        "num_samples": len(reference_data),
        "num_features": len(reference_data.columns),
        "reference_hash": reference_hash,
        "purpose": "frozen_baseline_for_drift_detection",
        "mutation_policy": "NEVER - reference is immutable",
        "features": reference_data.columns.tolist(),
    }

    # Save reference data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reference_file = output_path / "reference_data.csv"
    reference_data.to_csv(reference_file, index=False)
    print(f"\n4. ✅ Saved reference data: {reference_file}")

    # Save metadata
    metadata_file = output_path / "reference_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"5. ✅ Saved metadata: {metadata_file}")

    print("\n" + "=" * 60)
    print("REFERENCE BOOTSTRAPPING COMPLETE")
    print("=" * 60)
    print("\n⚠️  WARNING: Do NOT modify reference_data.csv")
    print("   If reference needs updating, re-run this script")
    print("   and document the change in your experiment log.")
    print()

    return reference_data, metadata


def verify_reference_integrity(reference_dir: str = "/app/monitoring/reference"):
    """
    Verify reference data hasn't been corrupted or modified.
    Run this periodically to ensure data integrity.
    """
    reference_file = Path(reference_dir) / "reference_data.csv"
    metadata_file = Path(reference_dir) / "reference_metadata.json"

    if not reference_file.exists():
        raise FileNotFoundError("Reference data not found. Run bootstrap first.")

    # Load data and metadata
    df = pd.read_csv(reference_file)
    with open(metadata_file) as f:
        metadata = json.load(f)

    # Recompute hash
    current_hash = compute_reference_hash(df)
    expected_hash = metadata["reference_hash"]

    if current_hash != expected_hash:
        raise ValueError(
            f"Reference data integrity check FAILED!\n"
            f"Expected: {expected_hash[:16]}...\n"
            f"Got:      {current_hash[:16]}...\n"
            f"Reference data may have been modified."
        )

    print("✅ Reference integrity verified")
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_reference_integrity()
    else:
        bootstrap_from_training_data()
