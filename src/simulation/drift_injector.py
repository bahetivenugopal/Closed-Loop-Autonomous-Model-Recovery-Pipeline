"""
Scientific drift injection engine.

Supports three types of drift:
1. Covariate Shift: Feature distribution changes
2. Population Shift: Class balance changes
3. Concept Drift: Feature-target relationship changes

All drift is logged for reproducibility.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from .drift_logger import get_drift_logger

logger = logging.getLogger(__name__)


class DriftInjector:
    """
    Inject controlled, logged drift into data.

    Critical: All drift is intentional and reproducible.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize drift injector.

        Args:
            random_seed: For reproducible drift
        """
        self.random_seed = random_seed
        self.drift_logger = get_drift_logger()
        np.random.seed(random_seed)

    # ================================================================
    # 1. COVARIATE SHIFT: Feature distribution changes
    # ================================================================

    def inject_covariate_shift_scaling(
        self, data: pd.DataFrame, feature: str, scale_factor: float, reason: str
    ) -> pd.DataFrame:
        """
        Scale a feature's distribution.

        Example: MonthlyIncome increases by 30% (economic improvement)

        Args:
            data: Input dataframe
            feature: Feature to shift
            scale_factor: Multiplier (e.g., 1.3 for +30%)
            reason: Why this drift is being injected

        Returns:
            Modified dataframe
        """
        data = data.copy()
        original_mean = data[feature].mean()

        data[feature] = data[feature] * scale_factor

        new_mean = data[feature].mean()

        logger.info(f"Covariate shift (scaling): {feature}")
        logger.info(f"  Original mean: {original_mean:.2f}")
        logger.info(f"  New mean: {new_mean:.2f}")
        logger.info(f"  Scale factor: {scale_factor}")

        # Log drift event
        self.drift_logger.log_drift_event(
            drift_type="covariate_shift_scaling",
            affected_features=[feature],
            magnitude=scale_factor,
            reason=reason,
            metadata={"original_mean": float(original_mean), "new_mean": float(new_mean)},
        )

        return data

    def inject_covariate_shift_location(
        self, data: pd.DataFrame, feature: str, shift_amount: float, reason: str
    ) -> pd.DataFrame:
        """
        Shift a feature's location (mean).

        Example: age increases by 5 years (population aging)

        Args:
            data: Input dataframe
            feature: Feature to shift
            shift_amount: Amount to add
            reason: Why this drift is being injected

        Returns:
            Modified dataframe
        """
        data = data.copy()
        original_mean = data[feature].mean()

        data[feature] = data[feature] + shift_amount

        new_mean = data[feature].mean()

        logger.info(f"Covariate shift (location): {feature}")
        logger.info(f"  Original mean: {original_mean:.2f}")
        logger.info(f"  New mean: {new_mean:.2f}")
        logger.info(f"  Shift: +{shift_amount}")

        # Log drift event
        self.drift_logger.log_drift_event(
            drift_type="covariate_shift_location",
            affected_features=[feature],
            magnitude=shift_amount,
            reason=reason,
            metadata={"original_mean": float(original_mean), "new_mean": float(new_mean)},
        )

        return data

    # ================================================================
    # 2. POPULATION SHIFT: Class balance changes
    # ================================================================

    def inject_population_shift(
        self, data: pd.DataFrame, target_column: str, new_positive_ratio: float, reason: str
    ) -> pd.DataFrame:
        """
        Change class balance.

        Example: Default rate increases from 6% to 15% (recession)

        Args:
            data: Input dataframe
            target_column: Target variable name
            new_positive_ratio: Desired ratio of positive class (0-1)
            reason: Why this drift is being injected

        Returns:
            Resampled dataframe
        """
        original_ratio = data[target_column].mean()

        # Separate classes
        positives = data[data[target_column] == 1]
        negatives = data[data[target_column] == 0]

        # Calculate new counts
        total_desired = len(data)
        n_positives_desired = int(total_desired * new_positive_ratio)
        n_negatives_desired = total_desired - n_positives_desired

        # Resample
        positives_resampled = positives.sample(
            n=n_positives_desired, replace=True, random_state=self.random_seed
        )
        negatives_resampled = negatives.sample(
            n=n_negatives_desired, replace=True, random_state=self.random_seed
        )

        # Combine
        data_shifted = pd.concat([positives_resampled, negatives_resampled], ignore_index=True)
        data_shifted = data_shifted.sample(frac=1, random_state=self.random_seed).reset_index(
            drop=True
        )

        new_ratio = data_shifted[target_column].mean()

        logger.info(f"Population shift: {target_column}")
        logger.info(f"  Original positive ratio: {original_ratio:.3f}")
        logger.info(f"  New positive ratio: {new_ratio:.3f}")

        # Log drift event
        self.drift_logger.log_drift_event(
            drift_type="population_shift",
            affected_features=[target_column],
            magnitude=new_positive_ratio,
            reason=reason,
            metadata={
                "original_positive_ratio": float(original_ratio),
                "new_positive_ratio": float(new_ratio),
            },
        )

        return data_shifted

    # ================================================================
    # 3. CONCEPT DRIFT: Relationship between features and target changes
    # ================================================================

    def inject_concept_drift_noise(
        self, data: pd.DataFrame, feature: str, noise_std: float, reason: str
    ) -> pd.DataFrame:
        """
        Add noise to feature-target relationship.

        Example: Credit score becomes less predictive (random noise added)

        Args:
            data: Input dataframe
            feature: Feature to add noise to
            noise_std: Standard deviation of noise
            reason: Why this drift is being injected

        Returns:
            Modified dataframe
        """
        data = data.copy()

        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, size=len(data))
        data[feature] = data[feature] + noise

        logger.info(f"Concept drift (noise): {feature}")
        logger.info(f"  Noise std: {noise_std}")

        # Log drift event
        self.drift_logger.log_drift_event(
            drift_type="concept_drift_noise",
            affected_features=[feature],
            magnitude=noise_std,
            reason=reason,
            metadata={"noise_type": "gaussian", "noise_std": float(noise_std)},
        )

        return data

    def inject_concept_drift_flip(
        self, data: pd.DataFrame, target_column: str, flip_probability: float, reason: str
    ) -> pd.DataFrame:
        """
        Randomly flip target labels.

        Example: Label noise increases (data quality degradation)

        Args:
            data: Input dataframe
            target_column: Target variable
            flip_probability: Probability of flipping each label
            reason: Why this drift is being injected

        Returns:
            Modified dataframe
        """
        data = data.copy()

        # Randomly flip labels
        flip_mask = np.random.random(len(data)) < flip_probability
        data.loc[flip_mask, target_column] = 1 - data.loc[flip_mask, target_column]

        n_flipped = flip_mask.sum()

        logger.info(f"Concept drift (label flip): {target_column}")
        logger.info(f"  Flipped {n_flipped} labels ({n_flipped / len(data) * 100:.1f}%)")

        # Log drift event
        self.drift_logger.log_drift_event(
            drift_type="concept_drift_flip",
            affected_features=[target_column],
            magnitude=flip_probability,
            reason=reason,
            metadata={"labels_flipped": int(n_flipped), "flip_rate": float(flip_probability)},
        )

        return data
