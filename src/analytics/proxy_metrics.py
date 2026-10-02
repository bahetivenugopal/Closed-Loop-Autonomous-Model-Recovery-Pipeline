import pandas as pd
import numpy as np
from typing import Any, Dict, List, Union, TypedDict, cast
from scipy.stats import entropy as scipy_entropy
import logging

logger = logging.getLogger(__name__)


class ProxyStats(TypedDict):
    num_predictions: int
    positive_rate: float
    probability_mean: float
    probability_std: float
    probability_median: float
    probability_q25: float
    probability_q75: float


class NoData(TypedDict):
    status: str


ProxyStatsOrNoData = Union[ProxyStats, NoData]


def compute_prediction_distribution_stats(predictions: pd.DataFrame) -> ProxyStatsOrNoData:
    if len(predictions) == 0:
        return cast(ProxyStatsOrNoData, {"status": "no_data"})

    return {
        "num_predictions": len(predictions),
        "positive_rate": float(predictions["prediction"].mean()),
        "probability_mean": float(predictions["probability"].mean()),
        "probability_std": float(predictions["probability"].std()),
        "probability_median": float(predictions["probability"].median()),
        "probability_q25": float(predictions["probability"].quantile(0.25)),
        "probability_q75": float(predictions["probability"].quantile(0.75)),
    }


def compute_probability_entropy(predictions: pd.DataFrame) -> float:
    if len(predictions) == 0:
        return 0.0

    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(predictions["probability"], bins=bins)
    prob_dist = hist / hist.sum() if hist.sum() > 0 else hist
    return float(scipy_entropy(prob_dist + 1e-10))


def compute_time_windowed_trends(
    predictions: pd.DataFrame, windows: List[str] = ["1H", "6H", "24H"]
) -> Dict[str, Any]:
    if len(predictions) == 0:
        return cast(Dict[str, Any], {"status": "no_data"})

    predictions = predictions.copy()
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    predictions = predictions.sort_values("timestamp")
    current_time = predictions["timestamp"].max()

    results: Dict[str, Any] = {}

    for window in windows:
        cutoff_time = current_time - pd.Timedelta(window)
        windowed_data = predictions[predictions["timestamp"] > cutoff_time]

        if len(windowed_data) == 0:
            results[f"window_{window}"] = {"status": "no_data"}
            continue

        results[f"window_{window}"] = {
            "count": len(windowed_data),
            "positive_rate": float(windowed_data["prediction"].mean()),
            "probability_mean": float(windowed_data["probability"].mean()),
            "probability_std": float(windowed_data["probability"].std()),
            "entropy": compute_probability_entropy(windowed_data),
        }

    return results


def compute_rate_of_change(
    current_stats: Dict[str, Any], previous_stats: Dict[str, Any], time_delta_hours: float
) -> Dict[str, Any]:
    if not current_stats or not previous_stats:
        return cast(Dict[str, Any], {"status": "insufficient_data"})

    positive_rate_delta = current_stats.get("positive_rate", 0) - previous_stats.get(
        "positive_rate", 0
    )
    probability_mean_delta = current_stats.get("probability_mean", 0) - previous_stats.get(
        "probability_mean", 0
    )

    return {
        "positive_rate_change_per_hour": (
            positive_rate_delta / time_delta_hours if time_delta_hours > 0 else 0
        ),
        "probability_mean_change_per_hour": (
            probability_mean_delta / time_delta_hours if time_delta_hours > 0 else 0
        ),
        "time_delta_hours": time_delta_hours,
    }


def analyze_proxy_metrics(predictions: pd.DataFrame) -> Dict[str, Any]:
    logger.info(f"Analyzing proxy metrics for {len(predictions)} predictions")

    results: Dict[str, Any] = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "overall_stats": compute_prediction_distribution_stats(predictions),
        "entropy": compute_probability_entropy(predictions),
        "time_windowed": compute_time_windowed_trends(predictions),
    }

    logger.info(
        f"Proxy metrics computed: {results['overall_stats'].get('num_predictions', 0)} samples"
    )
    return results


class ProxyMetricsCalculator:
    def __init__(self, predictions: pd.DataFrame):
        self.predictions = predictions

    def compute_overall_stats(self) -> ProxyStatsOrNoData:
        return compute_prediction_distribution_stats(self.predictions)

    def compute_entropy(self) -> float:
        return compute_probability_entropy(self.predictions)

    def compute_time_windowed_trends(
        self, windows: List[str] = ["1H", "6H", "24H"]
    ) -> Dict[str, Any]:
        return compute_time_windowed_trends(self.predictions, windows)

    def analyze(self) -> Dict[str, Any]:
        return analyze_proxy_metrics(self.predictions)
