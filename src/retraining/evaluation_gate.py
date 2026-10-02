"""
Evaluation gate with COVERAGE and COOLDOWN checks.

NEW GATES:
4. Minimum evaluation coverage (% of labeled predictions)
5. Promotion cooldown (prevent retrain storms)

PHILOSOPHY: Rejection = Success
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class EvaluationGate:
    """
    Multi-criteria evaluation gate.

    ⚠️ COOLDOWN AUTHORITY: This class is the SOLE authority for promotion cooldown.
    ModelPromoter does NOT check cooldown - it trusts this gate's decision.

    Gates (ALL must pass):
    1. Sufficient samples (absolute count)
    2. Primary metric improvement (F1 score)
    3. Calibration maintained (Brier score)
    4. Minimum label coverage (%)
    5. Promotion cooldown (days)          # ✅ ENFORCED HERE ONLY
    6. No segment regression

    """

    def __init__(
        self,
        min_f1_improvement_pct: float = 2.0,
        max_brier_degradation: float = 0.01,
        max_segment_regression_pct: float = 5.0,
        min_samples_for_decision: int = 200,
        min_coverage_pct: float = 30.0,  # ✅ NEW: Min 30% labeled
        promotion_cooldown_days: int = 7,  # ✅ NEW: 7 day cooldown
        decisions_path: str = "/app/monitoring/retraining/decisions",
    ):
        """
        Initialize gate with thresholds.

        Args:
            min_f1_improvement_pct: Min F1 improvement (%)
            max_brier_degradation: Max Brier increase
            max_segment_regression_pct: Max segment F1 drop (%)
            min_samples_for_decision: Min absolute samples
            min_coverage_pct: Min % of predictions with labels
            promotion_cooldown_days: Min days between promotions
            decisions_path: Where promotion decisions are stored
        """
        self.min_f1_improvement_pct = min_f1_improvement_pct
        self.max_brier_degradation = max_brier_degradation
        self.max_segment_regression_pct = max_segment_regression_pct
        self.min_samples_for_decision = min_samples_for_decision
        self.min_coverage_pct = min_coverage_pct
        self.promotion_cooldown_days = promotion_cooldown_days
        self.decisions_path = Path(decisions_path)

        logger.info("=" * 80)
        logger.info("Evaluation Gate Initialized")
        logger.info("=" * 80)
        logger.info(f"  Min F1 improvement: {min_f1_improvement_pct}%")
        logger.info(f"  Max Brier degradation: {max_brier_degradation}")
        logger.info(f"  Max segment regression: {max_segment_regression_pct}%")
        logger.info(f"  Min samples: {min_samples_for_decision}")
        logger.info(f"  Min coverage: {min_coverage_pct}%")  # ✅ NEW
        logger.info(f"  Promotion cooldown: {promotion_cooldown_days} days")  # ✅ NEW
        logger.info("=" * 80)

    def evaluate(
        self,
        production_metrics: Dict,
        shadow_metrics: Dict,
        comparison: Dict,
        coverage_stats: Dict = None,  # Keep optional for now but validate below
    ) -> Tuple[bool, Dict]:
        """
        Run all evaluation gates.

        Args:
            production_metrics: Production model metrics
            shadow_metrics: Shadow model metrics
            comparison: Comparison results
            coverage_stats: Label coverage statistics

        Returns:
            (should_promote, decision_details)
        """
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING EVALUATION GATE")
        logger.info("=" * 80)

        decision: Dict[str, Any] = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "gate_results": {},
            "final_decision": False,
            "reason": [],
        }

        # ✅ VALIDATE COVERAGE STATS (fail-closed if missing)
        if self.min_coverage_pct > 0 and coverage_stats is None:
            error_msg = (
                f"Coverage gating enabled (min={self.min_coverage_pct}%) "
                "but coverage_stats not provided. FAILING CLOSED."
            )
            logger.error(f"🚨 {error_msg}")
            decision["reason"].append(error_msg)
            decision["final_decision"] = False
            decision["gate_results"]["coverage_validation"] = {
                "passed": False,
                "error": "coverage_stats_missing",
            }
            return False, decision

        # Gate 1: Sufficient samples
        num_samples = shadow_metrics.get("num_samples", 0)
        gate1_pass = num_samples >= self.min_samples_for_decision

        decision["gate_results"]["sufficient_samples"] = {
            "passed": gate1_pass,
            "num_samples": num_samples,
            "min_required": self.min_samples_for_decision,
        }

        logger.info("\n[Gate 1] Sufficient Samples")
        logger.info(f"  Samples: {num_samples} (min: {self.min_samples_for_decision})")
        logger.info(f"  Result: {'✅ PASS' if gate1_pass else '❌ FAIL'}")

        if not gate1_pass:
            decision["reason"].append(
                f"Insufficient samples: {num_samples} < {self.min_samples_for_decision}"
            )
            decision["final_decision"] = False
            self._log_rejection(decision, "insufficient_samples")
            return False, decision

        # Gate 2: Label coverage (✅ NEW)
        if coverage_stats:
            if "coverage_pct" in coverage_stats and coverage_stats["coverage_pct"] is not None:
                coverage_pct = float(coverage_stats["coverage_pct"])
            elif (
                "label_coverage_pct" in coverage_stats
                and coverage_stats["label_coverage_pct"] is not None
            ):
                coverage_pct = float(coverage_stats["label_coverage_pct"])
            elif "coverage_rate" in coverage_stats and coverage_stats["coverage_rate"] is not None:
                coverage_pct = float(coverage_stats["coverage_rate"]) * 100.0
            else:
                coverage_pct = 0.0

            gate2_pass = coverage_pct >= self.min_coverage_pct

            decision["gate_results"]["minimum_coverage"] = {
                "passed": gate2_pass,
                "coverage_pct": coverage_pct,
                "min_required_pct": self.min_coverage_pct,
            }

            logger.info("\n[Gate 2] Minimum Coverage")
            logger.info(f"  Coverage: {coverage_pct:.1f}% (min: {self.min_coverage_pct}%)")
            logger.info(f"  Result: {'✅ PASS' if gate2_pass else '❌ FAIL'}")

            if not gate2_pass:
                decision["reason"].append(
                    f"Insufficient label coverage: {coverage_pct:.1f}% < {self.min_coverage_pct}%"
                )
                decision["final_decision"] = False
                self._log_rejection(decision, "insufficient_coverage")
                return False, decision

        # Gate 3: Promotion cooldown (✅ NEW)
        gate3_pass, cooldown_msg = self._check_promotion_cooldown()

        decision["gate_results"]["promotion_cooldown"] = {
            "passed": gate3_pass,
            "message": cooldown_msg,
        }

        logger.info("\n[Gate 3] Promotion Cooldown")
        logger.info(f"  {cooldown_msg}")
        logger.info(f"  Result: {'✅ PASS' if gate3_pass else '❌ FAIL'}")

        if not gate3_pass:
            decision["reason"].append(cooldown_msg)
            decision["final_decision"] = False
            self._log_rejection(decision, "promotion_cooldown")
            return False, decision

        # Gate 4: Primary metric improvement
        f1_improvement_pct = comparison.get("f1_improvement_pct", 0)
        gate4_pass = f1_improvement_pct >= self.min_f1_improvement_pct

        decision["gate_results"]["metric_improvement"] = {
            "passed": gate4_pass,
            "f1_improvement_pct": f1_improvement_pct,
            "threshold": self.min_f1_improvement_pct,
        }

        logger.info("\n[Gate 4] Metric Improvement")
        logger.info(
            f"  F1 improvement: {f1_improvement_pct:.2f}% (min: {self.min_f1_improvement_pct}%)"
        )
        logger.info(f"  Result: {'✅ PASS' if gate4_pass else '❌ FAIL'}")

        if not gate4_pass:
            decision["reason"].append(
                f"Insufficient F1 improvement: {f1_improvement_pct:.2f}% < {self.min_f1_improvement_pct}%"
            )
            decision["final_decision"] = False
            self._log_rejection(decision, "insufficient_improvement")
            return False, decision

        # Gate 5: Calibration maintained
        brier_change = comparison.get("brier_change", float("inf"))
        gate5_pass = brier_change <= self.max_brier_degradation

        decision["gate_results"]["calibration_maintained"] = {
            "passed": gate5_pass,
            "brier_change": brier_change,
            "threshold": self.max_brier_degradation,
        }

        logger.info("\n[Gate 5] Calibration Maintained")
        logger.info(f"  Brier change: {brier_change:.4f} (max: {self.max_brier_degradation})")
        logger.info(f"  Result: {'✅ PASS' if gate5_pass else '❌ FAIL'}")

        if not gate5_pass:
            decision["reason"].append(
                f"Calibration degraded: Brier change {brier_change:.4f} > {self.max_brier_degradation}"
            )
            decision["final_decision"] = False
            self._log_rejection(decision, "calibration_degraded")
            return False, decision

        # Gate 6: No segment regression
        gate6_pass, segment_issues = self._check_segment_regression(
            production_metrics.get("segment_performance", {}),
            shadow_metrics.get("segment_performance", {}),
        )

        decision["gate_results"]["no_segment_regression"] = {
            "passed": gate6_pass,
            "issues": segment_issues,
        }

        logger.info("\n[Gate 6] No Segment Regression")
        logger.info(f"  Issues found: {len(segment_issues)}")
        logger.info(f"  Result: {'✅ PASS' if gate6_pass else '❌ FAIL'}")

        if not gate6_pass:
            decision["reason"].append(f"Segment regression: {segment_issues}")
            decision["final_decision"] = False
            self._log_rejection(decision, "segment_regression")
            return False, decision

        # ✅ ALL GATES PASSED
        logger.info("\n" + "=" * 80)
        logger.info("🎉 ALL GATES PASSED")
        logger.info("=" * 80)

        decision["final_decision"] = True
        decision["reason"] = [
            f"All gates passed. F1 improved {f1_improvement_pct:.2f}%, "
            f"calibration maintained (Brier: {brier_change:+.4f}), "
            f"no segment regression."
        ]

        logger.info("  Decision: PROMOTE")
        logger.info(f"  Reason: {decision['reason'][0]}")

        return True, decision

    def _check_promotion_cooldown(self) -> Tuple[bool, str]:
        """
        Check if enough time passed since last promotion.

        Returns:
            (cooldown_ok, message)
        """
        if not self.decisions_path.exists():
            return True, "No previous promotions found"

        # Find most recent promotion
        promotion_files = sorted(self.decisions_path.glob("decision_*.json"), reverse=True)

        last_promotion_time = None

        for filepath in promotion_files:
            try:
                with open(filepath, "r") as f:
                    record = json.load(f)

                if record.get("action") == "promote":
                    last_promotion_time = pd.to_datetime(record["timestamp"])
                    break
            except Exception as e:
                logger.warning(f"Could not read {filepath}: {e}")

        if last_promotion_time is None:
            return True, "No previous promotions found"

        # Check cooldown
        now = pd.Timestamp.now()
        days_since = (now - last_promotion_time).days

        if days_since < self.promotion_cooldown_days:
            days_remaining = self.promotion_cooldown_days - days_since
            return False, (
                f"Cooldown active: last promotion {days_since} days ago, "
                f"need {days_remaining} more days"
            )

        return True, f"Cooldown satisfied: {days_since} days since last promotion"

    def _check_segment_regression(
        self, prod_segments: Dict, shadow_segments: Dict
    ) -> Tuple[bool, List[str]]:
        """
        Check segment performance.

        ⚠️ EXPLICIT HANDLING: If segment exists in prod but missing in shadow,
        it is logged explicitly as insufficient data (not silently ignored).
        """
        if not prod_segments or not shadow_segments:
            return True, []

        issues = []
        missing_segments = []  # ✅ Track missing segments

        for feature, prod_feature_segments in prod_segments.items():
            shadow_feature_segments = shadow_segments.get(feature, {})

            for segment, prod_metrics in prod_feature_segments.items():
                shadow_metrics = shadow_feature_segments.get(segment, {})

                # ✅ EXPLICIT: Segment missing in shadow
                if not shadow_metrics:
                    missing_segments.append(f"{feature}={segment}")
                    logger.warning(
                        f"⚠️ Segment exists in production but not in shadow: "
                        f"{feature}={segment} (insufficient shadow data)"
                    )
                    # Decision: treat as non-blocking but logged
                    # Alternative: treat as regression issue by uncommenting below
                    # issues.append(f"{feature}={segment}: missing in shadow (insufficient data)")
                    continue

                # Check regression for segments that exist in both
                prod_f1 = prod_metrics.get("f1_score", 0)
                shadow_f1 = shadow_metrics.get("f1_score", 0)

                if prod_f1 > 0:
                    regression_pct = ((shadow_f1 - prod_f1) / prod_f1) * 100

                    if regression_pct < -self.max_segment_regression_pct:
                        issues.append(f"{feature}={segment}: F1 dropped {abs(regression_pct):.1f}%")

        # ✅ Log summary of missing segments
        if missing_segments:
            logger.warning(
                f"⚠️ {len(missing_segments)} segment(s) missing in shadow: {missing_segments}"
            )
            # Optionally add to decision metadata (non-blocking)
            # To make it blocking, add to issues instead

        return len(issues) == 0, issues

    def _log_rejection(self, decision: Dict, reason_code: str):
        """
        Log rejection as SUCCESSFUL system behavior.

        ✅ CRITICAL: Rejection is NOT a failure.
        It's the gate working correctly.
        """
        logger.info("\n" + "=" * 80)
        logger.info("✅ REJECTION = SUCCESSFUL GATE OPERATION")
        logger.info("=" * 80)
        logger.info(f"  Reason code: {reason_code}")
        logger.info(f"  Details: {decision['reason']}")
        logger.info("  System prevented deployment of inadequate model.")
        logger.info("  This is EXPECTED and CORRECT behavior.")
        logger.info("=" * 80)
