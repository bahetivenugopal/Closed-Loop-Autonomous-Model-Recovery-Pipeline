"""
Unit tests for evaluation gate logic.

CRITICAL: These tests verify the decision-making logic
that determines whether shadow models get promoted.
"""

import sys

import pytest
from src.retraining.evaluation_gate import EvaluationGate

sys.path.append("/app")


class TestEvaluationGate:
    """Test suite for EvaluationGate class."""

    def test_gate_passes_on_good_shadow_model(self, temp_monitoring_dir):
        """Test that gate approves a clearly better model."""
        gate = EvaluationGate(
            min_f1_improvement_pct=2.0,
            max_brier_degradation=0.01,
            min_samples_for_decision=200,
            min_coverage_pct=0.0,  # Disable coverage check for test
            promotion_cooldown_days=0,
            decisions_path=temp_monitoring_dir,
        )

        production_metrics = {
            "num_samples": 250,
            "primary_metrics": {"f1_score": 0.80, "roc_auc": 0.85},
            "calibration_metrics": {"brier_score": 0.15},
        }

        shadow_metrics = {
            "num_samples": 250,
            "primary_metrics": {"f1_score": 0.84, "roc_auc": 0.87},  # 5% improvement
            "calibration_metrics": {"brier_score": 0.145},  # Slight improvement
        }

        comparison = {"f1_improvement_pct": 5.0, "brier_change": -0.005}
        coverage_stats = {
            "lines": 95,
            "branches": 80,
            "label_coverage_pct": 50.0,  # 50% coverage > 25% minimum
            "labeled_samples": 125,
            "total_samples": 250,
        }

        should_promote, decision = gate.evaluate(
            production_metrics, shadow_metrics, comparison, coverage_stats
        )

        assert should_promote is True
        assert decision["final_decision"] is True
        assert all(gate["passed"] for gate in decision["gate_results"].values())

    def test_gate_fails_on_insufficient_improvement(self, temp_monitoring_dir):
        """Test that gate rejects model with insufficient F1 improvement."""
        gate = EvaluationGate(
            min_f1_improvement_pct=2.0,
            min_coverage_pct=0.0,
            promotion_cooldown_days=0,
            decisions_path=temp_monitoring_dir,
        )

        production_metrics = {
            "num_samples": 250,
            "primary_metrics": {"f1_score": 0.80},
            "calibration_metrics": {"brier_score": 0.15},
        }

        shadow_metrics = {
            "num_samples": 250,
            "primary_metrics": {"f1_score": 0.805},  # Only 0.625% improvement
            "calibration_metrics": {"brier_score": 0.15},
        }

        comparison = {"f1_improvement_pct": 0.625, "brier_change": 0.0}
        coverage_stats = {
            "lines": 95,
            "branches": 80,
            "label_coverage_pct": 50.0,
            "labeled_samples": 125,
            "total_samples": 250,
        }

        should_promote, decision = gate.evaluate(
            production_metrics, shadow_metrics, comparison, coverage_stats
        )

        assert should_promote is False
        assert decision["final_decision"] is False
        # Check for improvement gate failure
        assert any(
            "Insufficient F1 improvement" in str(reason) for reason in decision.get("reason", [])
        ) or "Insufficient F1 improvement" in str(decision.get("gate_results", {}))

    def test_gate_fails_on_calibration_degradation(self, temp_monitoring_dir):
        """Test that gate rejects model with worse calibration."""
        gate = EvaluationGate(
            min_f1_improvement_pct=2.0,
            max_brier_degradation=0.01,
            min_coverage_pct=0.0,
            promotion_cooldown_days=0,
            decisions_path=temp_monitoring_dir,
        )

        production_metrics = {
            "num_samples": 250,
            "primary_metrics": {"f1_score": 0.80},
            "calibration_metrics": {"brier_score": 0.10},
        }

        shadow_metrics = {
            "num_samples": 250,
            "primary_metrics": {"f1_score": 0.85},  # Better F1
            "calibration_metrics": {"brier_score": 0.15},  # But worse calibration
        }

        comparison = {"f1_improvement_pct": 6.25, "brier_change": 0.05}  # Exceeds threshold
        coverage_stats = {
            "lines": 95,
            "branches": 80,
            "label_coverage_pct": 50.0,
            "labeled_samples": 125,
            "total_samples": 250,
        }

        should_promote, decision = gate.evaluate(
            production_metrics, shadow_metrics, comparison, coverage_stats
        )

        assert should_promote is False
        # Check for calibration gate failure
        assert any(
            "Calibration degraded" in str(reason) for reason in decision.get("reason", [])
        ) or "Calibration degraded" in str(decision.get("gate_results", {}))

    def test_gate_fails_on_insufficient_samples(self, temp_monitoring_dir):
        """Test that gate requires minimum sample size."""
        gate = EvaluationGate(
            min_samples_for_decision=200,
            promotion_cooldown_days=0,
            decisions_path=temp_monitoring_dir,
        )

        production_metrics = {"num_samples": 150}
        shadow_metrics = {"num_samples": 150}
        comparison = {}
        coverage_stats = {
            "lines": 95,
            "branches": 80,
            "label_coverage_pct": 50.0,
        }

        should_promote, decision = gate.evaluate(
            production_metrics, shadow_metrics, comparison, coverage_stats
        )

        assert should_promote is False
        assert "Insufficient samples" in decision["reason"][0]

    def test_threshold_customization(self, temp_monitoring_dir):
        """Test that custom thresholds work correctly."""
        # Stricter gate
        strict_gate = EvaluationGate(
            min_f1_improvement_pct=5.0,
            max_brier_degradation=0.005,  # Require 5% improvement
            promotion_cooldown_days=0,
            decisions_path=temp_monitoring_dir,
        )

        production_metrics = {
            "num_samples": 250,
            "primary_metrics": {"f1_score": 0.80},
            "calibration_metrics": {"brier_score": 0.15},
        }

        shadow_metrics = {
            "num_samples": 250,
            "primary_metrics": {"f1_score": 0.83},  # 3.75% improvement
            "calibration_metrics": {"brier_score": 0.15},
        }

        comparison = {"f1_improvement_pct": 3.75, "brier_change": 0.0}

        should_promote, decision = strict_gate.evaluate(
            production_metrics, shadow_metrics, comparison
        )

        # Should fail strict gate
        assert should_promote is False
