"""
Retraining pipeline DAG for self-healing ML system.

This DAG:
1. Checks if retraining is needed (DATA + DRIFT signals)
2. Trains shadow model with temporal windows
3. Performs replay-based evaluation
4. Runs evaluation gate
5. Promotes or rejects shadow model

TRIGGERS:
  - Drift threshold exceeded (immediate response)
  - Scheduled weekly run (routine maintenance)
  - Manual trigger (operator intervention)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import pandas as pd
import os
import sys
import logging

sys.path.append("/app")

logger = logging.getLogger(__name__)


# ============================================================
# Task Functions
# ============================================================


def check_retraining_needed(**context):
    """
    Task 1: Determine if retraining should proceed.

    Checks (in order):
    1. Sufficient labeled data available?
    2. Coverage acceptable?
    3. ✅ DRIFT SIGNALS DETECTED? (Phase 3 integration)

    Triggers:
      - SCHEDULED: Weekly run + sufficient data → ROUTINE retraining

    Returns:
        'train_shadow_model' or 'skip_retraining'
    """
    from src.storage.prediction_logger import get_prediction_logger
    from src.storage.label_store import get_label_store
    from src.analytics.drift_signals import DriftSignalChecker  # ✅ NEW
    from src.storage.repositories import RetrainingDecisionsRepository  # ✅ MOVED HERE

    logger.info("=" * 80)
    logger.info("CHECKING RETRAINING CONDITIONS")
    logger.info("=" * 80)

    try:
        pred_logger = get_prediction_logger()
        label_store = get_label_store()

        # ============================================================
        # Step 1: Check Data Availability
        # ============================================================
        logger.info("\n[Step 1/3] Checking data availability...")

        predictions_df = pred_logger.get_recent_predictions(days=30)

        if predictions_df.empty:
            logger.warning("No predictions in last 30 days. Skipping retraining.")
            return "skip_retraining"

        # Check label coverage
        coverage = label_store.get_label_coverage(predictions_df)

        logger.info(f"  Total predictions: {coverage['total_predictions']}")
        logger.info(f"  Labeled predictions: {coverage['labeled_predictions']}")
        logger.info(f"  Coverage: {coverage['coverage_rate']*100:.1f}%")

        # Minimum requirements
        MIN_LABELED_SAMPLES = 200
        MIN_COVERAGE_PCT = 0.2  # 20%

        has_sufficient_data = (
            coverage["labeled_predictions"] >= MIN_LABELED_SAMPLES
            and coverage["coverage_rate"] >= MIN_COVERAGE_PCT
        )

        if not has_sufficient_data:
            logger.warning(
                f"Insufficient data: {coverage['labeled_predictions']} labeled "
                f"({coverage['coverage_rate']*100:.1f}% coverage)"
            )

        # ============================================================
        # Step 2: Check Drift Signals (NEW - Phase 3 Integration)
        # ============================================================
        logger.info("\n[Step 2/3] Checking drift signals from Phase 3 monitoring...")

        drift_checker = DriftSignalChecker(
            drift_threshold=0.3,  # 30% of features drifted triggers retraining
            lookback_hours=24,  # Check last 24 hours of drift reports
        )

        drift_detected, drift_details = drift_checker.check_drift_signals()

        logger.info(f"  Drift detected: {drift_detected}")
        # Use the new, explicit fields coming from DriftSignalChecker
        logger.info(f"  Observed drift_share: {drift_details.get('drift_share', 0.0):.3f}")
        logger.info(f"  Num drifted features: {drift_details.get('num_drifted_features', 0)}")
        if drift_detected:
            logger.warning(f"  ⚠️ Drift share: {drift_details['drift_share']:.3f}")
            logger.warning(f"  ⚠️ Threshold: {drift_checker.drift_threshold}")
            drifted_features = drift_details.get("drifted_feature_names", [])
            if drifted_features:
                logger.warning(f"  ⚠️ Drifted features: {drifted_features}")
            else:
                logger.warning(
                    "  ⚠️ Dataset-level drift detected, but no reliable feature attribution available"
                )

            logger.warning(
                f"  ⚠️ Drift events: {len(drift_details.get('drift_timestamps', []))}"
            )  # OBSERVATION-ONLY MODE: Log detection but no automated action
            logger.warning("=" * 80)
            logger.warning("⚠️ DRIFT DETECTED (OBSERVATION-ONLY MODE)")
            logger.warning(f"Observed drift_share: {drift_details.get('drift_share', 0.0):.3f}")
            logger.warning(f"Drifted features: {drift_details.get('drifted_feature_names', [])}")
            logger.warning("No automated retraining triggered.")
            logger.warning("=" * 80)
        else:
            logger.info(
                f"  No significant drift (observed drift_share: {drift_details.get('drift_share', 0.0):.3f})"
            )

        # ============================================================
        # Step 3: Make Retraining Decision
        # ============================================================
        logger.info("\n[Step 3/3] Making retraining decision...")

        # Store common info for downstream tasks
        context["task_instance"].xcom_push(key="coverage_stats", value=coverage)
        context["task_instance"].xcom_push(
            key="labeled_count", value=coverage["labeled_predictions"]
        )
        context["task_instance"].xcom_push(key="drift_detected", value=drift_detected)
        context["task_instance"].xcom_push(key="drift_details", value=drift_details)

        # ============================================================
        # Decision Logic (Priority Order)
        # ============================================================

        # Only decision: SCHEDULED RETRAINING (routine)
        if has_sufficient_data:
            logger.info("=" * 80)
            logger.info("✅ SCHEDULED RETRAINING")
            logger.info(f"Labeled samples: {coverage['labeled_predictions']} ✓")
            logger.info(f"Coverage: {coverage['coverage_rate']*100:.1f}% ✓")
            logger.info("Proceeding with SCHEDULED retraining...")
            logger.info("=" * 80)

            context["task_instance"].xcom_push(key="trigger_reason", value="scheduled")
            return "train_shadow_model"

        # No conditions met - SKIP
        logger.info("=" * 80)
        logger.info("⏸️ SKIPPING RETRAINING")
        logger.info("Reasons:")
        logger.info(f"  - Drift detected: {drift_detected}")
        logger.info(f"  - Sufficient data: {has_sufficient_data}")
        logger.info("Will retry on next scheduled run or when drift exceeds threshold.")
        logger.info("=" * 80)

        # In check_retraining_needed(), at the end, BEFORE return statement:
        # ============================================================
        # ✅ NEW: Log decision to database
        # ============================================================
        try:
            decisions_repo = RetrainingDecisionsRepository()

            decision_action = "train" if has_sufficient_data else "skip"

            decisions_repo.insert(
                timestamp=pd.Timestamp.now(),
                trigger_reason="scheduled",  # No auto-drift trigger
                action=decision_action,
                drift_context={
                    "feature_drift_ratio": drift_details.get("drift_share", 0.0),  # ✅ RENAMED
                    "num_drifted_features": len(drift_details.get("drifted_feature_names", [])),
                    "dataset_drift_detected": drift_detected,
                    "drifted_features": drift_details.get("drifted_feature_names", []),
                },
                data_context={
                    "labeled_samples": coverage["labeled_predictions"],
                    "coverage_pct": coverage["coverage_rate"] * 100,
                },
                decision_details={
                    "reason": (
                        "Insufficient data"
                        if not has_sufficient_data
                        else "Proceeding with training"
                    ),
                    "shadow_model_version": None,
                    "production_model_version": None,
                },
            )

            logger.info("✅ Decision logged to database")

        except Exception as e:
            logger.warning(f"Decision logging failed (non-critical): {e}")

        # Now return
        if has_sufficient_data:
            context["task_instance"].xcom_push(key="trigger_reason", value="scheduled")
            return "train_shadow_model"

        return "skip_retraining"

    except Exception as e:
        logger.error(f"Error checking retraining conditions: {e}")
        logger.exception("Full traceback:")
        return "skip_retraining"


def train_shadow_model(**context):
    """
    Task 2: Train shadow model using temporal windows.

    Uses:
    - Temporal train/eval split (no random splits)
    - Recent labeled data
    - Same architecture as production
    - Trigger reason from upstream task (drift vs scheduled)
    """
    from src.storage.prediction_logger import get_prediction_logger
    from src.storage.label_store import get_label_store
    from src.retraining.shadow_trainer import ShadowModelTrainer

    logger.info("=" * 80)
    logger.info("TRAINING SHADOW MODEL")
    logger.info("=" * 80)

    try:
        # Get trigger reason from upstream task
        trigger_reason = (
            context["task_instance"].xcom_pull(
                task_ids="check_retraining_needed", key="trigger_reason"
            )
            or "scheduled"
        )

        # Get drift details if available
        drift_details = context["task_instance"].xcom_pull(
            task_ids="check_retraining_needed", key="drift_details"
        )

        logger.info(f"Trigger reason: {trigger_reason.upper()}")
        if trigger_reason == "drift_detected" and drift_details:
            logger.info(f"Observed drift_share: {drift_details.get('drift_share', 'N/A')}")
            logger.info(f"Drifted features: {drift_details.get('drifted_feature_names', [])}")

        # Get data
        pred_logger = get_prediction_logger()
        predictions_df = pred_logger.get_predictions_with_features()

        label_store = get_label_store()
        labels_df = label_store.get_labeled_predictions()

        if predictions_df.empty or labels_df.empty:
            raise ValueError("No data available for training")

        # Compute temporal windows dynamically from data
        predictions_df["application_date"] = pd.to_datetime(predictions_df["application_date"])
        sorted_preds = predictions_df.sort_values("application_date").reset_index(drop=True)
        split_idx = int(len(sorted_preds) * 0.7)  # 70% train, 30% eval
        train_end = sorted_preds.iloc[split_idx]["application_date"]
        eval_start = train_end + pd.Timedelta(seconds=1)
        eval_end = sorted_preds["application_date"].max() + pd.Timedelta(seconds=1)

        logger.info(f"Temporal windows:")
        logger.info(f"  Train: up to {train_end.date()}")
        logger.info(f"  Eval: {eval_start.date()} to {eval_end.date()}")

        trainer = ShadowModelTrainer()

        (
            X_train,
            X_eval,
            y_train,
            y_eval,
            validation_status,
        ) = trainer.prepare_training_data_temporal(
            predictions_df=predictions_df,
            labels_df=labels_df,
            train_end_date=str(train_end),
            eval_start_date=str(eval_start),
            eval_end_date=str(eval_end),
        )

        # Train shadow model with validation status and trigger reason
        model, run_id, train_result = trainer.train_shadow_model(
            X_train,
            y_train,
            X_eval,
            y_eval,
            trigger_reason=f"airflow_{trigger_reason}",  # e.g., "airflow_drift_detected"
            validation_status=validation_status,
        )

        # Check if training was aborted
        if train_result.get("status") == "aborted":
            logger.warning(f"⚠️ Shadow model training aborted: {train_result.get('reason')}")
            context["task_instance"].xcom_push(key="training_status", value="aborted")
            return run_id

        logger.info(f"✅ Shadow model trained: {run_id}")

        # Get shadow model version from MLflow
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        shadow_versions = client.search_model_versions(
            f"name='credit-risk-model' and run_id='{run_id}'"
        )

        shadow_version = shadow_versions[0].version if shadow_versions else None

        # Store for next tasks
        context["task_instance"].xcom_push(key="shadow_run_id", value=run_id)
        context["task_instance"].xcom_push(key="shadow_version", value=shadow_version)
        context["task_instance"].xcom_push(key="training_status", value="success")

        return run_id

    except Exception as e:
        logger.error(f"Shadow model training failed: {e}")
        logger.exception("Full traceback:")
        raise


def evaluate_models_replay(**context):
    """
    Task 3: Replay-based evaluation (CRITICAL: identical samples).

    Re-scores evaluation data with:
    - Production model
    - Shadow model

    Ensures fair comparison on same data.
    """
    from src.storage.prediction_logger import get_prediction_logger
    from src.storage.label_store import get_label_store
    from src.analytics.model_evaluator import ModelEvaluator
    from src.retraining.shadow_trainer import ShadowModelTrainer
    import mlflow

    logger.info("=" * 80)
    logger.info("REPLAY-BASED EVALUATION")
    logger.info("=" * 80)

    try:
        # Get shadow model info
        shadow_version = context["task_instance"].xcom_pull(
            task_ids="train_shadow_model", key="shadow_version"
        )

        if not shadow_version:
            raise ValueError("Shadow model version not found")

        # Get production model version
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        prod_versions = client.get_latest_versions("credit-risk-model", stages=["Production"])

        if not prod_versions:
            logger.warning("No production model found. First deployment scenario.")
            # First deployment - no comparison needed
            context["task_instance"].xcom_push(key="is_first_deployment", value=True)
            return

        prod_version = prod_versions[0].version

        # Get evaluation data (predictions with labels)
        pred_logger = get_prediction_logger()
        label_store = get_label_store()

        predictions_df = pred_logger.get_predictions_with_features()
        labeled_df = label_store.get_labeled_predictions(predictions_df)

        if labeled_df.empty:
            raise ValueError("No labeled predictions for evaluation")

        # Use same eval window as training
        labeled_df["application_date"] = pd.to_datetime(labeled_df["application_date"])
        predictions_df["application_date"] = pd.to_datetime(predictions_df["application_date"])

        sorted_preds = predictions_df.sort_values("application_date").reset_index(drop=True)
        split_idx = int(len(sorted_preds) * 0.7)
        train_end = sorted_preds.iloc[split_idx]["application_date"]
        eval_start = train_end + pd.Timedelta(seconds=1)
        eval_end = sorted_preds["application_date"].max() + pd.Timedelta(seconds=1)

        eval_df = labeled_df[
            (labeled_df["application_date"] >= eval_start)
            & (labeled_df["application_date"] < eval_end)
        ].copy()

        if len(eval_df) < 100:
            raise ValueError(f"Insufficient evaluation samples: {len(eval_df)}")

        logger.info(f"Evaluation samples: {len(eval_df)}")

        # Replay-based evaluation
        evaluator = ModelEvaluator()

        feature_columns = ShadowModelTrainer.FEATURE_COLUMNS

        prod_metrics, shadow_metrics = evaluator.replay_evaluation(
            eval_df=eval_df,
            production_model_version=prod_version,
            shadow_model_version=shadow_version,
            feature_columns=feature_columns,
        )

        # Compare
        comparison = evaluator.compare_models(prod_metrics, shadow_metrics)

        logger.info("=" * 80)
        logger.info(f"F1 improvement: {comparison['f1_improvement_pct']:+.2f}%")
        logger.info(f"Brier change: {comparison['brier_change']:+.4f}")
        logger.info("=" * 80)

        # Store for gate
        context["task_instance"].xcom_push(key="prod_metrics", value=prod_metrics)
        context["task_instance"].xcom_push(key="shadow_metrics", value=shadow_metrics)
        context["task_instance"].xcom_push(key="comparison", value=comparison)
        context["task_instance"].xcom_push(key="is_first_deployment", value=False)

    except Exception as e:
        logger.error(f"Replay evaluation failed: {e}")
        logger.exception("Full traceback:")
        raise


def run_evaluation_gate(**context):
    """
    Task 4: Run evaluation gate (multi-criteria decision).

    Includes drift context in decision metadata.

    Returns:
        'promote_shadow_model' or 'reject_shadow_model'
    """
    from src.retraining.evaluation_gate import EvaluationGate
    from src.storage.repositories import RetrainingDecisionsRepository  # ✅ MOVED HERE

    logger.info("=" * 80)
    logger.info("RUNNING EVALUATION GATE")
    logger.info("=" * 80)

    try:
        # Check if first deployment
        is_first_deployment = context["task_instance"].xcom_pull(
            task_ids="evaluate_models_replay", key="is_first_deployment"
        )

        if is_first_deployment:
            logger.info("✅ First deployment - auto-promoting")
            decision = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "final_decision": True,
                "reason": ["First model deployment - no comparison needed"],
            }
            context["task_instance"].xcom_push(key="decision", value=decision)
            return "promote_shadow_model"

        # Get evaluation results
        prod_metrics = context["task_instance"].xcom_pull(
            task_ids="evaluate_models_replay", key="prod_metrics"
        )
        shadow_metrics = context["task_instance"].xcom_pull(
            task_ids="evaluate_models_replay", key="shadow_metrics"
        )
        comparison = context["task_instance"].xcom_pull(
            task_ids="evaluate_models_replay", key="comparison"
        )
        coverage_stats = context["task_instance"].xcom_pull(
            task_ids="check_retraining_needed", key="coverage_stats"
        )

        # Get trigger context (drift vs scheduled)
        trigger_reason = context["task_instance"].xcom_pull(
            task_ids="check_retraining_needed", key="trigger_reason"
        )
        drift_details = context["task_instance"].xcom_pull(
            task_ids="check_retraining_needed", key="drift_details"
        )

        # Initialize gate
        gate = EvaluationGate(
            min_f1_improvement_pct=2.0,
            max_brier_degradation=0.01,
            max_segment_regression_pct=5.0,
            min_samples_for_decision=200,
            min_coverage_pct=30.0,
            promotion_cooldown_days=7,
        )

        # Run gate
        should_promote, decision = gate.evaluate(
            production_metrics=prod_metrics,
            shadow_metrics=shadow_metrics,
            comparison=comparison,
            coverage_stats=coverage_stats,
        )
        # ✅ NEW: Log gate decision to database
        try:
            decisions_repo = RetrainingDecisionsRepository()

            # Normalize data context
            data_context = {}
            if coverage_stats:
                data_context = {
                    "labeled_samples": (
                        coverage_stats.get("labeled_predictions")
                        if "labeled_predictions" in coverage_stats
                        else coverage_stats.get("labeled_samples")
                    ),
                    "coverage_pct": (
                        coverage_stats.get("coverage_rate", 0) * 100.0
                        if "coverage_rate" in coverage_stats
                        else coverage_stats.get(
                            "coverage_pct", coverage_stats.get("label_coverage_pct", 0.0)
                        )
                    ),
                }

            decisions_repo.insert(
                timestamp=pd.Timestamp.now(),
                trigger_reason=trigger_reason,
                action="promote" if should_promote else "reject",
                drift_context={
                    "feature_drift_ratio": (
                        drift_details.get("drift_share", 0.0) if drift_details else 0.0
                    ),
                    "num_drifted_features": (
                        len(drift_details.get("drifted_feature_names", [])) if drift_details else 0
                    ),
                    "dataset_drift_detected": drift_details is not None
                    and drift_details.get("drift_share", 0) > 0,
                    "drifted_features": (
                        drift_details.get("drifted_feature_names", []) if drift_details else []
                    ),
                },
                data_context=data_context,
                decision_details={
                    "reason": "; ".join(decision["reason"]),
                    "failed_gate": (
                        decision.get("gate_results", {}).get("failed_gate")
                        if not should_promote
                        else None
                    ),
                    "shadow_model_version": context["task_instance"].xcom_pull(
                        task_ids="train_shadow_model", key="shadow_version"
                    ),
                    "production_model_version": None,  # Will be retrieved during promotion if needed
                    "f1_improvement_pct": comparison.get("f1_improvement_pct"),
                    "brier_change": comparison.get("brier_change"),
                },
            )

            logger.info("✅ Gate decision logged to database")

        except Exception as e:
            logger.warning(f"Gate decision logging failed (non-critical): {e}")

        # ✅ NEW: Add drift context to decision metadata
        decision["trigger_context"] = {
            "trigger_reason": trigger_reason,
            "drift_detected": drift_details is not None and drift_details.get("drift_share", 0) > 0,
            "drift_details": drift_details,
        }

        # ✅ Add explicit drift interpretation for clarity
        drift_share = drift_details.get("drift_share", 0.0) if drift_details else 0.0
        drifted_feature_names = (
            drift_details.get("drifted_feature_names", []) if drift_details else []
        )

        if drift_details and drift_share > 0:
            drift_interpretation = (
                f"Dataset-level drift detected (drift_share={drift_share:.3f}). "
                f"Feature attribution available for {len(drifted_feature_names)} features."
            )
        else:
            drift_interpretation = "No significant drift detected."

        decision["drift_interpretation"] = drift_interpretation

        logger.info("=" * 80)
        logger.info(f"Gate Decision: {'PROMOTE' if should_promote else 'REJECT'}")
        logger.info(f"Trigger: {trigger_reason.upper()}")
        if trigger_reason == "drift_detected":
            logger.info(f"Drift score: {drift_details.get('drift_share', 'N/A')}")
        logger.info(f"Drift interpretation: {drift_interpretation}")
        logger.info(f"Reason: {decision['reason']}")
        logger.info("=" * 80)

        # Store decision
        context["task_instance"].xcom_push(key="decision", value=decision)

        return "promote_shadow_model" if should_promote else "reject_shadow_model"

    except Exception as e:
        logger.error(f"Evaluation gate failed: {e}")
        logger.exception("Full traceback:")
        return "reject_shadow_model"


def promote_model(**context):
    """Task 5a: Promote shadow model to production."""
    from src.retraining.model_promoter import ModelPromoter

    logger.info("=" * 80)
    logger.info("PROMOTING SHADOW MODEL")
    logger.info("=" * 80)

    try:
        shadow_run_id = context["task_instance"].xcom_pull(
            task_ids="train_shadow_model", key="shadow_run_id"
        )
        decision = context["task_instance"].xcom_pull(
            task_ids="run_evaluation_gate", key="decision"
        )

        # Log trigger context
        trigger_context = decision.get("trigger_context", {})
        trigger_reason = trigger_context.get("trigger_reason", "unknown")

        logger.info(f"Promotion trigger: {trigger_reason.upper()}")
        if trigger_reason == "drift_detected":
            drift_details = trigger_context.get("drift_details", {})
            logger.info(f"Responding to drift: {drift_details.get('drift_share', 'N/A')}")

        promoter = ModelPromoter()
        result = promoter.promote_to_production(
            shadow_run_id=shadow_run_id, evaluation_decision=decision, promoted_by="airflow_dag"
        )

        if result["success"]:
            logger.info(f"✅ Promotion successful: v{result['new_production_version']}")
            if trigger_reason == "drift_detected":
                logger.info("🔄 Drift-triggered retraining completed successfully")
        else:
            logger.error(f"❌ Promotion failed: {result.get('error')}")
            raise ValueError(result.get("error"))

    except Exception as e:
        logger.error(f"Promotion task failed: {e}")
        logger.exception("Full traceback:")
        raise


def reject_model(**context):
    """Task 5b: Reject shadow model (SUCCESSFUL outcome)."""
    from src.retraining.model_promoter import ModelPromoter

    logger.info("=" * 80)
    logger.info("REJECTING SHADOW MODEL (SUCCESSFUL GATE OPERATION)")
    logger.info("=" * 80)

    try:
        shadow_run_id = context["task_instance"].xcom_pull(
            task_ids="train_shadow_model", key="shadow_run_id"
        )
        decision = context["task_instance"].xcom_pull(
            task_ids="run_evaluation_gate", key="decision"
        )

        # Log trigger context
        trigger_context = decision.get("trigger_context", {})
        trigger_reason = trigger_context.get("trigger_reason", "unknown")

        logger.info(f"Rejection context: {trigger_reason.upper()}")
        if trigger_reason == "drift_detected":
            logger.warning("⚠️ Drift was detected but shadow model did not meet quality gates")
            logger.warning("System correctly prevented deployment of inadequate model")

        promoter = ModelPromoter()
        result = promoter.reject_shadow_model(
            shadow_run_id=shadow_run_id, evaluation_decision=decision, rejected_by="airflow_dag"
        )

        logger.info("✅ Rejection recorded successfully")
        logger.info("   Gate prevented inadequate model deployment")

    except Exception as e:
        logger.error(f"Rejection task failed: {e}")
        logger.exception("Full traceback:")
        raise


def log_skip(**context):
    """Log when retraining is skipped."""
    logger.info("=" * 80)
    logger.info("RETRAINING SKIPPED")
    logger.info("Conditions not met:")
    logger.info("  - No significant drift detected")
    logger.info("  - Insufficient labeled data for scheduled retraining")
    logger.info("Will retry on next schedule or when drift threshold exceeded.")
    logger.info("=" * 80)


# ============================================================
# DAG Definition
# ============================================================

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retraining_pipeline",
    default_args=default_args,
    description="Self-healing ML retraining pipeline with drift detection and evaluation gates",
    schedule_interval="@weekly",  # Run weekly (but can also trigger on drift)
    start_date=days_ago(1),
    catchup=False,
    tags=["mlops", "retraining", "phase4", "self-healing"],  # ✅ Added 'self-healing' tag
) as dag:
    # Task 1: Check conditions (branching) - NOW INCLUDES DRIFT CHECK
    check_task = BranchPythonOperator(
        task_id="check_retraining_needed",
        python_callable=check_retraining_needed,
        provide_context=True,
    )

    # Task 2: Train shadow model
    train_task = PythonOperator(
        task_id="train_shadow_model", python_callable=train_shadow_model, provide_context=True
    )

    # Task 3: Replay evaluation
    evaluate_task = PythonOperator(
        task_id="evaluate_models_replay",
        python_callable=evaluate_models_replay,
        provide_context=True,
    )

    # Task 4: Evaluation gate (branching)
    gate_task = BranchPythonOperator(
        task_id="run_evaluation_gate", python_callable=run_evaluation_gate, provide_context=True
    )

    # Task 5a: Promote
    promote_task = PythonOperator(
        task_id="promote_shadow_model", python_callable=promote_model, provide_context=True
    )

    # Task 5b: Reject
    reject_task = PythonOperator(
        task_id="reject_shadow_model", python_callable=reject_model, provide_context=True
    )

    # Task 6: Skip handler
    skip_task = PythonOperator(
        task_id="skip_retraining", python_callable=log_skip, provide_context=True
    )

    # Define workflow
    check_task >> [train_task, skip_task]
    train_task >> evaluate_task >> gate_task
    gate_task >> [promote_task, reject_task]
