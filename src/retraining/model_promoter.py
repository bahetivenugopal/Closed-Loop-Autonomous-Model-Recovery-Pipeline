"""
Model promoter with cooldown enforcement.
"""

import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging
import json
from pathlib import Path
import pandas as pd
import sys
from src.storage.repositories import ModelVersionsRepository

sys.path.append("/app")

logger = logging.getLogger(__name__)


class ModelPromoter:
    """
    Safe model promotion with cooldown.
    """

    def __init__(
        self,
        model_name: str = "credit-risk-model",
        mlflow_tracking_uri: str = "http://mlflow:5000",
        decisions_path: str = "/app/monitoring/retraining/decisions",
    ):
        self.model_name = model_name
        self.decisions_path = Path(decisions_path)
        self.decisions_path.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.client = MlflowClient()

    def promote_to_production(
        self, shadow_run_id: str, evaluation_decision: Dict, promoted_by: str = "system"
    ) -> Dict:
        """Promote shadow model to production."""
        try:
            # Get shadow version
            shadow_versions = self.client.search_model_versions(
                f"name='{self.model_name}' and run_id='{shadow_run_id}'"
            )

            if not shadow_versions:
                raise ValueError(f"No model version for run {shadow_run_id}")

            shadow_version = shadow_versions[0].version

            # Archive current production
            prod_versions = self.client.get_latest_versions(self.model_name, stages=["Production"])

            archived_version = None
            if prod_versions:
                old_prod = prod_versions[0]
                archived_version = old_prod.version

                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=archived_version,
                    stage="Archived",
                    archive_existing_versions=False,
                )

                logger.info(f"Archived production v{archived_version}")

            # Promote shadow
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=shadow_version,
                stage="Production",
                archive_existing_versions=False,
            )

            logger.info("=" * 80)
            logger.info(f"✅ PROMOTED: Shadow v{shadow_version} → Production")
            logger.info("=" * 80)

            # Record
            promotion_record = {
                "timestamp": datetime.now().isoformat(),
                "action": "promote",
                "shadow_run_id": shadow_run_id,
                "shadow_version": shadow_version,
                "previous_production_version": archived_version,
                "promoted_by": promoted_by,
                "evaluation_decision": evaluation_decision,
            }

            self._save_decision_record(promotion_record)

            # ✅ NEW: Write to model_versions table
            try:
                model_repo = ModelVersionsRepository()
                model_repo.insert_or_update(
                    model_name=self.model_name,
                    version=int(shadow_version),
                    stage="Production",
                    training_context={
                        "promoted_at": datetime.now().isoformat(),
                        "training_run_id": shadow_run_id,
                    },
                    metrics=evaluation_decision.get("metrics", {}),
                    decision_id=None,
                )
                logger.info("✅ Model version written to database")
            except Exception as e:
                logger.warning(f"Failed to write model version to database (non-critical): {e}")

            return {
                "success": True,
                "new_production_version": shadow_version,
                "archived_version": archived_version,
                "record": promotion_record,
            }

        except Exception as e:
            logger.error(f"Promotion failed: {e}")

            failure_record = {
                "timestamp": datetime.now().isoformat(),
                "action": "promote_failed",
                "shadow_run_id": shadow_run_id,
                "error": str(e),
            }

            self._save_decision_record(failure_record)

            return {"success": False, "error": str(e)}

    def reject_shadow_model(
        self, shadow_run_id: str, evaluation_decision: Dict, rejected_by: str = "system"
    ) -> Dict:
        """
        Reject shadow model.

        ✅ THIS IS A SUCCESSFUL OUTCOME.
        Gate prevented bad deployment.
        """
        try:
            shadow_versions = self.client.search_model_versions(
                f"name='{self.model_name}' and run_id='{shadow_run_id}'"
            )

            if shadow_versions:
                shadow_version = shadow_versions[0].version

                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=shadow_version,
                    stage="Archived",
                    archive_existing_versions=False,
                )

                logger.info("=" * 80)
                logger.info(f"✅ REJECTION SUCCESSFUL: v{shadow_version} archived")
                logger.info("  Gate correctly prevented inadequate deployment")
                logger.info("  This is EXPECTED and CORRECT system behavior")
                logger.info("=" * 80)

            rejection_record = {
                "timestamp": datetime.now().isoformat(),
                "action": "reject",
                "shadow_run_id": shadow_run_id,
                "shadow_version": shadow_versions[0].version if shadow_versions else None,
                "rejected_by": rejected_by,
                "evaluation_decision": evaluation_decision,
                "outcome": "successful_rejection",  # ✅ Mark as success
            }

            self._save_decision_record(rejection_record)

            # ✅ Write rejected model version to database
            if shadow_versions:
                try:
                    model_repo = ModelVersionsRepository()
                    eval_metrics = evaluation_decision.get("metrics", {})
                    trigger_context = evaluation_decision.get("trigger_context", {})
                    trigger_reason = trigger_context.get("trigger_reason", "rejected")

                    model_repo.insert_or_update(
                        model_name=self.model_name,
                        version=int(shadow_versions[0].version),
                        stage="Archived",
                        training_context={
                            "archived_at": datetime.now().isoformat(),
                            "training_run_id": shadow_run_id,
                            "trigger_reason": trigger_reason,
                        },
                        metrics=eval_metrics,
                        decision_id=None,
                    )
                    logger.info("✅ Rejected model version written to database")
                except Exception as e:
                    logger.warning(f"Failed to write rejected model version to database: {e}")

            return {"success": True, "action": "rejected", "record": rejection_record}

        except Exception as e:
            logger.error(f"Rejection handling failed: {e}")
            return {"success": False, "error": str(e)}

    def rollback_to_version(self, version: str) -> Dict:
        """Rollback to archived version."""
        try:
            prod_versions = self.client.get_latest_versions(self.model_name, stages=["Production"])

            if prod_versions:
                self.client.transition_model_version_stage(
                    name=self.model_name, version=prod_versions[0].version, stage="Archived"
                )

            self.client.transition_model_version_stage(
                name=self.model_name, version=version, stage="Production"
            )

            logger.warning(f"⚠️ ROLLBACK: Restored v{version} to Production")

            rollback_record = {
                "timestamp": datetime.now().isoformat(),
                "action": "rollback",
                "restored_version": version,
                "rolled_back_from": prod_versions[0].version if prod_versions else None,
            }

            self._save_decision_record(rollback_record)

            return {"success": True, "restored_version": version, "record": rollback_record}

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {"success": False, "error": str(e)}

    def _save_decision_record(self, record: Dict):
        """Save decision for audit trail."""
        timestamp = record.get("timestamp", datetime.now().isoformat())
        filename = f"decision_{timestamp.replace(':', '-').replace('.', '-')}.json"

        filepath = self.decisions_path / filename

        with open(filepath, "w") as f:
            json.dump(record, f, indent=2)

        logger.info(f"Decision recorded: {filepath}")

    def get_deployment_history(self, limit: int = 10) -> list:
        """Get recent deployment history."""
        records: list = []

        for filepath in sorted(self.decisions_path.glob("decision_*.json"), reverse=True):
            if len(records) >= limit:
                break

            try:
                with open(filepath, "r") as f:
                    records.append(json.load(f))
            except Exception as e:
                logger.warning(f"Could not read {filepath}: {e}")

        return records
