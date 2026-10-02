#!/bin/bash

# ============================================================
# Phase 5: Complete Self-Healing Demo
# ============================================================
# Runs all 5 demo scenarios in sequence
# Total runtime: ~40 minutes
# ============================================================

set -e  # Exit on error

echo "================================================================================"
echo "PHASE 5: SELF-HEALING ML SYSTEM - COMPLETE DEMONSTRATION"
echo "================================================================================"
echo ""
echo "This demo will:"
echo "  1. Establish baseline (6 min)"
echo "  2. Inject covariate shift (6 min)"
echo "  3. Inject population shift (6 min)"
echo "  4. Demonstrate manual retraining trigger (2 min)"
echo "  5. Demonstrate rollback (2 min)"
echo "  6. Run evaluation audit (1 min)"
echo ""
echo "Total estimated time: ~35 minutes"
echo ""
read -p "Press Enter to start demo..."

# ============================================================
# Demo 1: Baseline
# ============================================================
echo ""
echo "================================================================================"
echo "DEMO 1/5: BASELINE ESTABLISHMENT"
echo "================================================================================"
echo ""
docker compose exec api python scripts/phase5/demo_01_baseline.py

echo ""
echo "✅ Demo 1 complete. Baseline established."
echo ""
read -p "Press Enter to continue to Demo 2..."

# ============================================================
# Demo 2: Covariate Shift
# ============================================================
echo ""
echo "================================================================================"
echo "DEMO 2/5: COVARIATE SHIFT INJECTION"
echo "================================================================================"
echo ""
docker compose exec api python scripts/phase5/demo_02_covariate_shift.py

echo ""
echo "✅ Demo 2 complete. Covariate shift injected."
echo ""
read -p "Press Enter to continue to Demo 3..."

# ============================================================
# Demo 3: Population Shift
# ============================================================
echo ""
echo "================================================================================"
echo "DEMO 3/5: POPULATION SHIFT INJECTION"
echo "================================================================================"
echo ""
docker compose exec api python scripts/phase5/demo_03_population_shift.py

echo ""
echo "✅ Demo 3 complete. Population shift injected."
echo ""
read -p "Press Enter to continue to Demo 4..."

# ============================================================
# Demo 4: Manual Trigger
# ============================================================
echo ""
echo "================================================================================"
echo "DEMO 4/5: MANUAL RETRAINING TRIGGER"
echo "================================================================================"
echo ""
docker compose exec api python scripts/phase5/demo_04_manual_trigger.py

echo ""
echo "✅ Demo 4 complete. Manual trigger workflow demonstrated."
echo ""
read -p "Press Enter to continue to Demo 5..."

# ============================================================
# Demo 5: Rollback
# ============================================================
echo ""
echo "================================================================================"
echo "DEMO 5/5: ROLLBACK & REJECTION DEMONSTRATION"
echo "================================================================================"
echo ""
docker compose exec api python scripts/phase5/demo_05_rollback.py

echo ""
echo "✅ Demo 5 complete."
echo ""
read -p "Press Enter to continue to Demo 6..."

# ============================================================
# Demo 6: Evaluation Audit
# ============================================================
echo ""
echo "================================================================================"
echo "DEMO 6/6: EVALUATION AUDIT"
echo "================================================================================"
echo ""
docker compose exec api python scripts/phase5/evaluation_audit.py

echo ""
echo "✅ Demo 6 complete."
echo ""

# ============================================================
# Summary
# ============================================================
echo ""
echo "================================================================================"
echo "🎉 PHASE 5 DEMO SUITE COMPLETE"
echo "================================================================================"
echo ""
echo "All 5 demos executed successfully!"
echo ""
echo "Review results:"
echo "  - Drift logs:      cat monitoring/drift_injections/drift_log.json | jq ."
echo "  - Drift reports:   ls -lh monitoring/reports/drift_reports/"
echo "  - Decisions:       ls -lh monitoring/retraining/decisions/"
echo "  - MLflow models:   http://localhost:5000"
echo "  - Airflow DAGs:    http://localhost:8080"
echo ""
echo "Next steps:"
echo "  - Analyze drift → retraining → decision flow"
echo "  - Create presentation/video walkthrough"
echo "  - Move to Phase 6 (production hardening)"
echo ""
