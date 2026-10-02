Param(
    [switch]$NonInteractive
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Title,
        [string]$Command
    )

    Write-Host ""
    Write-Host "================================================================================"
    Write-Host $Title
    Write-Host "================================================================================"
    Write-Host ""

    Invoke-Expression $Command
}

function Wait-Continue {
    param([string]$Message)
    if (-not $NonInteractive) {
        Read-Host $Message | Out-Null
    }
}

Write-Host "================================================================================"
Write-Host "PHASE 5: SELF-HEALING ML SYSTEM - COMPLETE DEMONSTRATION"
Write-Host "================================================================================"
Write-Host ""
Write-Host "This demo will:"
Write-Host "  1. Establish baseline"
Write-Host "  2. Inject covariate shift"
Write-Host "  3. Inject population shift"
Write-Host "  4. Demonstrate manual retraining trigger"
Write-Host "  5. Demonstrate rollback"
Write-Host "  6. Run evaluation audit"
Write-Host ""

Wait-Continue "Press Enter to start demo"

Invoke-Step -Title "DEMO 1/6: BASELINE ESTABLISHMENT" -Command "docker compose exec api python scripts/phase5/demo_01_baseline.py"
Wait-Continue "Press Enter to continue to Demo 2"

Invoke-Step -Title "DEMO 2/6: COVARIATE SHIFT INJECTION" -Command "docker compose exec api python scripts/phase5/demo_02_covariate_shift.py"
Wait-Continue "Press Enter to continue to Demo 3"

Invoke-Step -Title "DEMO 3/6: POPULATION SHIFT INJECTION" -Command "docker compose exec api python scripts/phase5/demo_03_population_shift.py"
Wait-Continue "Press Enter to continue to Demo 4"

Invoke-Step -Title "DEMO 4/6: MANUAL RETRAINING TRIGGER" -Command "docker compose exec api python scripts/phase5/demo_04_manual_trigger.py"
Wait-Continue "Press Enter to continue to Demo 5"

Invoke-Step -Title "DEMO 5/6: ROLLBACK & REJECTION" -Command "docker compose exec api python scripts/phase5/demo_05_rollback.py"
Wait-Continue "Press Enter to continue to Evaluation Audit"

Invoke-Step -Title "DEMO 6/6: EVALUATION AUDIT" -Command "docker compose exec api python scripts/phase5/evaluation_audit.py"

Write-Host ""
Write-Host "================================================================================"
Write-Host "PHASE 5 DEMO SUITE COMPLETE"
Write-Host "================================================================================"
Write-Host ""
$airflowPort = if ($env:AIRFLOW_WEBSERVER_HOST_PORT) { $env:AIRFLOW_WEBSERVER_HOST_PORT } else { "8081" }
Write-Host "Review dashboards:"
Write-Host "  MLflow:  http://localhost:5000"
Write-Host "  Airflow: http://localhost:$airflowPort"
Write-Host "  API:     http://localhost:8000/docs"
Write-Host ""
