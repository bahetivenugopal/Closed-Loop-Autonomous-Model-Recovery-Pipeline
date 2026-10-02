# 🔄 Self-Healing MLOps Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-2.9.2-0194E2?style=for-the-badge&logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13-336791?style=for-the-badge&logo=postgresql)

**Production-grade ML system with automated drift detection and intelligent retraining**

[Quick Start](#-quick-start) • [Architecture](docs/architecture.md) • [Documentation](#-documentation) • [API Docs](http://localhost:8000/docs)

</div>

---

## 🎯 Overview

A **production-ready MLOps pipeline** that continuously monitors model performance, detects data drift, and automatically retrains models using shadow deployment and multi-criteria evaluation gates.

### Why This Project Stands Out

| Typical ML Project | This Implementation |
|-------------------|---------------------|
| Train → Deploy → Done | Continuous monitoring + auto-retraining |
| No drift detection | Statistical drift detection (Evidently AI) |
| Manual model updates | Automated shadow models + evaluation gates |
| Local development only | Full Docker stack + CI/CD pipeline |
| Instant labels assumed | Handles real-world delayed feedback |

**Built for:** Credit risk prediction with real-world constraints and production MLOps patterns.

---

## ✨ Key Features

- 🎯 **Real-Time Predictions** - FastAPI service with <100ms latency
- 📊 **Drift Detection** - Statistical monitoring without requiring labels (Evidently AI)
- 🔄 **Smart Retraining** - Shadow models with 4-gate evaluation (performance, calibration, fairness, validation)
- 🗄️ **PostgreSQL Backend** - Scalable storage for predictions, labels, and metrics
- 📈 **MLflow Integration** - Full experiment tracking and model registry
- 🐳 **Production Ready** - Docker Compose orchestration with 8 services
- 🧪 **Tested** - Unit and integration suites, fully mocked so they run without the stack up
- 🚀 **CI/CD Pipeline** - GitHub Actions with automated quality checks

---

## 🏗️ Architecture
<img width="1920" height="1080" alt="Self- Healing MLOPs Pipline Architecture" src="https://github.com/user-attachments/assets/ed99e993-d58e-4e58-94d7-632d5d592858" />

**[📖 Detailed Architecture](docs/architecture.md)** | **[🔍 Evaluation Gates](docs/evaluation_gates.md)**

---

## 🚀 Quick Start

### Prerequisites
```bash
docker --version         # 20.10+
docker compose version   # 2.0+  (Compose V2 — a subcommand, not a hyphen)
python --version         # 3.10   (3.11+ cannot install this project's pinned dependencies)
```

Allow Docker Desktop at least **8 GB** of memory — the stack runs two PostgreSQL instances plus
Airflow's webserver and scheduler.

### Setup (Build Locally)

> **Full step-by-step walkthrough with explanations: [SETUP_GUIDE.md](SETUP_GUIDE.md).**
> The condensed version is below.

```bash
# Clone repository
git clone https://github.com/nimish1106/self-healing-mlops.git
cd self-healing-mlops

# Create env file, then EDIT IT — .env.example contains placeholders, not working values.
# AIRFLOW_FERNET_KEY in particular must be a real key or Airflow will not start:
#   python -c "import base64,os;print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
cp .env.example .env

# Start the foundation first (the database schema auto-applies on first boot)
docker compose up -d postgres postgres-mlops mlflow

# Bootstrap reference data for drift detection, then train the initial model.
# NOTE: docker-compose.yml has no `bootstrap` or `trainer` service — run them as
# one-off containers from the api image:
docker compose run --rm api python scripts/bootstrap_reference.py
docker compose run --rm api python -m src.train_model_mlflow
# (training auto-registers the model and promotes it to Production)

# Now start serving and monitoring
docker compose up -d api monitoring-job

# Airflow (optional, needed only for the retraining pipeline)
docker compose up -d airflow-webserver airflow-scheduler
# The admin user is NOT created automatically in this compose file — create it:
docker exec airflow-webserver airflow users create \
  --username admin --password admin --firstname Admin --lastname User \
  --role Admin --email admin@example.com
```

> ⚠️ The API cannot start serving until a model is in the `Production` stage. Always bootstrap and
> train **before** starting `api`.

### Setup (Share Mode With Prebuilt Images)
Use this mode when sharing with a friend so they only need to pull images.

```bash
# Clone repository
git clone https://github.com/nimish1106/self-healing-mlops.git
cd self-healing-mlops

# Create env file from template
cp .env.example .env

# Pull all images declared in share compose file
docker compose -f docker-compose.share.yml pull

# Start core infra first
docker compose -f docker-compose.share.yml up -d postgres postgres-mlops mlflow airflow-webserver airflow-scheduler pgadmin

# Initialize reference data and train initial model
docker compose -f docker-compose.share.yml run --rm bootstrap
docker compose -f docker-compose.share.yml up trainer

# Start API and monitoring after model is in Production
docker compose -f docker-compose.share.yml up -d api monitoring
```

### Restore Images/Containers On Your Own Machine
If local images/containers were removed, run:

```bash
docker pull nimish1106/sh-mlops-base:v4
docker pull nimish1106/sh-mlops-monitoring:v1
docker pull nimish1106/sh-mlops-airflow-scheduler:v1
docker pull nimish1106/sh-mlops-airflow-web:v1
docker pull ghcr.io/mlflow/mlflow:v2.9.2
docker pull postgres:13
docker pull dpage/pgadmin4

cp .env.example .env
docker compose -f docker-compose.share.yml up -d
docker compose -f docker-compose.share.yml run --rm bootstrap
docker compose -f docker-compose.share.yml up trainer
docker compose -f docker-compose.share.yml up -d api monitoring
```

> These image tags are the defaults in `docker-compose.share.yml` (`BASE_IMAGE` defaults to **v4**,
> not v3). In share mode the Airflow UI is on port **8080** and the monitoring service is named
> `monitoring`; in build mode they are **8081** and `monitoring-job`.

### Verify Installation
```bash
# Check services
docker compose ps

# Test API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.77,
    "age": 45,
    "NumberOfTime30_59DaysPastDueNotWorse": 2,
    "DebtRatio": 0.80,
    "MonthlyIncome": 9120,
    "NumberOfOpenCreditLinesAndLoans": 13,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 6,
    "NumberOfTime60_89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2
  }'
```

**Expected Response:**
```json
{
  "prediction": 0,
  "probability": 0.087,
  "model_version": "1",
  "prediction_id": "pred_20260115_143022_918233",
  "timestamp": "2026-01-15T14:30:22.918233",
  "request_id": "req_4f2a9c7b1e03"
}
```

> On Windows PowerShell, `curl` is an alias for `Invoke-WebRequest` and will not accept the flags
> above. Use `Invoke-RestMethod` instead, or run the command from Git Bash. See
> [SETUP_GUIDE.md](SETUP_GUIDE.md).

---

## 📂 Project Structure
```
self-healing-mlops/
├── src/
│   ├── api_mlflow.py              # FastAPI prediction service
│   ├── train_model_mlflow.py      # Initial model training
│   ├── config.py                  # Env-driven settings
│   ├── features/schema.py         # Canonical feature order (single source of truth)
│   ├── storage/                   # PostgreSQL repositories, prediction & label stores
│   ├── analytics/                 # Drift detection, proxy metrics, model evaluator
│   ├── monitoring/                # Monitoring job
│   ├── retraining/                # Shadow trainer, evaluation gate, model promoter
│   ├── simulation/                # Traffic & drift simulators
│   └── orchestration/             # SimpleScheduler (the 5-min monitoring loop)
│
├── airflow/dags/
│   └── retraining_pipeline.py     # The retraining DAG
│
├── scripts/
│   ├── bootstrap_reference.py     # Freeze the drift baseline
│   ├── generate_fake_predictions.py  # Traffic generator with injectable drift
│   ├── db/schema.sql              # PostgreSQL schema (auto-applied on first boot)
│   └── phase5/                    # End-to-end demo scripts
│
├── tests/
│   ├── unit/                      # Unit tests (drift, gates, metrics)
│   └── integration/               # API & workflow tests (all mocked)
│
├── docs/
│   ├── architecture.md            # System design & diagrams
│   ├── api.md                     # API reference
│   ├── database.md                # Schema & common queries
│   ├── evaluation_gates.md        # Retraining decision logic
│   ├── runbook.md                 # Operations guide
│   ├── troubleshooting.md         # Common issues & fixes
│   └── TECHSTACK.md               # Full technology inventory
│
├── monitoring/                    # Monitoring outputs (bind-mounted into containers)
│   ├── reference/                 # Frozen baseline (immutable)
│   ├── reports/drift_reports/     # Drift reports (HTML + JSON)
│   └── retraining/decisions/      # Promotion/rejection audit trail
│
├── kindsetup/                     # Kubernetes (kind) manifests — alternative deployment
├── docker-compose.yml             # Build-from-source stack (8 services)
├── docker-compose.share.yml       # Prebuilt-image stack (10 services)
├── SETUP_GUIDE.md                 # Step-by-step local setup
└── .github/workflows/ci-cd.yml    # GitHub Actions pipeline
```

> Note: `src/orchestration/` holds the monitoring scheduler, **not** the Airflow DAGs. DAGs live in
> `airflow/dags/`.

---

## 🗄️ Services & Ports

| Service | Compose service | Container name | URL | Purpose |
|---------|-----------------|----------------|-----|---------|
| **FastAPI** | `api` | `mlops-api` | http://localhost:8000 | Prediction API + docs |
| **MLflow** | `mlflow` | `mlflow-server` | http://localhost:5000 | Experiment tracking + registry |
| **Airflow** | `airflow-webserver` | `airflow-webserver` | http://localhost:8081 | Workflow orchestration |
| **PgAdmin** | `pgadmin` | `pgadmin` | http://localhost:5050 | Database UI |
| **PostgreSQL** | `postgres-mlops` | `mlops-postgres` | localhost:5433 | Predictions & labels storage |
| **Monitoring** | `monitoring-job` | `mlops-monitoring-job` | — | 5-min drift detection loop |
| **Airflow DB** | `postgres` | `airflow-postgres` | — | Airflow metadata |

> Service names and container names differ. Use the **service** name with `docker compose ...` and
> the **container** name with `docker exec ...`.

**Airflow Login:** whatever you set in `.env` (`AIRFLOW_USERNAME` / `AIRFLOW_PASSWORD`) — but note
the build-mode compose file does not pass those into the container, so the admin user must be
created manually (see Quick Start above).
**PgAdmin Login:** the values you set for `PGADMIN_EMAIL` / `PGADMIN_PASSWORD` in `.env`.
Airflow's port is controlled by `AIRFLOW_WEBSERVER_HOST_PORT` (default `8081` in build mode).

---

## 🧪 Testing
```bash
# All tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# View coverage report
open htmlcov/index.html
```

All tests — unit **and** integration — mock MLflow and PostgreSQL via `tests/conftest.py`, so the
suite runs without the Docker stack up.

**Coverage:** the committed `coverage.xml` snapshot reports **257/896 lines (28.7%)** from a partial
run. Regenerate a current figure with `pytest tests/ --cov=src --cov-report=term-missing`.

---

## 📊 Monitoring & Operations

### Daily Health Check
```bash
# Check all services
docker compose ps

# API health and readiness
curl http://localhost:8000/health | jq
curl http://localhost:8000/ready | jq

# Prediction stats
curl http://localhost:8000/monitoring/stats | jq

# Database row count
docker exec mlops-postgres psql -U "$POSTGRES_USER" -d mlops \
  -c "SELECT COUNT(*) FROM predictions;"
```

### View Drift Reports
```bash
# Latest drift report (interactive HTML)
ls -t monitoring/reports/drift_reports/drift_report_*.html | head -1

# Machine-readable summary for the same run
jq '{drift_share, num_drifted_features, dataset_drift_detected}' \
  "$(ls -t monitoring/reports/drift_reports/drift_summary_*.json | head -1)"

# Monitoring time series lives in PostgreSQL, not on disk
docker exec mlops-postgres psql -U "$POSTGRES_USER" -d mlops \
  -c "SELECT timestamp, num_predictions, feature_drift_ratio FROM monitoring_metrics ORDER BY timestamp DESC LIMIT 5;"
```

### Common Operations
```bash
# Restart service
docker compose restart api

# View logs
docker compose logs -f monitoring-job

# Trigger retraining (Airflow)
docker exec airflow-scheduler airflow dags trigger retraining_pipeline

# Rollback model (via MLflow UI)
# Navigate to Models → Select previous version → Transition to Production
```

**[📋 Full Operations Runbook](docs/runbook.md)**

---

## ⚙️ Configuration

Key environment variables (set in `.env`, consumed by `docker-compose.yml` and `src/config.py`):
```yaml
# Database
POSTGRES_HOST: postgres-mlops     # Docker service name, not localhost
MLOPS_DB_NAME: mlops

# MLflow
MLFLOW_TRACKING_URI: http://mlflow:5000

# Monitoring
MONITORING_INTERVAL: 300          # 5 minutes
MONITORING_LOOKBACK: 24           # 24 hours

# Airflow
AIRFLOW_WEBSERVER_HOST_PORT: 8081
AIRFLOW_FERNET_KEY: <real key required — see SETUP_GUIDE.md>
```

**Retraining gate thresholds are not environment variables.** They are hardcoded as constructor
arguments in `run_evaluation_gate()` inside `airflow/dags/retraining_pipeline.py`:

```python
EvaluationGate(
    min_f1_improvement_pct=2.0,      # Require 2% F1 improvement
    max_brier_degradation=0.01,      # Max calibration loss
    max_segment_regression_pct=5.0,  # Max per-segment F1 drop
    min_samples_for_decision=200,    # Statistical validity
    min_coverage_pct=30.0,           # Min labelled fraction
    promotion_cooldown_days=7,       # Days between promotions
)
```

**[🔍 Evaluation Gate Details](docs/evaluation_gates.md)**

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Setup Guide](SETUP_GUIDE.md) | Step-by-step local setup, from a bare machine |
| [Next Steps](NEXT_STEPS.md) | Current state of this working copy and what remains to get it running |
| [Architecture](docs/architecture.md) | System design with Mermaid diagrams |
| [API Reference](docs/api.md) | Complete endpoint documentation |
| [Database](docs/database.md) | Schema, views and common queries |
| [Evaluation Gates](docs/evaluation_gates.md) | Retraining decision criteria |
| [Operations Runbook](docs/runbook.md) | Day-to-day operations guide |
| [Troubleshooting](docs/troubleshooting.md) | Common issues & solutions |
| [Tech Stack](docs/TECHSTACK.md) | Full technology inventory |

---

## 🔄 CI/CD Pipeline

**GitHub Actions Workflow:**
```
Push to main/develop
  ↓
├─ Code Quality (Black, Flake8, MyPy)
├─ Data Validation (Pandera schemas)
├─ Unit Tests (pytest)
├─ Integration Tests
├─ Docker Build
└─ Deploy (main branch only)
```

![CI](https://github.com/nimish1106/self-healing-mlops/workflows/CI/badge.svg)

---

## 🐛 Troubleshooting

**Services won't start:**
```bash
docker compose down -v          # WARNING: -v destroys the databases
docker compose up -d postgres postgres-mlops mlflow
# schema.sql re-applies automatically on a fresh volume; then re-run bootstrap + training
```

**"Insufficient samples" / monitoring reports `insufficient_data`:**
The monitoring job needs **200+** predictions inside the lookback window. Generate traffic:
```bash
docker compose exec api python scripts/generate_fake_predictions.py \
  --num-samples 1000 --coverage 50 --drift-phase 0
```

**Airflow UI rejects your login:**
The build-mode compose file never passes `AIRFLOW_USERNAME` into the container, so no admin user is
created. Create one:
```bash
docker exec airflow-webserver airflow users create \
  --username admin --password admin --firstname Admin --lastname User \
  --role Admin --email admin@example.com
```

**Database connection issues:**
```bash
docker logs mlops-postgres
docker compose restart postgres-mlops
```

**[📖 Full Troubleshooting Guide](docs/troubleshooting.md)**

---

## 🤝 Contributing
```bash
# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
pytest tests/ -v
black src/ tests/
flake8 src/ tests/

# Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```


---

## 🙏 Acknowledgments

- **Dataset:** [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle)
- **MLflow:** Experiment tracking framework
- **Evidently AI:** Open-source ML monitoring
- **FastAPI:** Modern Python web framework

---


<div align="center">

**Built with discipline • Deployed with confidence • Monitored with precision**

⭐ Star this repo if you find it useful!

[Report Bug](https://github.com/nimish1106/self-healing-mlops/issues) • [Request Feature](https://github.com/nimish1106/self-healing-mlops/issues) • [Documentation](docs/)

</div>
