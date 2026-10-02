
# Operations Runbook

**Self-Healing MLOps Pipeline — Production Operations**

> Commands below assume the local build stack (`docker-compose.yml`). Service names are what you
> pass to `docker compose`; container names are what you pass to `docker exec`. They are **not** the
> same — see the table in [troubleshooting.md](troubleshooting.md).

---

## 1. Daily Health Checks (5 minutes)

```bash
# Services
docker compose ps

# API
curl http://localhost:8000/health
curl http://localhost:8000/ready

# Monitoring stats
curl http://localhost:8000/monitoring/stats

# Database
docker exec mlops-postgres psql -U "$POSTGRES_USER" -d mlops -c "SELECT 1;"

# MLflow (no /health endpoint on MLflow 2.9.2 — the root path is the liveness check)
curl -I http://localhost:5000/
```

**Healthy State**

* All containers: `Up`
* API `/health`: `status=healthy` and `model_loaded=true`
* API `/ready`: returns `200` (not `503`)
* MLflow root returns `200`
* No repeated errors in `docker compose logs monitoring-job`

> `/health` always returns HTTP 200 — it reports status in the body. Use `/ready` when you need a
> pass/fail exit code.

---

## 2. Key Metrics & Thresholds

| Metric            | Normal       | Warning | Critical | Action                |
| ----------------- | ------------ | ------- | -------- | --------------------- |
| Prediction volume | 100–1000/day | <50/hr  | <10/hr   | Check API + upstream  |
| Drift share       | <30%         | 30–50%  | >50%     | Investigate / retrain |
| F1 score          | ≥ baseline   | −5%     | −10%     | Rollback / retrain    |
| API latency       | <200ms       | >500ms  | >2s      | Restart / scale       |
| DB size           | <5GB         | >5GB    | >10GB    | Archive data          |
| Disk usage        | <80%         | >80%    | >95%     | Cleanup immediately   |

---

## 3. Common Failures & Fixes

### API Down

```bash
docker compose ps api
docker compose logs --tail=100 api
docker compose restart api
curl http://localhost:8000/health
```

**Common causes**

* **No model in `Production`** → the most frequent cause. Train and promote one (see §5 Fallback),
  or promote an existing version in the MLflow UI.
* DB connection error → restart Postgres
* OOM → increase Docker Desktop memory

---

### PostgreSQL Down

```bash
docker ps | grep mlops-postgres
docker logs mlops-postgres --tail=100
docker compose restart postgres-mlops
```

If corrupted (last resort — **destroys all predictions, labels and decisions**):

```bash
docker compose down
docker volume ls | grep postgres-mlops     # find the real prefixed name first
docker volume rm <project>_postgres-mlops-volume
docker compose up -d postgres-mlops
# schema.sql re-applies automatically on the fresh volume; restore from backup if you have one
```

> The volume name is prefixed with the Compose project name, which defaults to the folder name.
> Always confirm with `docker volume ls` before removing.

---

### MLflow Down

```bash
docker compose logs --tail=100 mlflow
docker compose restart mlflow
curl -I http://localhost:5000/
```

If the backing store is corrupted → delete `mlflow/mlflow.db` and recreate (experiment history and
the registry are lost; the model artifacts under `mlflow/artifacts/` survive).

---

## 4. Drift Handling (>50%)

**Steps**

```bash
# Open the latest drift report (HTML, interactive)
ls -t monitoring/reports/drift_reports/drift_report_*.html | head -1

# Inspect which features drifted, from the matching JSON summary
jq '[.features[] | select(.drift_detected) | .feature]' \
  "$(ls -t monitoring/reports/drift_reports/drift_summary_*.json | head -1)"

# Or query the time series in PostgreSQL
docker exec mlops-postgres psql -U "$POSTGRES_USER" -d mlops -c \
  "SELECT timestamp, feature_drift_ratio, num_drifted_features, dataset_drift_detected
   FROM monitoring_metrics ORDER BY timestamp DESC LIMIT 10;"
```

**Decision**

```
Pipeline issue?          → Fix pipeline (NO retraining)
Real distribution shift? → Trigger retraining
Expected/seasonal?       → Document & monitor
```

**Trigger retraining**

```bash
docker exec airflow-scheduler airflow dags trigger retraining_pipeline
```

> Drift alone does **not** auto-trigger retraining. The DAG logs drift in observation-only mode;
> retraining proceeds on the scheduled/data-sufficiency path or a manual trigger like the one above.

---

## 5. Retraining Failed

```bash
# Airflow UI (task logs are easiest to read here)
# http://localhost:8081  →  DAG: retraining_pipeline  →  click the failed task  →  Log

# Or tail the scheduler
docker compose logs --tail=200 airflow-scheduler
```

**Typical causes**

* Fewer than 200 labelled predictions, or coverage below 20%
* Evaluation gate rejection (this is a **successful** outcome, not a failure — check
  `retraining_decisions.failed_gate`)
* `train_shadow_model` raising `NameError: name 'os' is not defined` — a known bug, see
  [troubleshooting.md](troubleshooting.md)

**Fallback: train and promote manually**

```bash
docker compose run --rm api python -m src.train_model_mlflow
```

This trains on `data/cs-training.csv`, registers the model and promotes it to `Production`
automatically.

---

## 6. Rollback Model (Fast)

```bash
# MLflow UI → http://localhost:5000
# Models → credit-risk-model → pick the previous version → Transition to Production

# The API polls the registry per request and hot-reloads, but a restart forces it immediately:
docker compose restart api
curl http://localhost:8000/model/info
```

Rollback should take **<5 minutes**.

---

## 7. Backup (Daily)

```bash
mkdir -p backups
docker exec mlops-postgres pg_dump -U "$POSTGRES_USER" mlops > backups/mlops_$(date +%Y%m%d).sql
gzip backups/mlops_*.sql
```

Keep last **30 days** only. Back up `mlflow/` too — it holds the model artifacts and registry.

---

## 8. High-Level Monitoring Commands

```bash
docker stats --no-stream
df -h
du -sh monitoring/ mlflow/ data/
```

---

## 9. Escalation

| Issue             | Severity | Response  |
| ----------------- | -------- | --------- |
| API Down          | P0       | Immediate |
| DB Failure        | P0       | Immediate |
| Drift >50%        | P1       | <4 hours  |
| Model degradation | P2       | Next day  |

**Escalation path**

```
On-call → Team Lead → Engineering Manager
```

---

## References

* [`api.md`](api.md)
* [`database.md`](database.md)
* [`architecture.md`](architecture.md)
* [`troubleshooting.md`](troubleshooting.md)
* [`evaluation_gates.md`](evaluation_gates.md)

---

**Last verified against the codebase:** September 2026
