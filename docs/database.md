# Database Documentation

**PostgreSQL schema and core operations for the Self-Healing MLOps system**

Source of truth: [`scripts/db/schema.sql`](../scripts/db/schema.sql). That file is mounted into the
container at `/docker-entrypoint-initdb.d/01_schema.sql` and runs **automatically the first time the
database volume is created**. The API also re-applies it on startup (every statement is
`CREATE ... IF NOT EXISTS`, so it is safe to run repeatedly).

---

## Overview

| | |
|---|---|
| **Engine** | PostgreSQL 13 |
| **Compose service** | `postgres-mlops` |
| **Container name** | `mlops-postgres` |
| **Database** | `mlops` |
| **Port** | `5432` inside the Docker network · `5433` on the host |

Credentials come from `.env` (`POSTGRES_USER` / `POSTGRES_PASSWORD`). They are used **to create** the
database on first boot — changing them later has no effect until the volume is removed.

⚠️ Default development credentials are not suitable for production.

### Connection strings

```text
# From another container on the mlops-network
postgresql://<POSTGRES_USER>:<POSTGRES_PASSWORD>@postgres-mlops:5432/mlops

# From your host machine (psql, PgAdmin, a notebook)
postgresql://<POSTGRES_USER>:<POSTGRES_PASSWORD>@localhost:5433/mlops
```

> Note the container name differs from the service name. Use `docker exec mlops-postgres ...`, but
> `docker compose restart postgres-mlops`.

---

## Schema

Five tables and four views. There is a second PostgreSQL instance (`postgres` / container
`airflow-postgres`) that holds Airflow's own metadata — it is unrelated to the tables below.

### `predictions`

Durable record of every inference request. The full feature vector is stored as `JSONB` so that
evaluation can **replay** historical requests against a new model.

```sql
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id    TEXT PRIMARY KEY,
    timestamp        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_version    TEXT NOT NULL,
    prediction       INT NOT NULL CHECK (prediction IN (0, 1)),
    probability      FLOAT NOT NULL CHECK (probability >= 0.0 AND probability <= 1.0),
    application_date TIMESTAMP WITH TIME ZONE,
    features         JSONB NOT NULL,
    request_id       TEXT,
    created_at       TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

Indexes: `timestamp DESC`, `model_version`, `application_date DESC`, `request_id`.

`application_date` is the *business* timestamp used for temporal train/eval splits, separate from
`timestamp` (when the request was served).

---

### `labels`

Delayed ground truth, keyed one-to-one against a prediction. `ON DELETE CASCADE` means removing a
prediction removes its label.

```sql
CREATE TABLE IF NOT EXISTS labels (
    prediction_id   TEXT PRIMARY KEY REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    true_label      INT NOT NULL CHECK (true_label IN (0, 1)),
    label_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    label_source    TEXT DEFAULT 'manual',
    days_delayed    INT,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

Indexes: `label_timestamp DESC`, `true_label`.

---

### `monitoring_metrics`

One row per monitoring run (default: every 5 minutes). Proxy metrics and drift results are
flattened into columns for fast time-series queries.

```sql
CREATE TABLE IF NOT EXISTS monitoring_metrics (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp              TIMESTAMP NOT NULL,
    lookback_hours         INT NOT NULL,
    num_predictions        INT NOT NULL,
    positive_rate          FLOAT,
    probability_mean       FLOAT,
    probability_std        FLOAT,
    entropy                FLOAT,
    dataset_drift_detected BOOLEAN DEFAULT FALSE,
    feature_drift_ratio    FLOAT DEFAULT 0.0,
    num_drifted_features   INT DEFAULT 0,
    drift_summary_ref      TEXT,
    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT monitoring_metrics_timestamp_unique UNIQUE (timestamp)
);
```

`drift_summary_ref` is a **path pointer**, not the report itself — the full Evidently output lives on
disk under `monitoring/reports/drift_reports/`.

---

### `retraining_decisions`

Audit trail of every retraining decision, including the ones that chose *not* to act.

```sql
CREATE TABLE IF NOT EXISTS retraining_decisions (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp                TIMESTAMP NOT NULL,
    trigger_reason           TEXT NOT NULL CHECK (trigger_reason IN ('scheduled', 'manual', 'drift_alert')),
    feature_drift_ratio      FLOAT,
    num_drifted_features     INT,
    dataset_drift_detected   BOOLEAN,
    drifted_features         TEXT[],
    labeled_samples          INT,
    coverage_pct             FLOAT,
    action                   TEXT NOT NULL CHECK (action IN ('train', 'skip', 'promote', 'reject')),
    failed_gate              TEXT,
    reason                   TEXT,
    shadow_model_version     INT,
    production_model_version INT,
    f1_improvement_pct       FLOAT,
    brier_change             FLOAT,
    drift_summary_ref        TEXT,
    evaluation_report_ref    TEXT,
    created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT retraining_decisions_timestamp_unique UNIQUE (timestamp)
);
```

A row with `action = 'reject'` is a **successful** outcome — the gate did its job. See
[evaluation_gates.md](evaluation_gates.md).

---

### `model_versions`

Model lineage and governance, mirroring the MLflow registry so lifecycle history is queryable in
SQL.

```sql
CREATE TABLE IF NOT EXISTS model_versions (
    model_name                     TEXT NOT NULL,
    version                        INT NOT NULL,
    stage                          TEXT NOT NULL CHECK (stage IN ('Staging', 'Production', 'Archived', 'None')),
    trained_at                     TIMESTAMP,
    promoted_at                    TIMESTAMP,
    archived_at                    TIMESTAMP,
    trigger_reason                 TEXT,
    training_run_id                TEXT,
    f1_score                       FLOAT,
    brier_score                    FLOAT,
    num_samples                    INT,
    feature_drift_ratio_at_training FLOAT,
    decision_id                    UUID REFERENCES retraining_decisions(id),
    created_at                     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at                     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_name, version)
);

-- Governance invariant: at most one Production model per model_name
CREATE UNIQUE INDEX IF NOT EXISTS one_production_model
ON model_versions(model_name)
WHERE stage = 'Production';
```

That partial unique index is the important part: the database itself refuses to let two versions of
the same model sit in `Production` simultaneously.

---

## Views

| View | What it gives you |
|---|---|
| `v_labeled_predictions` | `predictions` INNER JOIN `labels` — the evaluation dataset |
| `v_recent_monitoring` | Last 100 monitoring runs, newest first |
| `v_decision_history` | All retraining decisions, newest first |
| `v_model_timeline` | Model versions joined to the decision that produced them |

---

## Common queries

### Prediction volume (last 24h)

```sql
SELECT COUNT(*)
FROM predictions
WHERE timestamp > NOW() - INTERVAL '24 hours';
```

### Label coverage

```sql
SELECT
  COUNT(l.prediction_id)::float / NULLIF(COUNT(p.prediction_id), 0) AS coverage
FROM predictions p
LEFT JOIN labels l USING (prediction_id);
```

### Accuracy by model version (labelled rows only)

```sql
SELECT
  model_version,
  AVG((prediction = true_label)::int)::float AS accuracy,
  COUNT(*) AS n
FROM v_labeled_predictions
GROUP BY model_version
ORDER BY model_version;
```

### Recent drift signal

```sql
SELECT timestamp, num_predictions, feature_drift_ratio, dataset_drift_detected
FROM monitoring_metrics
ORDER BY timestamp DESC
LIMIT 10;
```

### Why was the last shadow model rejected?

```sql
SELECT timestamp, action, failed_gate, reason, f1_improvement_pct, brier_change
FROM retraining_decisions
WHERE action IN ('promote', 'reject')
ORDER BY timestamp DESC
LIMIT 5;
```

### Read a single feature out of the JSONB column

```sql
SELECT prediction_id,
       (features->>'MonthlyIncome')::float AS monthly_income,
       (features->>'age')::int             AS age
FROM predictions
ORDER BY timestamp DESC
LIMIT 20;
```

---

## Maintenance

```sql
VACUUM ANALYZE predictions;
VACUUM ANALYZE labels;
VACUUM ANALYZE monitoring_metrics;
```

`predictions` grows fastest — it is append-only by design and holds a JSONB blob per row.

---

## Backup & recovery

```bash
# Full backup
docker exec mlops-postgres pg_dump -U <POSTGRES_USER> mlops > mlops_backup.sql

# Restore
cat mlops_backup.sql | docker exec -i mlops-postgres psql -U <POSTGRES_USER> -d mlops
```

### Inspecting interactively

```bash
docker exec -it mlops-postgres psql -U <POSTGRES_USER> -d mlops

# then, at the psql prompt:
\dt                      -- list tables
\d+ predictions          -- describe a table
\dv                      -- list views
```

Or use PgAdmin at `http://localhost:5050`, connecting to host `postgres-mlops`, port `5432`.

---

## Performance notes

* Indexed on `timestamp`, `model_version`, `application_date`, `request_id` and the drift columns.
* `JSONB` is used only for the feature vector, where schema flexibility genuinely pays off.
* `gen_random_uuid()` is built into PostgreSQL 13 — no `pgcrypto` extension needed.

---

## Security notes

* Change the default credentials before any shared deployment.
* Use read-only roles for analytics access.
* Enable SSL and external connection pooling outside development.

---

**Last verified against `scripts/db/schema.sql`:** September 2026
