# NEXT_STEP.md — Getting This Project Running

**Last updated:** 2026-10-02
**Machine:** Windows 11, project at `C:\Storage\Academics\Year BTech\Sem - 07\EDAI7\Project`
**Goal:** bring the full stack up — predictions served, drift detected, retraining pipeline runnable.

This file tracks only **running the project**: what is verified working, what broke, what is fixed,
and what is left to do.

---

## 1. Where we are right now

| Phase | Status |
|---|---|
| Docker Desktop installed & engine running | ✅ Done |
| `.env` created and validated | ✅ Done |
| Third-party images pulled | ✅ Done (4 of 4) |
| `monitoring-job` image built | ✅ Done (359 MB) |
| `api` image built | 🔄 **In progress** |
| `airflow-*` images built | ⬜ Not started (blocker fixed, needs rebuild) |
| Reference baseline bootstrapped | ⬜ Pending |
| Initial model trained & promoted | ⬜ Pending |
| API serving predictions | ⬜ Pending |
| Monitoring loop running | ⬜ Pending |
| Airflow up + admin user created | ⬜ Pending |
| Traffic generated / drift observed | ⬜ Pending |
| Retraining DAG run | ⬜ Pending |

**Nothing is running yet.** No containers exist. `project-monitoring-job` has built successfully;
`project-api` is still installing its pip dependencies (the heaviest step — `requirements.txt`
includes `apache-airflow` itself, ~2 GB of wheels).

---

## 2. Environment — verified facts

Everything below was checked directly on this machine, not assumed.

| Item | Value |
|---|---|
| Docker | 29.8.0, Compose v5.5.1 |
| Docker install path | `C:\Users\DELL\AppData\Local\Programs\DockerDesktop` (user-local, **not** Program Files) |
| Engine | Linux containers, **7.6 GiB** RAM, 12 CPUs |
| Ports 5000 / 5050 / 5433 / 8000 / 8081 | All free |
| Existing containers / volumes | None — clean slate |
| `docker compose config` | Valid; all 8 services resolve |
| Host Python | **3.12.3 only** — no 3.10 |
| `data/cs-training.csv` | 150,000 rows ✅ |
| `data/processed/cs-training-temporal.csv` | 150,000 rows, has `application_date` column ✅ |

### `.env` validation result

All 24 variables that `docker-compose.yml` references are present. No placeholder values left.
Specifically confirmed:

- `AIRFLOW_FERNET_KEY` decodes to **exactly 32 bytes** → valid (this is the one that silently
  crash-loops Airflow when wrong)
- `AIRFLOW_SECRET_KEY` — 32 chars
- `POSTGRES_HOST=postgres-mlops`, `AIRFLOW_DB_HOST=postgres`,
  `MLFLOW_TRACKING_URI=http://mlflow:5000` — all correct service-name wiring
- `AIRFLOW_WEBSERVER_HOST_PORT=8081`, `MONITORING_INTERVAL=300`, `MONITORING_LOOKBACK=24`

### Images present

| Image | Size | Source |
|---|---|---|
| `postgres:13` | 624 MB | pulled |
| `ghcr.io/mlflow/mlflow:v2.9.2` | 1.06 GB | pulled |
| `dpage/pgadmin4:latest` | 791 MB | pulled |
| `apache/airflow:2.7.3-python3.10` | 2.08 GB | pulled (build base) |
| `project-monitoring-job:latest` | 359 MB | ✅ built |
| `project-api:latest` | — | 🔄 building now |
| `project-airflow-webserver`, `project-airflow-scheduler` | — | not yet built |

---

## 3. Problems found

### P1 — Docker engine was not running ✅ RESOLVED

Docker Desktop was installed but the daemon was down
(`open //./pipe/docker_engine: The system cannot find the file specified`). The CLI being on PATH
does **not** mean the engine is up.

Launched `C:\Users\DELL\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe`; engine came up in
about a minute.

> Docker Desktop does not auto-start on login here. **Every session must start it first** and wait
> for `docker info` to return a `ServerVersion` before running any compose command.

---

### P2 — `Dockerfile.airflow` could not build ✅ FIXED (code change, approved)

**Symptom:** `docker compose build` failed:

```
E: The repository 'https://apt.postgresql.org/pub/repos/apt bullseye-pgdg Release'
   does not have a Release file.
target airflow-scheduler: failed ... exit code: 100
```

**Root cause:** the base image `apache/airflow:2.7.3-python3.10` is Debian **11.8 (bullseye)** and
ships this APT source:

```
deb https://apt.postgresql.org/pub/repos/apt/ bullseye-pgdg main
```

Upstream has retired that path, so `apt-get update` fails. This is an **external change, not a
project bug** — the Dockerfile was fine when written.

**Second-order damage:** when the Airflow target failed, BuildKit **cancelled the in-flight `api`
and `monitoring-job` builds**. The first `docker compose build` therefore produced *zero* project
images, which made it look like nothing had built rather than one target failing.

> Also note: `docker compose build 2>&1 | tail -60` reports the **exit code of `tail`**, not the
> build. It printed `exit code 0` on a failed build. Check `${PIPESTATUS[0]}` or don't pipe.

**Fix applied:** removed the `apt-get` block from `Dockerfile.airflow` entirely. This is safe —
`requirements.txt` pins `psycopg2-binary` (line 24), which ships prebuilt wheels, so **nothing in
this project needs `gcc`, `g++` or `libpq-dev`**. That block was vestigial. A comment in the file
records why.

**Still to verify:** the Airflow image has not been rebuilt since the fix.

---

### P3 — Reference baseline fails its own integrity check ⚠️ FIX PENDING (bootstrap will clear it)

**Symptom:** the baseline committed to the repo does not validate:

```
stored hash (reference_metadata.json): 3c7679fc25aa259635e06d44...
recomputed from reference_data.csv  : 182c23789d397ef2500c94b3...
```

**Why this matters:** `MonitoringJob.__init__` calls `verify_reference_integrity()`, which **raises
`ValueError`** on mismatch. The monitoring loop will not start at all until this is resolved.

**Root cause:** `compute_reference_hash()` hashes `df.to_csv(index=False)` *after* a
`pd.read_csv()` round-trip — not the raw file bytes. Different pandas versions format floats
slightly differently, so a baseline written by one version fails verification under another.

**Fix:** re-run the bootstrap (already step 4 below). Critically, run it **inside the container**,
so the pandas that writes the hash is the same one that later verifies it:

```bash
docker compose run --rm api python scripts/bootstrap_reference.py
```

Do **not** bootstrap with host Python — it will reintroduce the mismatch.

---

### P4 — Retraining DAG crashed on a missing import ✅ FIXED (code change, approved)

`airflow/dags/retraining_pipeline.py` called `os.getenv(...)` in both `train_shadow_model()` and
`evaluate_models_replay()`, but never imported `os` → `NameError` as soon as the shadow-training
task ran, killing the retraining half of the pipeline.

**Fix applied:** added `import os` (line 23). Verified the module parses and both `os.getenv` call
sites resolve.

---

### P5 — Airflow creates no admin user ⬜ PENDING (manual step required)

The compose startup command runs:

```
airflow users create --username $${AIRFLOW_USERNAME} ...
```

but `AIRFLOW_USERNAME` is **not in the `airflow-webserver` service's `environment:` list**. It
expands to an empty string, the command fails, and `|| true` swallows the error. The webserver
starts with **zero users**, so the UI login is impossible.

Setting it in `.env` does not help — compose only passes variables that the service explicitly
declares.

**Workaround** (step 8 below): create the user by hand after the container is up.

---

### P6 — `docker-compose.yml` has no `trainer` or `bootstrap` service ⬜ WORKAROUND IN PLACE

Those two services exist only in `docker-compose.share.yml`. Any instruction saying
`docker compose up trainer` or `docker compose run --rm bootstrap` will fail with
"no such service" on the build stack.

**Workaround:** run them as one-off containers from the `api` image, which already contains
`src/`, `scripts/` and `data/`:

```bash
docker compose run --rm api python scripts/bootstrap_reference.py
docker compose run --rm api python -m src.train_model_mlflow
```

---

### P7 — No Python 3.10 on the host ⬜ OPEN (does not block Docker)

Only Python 3.12.3 is installed. This project pins `pydantic==1.10.13`, `numpy==1.26.0` and
`apache-airflow==2.7.3`, none of which build on 3.12.

**Impact:** limited to running `pytest` and lint tools *natively*. The entire Docker stack is
unaffected.

To enable host-side tests later: install Python 3.10.11 (tick "py launcher", do **not** tick "Add to
PATH"), then create a venv and install the runtime deps **excluding `apache-airflow`** — Airflow
does not install on native Windows, and the tests never import it.

---

## 4. Remaining steps, in order

Run from the project root. Docker Desktop must be running first.

```bash
# 1. Finish building the API image (IN PROGRESS; monitoring-job already done)
docker compose build api

# 2. Build Airflow images (now that Dockerfile.airflow is fixed)
docker compose build airflow-webserver airflow-scheduler

# 3. Start the foundation — nothing works until these are healthy.
#    schema.sql auto-applies on the first init of the postgres-mlops volume.
docker compose up -d postgres postgres-mlops mlflow
docker compose ps                      # wait for (healthy)
curl -I http://localhost:5000/         # MLflow must answer 200

# 4. Regenerate the drift baseline (clears P3)
docker compose run --rm api python scripts/bootstrap_reference.py

# 5. Train + auto-promote the first model.
#    The API cannot serve until a model is in the Production stage.
docker compose run --rm api python -m src.train_model_mlflow

# 6. Start the API
docker compose up -d api
curl http://localhost:8000/health      # expect model_loaded: true
curl http://localhost:8000/ready       # expect 200, not 503

# 7. Start monitoring (expect "insufficient_data" until 200+ predictions exist — normal)
docker compose up -d monitoring-job

# 8. Start Airflow, then create the admin user by hand (clears P5)
docker compose up -d airflow-webserver airflow-scheduler
docker exec airflow-webserver airflow users create \
  --username admin --password admin --firstname Admin --lastname User \
  --role Admin --email admin@example.com
# then open http://localhost:8081 and UNPAUSE retraining_pipeline

# 9. Generate clean traffic (1000 predictions, 50% labelled)
docker compose exec api python scripts/generate_fake_predictions.py \
  --num-samples 1000 --coverage 50 --drift-phase 0

# 10. Watch monitoring flip from insufficient_data to real drift numbers
docker compose logs -f monitoring-job

# 11. Inject severe drift and confirm it is detected
docker compose exec api python scripts/generate_fake_predictions.py \
  --num-samples 1000 --coverage 50 --drift-phase 3
ls -t monitoring/reports/drift_reports/drift_report_*.html | head -1

# 12. Trigger retraining
docker exec airflow-scheduler airflow dags trigger retraining_pipeline
```

---

## 5. Behaviour to expect — not bugs

These will look like failures but are the system working as designed. Do not debug them.

| What you see | Why |
|---|---|
| Monitoring logs `insufficient_data` | Needs **200+** predictions in the 24h window before computing anything. Deliberate guard against conclusions from noise. |
| `/health` returns 200 while reporting `unhealthy` | `/health` is a liveness probe and always returns 200 — read the body. Use `/ready` for a pass/fail code. |
| Drift detected but **no retraining triggered** | The DAG runs drift in **observation-only** mode. It logs drift; only the scheduled data-sufficiency path proceeds to training. |
| Gate 6 (segment regression) always passes | `ModelEvaluator()` is built with no `segment_features`, so there is nothing to compare. The logic exists but is not wired up. |
| Shadow model **rejected** | A success, not a failure — the gate did its job. Check `retraining_decisions.failed_gate` for which one stopped it. |
| First model promoted with no gate checks | Intentional: with nothing in Production there is nothing to compare against. Gates govern *replacements*. |
| Retraining skipped every run | Needs ≥200 labelled predictions **and** ≥20% coverage. Raise `--coverage` when generating traffic. |

---

## 6. Gotchas that cost time

- **Service names ≠ container names.** `docker compose` takes the service; `docker exec` / `docker
  logs` take the container.

  | Service | Container | Port |
  |---|---|---|
  | `postgres` | `airflow-postgres` | — |
  | `postgres-mlops` | `mlops-postgres` | 5433 |
  | `mlflow` | `mlflow-server` | 5000 |
  | `airflow-webserver` | `airflow-webserver` | 8081 |
  | `airflow-scheduler` | `airflow-scheduler` | — |
  | `api` | `mlops-api` | 8000 |
  | `monitoring-job` | `mlops-monitoring-job` | — |
  | `pgadmin` | `pgadmin` | 5050 |

- **Airflow is on 8081**, not 8080 (`AIRFLOW_WEBSERVER_HOST_PORT`).
- **The monitoring service is `monitoring-job`**, not `monitoring` (that name is share-mode only).
- **`curl` in PowerShell** is an alias for `Invoke-WebRequest` and rejects `-X` / `-d`. Use
  `Invoke-RestMethod`, or run curl from Git Bash.
- **Changing `POSTGRES_USER`/`POSTGRES_PASSWORD` after first boot does nothing.** They only apply
  when the volume is first created. Changing them later just makes the app present credentials the
  server does not know. You must delete the volume.
- **`docker compose down -v` destroys all data** — predictions, labels, decisions. After it, re-run
  steps 4 and 5; the schema recreates itself but is empty and no model is in Production.
- **Never pipe a build through `tail`** if you care about the exit code (see P2).

---

## 7. Open decisions / risks

- **Airflow images are unverified.** The Dockerfile fix is in, but the build has not been run. If
  `pip install -r requirements.txt` fails inside the Airflow base image for an unrelated reason,
  the fallback is `docker-compose.share.yml`'s prebuilt `nimish1106/sh-mlops-airflow-*:v1` images
  via an override file.
- **7.6 GiB RAM is tight** for two PostgreSQL instances + Airflow webserver and scheduler + API +
  monitoring. If containers get OOM-killed, raise Docker Desktop's memory limit or run Airflow only
  when actually needed for retraining.
- **MLflow uses a SQLite backend** (`sqlite:///mlflow/mlflow.db`). Fine for one node; it is the
  first concurrency ceiling this stack will hit.
- **`data/processed/cs-training-temporal.csv` is gitignored** but required by
  `generate_fake_predictions.py` and the `scripts/phase5/` demos. It is present in this copy; a
  fresh `git clone` will not have it. Anyone cloning must copy it across or steps 9–11 fail with
  `FileNotFoundError`.

---

## 8. Code changes made (both explicitly approved)

Everything else in the repository is untouched.

| File | Change | Why |
|---|---|---|
| `Dockerfile.airflow` | Removed the `apt-get install gcc g++ libpq-dev` block | It broke the build via a retired Debian 11 APT source, and nothing needs it — `psycopg2-binary` ships wheels |
| `airflow/dags/retraining_pipeline.py` | Added `import os` (line 23) | `os.getenv` was called in two functions without the import, raising `NameError` |
