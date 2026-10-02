# Setup Guide — Self-Healing MLOps Pipeline

A step-by-step walkthrough for getting this project running on a fresh Windows machine. Every step
says **what** it does and **why**, so you are not just pasting commands.

Written for the **build-from-source** path (`docker-compose.yml`) — everything is built from this
repository, so nothing depends on images hosted in someone else's account.

> **Why this guide exists instead of just following the README:** the README's quick-start calls
> services that don't exist in the build compose file, uses a container name that doesn't match,
> points at the wrong Airflow port, and ships a `.env` template containing an invalid encryption
> key. Those have now been corrected across the docs, and this guide is the verified path.

---

## 0. What you are about to build

Eight containers on one private Docker network:

```
      you ──► api (8000) ──────► mlflow (5000) ──── model registry
                 │                    ▲
                 ▼                    │
         postgres-mlops (5433)   airflow-scheduler ──► retraining DAG
                 ▲                    ▲
                 │                    │
         monitoring-job          airflow-webserver (8081)
         (drift, every 5 min)
                                 postgres (Airflow's own bookkeeping)
                                 pgadmin (5050) — database UI
```

**Done** looks like: a prediction returned from `localhost:8000`, a model marked `Production` in
MLflow, and the monitoring loop writing drift reports.

### Time and disk

| | |
|---|---|
| First image build | 15–25 minutes (Airflow is the slow layer) |
| Disk needed | ~8 GB for images + volumes |
| Every run after | under a minute |

---

## Part A — Things only you can do

These need a human: installers, an account, secrets. Work through them, then the rest can be run
for you.

### A0. Free disk space — do this FIRST

**This is a hard prerequisite, not a nice-to-have.** Docker Desktop keeps every image and container
in a single virtual disk on your **C: drive** (`%LOCALAPPDATA%\Docker\wsl\disk\docker_data.vhdx`),
regardless of where this project lives.

**You need at least 40 GB free on C:.** This stack pulls and builds roughly 8–10 GB of images, and
the VHDX only ever grows.

Check before you start:

```powershell
$v = Get-Volume -DriveLetter C
"Free: {0:N1} GB of {1:N1} GB ({2:N1}% free)" -f ($v.SizeRemaining/1GB), ($v.Size/1GB), (100*$v.SizeRemaining/$v.Size)
```

**What happens if you ignore this** (observed on this machine, not hypothetical): image layers get
written into a disk with no room, silently corrupt, and then a perfectly normal `linux/amd64` image
fails with `exec /bin/sh: exec format error` — an error that looks like an architecture problem and
sends you chasing the wrong thing entirely. Crash dumps also can't be written, so if the machine
does become unstable you lose the evidence needed to diagnose it.

**If you can't free up C:,** relocate Docker's storage to a drive that has room:

> Docker Desktop → **Settings → Resources → Advanced → Disk image location** → move it to e.g.
> `D:\DockerData`. Apply & Restart. Do this while the image library is still small — moving it later
> copies the whole VHDX.

### A1. Install Docker Desktop

**Why:** every piece of this project runs in a container. Docker Compose reads
`docker-compose.yml` and starts all eight services with one command, on a private network where
they find each other by name.

1. Download **Docker Desktop for Windows** from <https://www.docker.com/products/docker-desktop/>
2. During install, keep **"Use WSL 2 instead of Hyper-V"** ticked. Your machine already has WSL 2
   with Ubuntu, so this will work without extra setup.
3. Reboot if prompted, then launch Docker Desktop and wait for the whale icon to stop animating.
4. **Check available memory.** With the WSL 2 backend, Docker Desktop has no memory slider — WSL 2
   takes **50% of total RAM** by default, and there is nothing to configure unless that isn't
   enough. On a 16 GB machine that is ~8 GB, which is what this stack wants, so **do nothing**.

   Only if the build struggles, create `C:\Users\<you>\.wslconfig` with:

   ```ini
   [wsl2]
   memory=10GB
   processors=8
   ```

   Then run `wsl --shutdown` in PowerShell and restart Docker Desktop. Check your current figure
   with:

   ```powershell
   "{0:N1} GB total RAM -> WSL2 gets ~{1:N1} GB" -f (($r=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory)/1GB), ($r/1GB/2)
   ```

Verify (in PowerShell or Git Bash):

```bash
docker --version          # expect 20.10 or newer
docker compose version    # expect v2.x — note: a subcommand, not "docker-compose"
docker run --rm hello-world
```

The last command proves Docker can pull and run an image. If it works, you're done here.

**Two things that bite on Windows:**

1. **`docker: command not found` in an already-open terminal.** Recent Docker Desktop installs
   per-user to `%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin` and adds it to your *user* PATH.
   Terminals opened before the install don't see it. Open a new terminal, or prepend it:

   ```powershell
   $env:PATH = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin;$env:PATH"
   ```

2. **The daemon must actually be running.** The CLI works fine with the engine down and fails with
   `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`. Docker Desktop
   is a GUI app — launch it from the Start menu and wait for the whale icon to go steady. Tick
   **Settings → General → "Start Docker Desktop when you sign in"** so it's always up.

### A2. Python — nothing to install

**Verified on this project: Python 3.12 works. All 65 tests pass.**

You may see advice (including an earlier draft of this guide) saying you need Python 3.10 because
the pins don't support 3.12. That was tested and is **wrong**:

* Every pinned package the tests need ships a `cp312` wheel, and `pydantic==1.10.13` ships a
  pure-Python wheel that works on 3.12.
* `pytest tests/ -q` → **65 passed** on Python 3.12.3.

One real gotcha, which is why the confusion exists: Python 3.12 venvs **no longer bundle
`setuptools`**, and `mlflow==2.9.2` still imports the deprecated `pkg_resources`. Without
`setuptools` installed you get `ModuleNotFoundError: No module named 'pkg_resources'`. The project's
`requirements.txt` already pins `setuptools==69.5.1` for exactly this reason — just don't omit it.
Part D handles this.

> **The containers are unaffected either way.** They build on `python:3.10-slim` as declared in the
> Dockerfiles, and that should not change. The host Python is only for running tests and linters.

**If you did want 3.10 anyway:** use **3.10.11**, not 3.10.21. Python 3.10.11 was the last 3.10
release shipped with a Windows installer; 3.10.12 onward are security-only, **source-only** releases
with no `.exe` to download. But you don't need it.

### A2b. One environment note: long paths

Windows caps paths at 260 characters by default, and deeply nested virtual environments blow past it
— `sklearn` fails with `ImportError: DLL load failed ... The filename or extension is too long`.
Keeping the venv at `.venv` inside the project root (as Part D does) stays well under the limit, so
this won't bite you. Worth knowing if you ever relocate it somewhere deeper.

### A3. Create your `.env` file

**Why this step breaks more setups than any other:** `docker-compose.yml` reads every credential
from `.env`. The shipped `.env.example` contains *placeholders*, not values — copying it unchanged
gives you a database user literally named `your_mlops_db_user` and an Airflow encryption key that
is not a valid key, so Airflow will crash-loop on startup.

**Step 1 — generate two secrets.** Run these and keep the output:

```powershell
# AIRFLOW_FERNET_KEY — must be exactly 32 random bytes, url-safe base64 encoded.
# Airflow uses it to encrypt stored connection passwords. An invalid value = crash on boot.
py -3.12 -c "import base64,os;print(base64.urlsafe_b64encode(os.urandom(32)).decode())"

# AIRFLOW_SECRET_KEY — signs web session cookies. Any random string works.
py -3.12 -c "import secrets;print(secrets.token_hex(16))"
```

**Step 2 — create `.env`** in the project root with this content, pasting your two generated values
where marked:

```ini
# Application & Security
SECRET_KEY=local-dev-secret-change-me

# MLOps Database (predictions, labels, metrics, decisions)
POSTGRES_HOST=postgres-mlops
POSTGRES_PORT=5432
MLOPS_DB_NAME=mlops
POSTGRES_USER=mlops
POSTGRES_PASSWORD=mlops

# Airflow's own metadata database
AIRFLOW_DB_HOST=postgres
AIRFLOW_DB_PORT=5432
AIRFLOW_DB_NAME=airflow
AIRFLOW_DB_USER=airflow
AIRFLOW_DB_PASSWORD=airflow

# Airflow webserver & security
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=admin
AIRFLOW_EMAIL=admin@example.com
AIRFLOW_WEBSERVER_HOST_PORT=8081
AIRFLOW_SECRET_KEY=<paste the token_hex value here>
AIRFLOW_FERNET_KEY=<paste the base64 value here>

# MLflow tracking server
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_USERNAME=mlflow
MLFLOW_PASSWORD=mlflow

# PgAdmin UI login
PGADMIN_EMAIL=admin@admin.com
PGADMIN_PASSWORD=admin

# Monitoring scheduler
MONITORING_INTERVAL=300
MONITORING_LOOKBACK=24
```

**Where each value comes from.** The key thing to understand: *almost none of these are credentials
issued to you by an external service.* There is no account to sign up for. They fall into four
groups:

| Group | Fields | What to do |
|---|---|---|
| **You invent them** | `POSTGRES_USER/PASSWORD`, `AIRFLOW_DB_USER/PASSWORD`, `AIRFLOW_USERNAME/PASSWORD`, `AIRFLOW_EMAIL`, `PGADMIN_EMAIL/PASSWORD` | Make them up. The containers *create* these accounts on first boot using whatever you write here. For local development, `mlops`/`mlops` and `admin`/`admin` are fine. |
| **Must be generated** | `AIRFLOW_FERNET_KEY`, `AIRFLOW_SECRET_KEY`, `SECRET_KEY` | Run the two commands above. The Fernet key in particular must be valid base64 of 32 bytes or Airflow crash-loops. |
| **Leave exactly as-is** | `POSTGRES_HOST`, `POSTGRES_PORT`, `MLOPS_DB_NAME`, `AIRFLOW_DB_*` (host/port/name), `MLFLOW_TRACKING_URI`, `AIRFLOW_WEBSERVER_HOST_PORT`, `MONITORING_*` | These are Docker service names, internal ports and tuning values. Changing them breaks service discovery. |
| **Unused — ignore** | `MLFLOW_USERNAME`, `MLFLOW_PASSWORD` | Present in `.env.example` but referenced **nowhere** in `docker-compose.yml` or `src/`. MLflow runs without authentication here. Set them to anything or delete the lines. |

**What the individual settings mean:**

| Variable | Why it is what it is |
|---|---|
| `POSTGRES_HOST=postgres-mlops` | A **Docker service name**, not `localhost`. Containers resolve each other by service name on the shared network. Using `localhost` here would make a container look inside itself. |
| `POSTGRES_USER` / `PASSWORD` | Used **to create** the database the first time the volume initialises. Changing them later does nothing until you delete the volume — a classic source of "password authentication failed". |
| `AIRFLOW_FERNET_KEY` | Must be valid base64 of 32 bytes or Airflow refuses to start. |
| `AIRFLOW_WEBSERVER_HOST_PORT=8081` | Port 8080 is a common collision on Windows. The compose file defaults to 8081; the old README said 8080. |
| `MONITORING_INTERVAL=300` | Run the drift check every 5 minutes. |
| `MONITORING_LOOKBACK=24` | Each check looks at the last 24 hours of predictions. |

> `.env` is gitignored. It never gets committed, which is why the template exists.

### A4. Check the ports are free

The stack needs **5000, 5050, 5433, 8000, 8081**:

```powershell
foreach ($p in 5000,5050,5433,8000,8081) {
  $r = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
  if ($r) { "PORT $p IN USE by PID $($r.OwningProcess)" } else { "port $p free" }
}
```

Anything in use must be stopped, or its host-side port changed in `docker-compose.yml`.

---

## Part B — Bringing the stack up

From here on, every step is a command in the project root.

### B1. Build the images

```bash
docker compose build
```

**What this does:** builds three images from the repo's Dockerfiles.

| Dockerfile | Produces | Contains |
|---|---|---|
| `Dockerfile` | the API image | Python 3.10, all runtime deps, plus `src/`, `scripts/` and `data/` baked in |
| `Dockerfile.monitoring` | the monitoring image | Same deps, runs the 5-minute scheduler loop |
| `Dockerfile.airflow` | the Airflow image | Airflow 2.7.3 + this project's dependencies — **the slow one** |

Go get a coffee. Subsequent builds are cached and take seconds.

### B2. Start the foundation

```bash
docker compose up -d postgres postgres-mlops mlflow
```

**Why only these three first:** nothing else can work until the databases accept connections and
MLflow is reachable. Starting them alone makes any failure obvious instead of buried in eight
services' logs.

**What happens invisibly:** `scripts/db/schema.sql` is mounted into
`/docker-entrypoint-initdb.d/`, and PostgreSQL runs everything in that folder **the first time the
data volume is created**. Your five tables and four views exist before you touch anything.

Wait for health, then confirm:

```bash
docker compose ps                    # postgres services should show (healthy)
curl -I http://localhost:5000/       # MLflow should answer 200
```

Open <http://localhost:5000> — an empty MLflow UI. Nothing has been trained yet.

### B3. Freeze the drift baseline

```bash
docker compose run --rm api python scripts/bootstrap_reference.py
```

**Why:** drift means "today's data differs from a known-good reference". That reference has to be
fixed — a moving yardstick measures nothing. This script takes the last 20% of
`data/cs-training.csv`, writes it to `monitoring/reference/reference_data.csv`, and records a
SHA-256 fingerprint in `reference_metadata.json` so later accidental edits are detected.

**Why `docker compose run` instead of a service:** `docker-compose.yml` has no `bootstrap` service
(only `docker-compose.share.yml` does). `run --rm` starts a throwaway container from the API image,
which already contains the script and the data, then deletes it.

Expect `✅ Saved reference data` and a hash. Confirm:

```bash
ls -la monitoring/reference/
```

### B4. Train the first model

```bash
docker compose run --rm api python -m src.train_model_mlflow
```

**What this does:**
1. Loads `data/cs-training.csv` (150,000 rows) and fingerprints it, so every run is traceable to
   an exact dataset.
2. Validates it against the canonical feature schema and extracts the 10 features in fixed order.
3. Trains a `RandomForestClassifier` on an 80/20 stratified split.
4. Logs parameters, metrics (F1, ROC AUC, Brier) and feature importances to MLflow.
5. Registers the model as `credit-risk-model` and **promotes it to `Production`**.

That last step matters: **the API will not serve until a model is in `Production`.**

Verify in the MLflow UI at <http://localhost:5000>:
- **Experiments → credit-risk-prediction** → one run with metrics
- **Models → credit-risk-model → version 1** → stage reads `Production`

### B5. Start the API

```bash
docker compose up -d api
docker compose logs --tail=30 api
```

Look for `✅ Loaded model version 1 from Production`.

```bash
curl http://localhost:8000/health     # status: healthy, model_loaded: true
curl http://localhost:8000/ready      # 200 = model loaded AND database reachable
```

Open <http://localhost:8000/docs> for the interactive Swagger UI.

**Make a real prediction — Git Bash:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"RevolvingUtilizationOfUnsecuredLines":0.77,"age":45,
       "NumberOfTime30_59DaysPastDueNotWorse":2,"DebtRatio":0.80,"MonthlyIncome":9120,
       "NumberOfOpenCreditLinesAndLoans":13,"NumberOfTimes90DaysLate":0,
       "NumberRealEstateLoansOrLines":6,"NumberOfTime60_89DaysPastDueNotWorse":0,
       "NumberOfDependents":2}'
```

**Or PowerShell** (where `curl` is an alias for `Invoke-WebRequest` and won't accept those flags):

```powershell
$body = @{
  RevolvingUtilizationOfUnsecuredLines = 0.77; age = 45
  NumberOfTime30_59DaysPastDueNotWorse = 2;    DebtRatio = 0.80
  MonthlyIncome = 9120; NumberOfOpenCreditLinesAndLoans = 13
  NumberOfTimes90DaysLate = 0; NumberRealEstateLoansOrLines = 6
  NumberOfTime60_89DaysPastDueNotWorse = 0; NumberOfDependents = 2
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post `
  -ContentType 'application/json' -Body $body
```

You get back a `prediction`, a `probability`, and a **`prediction_id`** — the key you'd later
submit the real outcome against via `POST /labels`.

### B6. Start monitoring

```bash
docker compose up -d monitoring-job
docker compose logs -f monitoring-job
```

**Expect `insufficient_data` at first — that is correct.** The job needs 200+ predictions inside
the 24-hour window before it will compute drift. It refuses to draw conclusions from a handful of
samples. Press `Ctrl+C` to stop tailing.

### B7. Start Airflow (optional, for retraining)

```bash
docker compose up -d airflow-webserver airflow-scheduler
```

Give it 60–90 seconds to initialise its database on first run.

**Then create the admin user — this step is required:**

```bash
docker exec airflow-webserver airflow users create \
  --username admin --password admin --firstname Admin --lastname User \
  --role Admin --email admin@example.com
```

**Why:** the compose file's startup command runs
`airflow users create --username $${AIRFLOW_USERNAME} ...`, but `AIRFLOW_USERNAME` is not in that
container's environment list. The variable expands to nothing, the command fails, and `|| true`
swallows the error — so the webserver starts with no users and you cannot log in.

Now open <http://localhost:8081>, log in with `admin` / `admin`, find `retraining_pipeline` and
**unpause it** with the toggle (DAGs are created paused by design).

---

## Part C — Watching the loop actually work

This is the part worth demoing.

### C1. Generate baseline traffic

```bash
docker compose exec api python scripts/generate_fake_predictions.py \
  --num-samples 1000 --coverage 50 --drift-phase 0
```

Sends 1,000 realistic requests through the real API, then submits ground-truth labels for a random
50% — simulating the partial, delayed feedback you get in the real world. `--drift-phase 0` means
clean data, no drift injected.

### C2. Watch monitoring turn on

```bash
docker compose logs -f monitoring-job
```

Within five minutes it flips from `insufficient_data` to real drift statistics. Check what landed:

```bash
ls -la monitoring/reports/drift_reports/

docker exec mlops-postgres psql -U mlops -d mlops -c \
  "SELECT timestamp, num_predictions, feature_drift_ratio, dataset_drift_detected
   FROM monitoring_metrics ORDER BY timestamp DESC LIMIT 5;"
```

### C3. Inject drift and watch it get caught

```bash
docker compose exec api python scripts/generate_fake_predictions.py \
  --num-samples 1000 --coverage 50 --drift-phase 3
```

Phase 3 simulates an economic downturn — debt ratios up 2.5–4×, incomes collapsed to 20–40%,
delinquencies climbing. Wait for the next monitoring cycle, then open the newest
`monitoring/reports/drift_reports/drift_report_*.html` in a browser. You'll see per-feature
distribution comparisons with the drifted features flagged.

### C4. Trigger retraining

```bash
docker exec airflow-scheduler airflow dags trigger retraining_pipeline
```

Follow it in the Airflow UI. The DAG checks data sufficiency, trains a shadow model on a temporal
split, replays both models over identical historical rows, and runs the six evaluation gates.

> ⚠️ **Known bug:** `airflow/dags/retraining_pipeline.py` calls `os.getenv(...)` without importing
> `os`, so `train_shadow_model` raises `NameError`. Steps C1–C3 and the gate logic are all
> demonstrable; this one task will fail until `import os` is added. Left unpatched pending a team
> decision — see `docs/troubleshooting.md`.

### C5. Inspect the database

Open <http://localhost:5050>, log in with your `PGADMIN_EMAIL` / `PGADMIN_PASSWORD`, then register
a server: host `postgres-mlops`, port `5432`, database `mlops`, user `mlops`.

Tables worth looking at: `predictions`, `labels`, `monitoring_metrics`, `retraining_decisions`,
`model_versions`. Query examples are in [`docs/database.md`](docs/database.md).

---

## Part D — Host Python environment (for tests)

**Why:** the test suite is fully mocked — it never touches a real database or MLflow — so it runs
natively and gives you fast feedback plus working IDE autocomplete.

Use your existing Python 3.12 — no separate install needed.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # PowerShell
# source .venv/Scripts/activate       # Git Bash
python -m pip install --upgrade pip
```

Install dev tools, then the runtime packages **individually**:

```powershell
pip install -r requirements-dev.txt

pip install setuptools==69.5.1 scikit-learn==1.4.0 pandas==2.2.0 numpy==1.26.0 `
  fastapi==0.103.2 "uvicorn[standard]==0.23.2" pydantic==1.10.13 httpx==0.24.1 `
  mlflow==2.9.2 evidently==0.4.15 scipy==1.11.4 plotly==5.18.0 `
  psycopg2-binary==2.9.9 sqlalchemy==1.4.51 python-dotenv==1.0.0 `
  joblib==1.3.2 requests==2.31.0 pytest-timeout==2.1.0 pyyaml
```

Two things about that list:

* **`setuptools==69.5.1` is not optional on Python 3.12.** 3.12 venvs stopped bundling it, and
  `mlflow==2.9.2` imports `pkg_resources` at module load. Omit it and every mlflow import dies with
  `ModuleNotFoundError: No module named 'pkg_resources'`.
* **`apache-airflow` is deliberately excluded.** `requirements.txt` pins it, but it does not install
  cleanly on native Windows and the test suite never imports it. Airflow only ever runs inside its
  container.

Run them:

```bash
pytest tests/ -v                                     # everything
pytest tests/unit -v                                 # just unit tests
pytest tests/ --cov=src --cov-report=term-missing    # with coverage
```

---

## Part E — Daily use

### Start / stop

```bash
docker compose up -d           # start everything
docker compose stop            # stop containers, keep all data
docker compose down            # remove containers, keep volumes (data survives)
docker compose down -v         # ⚠️ ALSO DELETES VOLUMES — all predictions, labels, decisions gone
```

After `down -v` you must re-run **B3** (bootstrap) and **B4** (train) — the schema re-creates
itself automatically, but it will be empty and no model will be in Production.

### Useful commands

```bash
docker compose ps                        # what's running and healthy
docker compose logs -f api               # follow a service's logs
docker compose restart api               # restart one service
docker stats --no-stream                 # resource usage
```

### Service names vs container names

They differ, and mixing them up causes most "no such container" errors. `docker compose` takes the
left column; `docker exec` and `docker logs` take the right.

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

---

## Quick reference — the whole thing, no prose

```bash
# after Docker Desktop is installed and .env is created
docker compose build
docker compose up -d postgres postgres-mlops mlflow
docker compose run --rm api python scripts/bootstrap_reference.py
docker compose run --rm api python -m src.train_model_mlflow
docker compose up -d api monitoring-job
docker compose up -d airflow-webserver airflow-scheduler
docker exec airflow-webserver airflow users create \
  --username admin --password admin --firstname Admin --lastname User \
  --role Admin --email admin@example.com
docker compose exec api python scripts/generate_fake_predictions.py \
  --num-samples 1000 --coverage 50 --drift-phase 0
```

| | |
|---|---|
| API docs | <http://localhost:8000/docs> |
| MLflow | <http://localhost:5000> |
| Airflow | <http://localhost:8081> |
| PgAdmin | <http://localhost:5050> |

---

## If something goes wrong

[`docs/troubleshooting.md`](docs/troubleshooting.md) covers the common failures: invalid Fernet
keys, the API reporting `model_loaded: false`, `insufficient_data` that never clears, Airflow login
problems, port conflicts, and the `import os` bug.

Other references: [`docs/runbook.md`](docs/runbook.md) for operations,
[`docs/architecture.md`](docs/architecture.md) for the design, and **`Project_Overview.docx`** for
the explanation of what the system does and why.
