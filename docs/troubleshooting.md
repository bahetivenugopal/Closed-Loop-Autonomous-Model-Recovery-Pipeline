# Troubleshooting

Common issues, in roughly the order you're likely to hit them. Commands assume the build stack
(`docker-compose.yml`).

---

## Naming: services vs containers

Most confusing errors come from mixing these up. `docker compose` takes the **service** name;
`docker exec` and `docker logs` take the **container** name.

| Service (for `docker compose`) | Container (for `docker exec` / `docker logs`) | Host port |
|---|---|---|
| `postgres` | `airflow-postgres` | — |
| `postgres-mlops` | `mlops-postgres` | 5433 |
| `mlflow` | `mlflow-server` | 5000 |
| `airflow-webserver` | `airflow-webserver` | 8081 |
| `airflow-scheduler` | `airflow-scheduler` | — |
| `api` | `mlops-api` | 8000 |
| `monitoring-job` | `mlops-monitoring-job` | — |
| `pgadmin` | `pgadmin` | 5050 |

`docker-compose.share.yml` differs: the monitoring service is `monitoring`, the API container is
`credit-risk-api`, and Airflow defaults to port `8080`.

---

## Docker won't run at all

### `docker: command not found`, but Docker Desktop is installed

Recent Docker Desktop versions install **per-user** to
`%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin` and add that to your *user* PATH — so
`C:\Program Files\Docker` won't exist, and terminals opened before the install won't see the CLI.
Open a new terminal, or prepend the path for the current one:

```powershell
$env:PATH = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin;$env:PATH"
```

### `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`

The CLI is installed but the **engine is not running**. Docker Desktop is a desktop application, not
a background service — the CLI exists and works whether or not the engine is up.

Launch Docker Desktop from the Start menu, wait for the whale icon to stop animating, then retry.
Enable **Settings → General → "Start Docker Desktop when you sign in"** to stop this recurring.

### `exec /bin/sh: exec format error` on an image that is the right architecture

Do **not** go down the architecture rabbit hole. If `docker info` reports `x86_64`,
`docker image inspect <img> --format "{{.Os}}/{{.Architecture}}"` reports `linux/amd64`, and your host
is AMD64 — the architectures are fine and the **image data is corrupt**.

The usual cause is **writing image layers to a full disk**. Docker keeps its whole image library in
one virtual disk on C: (`%LOCALAPPDATA%\Docker\wsl\disk\docker_data.vhdx`); when C: runs out of room
mid-pull the layers are written incompletely, and the corruption surfaces later as an exec failure.

Fix, in order:

1. **Free space on C: first** — at least 40 GB. Nothing else will hold if you skip this.
   ```powershell
   $v = Get-Volume -DriveLetter C
   "Free: {0:N1} GB ({1:N1}% free)" -f ($v.SizeRemaining/1GB), (100*$v.SizeRemaining/$v.Size)
   ```
   Or relocate the store: Docker Desktop → **Settings → Resources → Advanced → Disk image location**.
2. **Discard the corrupted layers.** Removing the bad image is often enough:
   ```powershell
   docker rmi -f python:3.10-slim
   docker builder prune -af
   ```
3. **If it persists, reset the image store.** Docker Desktop → **Troubleshoot** (bug icon) →
   **Clean / Purge data**. This deletes all images, containers and volumes — you will need to re-run
   bootstrap and training, but the project code and `.env` are untouched.
4. **Reset the WSL VM** if Docker Desktop itself won't start cleanly:
   ```powershell
   wsl --shutdown          # stops all WSL distros; safe, no data loss
   ```
   Then start Docker Desktop again.

### The machine reboots during builds

Image builds are among the heaviest sustained loads a laptop sees, so they tend to expose latent
instability rather than cause it. Check what actually happened instead of guessing:

```powershell
# Unexpected shutdowns and bugchecks in the last 7 days
Get-WinEvent -FilterHashtable @{LogName='System'; ID=41,1001,6008; StartTime=(Get-Date).AddDays(-7)} |
  Select-Object TimeCreated, Id, @{n='Msg';e={($_.Message -split "`n")[0]}} | Format-Table -AutoSize -Wrap

# Hardware error records
Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'} -MaxEvents 10
```

Reading the result:

* **Event 1001 with a bugcheck code** = a real BSOD. Several different codes (`0x50`, `0x7E`, `0x1E`)
  all reporting access violations is the classic signature of **failing RAM**. Run the built-in test:
  `mdsched.exe` (reboots and tests), or MemTest86 overnight for a thorough pass.
* **Event 41 with `BugcheckCode=0`** = no bugcheck was recorded — power loss, thermal cutoff or a
  hard hang.
* **`C:\Windows\Minidump` empty despite event 1001 naming dump files** = there was no disk space to
  write them. Another reason to fix C: first.

---

## Setup

### `docker compose up` fails with "variable is not set"

Your `.env` is missing or incomplete. `.env.example` contains **placeholders, not values** — copying
it verbatim is not enough. Every `${VAR}` referenced in `docker-compose.yml` must be defined.

### Airflow containers crash-loop on startup

Almost always an invalid `AIRFLOW_FERNET_KEY`. It must be 32 random bytes, url-safe base64 encoded:

```bash
python -c "import base64,os;print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
```

`your-airflow-fernet-key` from `.env.example` is not a valid key.

### I changed `POSTGRES_USER` / `POSTGRES_PASSWORD` and now nothing connects

Those variables are used **to create** the database the first time the volume is initialised.
Changing them later does not change the existing database — it just makes the app present
credentials the server doesn't recognise. Either set them back, or wipe the volume:

```bash
docker compose down
docker volume ls | grep postgres-mlops     # confirm the real prefixed name
docker volume rm <project>_postgres-mlops-volume
docker compose up -d postgres-mlops
```

This destroys all predictions, labels and decisions. Re-run bootstrap and training afterwards.

### Port already in use

The stack needs **5000, 5050, 5433, 8000, 8081**. On Windows, port 5000 is a frequent collision.
Find the culprit and either stop it or change the host-side port mapping in `docker-compose.yml`.

---

## API

### API container is up but `/health` says `model_loaded: false`

There is no model in the `Production` stage. The API starts anyway (by design — it logs a warning
rather than crash-looping), but it cannot serve predictions.

```bash
docker compose run --rm api python -m src.train_model_mlflow
```

Training registers the model and promotes it to `Production` automatically. Confirm in the MLflow UI
under **Models → credit-risk-model**.

### `/ready` returns 503 but `/health` returns 200

That is intentional. `/health` is a liveness probe and always returns 200 — read the `status` field
in the body. `/ready` is the readiness probe and returns 503 when the model isn't loaded or storage
is unreachable. The `detail` field names the failing check.

### `curl` fails on Windows PowerShell

In PowerShell, `curl` is an alias for `Invoke-WebRequest`, which does not understand `-X` or `-d`.
Either use `Invoke-RestMethod`:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post `
  -ContentType 'application/json' `
  -Body (Get-Content -Raw sample.json)
```

…or run the `curl` command from Git Bash instead.

### 422 Unprocessable Entity on `/predict`

All 10 features are required, and validation is strict: `age` must be 18–120, `MonthlyIncome` ≥ 0,
and every count field ≥ 0. The response body names the offending field.

---

## Monitoring

### Monitoring logs `insufficient_data` forever

The job needs **200+** predictions inside the lookback window (`MONITORING_LOOKBACK`, default 24h)
before it will compute anything. This is a guard against drawing conclusions from noise, not a bug.

```bash
docker compose exec api python scripts/generate_fake_predictions.py \
  --num-samples 1000 --coverage 50 --drift-phase 0
```

### `FileNotFoundError: Reference data not found`

The frozen baseline hasn't been created yet:

```bash
docker compose run --rm api python scripts/bootstrap_reference.py
```

### `Reference data integrity check FAILED`

`monitoring/reference/reference_data.csv` no longer hashes to the value stored in
`reference_metadata.json`. `MonitoringJob` verifies this on startup and **raises**, so the
monitoring loop will not run until it is resolved.

**First, check which Python you verified with.** The fingerprint is pandas-version-sensitive (see
below), so a baseline that is perfectly valid inside the container will appear corrupt if you check
it with a different pandas on your host. Verify in-container before concluding anything:

```bash
docker compose run --rm api python scripts/bootstrap_reference.py verify
```

If that reports `Reference integrity verified`, nothing is wrong. If it genuinely fails, regenerate:

```bash
docker compose run --rm api python scripts/bootstrap_reference.py
```

**Why a false failure is easy to trigger:** `compute_reference_hash()` hashes
`df.to_csv(index=False)` **after** a `pd.read_csv()` round-trip, not the raw file bytes. Different
pandas versions format floats slightly differently, so the same file hashes differently under
different versions. Observed on this project: pandas **2.2.0** inside the container vs **2.2.2** on
the host produce different hashes for the identical file — the host check reports a mismatch while
the container check passes.

Consequences:

* Always bootstrap **and verify inside the container**. The container's pandas is the one that will
  verify the hash at runtime, so it is the only opinion that matters.
* Never conclude the baseline is corrupt from a host-side check alone.

The baseline is meant to be immutable, because drift is only meaningful against a fixed yardstick.
If you regenerate it on a system that already has drift history, note it in your experiment log —
drift numbers from before and after are not comparable.

### No drift reports appear under `monitoring/reports/drift_reports/`

Reports are only written on runs that clear the 200-sample threshold. Check the job is actually
running: `docker compose logs -f monitoring-job`.

---

## Retraining

### `NameError: name 'os' is not defined` in `train_shadow_model`

**Known bug.** `airflow/dags/retraining_pipeline.py` calls `os.getenv(...)` inside
`train_shadow_model()` and `evaluate_models_replay()`, but the module never imports `os`. The task
fails as soon as it reaches that line.

Fix: add `import os` to the imports at the top of the DAG file. This has been left unpatched
deliberately so the team can decide when to apply it.

### DAG doesn't appear in the Airflow UI

Give the scheduler ~30 seconds to parse it, then check for import errors:

```bash
docker compose logs --tail=100 airflow-scheduler | grep -i error
docker exec airflow-scheduler airflow dags list
```

### DAG is listed but never runs

DAGs are created paused (`AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=True`). Unpause it in the UI,
or:

```bash
docker exec airflow-scheduler airflow dags unpause retraining_pipeline
docker exec airflow-scheduler airflow dags trigger retraining_pipeline
```

### I can't log into the Airflow UI

The build-mode compose command runs `airflow users create --username $${AIRFLOW_USERNAME} ...`, but
`AIRFLOW_USERNAME` is **not in that container's environment list**, so the command receives an empty
username and fails silently (`|| true` swallows it). Create the user manually:

```bash
docker exec airflow-webserver airflow users create \
  --username admin --password admin --firstname Admin --lastname User \
  --role Admin --email admin@example.com
```

### Retraining is skipped every time

`check_retraining_needed` requires **≥200 labelled predictions** and **≥20% label coverage**.
Generate traffic with a higher `--coverage` value. Note that drift alone does not trigger
retraining — the DAG runs drift detection in observation-only mode.

### The shadow model was rejected

That is a successful outcome, not a failure. Check which gate stopped it:

```bash
docker exec mlops-postgres psql -U "$POSTGRES_USER" -d mlops -c \
  "SELECT timestamp, action, failed_gate, reason FROM retraining_decisions ORDER BY timestamp DESC LIMIT 5;"
```

See [evaluation_gates.md](evaluation_gates.md).

---

## Tests

### `pip install -r requirements.txt` fails on Windows

Two separate causes:

1. **Python version.** The pins (`pydantic==1.10.13`, `numpy==1.26.0`, `apache-airflow==2.7.3`)
   require Python 3.10. They will not build on 3.11+.
2. **Airflow on native Windows.** `apache-airflow` does not install cleanly outside WSL/Linux. The
   test suite never imports it — install the runtime packages individually and skip the Airflow
   lines. See [SETUP_GUIDE.md](../SETUP_GUIDE.md).

### Tests fail trying to reach a database

They shouldn't — `tests/conftest.py` sets `TESTING=true` and points MLflow/Postgres at mock URLs
before any test runs. If you see real connection attempts, you are probably running a test file
directly instead of through `pytest` from the project root, so `conftest.py` never loaded.

---

## Demo scripts

### `FileNotFoundError: /app/data/processed/cs-training-temporal.csv`

That file is **gitignored** (`data/processed/*.csv`), so a fresh `git clone` will not contain it,
even though the scripts under `scripts/phase5/` and `scripts/generate_fake_predictions.py` require
it. Copy it from a teammate's working checkout into `data/processed/`.

### Demo scripts appear to hang

`scripts/phase5/demo_01_baseline.py` deliberately sleeps for 6 minutes waiting for the monitoring
loop to complete a cycle. That is expected.

---

## Related

* [runbook.md](runbook.md) — operational procedures
* [SETUP_GUIDE.md](../SETUP_GUIDE.md) — first-time setup
* [database.md](database.md) — schema and queries
