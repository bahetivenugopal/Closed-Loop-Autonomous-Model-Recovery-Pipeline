# API Reference

Source of truth: [`src/api_mlflow.py`](../src/api_mlflow.py). The request/response shapes below are
the Pydantic models defined in that file.

## Base URL

```
http://localhost:8000
```

Interactive Swagger UI: `http://localhost:8000/docs`

---

## Request correlation

Every request passes through a correlation middleware:

* If you send an `X-Request-ID` header, it is reused.
* If you don't, the API generates one (`req_<12 hex chars>`).
* The value is echoed back in the `X-Request-ID` response header and stored alongside the
  prediction, so a log line, a database row and a client-side trace can be tied together.

---

## Endpoints

### `GET /`

Service identification.

```json
{
  "service": "Credit Risk Prediction API",
  "version": "3.1.0",
  "phase": "3 - Monitoring",
  "status": "healthy",
  "model_version": "1"
}
```

---

### `GET /health`

**Liveness** probe — fast and non-blocking. Always returns `200`; read `status` to judge health.
`model_version` is `null` and `status` is `"unhealthy"` when no model is loaded.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1",
  "model_stage": "Production",
  "predictions_logged": 0,
  "message": null
}
```

> `predictions_logged` is currently always `0` — it is a placeholder field, not a live counter.
> Use `GET /monitoring/stats` for real prediction volume.

---

### `GET /ready`

**Readiness** probe — verifies the two things needed to actually serve traffic: a loaded model, and
reachable storage (a `SELECT 1` against PostgreSQL, falling back to a writable
`monitoring/predictions` directory).

Returns `200` when ready:

```json
{
  "status": "ready",
  "ready": true,
  "model_loaded": true,
  "model_version": "1",
  "model_stage": "Production",
  "storage_accessible": true,
  "checks": { "model_loaded": true, "storage_accessible": true },
  "message": "Service is ready to serve predictions"
}
```

Returns `503` when not ready, with the reason in `detail`.

---

### `GET /model/info`

Metadata for the model currently being served, read from the MLflow Model Registry.

```json
{
  "model_name": "credit-risk-model",
  "model_version": "1",
  "model_stage": "Production",
  "model_uri": "models:/credit-risk-model/Production",
  "run_id": "a1b2c3d4e5f6...",
  "creation_timestamp": 1737000000000,
  "last_updated_timestamp": 1737000000000
}
```

**Status codes:** `200` OK · `503` Model not loaded

---

### `POST /predict`

Single prediction. Before serving, the API checks the registry for a newer `Production` version and
hot-reloads it if one exists — no restart needed after a promotion.

**Request** — all 10 features are required, in any key order:

```json
{
  "RevolvingUtilizationOfUnsecuredLines": 0.766127,
  "age": 45,
  "NumberOfTime30_59DaysPastDueNotWorse": 2,
  "DebtRatio": 0.802982,
  "MonthlyIncome": 9120.0,
  "NumberOfOpenCreditLinesAndLoans": 13,
  "NumberOfTimes90DaysLate": 0,
  "NumberRealEstateLoansOrLines": 6,
  "NumberOfTime60_89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 2
}
```

**Validation rules:** `age` must be 18–120; `MonthlyIncome` ≥ 0; all count fields ≥ 0. Violations
return `422`.

**Response**

```json
{
  "prediction": 0,
  "probability": 0.0872,
  "model_version": "1",
  "prediction_id": "pred_20260115_143022_918233",
  "timestamp": "2026-01-15T14:30:22.918233",
  "request_id": "req_4f2a9c7b1e03"
}
```

* `prediction` — `0` = no default risk, `1` = default risk
* `probability` — probability of default (class 1)
* `prediction_id` — format `pred_<YYYYmmdd>_<HHMMSS>_<microseconds>`. **Keep this**: it is the key
  you submit ground truth against later via `POST /labels`.

**Status codes:** `200` OK · `422` Invalid input · `500` Prediction failed · `503` Model not loaded

> There is no batch endpoint and no per-request model-version override. To score many rows, loop
> over `POST /predict` — see
> [`scripts/generate_fake_predictions.py`](../scripts/generate_fake_predictions.py).

---

### `POST /labels`

Submit the ground-truth outcome for an earlier prediction. This is how delayed feedback enters the
system. Writes are idempotent — re-submitting the same `prediction_id` updates the existing row.

**Request**

```json
{
  "prediction_id": "pred_20260115_143022_918233",
  "true_label": 1,
  "label_source": "manual",
  "prediction_timestamp": "2026-01-15T14:30:22.918233"
}
```

`label_source` and `prediction_timestamp` are optional (`label_source` defaults to `"manual"`).

**Response**

```json
{
  "status": "success",
  "prediction_id": "pred_20260115_143022_918233",
  "true_label": 1,
  "message": "Label recorded successfully"
}
```

**Status codes:** `200` OK · `422` Invalid input · `500` Failed to record label

---

### `GET /monitoring/stats`

Summary statistics over logged predictions.

```json
{
  "total_predictions": 1250,
  "recent_100": {
    "count": 100,
    "positive_rate": 0.08,
    "probability_mean": 0.1234,
    "probability_std": 0.0921
  },
  "note": "For detailed monitoring, see monitoring job results"
}
```

When nothing has been logged yet:

```json
{ "status": "no_predictions", "message": "No predictions logged yet" }
```

**Status codes:** `200` OK · `500` Failed to compute statistics

---

## Errors

Error bodies are deliberately generic — stack traces, file paths and credentials are logged
server-side but never returned to the caller.

```json
// 422 Validation Error (FastAPI/Pydantic, includes field detail)
{ "detail": [ { "loc": ["body", "age"], "msg": "ensure this value is >= 18" } ] }

// 503 Model Not Loaded
{ "detail": "Model not loaded" }

// 500 Internal Error
{ "detail": "Prediction failed" }
```

---

## Where predictions are stored

Predictions are written to the **PostgreSQL `predictions` table**, with the full feature vector in a
`JSONB` column so evaluation can replay historical requests exactly. Labels go to the `labels`
table, keyed on `prediction_id`. See [database.md](database.md).

---

## Authentication & rate limiting

Not enabled. CORS origins come from the `ALLOWED_ORIGINS` environment variable (default `*`). For
production, add token auth (`HTTPBearer`) and rate limiting (`slowapi`).

---

## Runtime

Served by **Uvicorn** (`uvicorn src.api_mlflow:app --host 0.0.0.0 --port 8000`), single worker in the
default Compose setup.

---

## Related

* [troubleshooting.md](troubleshooting.md)
* [architecture.md](architecture.md)
* [database.md](database.md)
