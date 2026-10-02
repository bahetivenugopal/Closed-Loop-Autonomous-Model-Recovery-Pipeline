"""
Production API with prediction logging for Phase 3 monitoring and Phase 2 reliability hardening.

RESPONSIBILITIES:
- Serve predictions from Production model with strict validation
- Correlate requests and log predictions for monitoring
- Provide lightweight /health (liveness) and thorough /ready (readiness) probes
- Safe error handling: never leak stack traces, internal paths, or credentials
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import logging
import sys
from src.storage.prediction_logger import get_prediction_logger

# Add to path
sys.path.append("/app")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def is_testing() -> bool:
    """Check if running in test mode (evaluated at runtime, not import time)."""
    return os.getenv("TESTING", "false").lower() == "true"


MODEL_NAME = "credit-risk-model"
PRODUCTION_STAGE = "Production"

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Self-Healing MLOps Pipeline - Production Inference API",
    version="3.1.0",
)

# CORS Configuration
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_correlation_middleware(request: Request, call_next):
    """
    Request correlation middleware.
    Attaches incoming or generated X-Request-ID to request.state and response headers.
    """
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = f"req_{uuid.uuid4().hex[:12]}"
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Global state
model = None
model_version = None
model_uri = None
prediction_logger = None
_last_checked_version = None  # Track if we've checked for updates


def check_and_reload_model_if_needed():
    """
    Check if Production model has changed and reload if needed.
    This allows the API to pick up new models without restarting.
    Called before each prediction request.
    """
    if is_testing():
        return False

    global model, model_version, _last_checked_version  # noqa: F824

    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=[PRODUCTION_STAGE])

        if not versions:
            logger.warning("No model in Production stage")
            return False

        latest_version = versions[0].version

        if latest_version != _last_checked_version:
            logger.info(f"Production model updated: v{_last_checked_version} → v{latest_version}")
            load_production_model()
            _last_checked_version = latest_version
            logger.info(f"✅ Reloaded model to v{latest_version}")
            return True

        return False

    except Exception as e:
        logger.warning(f"Error checking for model updates: {e}")
        return False


class PredictionInput(BaseModel):
    """
    Input schema for predictions.
    Pydantic handles validation automatically.
    """

    RevolvingUtilizationOfUnsecuredLines: float
    age: int = Field(..., ge=18, le=120, description="Age must be 18-120")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., ge=0)
    DebtRatio: float
    MonthlyIncome: float = Field(..., ge=0, description="Monthly income must be positive")
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0)
    NumberOfTimes90DaysLate: int = Field(..., ge=0)
    NumberRealEstateLoansOrLines: int = Field(..., ge=0)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., ge=0)
    NumberOfDependents: int = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "RevolvingUtilizationOfUnsecuredLines": 0.766127,
                "age": 45,
                "NumberOfTime30_59DaysPastDueNotWorse": 2,
                "DebtRatio": 0.802982,
                "MonthlyIncome": 9120.0,
                "NumberOfOpenCreditLinesAndLoans": 13,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 6,
                "NumberOfTime60_89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2,
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for predictions."""

    prediction: int = Field(..., description="0=no default risk, 1=default risk")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of default")
    model_version: str = Field(..., description="Model version used for this prediction")
    prediction_id: str = Field(..., description="Unique ID for tracking this prediction")
    timestamp: str = Field(..., description="ISO timestamp of prediction")
    request_id: Optional[str] = Field(None, description="Correlated request ID")


class LabelInput(BaseModel):
    """Ground truth label submission schema."""

    prediction_id: str = Field(..., description="Correlated prediction ID")
    true_label: int = Field(..., ge=0, le=1, description="Ground truth outcome (0 or 1)")
    label_source: Optional[str] = Field("manual", description="Origin/source of label")
    prediction_timestamp: Optional[str] = Field(
        None, description="Original prediction timestamp if known"
    )


class LabelResponse(BaseModel):
    """Label submission response."""

    status: str
    prediction_id: str
    true_label: int
    message: Optional[str] = None


class RootResponse(BaseModel):
    """Root service information."""

    service: str
    version: str
    phase: str
    status: str
    model_version: Optional[str] = None


class HealthResponse(BaseModel):
    """Liveness check response."""

    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    model_stage: str
    predictions_logged: int = 0
    message: Optional[str] = None


class ReadinessResponse(BaseModel):
    """Production readiness check response."""

    status: str
    ready: bool
    model_loaded: bool
    model_version: Optional[str] = None
    model_stage: Optional[str] = None
    storage_accessible: bool
    checks: Dict[str, bool]
    message: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Model registry metadata response."""

    model_name: str
    model_version: Optional[str] = None
    model_stage: str
    model_uri: Optional[str] = None
    run_id: Optional[str] = None
    creation_timestamp: Optional[int] = None
    last_updated_timestamp: Optional[int] = None


def load_production_model():
    """
    Load Production model from MLflow Registry.
    If no Production model exists, API should log error and fail readiness.
    """
    global model, model_version, model_uri

    try:
        model_uri = f"models:/{MODEL_NAME}/{PRODUCTION_STAGE}"
        logger.info(f"Loading model: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)

        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=[PRODUCTION_STAGE])

        if not versions:
            raise ValueError(f"No model in {PRODUCTION_STAGE} stage")

        model_version = versions[0].version
        logger.info(f"✅ Loaded model version {model_version} from {PRODUCTION_STAGE}")

    except Exception as e:
        logger.error(f"❌ Failed to load Production model: {e}")
        if not is_testing():
            raise RuntimeError(
                "Cannot start API: No model in Production stage. "
                "Train a model and promote it to Production first."
            )


@app.on_event("startup")
async def startup_event():
    """Initialize API on startup."""
    global prediction_logger, _last_checked_version

    logger.info("=" * 70)
    logger.info("API STARTING UP")
    logger.info("=" * 70)

    try:
        load_production_model()
        _last_checked_version = model_version
    except Exception as e:
        logger.warning(f"Startup model load skipped/deferred: {e}")

    if not is_testing():
        try:
            from src.storage.db_manager import get_db_manager

            db = get_db_manager()
            schema_path = Path("/app/scripts/db/schema.sql")
            if schema_path.exists():
                db.execute_script(str(schema_path))
                logger.info("✅ Database schema verified on startup")
        except Exception as e:
            logger.warning(f"Database schema verification deferred: {e}")

    try:
        prediction_logger = get_prediction_logger()
        logger.info("✅ Prediction logger initialized")
    except Exception as e:
        logger.warning(f"Prediction logger initialization deferred: {e}")

    logger.info("=" * 70)
    logger.info("API INITIALIZATION COMPLETE")
    logger.info("=" * 70)


@app.get("/", response_model=RootResponse)
async def root():
    """Root endpoint - basic service identification."""
    return RootResponse(
        service="Credit Risk Prediction API",
        version="3.1.0",
        phase="3 - Monitoring",
        status="healthy",
        model_version=str(model_version) if model_version else None,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Liveness check probe.
    Fast, non-blocking check verifying that the API process is alive and responsive.
    """
    is_healthy = model is not None
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=is_healthy,
        model_version=str(model_version) if model_version else None,
        model_stage=PRODUCTION_STAGE,
        predictions_logged=0,
        message=None if is_healthy else "Model not loaded",
    )


@app.get("/ready", response_model=ReadinessResponse)
async def ready():
    """
    Readiness check probe.
    Verifies only the dependencies required for serving live inference:
    1. Model loaded in memory
    2. Prediction storage directory accessible
    """
    model_ready = model is not None and model_version is not None

    storage_ready = False
    if is_testing():
        storage_ready = True
    else:
        try:
            from src.storage.db_manager import get_db_manager

            db = get_db_manager()
            db.execute_query("SELECT 1")
            storage_ready = True
        except Exception:
            try:
                storage_path = Path("monitoring/predictions")
                storage_path.mkdir(parents=True, exist_ok=True)
                test_file = storage_path / ".ready_check"
                test_file.touch(exist_ok=True)
                storage_ready = True
            except Exception as e:
                logger.warning("Readiness check: storage inaccessible: %s", e)
                storage_ready = False

    is_ready = model_ready and storage_ready
    checks = {
        "model_loaded": model_ready,
        "storage_accessible": storage_ready,
    }

    if not is_ready:
        unready_reasons = []
        if not model_ready:
            unready_reasons.append("Model is not loaded")
        if not storage_ready:
            unready_reasons.append("Prediction storage is not accessible")

        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {', '.join(unready_reasons)}",
        )

    return ReadinessResponse(
        status="ready",
        ready=True,
        model_loaded=model_ready,
        model_version=str(model_version),
        model_stage=PRODUCTION_STAGE,
        storage_accessible=storage_ready,
        checks=checks,
        message="Service is ready to serve predictions",
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get current active Production model metadata.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not is_testing():
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(MODEL_NAME, stages=[PRODUCTION_STAGE])

            if versions:
                version_info = versions[0]
                return ModelInfoResponse(
                    model_name=MODEL_NAME,
                    model_version=str(model_version),
                    model_stage=PRODUCTION_STAGE,
                    model_uri=model_uri,
                    run_id=version_info.run_id,
                    creation_timestamp=version_info.creation_timestamp,
                    last_updated_timestamp=version_info.last_updated_timestamp,
                )
        except Exception as e:
            logger.warning("Could not fetch extended MLflow metadata: %s", e)

    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_version=str(model_version),
        model_stage=PRODUCTION_STAGE,
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput, request: Request):
    """
    Make prediction and log for monitoring.
    Safe error handling: internal details logged; safe generic error returned to client.
    """
    # Check if model has been updated and reload if needed
    check_and_reload_model_if_needed()

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = getattr(request.state, "request_id", None) or f"req_{uuid.uuid4().hex[:12]}"

    try:
        from src.features.schema import FEATURE_COLUMNS

        # Prepare features (canonical ordering matching training)
        input_dict = input_data.dict()
        df_features = pd.DataFrame([input_dict])[FEATURE_COLUMNS]
        features = df_features.to_numpy()

        # Predict
        preds = np.asarray(model.predict(features))
        prediction = int(preds[0])

        probs = np.asarray(model.predict_proba(features))
        probability = float(probs[0, 1] if probs.ndim == 2 else probs[1])

        # Generate unique prediction ID correlated with request
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        prediction_id = f"pred_{timestamp_str}"

        # Log prediction for monitoring and durable storage
        if prediction_logger:
            try:
                prediction_logger.log_prediction(
                    prediction_id=prediction_id,
                    features=input_data.dict(),
                    prediction=prediction,
                    probability=probability,
                    model_version=str(model_version),
                    request_id=request_id,
                )
            except Exception as e:
                logger.error(
                    "Failed to log prediction %s (request_id=%s): %s",
                    prediction_id,
                    request_id,
                    e,
                    exc_info=True,
                )

        return PredictionOutput(
            prediction=prediction,
            probability=probability,
            model_version=str(model_version),
            prediction_id=prediction_id,
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Prediction failed for request_id=%s: %s",
            request_id,
            e,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/labels", response_model=LabelResponse)
async def submit_label(input_data: LabelInput):
    """
    Submit ground truth outcome for a past prediction.
    Enforces idempotency and links to prediction_id for model evaluation.
    """
    try:
        from src.storage.label_store import get_label_store

        label_store = get_label_store()
        label_store.store_label(
            prediction_id=input_data.prediction_id,
            true_label=input_data.true_label,
            label_source=input_data.label_source or "manual",
            prediction_timestamp=input_data.prediction_timestamp,
        )
        return LabelResponse(
            status="success",
            prediction_id=input_data.prediction_id,
            true_label=input_data.true_label,
            message="Label recorded successfully",
        )
    except Exception as e:
        logger.error(
            f"Failed to record label for prediction {input_data.prediction_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to record label")


@app.get("/monitoring/stats")
async def monitoring_stats():
    """
    Get basic monitoring statistics summary.
    """
    try:
        from src.storage.repositories import PredictionsRepository

        repo = PredictionsRepository()
        total_count = repo.count()

        if total_count == 0:
            return {"status": "no_predictions", "message": "No predictions logged yet"}

        recent = repo.get_recent_predictions(days=30)
        if not recent:
            return {"status": "no_predictions", "message": "No recent predictions found"}

        df = pd.DataFrame(recent)
        recent_100 = df.head(100)

        prob_std = float(recent_100["probability"].std()) if len(recent_100) > 1 else 0.0
        if np.isnan(prob_std):
            prob_std = 0.0

        return {
            "total_predictions": total_count,
            "recent_100": {
                "count": len(recent_100),
                "positive_rate": float(recent_100["prediction"].mean()),
                "probability_mean": float(recent_100["probability"].mean()),
                "probability_std": prob_std,
            },
            "note": "For detailed monitoring, see monitoring job results",
        }

    except Exception as e:
        logger.error("Failed to compute stats: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to compute monitoring statistics")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
