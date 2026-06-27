#Pipeline
#--------
#RFID Boarding Event  →  /predict-duration/  →  ETA Notification
#RFID Exit Event      →  TripHistory logged
#Every 7 trips        →  /retrain/           →  updated .pkl saved
#"""

from __future__ import annotations

import logging
import math
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smartkids_guard")

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
MODEL_PATH = Path(os.getenv("MODEL_PATH", "smartkids_model.pkl"))

SCHOOL_LAT: float = float(os.getenv("SCHOOL_LAT", "30.0444"))   # Cairo default
SCHOOL_LON: float = float(os.getenv("SCHOOL_LON", "31.2357"))

# Cold-start defaults
DEFAULT_AVG_BUS_SPEED_KMH: float = 30.0   # km/h
DEFAULT_WAITING_TIME_MIN: float = 5.0     # minutes

FEATURE_NAMES: List[str] = [
    "DistanceToSchool",
    "HistoricalAverageDuration",
    "WeatherSeverity",
    "TrafficIndex",
    "HourOfDay",
    "DayOfWeek",
]

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class LocationPredictionRequest(BaseModel):
    """
    Payload sent by the Backend when an RFID boarding event fires.

    Fields
    ------
    child_id            : Unique student identifier.
    current_lat/lon     : Live GPS coordinates from the wearable.
    trip_history        : Last ≤7 completed actual-duration values (minutes).
                          Pass an empty list for new students (cold-start).
    weather_severity    : 0.0 (clear) → 1.0 (severe storm).
    traffic_index       : 1.0 (free flow) → 5.0 (gridlock).
    avg_bus_speed_kmh   : Optional override for cold-start formula.
    waiting_time_min    : Optional override for cold-start formula.
    """

    child_id: str = Field(..., examples=["child_001"])
    current_lat: float = Field(..., ge=-90.0, le=90.0)
    current_lon: float = Field(..., ge=-180.0, le=180.0)
    trip_history: List[float] = Field(
        default_factory=list,
        description="Actual durations (minutes) of the last ≤7 completed trips.",
    )
    weather_severity: float = Field(..., ge=0.0, le=1.0)
    traffic_index: float = Field(..., ge=1.0, le=5.0)
    avg_bus_speed_kmh: Optional[float] = Field(
        default=None, gt=0.0,
        description="Override for cold-start formula; defaults to 30.0 km/h.",
    )
    waiting_time_min: Optional[float] = Field(
        default=None, ge=0.0,
        description="Override for cold-start formula; defaults to 5.0 min.",
    )

    @field_validator("trip_history")
    @classmethod
    def cap_history_length(cls, v: List[float]) -> List[float]:
        """Keep only the most recent 7 records to match the algorithm spec."""
        if any(d <= 0 for d in v):
            raise ValueError("All trip durations must be positive (> 0 minutes).")
        return v[-7:]


class PredictionResponse(BaseModel):
    child_id: str
    predicted_duration_minutes: float = Field(..., description="Model output, rounded to 2 dp.")
    distance_to_school_km: float
    historical_avg_duration_minutes: Optional[float] = Field(
        None, description="None when cold-start baseline was used."
    )
    cold_start_used: bool
    hour_of_day: int
    day_of_week: int
    timestamp_utc: str


class TripRecord(BaseModel):
    """Single record in the retraining payload (mirrors TripHistory schema)."""

    distance_to_school: float = Field(..., gt=0)
    historical_avg_duration: float = Field(..., gt=0)
    weather_severity: float = Field(..., ge=0.0, le=1.0)
    traffic_index: float = Field(..., ge=1.0, le=5.0)
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    actual_duration: float = Field(..., gt=0, description="Ground-truth label.")


class RetrainRequest(BaseModel):
    """
    Backend sends the last 28 TripHistory records every 7 trips.
    Minimum viable dataset is 10 records (safety guard).
    """

    records: List[TripRecord] = Field(..., min_length=10)


class RetrainResponse(BaseModel):
    message: str
    records_used: int
    model_saved_at: str
    mae_minutes: float = Field(..., description="Mean Absolute Error on training set.")
    r2_score: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    mock_mae_minutes: float
    mock_r2_score: float
    server_utc: str

# ---------------------------------------------------------------------------
# Math Utilities
# ---------------------------------------------------------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the great-circle distance between two GPS points.

    Parameters
    ----------
    lat1, lon1 : Origin coordinates (decimal degrees).
    lat2, lon2 : Destination coordinates (decimal degrees).

    Returns
    -------
    float : Distance in kilometres.
    """
    R = 6_371.0  # Earth's mean radius in km

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def trimmed_mean_historical_duration(durations: List[float]) -> float:
    """
    Trimmed-Mean algorithm for HistoricalAverageDuration.

    Steps
    -----
    1. Accept the last 7 completed trip durations.
    2. Remove the single maximum value (upper outlier).
    3. Remove the single minimum value (lower outlier).
    4. Return the arithmetic mean of the remaining 5 values.

    Parameters
    ----------
    durations : Exactly 7 positive float values (minutes).

    Returns
    -------
    float : Trimmed mean in minutes.

    Raises
    ------
    ValueError : If fewer than 7 durations are supplied.
    """
    if len(durations) < 7:
        raise ValueError(
            f"Trimmed mean requires exactly 7 records; received {len(durations)}."
        )
    sorted_d = sorted(durations)
    trimmed = sorted_d[1:-1]          # drop min (index 0) and max (index -1)
    return float(np.mean(trimmed))


def cold_start_baseline(
    distance_km: float,
    avg_speed_kmh: float = DEFAULT_AVG_BUS_SPEED_KMH,
    waiting_min: float = DEFAULT_WAITING_TIME_MIN,
) -> float:
    """
    Deterministic fallback for students with no trip history.

    Formula
    -------
        InitialDuration = (DistanceToSchool / AverageBusSpeed) × 60 + WaitingTime

    Parameters
    ----------
    distance_km   : Haversine distance from current location to school (km).
    avg_speed_kmh : Average travel speed in km/h (default 30.0).
    waiting_min   : Fixed boarding/waiting offset in minutes (default 5.0).

    Returns
    -------
    float : Estimated trip duration in minutes.
    """
    travel_time_min = (distance_km / avg_speed_kmh) * 60.0
    return round(travel_time_min + waiting_min, 4)

# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------

def _build_fresh_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def load_model() -> Optional[Pipeline]:
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            logger.info("Model loaded from '%s'.", MODEL_PATH)
            return model
        except Exception as exc:
            logger.warning("Failed to load model: %s. Starting without one.", exc)
    else:
        logger.info("No pre-trained model found at '%s'. Cold-start mode active.", MODEL_PATH)
    return None


def save_model(model: Pipeline) -> None:
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved → '%s'.", MODEL_PATH)

# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self) -> None:
        self.model: Optional[Pipeline] = load_model()


app_state = AppState()

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SmartKids Guard — Arrival Prediction API",
    description=(
        "ML microservice for predicting student arrival times at school "
        "using a wearable RFID/GPS device. "
        "Graduation Project · HTI · Computer Science / Data Science."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health-check",
    tags=["Monitoring"],
)
def health_check() -> HealthResponse:
    """
    Returns service liveness and a mock evaluation snapshot for the
    graduation defense dashboard.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=app_state.model is not None,
        model_path=str(MODEL_PATH),
        mock_mae_minutes=3.42,
        mock_r2_score=0.91,
        server_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.post(
    "/predict-duration/",
    response_model=PredictionResponse,
    summary="Predict student arrival duration (triggered by RFID boarding event)",
    tags=["Prediction"],
    status_code=status.HTTP_200_OK,
)
def predict_duration(request: LocationPredictionRequest) -> PredictionResponse:
    """
    **Prediction Pipeline**

    1. Compute `DistanceToSchool` using the Haversine formula.
    2. Compute `HistoricalAverageDuration` via the trimmed-mean algorithm
       (or fall back to the cold-start baseline for new students).
    3. Capture server time → `HourOfDay`, `DayOfWeek`.
    4. Run the feature vector through the RandomForest pipeline.
    5. Return the predicted duration in minutes.

    If no trained model exists yet, the cold-start baseline is returned
    as the predicted duration.
    """
    now_utc = datetime.now(timezone.utc)
    hour_of_day = now_utc.hour
    day_of_week = now_utc.weekday()  # 0 = Monday … 6 = Sunday

    # --- Feature 1: Haversine distance ---------------------------------
    try:
        distance_km = haversine_distance(
            request.current_lat, request.current_lon,
            SCHOOL_LAT, SCHOOL_LON,
        )
    except Exception as exc:
        logger.error("Haversine calculation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not compute distance: {exc}",
        )

    if distance_km < 0.001:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Student appears to be at the school already (distance < 1 m).",
        )

    # --- Feature 2: HistoricalAverageDuration --------------------------
    cold_start_used = False
    historical_avg: Optional[float] = None
    speed_override = request.avg_bus_speed_kmh or DEFAULT_AVG_BUS_SPEED_KMH
    wait_override = request.waiting_time_min if request.waiting_time_min is not None else DEFAULT_WAITING_TIME_MIN

    if len(request.trip_history) >= 7:
        try:
            historical_avg = trimmed_mean_historical_duration(request.trip_history)
        except ValueError as exc:
            logger.warning("Trimmed-mean fallback: %s", exc)
            historical_avg = float(np.mean(request.trip_history))
    else:
        # Cold-start: not enough history — use deterministic baseline
        cold_start_used = True
        historical_avg = cold_start_baseline(distance_km, speed_override, wait_override)
        logger.info(
            "Cold-start activated for child '%s' (history=%d trips).",
            request.child_id, len(request.trip_history),
        )

    # --- Assemble feature vector ---------------------------------------
    feature_vector = np.array([[
        distance_km,
        historical_avg,
        request.weather_severity,
        request.traffic_index,
        float(hour_of_day),
        float(day_of_week),
    ]])

    # --- Inference -----------------------------------------------------
    if app_state.model is None:
        # No trained model yet → return cold-start baseline directly
        logger.warning("No model loaded; returning cold-start baseline for child '%s'.", request.child_id)
        predicted_duration = cold_start_baseline(distance_km, speed_override, wait_override)
        cold_start_used = True
    else:
        try:
            predicted_duration = float(app_state.model.predict(feature_vector)[0])
            predicted_duration = max(predicted_duration, 1.0)   # floor at 1 minute
        except Exception as exc:
            logger.error("Model inference failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction engine error: {exc}",
            )

    logger.info(
        "Prediction | child=%s | dist=%.3f km | hist_avg=%.2f min | "
        "weather=%.2f | traffic=%.2f | hour=%d | dow=%d | pred=%.2f min",
        request.child_id, distance_km, historical_avg,
        request.weather_severity, request.traffic_index,
        hour_of_day, day_of_week, predicted_duration,
    )

    return PredictionResponse(
        child_id=request.child_id,
        predicted_duration_minutes=round(predicted_duration, 2),
        distance_to_school_km=round(distance_km, 4),
        historical_avg_duration_minutes=round(historical_avg, 2) if not cold_start_used else None,
        cold_start_used=cold_start_used,
        hour_of_day=hour_of_day,
        day_of_week=day_of_week,
        timestamp_utc=now_utc.isoformat(),
    )


@app.post(
    "/retrain/",
    response_model=RetrainResponse,
    summary="Retrain the model on the latest TripHistory records",
    tags=["Training"],
    status_code=status.HTTP_200_OK,
)
def retrain_model(request: RetrainRequest) -> RetrainResponse:
    """
    **Retraining Loop** — called by the Backend every 7 new trips.

    Accepts the last 28 `TripHistory` records, retrains a fresh
    `StandardScaler → RandomForestRegressor(n_estimators=200)` pipeline,
    persists the updated `.pkl` file, and hot-swaps the in-memory model.

    Returns MAE and R² computed on the full training batch (indicative,
    not held-out; use for monitoring, not model selection).
    """
    records = request.records
    n = len(records)
    logger.info("Retraining requested with %d records.", n)

    # --- Build X, y ----------------------------------------------------
    try:
        X = np.array([[
            r.distance_to_school,
            r.historical_avg_duration,
            r.weather_severity,
            r.traffic_index,
            float(r.hour_of_day),
            float(r.day_of_week),
        ] for r in records])

        y = np.array([r.actual_duration for r in records])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse training records: {exc}",
        )

    # --- Train ---------------------------------------------------------
    try:
        pipeline = _build_fresh_pipeline()
        pipeline.fit(X, y)
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training error: {exc}",
        )

    # --- Evaluate on training set (indicative) -------------------------
    y_pred = pipeline.predict(X)
    mae = float(np.mean(np.abs(y - y_pred)))
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # --- Persist & hot-swap --------------------------------------------
    try:
        save_model(pipeline)
    except Exception as exc:
        logger.error("Model persistence failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save model: {exc}",
        )

    app_state.model = pipeline
    saved_at = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Retraining complete | records=%d | MAE=%.4f min | R²=%.4f | saved=%s",
        n, mae, r2, saved_at,
    )

    return RetrainResponse(
        message="Model retrained and hot-swapped successfully.",
        records_used=n,
        model_saved_at=saved_at,
        mae_minutes=round(mae, 4),
        r2_score=round(r2, 4),
    )