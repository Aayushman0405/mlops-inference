import os
import time
import mlflow
import pandas as pd
import logging
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException, Header, Response, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- App ----------------
app = FastAPI(
    title="ML Inference Service",
    description="Uber-style ETA Prediction Service",
    version="2.0.0"
)

# ---------------- Global Variables ----------------
model = None
model_version = None
model_loaded_time = None

EXPECTED_FEATURES = [
    "vendor_id",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "trip_distance",
    "pickup_hour",
    "pickup_day",
    "store_and_fwd_flag_Y"
]

# ---------------- Prometheus Metrics ----------------
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["endpoint", "model_version"]
)

REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Inference request latency in seconds",
    ["endpoint", "model_version"],
    buckets=[0.1, 0.25, 0.5, 1, 2.5, 5, 10]
)

REQUEST_ERRORS = Counter(
    "inference_request_errors_total",
    "Total inference request errors",
    ["endpoint", "error_type"]
)

MODEL_INFO = Counter(
    "model_info",
    "Model information",
    ["model_name", "model_version", "model_stage"]
)

# ---------------- Pydantic Models ----------------
class PredictionRequest(BaseModel):
    vendor_id: int = Field(..., ge=1, le=6, description="Vendor ID (1-6)")
    passenger_count: int = Field(..., ge=1, le=8, description="Number of passengers")
    pickup_longitude: float = Field(..., ge=-74.05, le=-73.75, description="Pickup longitude")
    pickup_latitude: float = Field(..., ge=40.60, le=40.90, description="Pickup latitude")
    dropoff_longitude: float = Field(..., ge=-74.05, le=-73.75, description="Dropoff longitude")
    dropoff_latitude: float = Field(..., ge=40.60, le=40.90, description="Dropoff latitude")
    trip_distance: float = Field(..., ge=0.1, le=100, description="Trip distance in miles")
    pickup_hour: int = Field(..., ge=0, le=23, description="Hour of pickup (0-23)")
    pickup_day: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    store_and_fwd_flag_Y: int = Field(0, ge=0, le=1, description="Store and forward flag")

    class Config:
        schema_extra = {
            "example": {
                "vendor_id": 1,
                "passenger_count": 1,
                "pickup_longitude": -73.98,
                "pickup_latitude": 40.75,
                "dropoff_longitude": -73.97,
                "dropoff_latitude": 40.76,
                "trip_distance": 1.5,
                "pickup_hour": 12,
                "pickup_day": 2,
                "store_and_fwd_flag_Y": 0
            }
        }

class PredictionResponse(BaseModel):
    prediction: float
    units: str = "minutes"
    model_version: str
    model_name: str
    features_used: List[str]
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    model_version: Optional[str] = None
    uptime_seconds: Optional[float] = None
    features: List[str]

# ---------------- Startup ----------------
@app.on_event("startup")
def load_model():
    global model, model_version, model_loaded_time
    
    try:
        model_loaded_time = time.time()
        
        # Get model info from environment
        model_name = os.getenv('MODEL_NAME', 'nyc_taxi_rf')
        model_alias = os.getenv('MODEL_ALIAS', 'production')
        model_uri = f"models:/{model_name}@{model_alias}"
        
        logger.info(f"Loading model from {model_uri}")
        
        # Configure MLflow for S3
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL', 
                                                         'http://rook-ceph-rgw-mlflow-store.rook-ceph.svc.cluster.local')
        
        # Load model
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Get model version
        client = mlflow.tracking.MlflowClient()
        model_version_detail = client.get_model_version_by_alias(model_name, model_alias)
        model_version = model_version_detail.version if model_version_detail else "unknown"
        
        # Record model info in metrics
        MODEL_INFO.labels(
            model_name=model_name,
            model_version=model_version,
            model_stage=model_alias
        ).inc()
        
        # Sanity check
        test_df = pd.DataFrame([{
            "vendor_id": 1,
            "passenger_count": 1,
            "pickup_longitude": -73.98,
            "pickup_latitude": 40.75,
            "dropoff_longitude": -73.97,
            "dropoff_latitude": 40.76,
            "trip_distance": 1.5,
            "pickup_hour": 12,
            "pickup_day": 2,
            "store_and_fwd_flag_Y": 0,
        }])
        
        test_pred = model.predict(test_df)
        logger.info(f"Model loaded successfully. Test prediction: {test_pred[0]}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model = None
        model_version = None

# ---------------- Health ----------------
@app.get("/health", response_model=HealthResponse)
async def health():
    uptime = time.time() - model_loaded_time if model_loaded_time else None
    
    return HealthResponse(
        status="ok" if model else "degraded",
        service="ml-inference",
        model_loaded=model is not None,
        model_version=model_version,
        uptime_seconds=uptime,
        features=EXPECTED_FEATURES,
    )

# ---------------- Metrics ----------------
@app.get("/metrics")
async def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

# ---------------- Readiness Probe ----------------
@app.get("/ready")
async def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

# ---------------- Liveness Probe ----------------
@app.get("/live")
async def live():
    return {"status": "alive"}

# ---------------- Prediction ----------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    req: PredictionRequest,
    request: Request,
    x_api_key: Optional[str] = Header(None),
):
    start_time = time.time()
    
    # Authenticate
    expected_key = os.getenv("API_KEY")
    if expected_key and x_api_key != expected_key:
        REQUEST_ERRORS.labels(endpoint="/predict", error_type="unauthorized").inc()
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Check model loaded
    if model is None:
        REQUEST_ERRORS.labels(endpoint="/predict", error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        input_dict = req.dict()
        df = pd.DataFrame([input_dict])
        
        # Ensure correct column order
        df = df[EXPECTED_FEATURES]
        
        # Predict
        prediction = model.predict(df)
        result = float(prediction[0])
        
        # Record metrics
        latency = time.time() - start_time
        REQUEST_COUNT.labels(endpoint="/predict", model_version=model_version or "unknown").inc()
        REQUEST_LATENCY.labels(endpoint="/predict", model_version=model_version or "unknown").observe(latency)
        
        return PredictionResponse(
            prediction=result,
            model_version=model_version or "unknown",
            model_name=os.getenv('MODEL_NAME', 'nyc_taxi_rf'),
            features_used=list(df.columns),
            latency_ms=round(latency * 1000, 2)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        REQUEST_ERRORS.labels(endpoint="/predict", error_type="prediction_error").inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ---------------- Batch Prediction ----------------
class BatchPredictionRequest(BaseModel):
    trips: List[PredictionRequest]
    batch_id: Optional[str] = None

@app.post("/predict/batch")
async def batch_predict(
    req: BatchPredictionRequest,
    x_api_key: Optional[str] = Header(None),
):
    start_time = time.time()
    
    # Authenticate
    expected_key = os.getenv("API_KEY")
    if expected_key and x_api_key != expected_key:
        REQUEST_ERRORS.labels(endpoint="/predict/batch", error_type="unauthorized").inc()
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if model is None:
        REQUEST_ERRORS.labels(endpoint="/predict/batch", error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert all trips to DataFrame
        trips_data = [trip.dict() for trip in req.trips]
        df = pd.DataFrame(trips_data)
        df = df[EXPECTED_FEATURES]
        
        # Batch predict
        predictions = model.predict(df)
        
        # Record metrics
        latency = time.time() - start_time
        REQUEST_COUNT.labels(endpoint="/predict/batch", model_version=model_version or "unknown").inc()
        REQUEST_LATENCY.labels(endpoint="/predict/batch", model_version=model_version or "unknown").observe(latency)
        
        return {
            "batch_id": req.batch_id or f"batch_{int(time.time())}",
            "predictions": [float(p) for p in predictions],
            "count": len(predictions),
            "model_version": model_version or "unknown",
            "latency_ms": round(latency * 1000, 2)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        REQUEST_ERRORS.labels(endpoint="/predict/batch", error_type="prediction_error").inc()
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
