🚖 MLOps Platform - NYC Taxi Trip Duration Prediction

A production-ready MLOps platform for training, tracking, and serving NYC Taxi trip duration prediction models with blue-green deployments, auto-scaling, and monitoring.

                                    ┌─────────────────┐
                                    │   Ingress       │
                                    │   (nginx)       │
                                    └────────┬────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    ▼                       ▼                       ▼
            ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
            │   MLflow UI   │       │    Inference  │       │   Prometheus  │
            │   (Port 5000) │       │   Service     │       │   Metrics     │
            └───────┬───────┘       │   (Port 8000) │       └───────────────┘
                    │               └───────┬───────┘
                    │                       │
                    ▼                       ▼
            ┌───────────────┐       ┌───────────────┐
            │   MariaDB     │       │   Blue/Green  │
            │   (Backend)   │       │  Deployments  │
            └───────────────┘       └───────┬───────┘
                                            │
                                    ┌───────▼───────┐
                                    │    HPA        │
                                    │   (Auto-scale)│
                                    └───────────────┘
                               ┌─────────────────────────┐
                               │     Rook/Ceph Storage   │
                               │  ├─ Block (MySQL)       │
                               │  ├─ FS (Artifacts)      │
                               │  └─ S3 (MLflow Models)  │
                               └─────────────────────────┘


💻 Tech Stack
Core Technologies
Kubernetes: Container orchestration (v1.19+)
Docker: Containerization
Rook/Ceph: Distributed storage (Block, Filesystem, S3-compatible object storage)

ML Platform
MLflow: Model tracking, registry, and management (v2.12.1)

Python: ML model development (v3.9+)
Scikit-learn: RandomForest model training
Pandas/Numpy: Data manipulation

Inference Service
FastAPI: High-performance REST API
Uvicorn: ASGI server
Prometheus: Metrics collection and monitoring
Pydantic: Data validation

Deployment & Operations
Blue-Green Deployment: Zero-downtime updates
Horizontal Pod Autoscaler (HPA): Auto-scaling based on CPU/memory
Ingress-Nginx: Load balancing and SSL termination
Cert-Manager: Automatic SSL certificate management


📁 Project Structure
mlops-platform/
├── inference/                 # Inference service
│   ├── app.py                # FastAPI application with prediction endpoints
│   ├── Dockerfile             # Container definition
│   ├── requirements.txt       # Python dependencies
│   ├── 01-configmap.yaml      # Configuration for inference service
│   ├── 02-secrets-template.yaml # Secret template (user must fill)
│   ├── 03-deployment-blue.yaml # Blue deployment with HPA
│   ├── 04-deployment-green.yaml # Green deployment with HPA
│   ├── 05-service.yaml        # Kubernetes services
│   └── 06-ingress.yaml        # Ingress with SSL support
│
├── mlflow/                    # MLflow tracking server
│   ├── train-script.py        # Training script with synthetic data
│   ├── 01-configmap.yaml      # MLflow configuration
│   ├── 02-secrets-template.yaml # Database and AWS secrets template
│   ├── 03-mysql-deployment.yaml # MySQL for MLflow backend
│   ├── 07-mlflow-deployment-s3.yaml # MLflow server with S3
│   ├── 08-mariadb-service.yaml # MariaDB service
│   └── 09-train-model-job.yaml # Kubernetes training job
│
├── storage/                   # Storage configuration
│   ├── 01-storage-class.yaml  # Rook/Ceph storage classes
│   ├── 02-ceph-rgw-bucket.yaml # S3-compatible bucket
│   └── 03-pv-claims.yaml       # PVC for MySQL and artifacts
│
├── namespaces/                # Kubernetes namespaces
│   └── 01-namespace-blue.yaml # mlops-blue namespace
│
├── scripts/                   # Utility scripts
│   ├── train_model.py         # Advanced training with hyperparameter tuning
│   ├── register_model.py      # Model stage management
│   └── switch-blue-green.sh   # Blue-green deployment switcher
│
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
└── Makefile                   # Automation commands


📊 MLflow Tracking Server
Access URL: https://mlflow.aayushmandev.space

The MLflow server provides:

Experiment Tracking: Log parameters, metrics, and artifacts

Model Registry: Version control for models with staging/production stages

Artifact Storage: S3-compatible storage via Rook/Ceph

Features
Backend store: MariaDB for metadata

Artifact store: S3 bucket for models and artifacts

Auto-registration of models with production stage promotion

🏋️ Training Pipeline
Model Logic
The model predicts taxi trip duration based on:

Features:
vendor_id: Taxi vendor (1-6)
passenger_count: Number of passengers
pickup_longitude/latitude: Pickup coordinates
dropoff_longitude/latitude: Dropoff coordinates
trip_distance: Distance in miles
pickup_hour: Hour of day (0-23)
pickup_day: Day of week (0-6)
store_and_fwd_flag_Y: Store and forward flag
Target: trip_duration in minutes

The relationship is modeled as:
trip_duration = 3 + 
                0.5 * trip_distance + 
                0.1 * passenger_count +
                0.02 * (longitude_diff * 100) +
                0.02 * (latitude_diff * 100) +
                random_noise
Training Features
Synthetic Data Generation: Creates realistic training data for testing
RandomForest Model: 150 estimators with max depth 20
Automatic Registration: Models auto-registered to MLflow registry
Production Promotion: Best model automatically promoted to Production stage

Advanced Training (scripts/train_model.py)
The advanced training script includes:
Hyperparameter Tuning: Using Optuna for automated hyperparameter search
Cross-Validation: 3-fold CV for robust model selection
Feature Engineering: Sine/cosine transformations for cyclical features
Multiple Metrics: MAE, RMSE, R² tracking
Feature Importance: Automatic logging of feature importance plots

🌐 Inference Service
FastAPI Application Features
The inference service (inference/app.py) provides:
RESTful API: Modern async FastAPI implementation
Model Loading: Dynamic loading from MLflow registry
Request Validation: Pydantic models for type safety
Authentication: API key validation for all endpoints
Prometheus Metrics: Built-in monitoring
Health Checks: Readiness and liveness probes
Batch Processing: Support for batch predictions

API Endpoints
Endpoint	Method	Description
/	GET	Welcome message
/health	GET	Service health status
/ready	GET	Readiness probe for k8s
/live	GET	Liveness probe for k8s
/metrics	GET	Prometheus metrics
/predict	POST	Single prediction
/predict/batch	POST	Batch predictions
/docs	GET	Swagger UI documentation


📚 API Documentation
Interactive API Docs
Once deployed, visit:

Swagger UI: https://mlops-inference.aayushmandev.space/docs


Authentication
All prediction endpoints require an API key in the header:
X-API-Key: your-api-key-here


Request/Response Examples
Single Prediction

Request:
curl -X POST https://mlops-inference.aayushmandev.space/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
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
  }'


Batch Prediction
curl -X POST https://mlops-inference.aayushmandev.space/predict/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "batch_id": "test-batch-001",
    "trips": [
      {
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
      },
      {
        "vendor_id": 2,
        "passenger_count": 2,
        "pickup_longitude": -73.99,
        "pickup_latitude": 40.76,
        "dropoff_longitude": -73.96,
        "dropoff_latitude": 40.77,
        "trip_distance": 2.5,
        "pickup_hour": 18,
        "pickup_day": 5,
        "store_and_fwd_flag_Y": 0
      }
    ]
  }'




🔄 Blue-Green Deployment
Strategy
Blue: Current production environment (mlops-blue namespace)
Green: Staging environment for testing (mlops-green namespace)
Service: Points to active environment (blue by default)

Switching Environments
# Switch to green environment
./scripts/switch-blue-green.sh green
# Switch back to blue
./scripts/switch-blue-green.sh blue


Zero-Downtime Updates

Deploy new version to green environment
Run tests against green service
Switch traffic by updating service selector
Monitor for issues
Scale down blue if everything is healthy

Kubernetes Health Checks
Readiness Probe (/ready): Pod is ready to receive traffic
Liveness Probe (/live): Pod is alive and running
Startup Probe: Initial startup period



Horizontal Pod Autoscaling:
metrics:
- type: Resource
  resource:
    name: cpu
    target:
      type: Utilization
      averageUtilization: 70
- type: Resource
  resource:
    name: memory
    target:
      type: Utilization
      averageUtilization: 80

Scaling: 3-10 replicas based on CPU/Memory usage


🔒 Security
Secrets Management
Never commit secrets to git (use .gitignore)

Template files provided for reference

Environment variables for local development

Kubernetes secrets for production

Authentication
API key validation for all prediction endpoints

Keys stored as Kubernetes secrets

Rotate keys regularly

Network Security
TLS/SSL for all public endpoints

Internal services use ClusterIP

Network policies (if enabled)

Storage Security
Ceph encryption at rest

S3 bucket with access credentials

Database credentials in secrets

🎯 What This Platform Achieves
End-to-End MLOps: Complete pipeline from training to production
Reproducibility: All experiments tracked in MLflow
Scalability: Auto-scaling inference service
High Availability: Blue-green deployments with zero downtime
Monitoring: Built-in Prometheus metrics
Security: API authentication and secret management
Storage: Distributed, fault-tolerant storage with Ceph
Documentation: Self-documenting API with Swagger/ReDoc


🛠 Makefile Commands

make help           # Show available commands
make setup          # Create namespaces and storage
make deploy-mlflow  # Deploy MLflow tracking server
make deploy-inference # Deploy inference service
make train          # Submit training job
make test           # Test inference endpoint
make clean          # Remove all deployments
make switch-blue-green environment=green # Switch to green

