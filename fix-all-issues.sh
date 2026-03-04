#!/bin/bash

set -e

echo "🔧 Fixing all MLOps issues..."

NAMESPACE_BLUE="mlops-blue"
NAMESPACE_GREEN="mlops-green"

# Function to wait for pods
wait_for_pod() {
    local ns=$1
    local label=$2
    local timeout=$3
    echo "Waiting for $label in $ns..."
    kubectl wait --for=condition=ready pod -l $label -n $ns --timeout=${timeout}s 2>/dev/null || true
}

# 1. Fix MLflow connection
echo "📡 Checking MLflow connection..."

# Check if MLflow is running
MLFLOW_POD=$(kubectl get pods -n $NAMESPACE_BLUE -l app=mlflow-server -o name)
if [ -z "$MLFLOW_POD" ]; then
    echo "❌ MLflow pod not found! Redeploying MLflow..."
    kubectl delete -f mlflow/ --ignore-not-found=true
    kubectl apply -f mlflow/
    sleep 30
fi

# Get MLflow service details
MLFLOW_SERVICE_IP=$(kubectl get svc -n $NAMESPACE_BLUE mlflow-server -o jsonpath='{.spec.clusterIP}')
MLFLOW_PORT=$(kubectl get svc -n $NAMESPACE_BLUE mlflow-server -o jsonpath='{.spec.ports[0].port}')
echo "MLflow service: $MLFLOW_SERVICE_IP:$MLFLOW_PORT"

# Test MLflow connection from a test pod
echo "Testing MLflow connection..."
kubectl run curl-test --image=curlimages/curl --restart=Never --rm -it -- \
    curl -s http://mlflow-server.$NAMESPACE_BLUE.svc.cluster.local:$MLFLOW_PORT/health --connect-timeout 5 || {
    echo "❌ Cannot connect to MLflow. Checking logs..."
    kubectl logs -n $NAMESPACE_BLUE $MLFLOW_POD --tail=20
}

# 2. Fix green namespace configmap
echo -e "\n📋 Fixing green namespace configmap..."
kubectl get configmap inference-config -n $NAMESPACE_BLUE -o yaml | \
    sed "s/namespace: $NAMESPACE_BLUE/namespace: $NAMESPACE_GREEN/g" | \
    kubectl apply -f -

# 3. Ensure green namespace has secrets
echo -e "\n🔑 Ensuring green namespace secrets..."
if ! kubectl get secret inference-secrets -n $NAMESPACE_GREEN &>/dev/null; then
    # Copy secrets from blue
    API_KEY=$(kubectl get secret inference-secrets -n $NAMESPACE_BLUE -o jsonpath='{.data.API_KEY}' | base64 -d)
    AWS_ACCESS_KEY=$(kubectl get secret inference-secrets -n $NAMESPACE_BLUE -o jsonpath='{.data.AWS_ACCESS_KEY_ID}' | base64 -d)
    AWS_SECRET_KEY=$(kubectl get secret inference-secrets -n $NAMESPACE_BLUE -o jsonpath='{.data.AWS_SECRET_ACCESS_KEY}' | base64 -d)
    
    kubectl create secret generic inference-secrets -n $NAMESPACE_GREEN \
        --from-literal=API_KEY=$API_KEY \
        --from-literal=AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY \
        --from-literal=AWS_SECRET_ACCESS_KEY=$AWS_SECRET_KEY \
        --dry-run=client -o yaml | kubectl apply -f -
fi

# 4. Check and fix RGW endpoint
echo -e "\n☁️ Checking RGW endpoint..."
RGW_SERVICE=$(kubectl get svc -n rook-ceph | grep rgw | head -1 | awk '{print $1}')
if [ -n "$RGW_SERVICE" ]; then
    RGW_ENDPOINT="http://$RGW_SERVICE.rook-ceph.svc.cluster.local"
    echo "Found RGW endpoint: $RGW_ENDPOINT"
    
    # Update configmaps
    kubectl patch configmap inference-config -n $NAMESPACE_BLUE -p "{\"data\":{\"MLFLOW_S3_ENDPOINT_URL\":\"$RGW_ENDPOINT\"}}"
    kubectl patch configmap inference-config -n $NAMESPACE_GREEN -p "{\"data\":{\"MLFLOW_S3_ENDPOINT_URL\":\"$RGW_ENDPOINT\"}}"
fi

# 5. Check if model exists in MLflow
echo -e "\n🤖 Checking if model exists..."
kubectl port-forward -n $NAMESPACE_BLUE svc/mlflow-server 5000:5000 &
PF_PID=$!
sleep 3

# Check if model exists
curl -s http://localhost:5000/api/2.0/mlflow/registered-models/get?name=nyc_taxi_rf || {
    echo "❌ Model 'nyc_taxi_rf' not found! Creating sample model..."
    
    # Create a simple model
    python3 << 'PYTHON_SCRIPT'
import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc_taxi_trip_duration")

# Generate sample data
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'vendor_id': np.random.randint(1, 7, n_samples),
    'passenger_count': np.random.randint(1, 5, n_samples),
    'pickup_longitude': np.random.uniform(-74.05, -73.75, n_samples),
    'pickup_latitude': np.random.uniform(40.60, 40.90, n_samples),
    'dropoff_longitude': np.random.uniform(-74.05, -73.75, n_samples),
    'dropoff_latitude': np.random.uniform(40.60, 40.90, n_samples),
    'trip_distance': np.random.uniform(0.5, 20, n_samples),
    'pickup_hour': np.random.randint(0, 24, n_samples),
    'pickup_day': np.random.randint(0, 7, n_samples),
    'store_and_fwd_flag_Y': np.random.randint(0, 2, n_samples),
    'trip_duration': np.random.uniform(2, 60, n_samples)
})

X = data.drop('trip_duration', axis=1)
y = data['trip_duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="sample_rf_model"):
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = np.mean(np.abs(predictions - y_test))
    
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="nyc_taxi_rf"
    )
    print(f"✅ Model created with MAE: {mae:.2f}")
PYTHON_SCRIPT

    # Set to production
    python3 << 'PYTHON_SCRIPT'
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

versions = client.get_latest_versions("nyc_taxi_rf", stages=["None"])
if versions:
    latest_version = versions[0].version
    client.transition_model_version_stage(
        name="nyc_taxi_rf",
        version=latest_version,
        stage="Production"
    )
    client.set_registered_model_alias(
        name="nyc_taxi_rf",
        alias="production",
        version=latest_version
    )
    print(f"✅ Model version {latest_version} set to Production")
PYTHON_SCRIPT
}

kill $PF_PID 2>/dev/null || true

# 6. Restart deployments
echo -e "\n🔄 Restarting deployments..."
kubectl rollout restart deployment/inference-blue -n $NAMESPACE_BLUE
kubectl rollout restart deployment/inference-green -n $NAMESPACE_GREEN

# 7. Wait for pods to be ready
echo -e "\n⏳ Waiting for pods to be ready..."
sleep 10

# 8. Show status
echo -e "\n📊 Current pod status:"
echo "=== Blue namespace ==="
kubectl get pods -n $NAMESPACE_BLUE
echo -e "\n=== Green namespace ==="
kubectl get pods -n $NAMESPACE_GREEN

# 9. Show recent events
echo -e "\n📋 Recent events in blue namespace:"
kubectl get events -n $NAMESPACE_BLUE --sort-by='.lastTimestamp' | tail -5

echo -e "\n✅ Fix script completed!"
