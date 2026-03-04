import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
import time
import sys
import os

def main():
    print("🚀 Training NYC Taxi Duration Model...")
    
    # Use the correct service name
    mlflow_tracking_uri = "http://mlflow.mlflow.svc.cluster.local:5000"
    print(f"Connecting to MLflow at: {mlflow_tracking_uri}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Test MLflow connection
    try:
        experiments = mlflow.search_experiments()
        print(f"✅ Connected to MLflow. Found {len(experiments)} experiments")
    except Exception as e:
        print(f"❌ Failed to connect to MLflow: {e}")
        print("Checking service details...")
        import socket
        try:
            ip = socket.gethostbyname('mlflow.mlflow.svc.cluster.local')
            print(f"IP address of mlflow.mlflow.svc.cluster.local: {ip}")
        except:
            print("Could not resolve hostname")
        sys.exit(1)
    
    # Generate synthetic training data
    print("📊 Generating training data...")
    np.random.seed(42)
    n_samples = 5000
    
    data = pd.DataFrame({
        "vendor_id": np.random.randint(1, 7, n_samples),
        "passenger_count": np.random.randint(1, 6, n_samples),
        "pickup_longitude": np.random.uniform(-74.05, -73.75, n_samples),
        "pickup_latitude": np.random.uniform(40.60, 40.90, n_samples),
        "dropoff_longitude": np.random.uniform(-74.05, -73.75, n_samples),
        "dropoff_latitude": np.random.uniform(40.60, 40.90, n_samples),
        "trip_distance": np.random.uniform(0.5, 30.0, n_samples),
        "pickup_hour": np.random.randint(0, 24, n_samples),
        "pickup_day": np.random.randint(0, 7, n_samples),
        "store_and_fwd_flag_Y": np.random.randint(0, 2, n_samples),
    })
    
    # Generate target (trip duration in minutes)
    data["trip_duration"] = (
        3 + 
        0.5 * data["trip_distance"] + 
        0.1 * data["passenger_count"] +
        0.02 * np.abs(data["pickup_longitude"] - data["dropoff_longitude"]) * 100 +
        0.02 * np.abs(data["pickup_latitude"] - data["dropoff_latitude"]) * 100 +
        np.random.normal(0, 3, n_samples)
    )
    
    X = data.drop("trip_duration", axis=1)
    y = data["trip_duration"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("🏋️ Training RandomForest model...")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_pred - y_test))
    print(f"✅ Model trained - MAE: {mae:.2f} minutes")
    
    # Log to MLflow
    with mlflow.start_run(run_name="nyc_taxi_rf_production") as run:
        mlflow.log_metric("mae", mae)
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("max_depth", 20)
        
        # Register model
        print("📝 Registering model as 'nyc_taxi_rf'...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="nyc_taxi_rf"
        )
        
        print(f"✅ Model logged with run_id: {run.info.run_id}")
        
        # Wait for registration
        time.sleep(5)
        
        # Get the latest version and set to production
        client = MlflowClient()
        versions = client.get_latest_versions("nyc_taxi_rf")
        if versions:
            latest_version = versions[0].version
            print(f"📦 Model version: {latest_version}")
            
            # Transition to Production
            client.transition_model_version_stage(
                name="nyc_taxi_rf",
                version=latest_version,
                stage="Production"
            )
            print(f"✅ Model version {latest_version} set to Production")
    
    print("\n🎉 Training job completed successfully!")

if __name__ == "__main__":
    main()
