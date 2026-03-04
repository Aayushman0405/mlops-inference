#!/usr/bin/env python3
"""
Script to manage model stages in MLflow
"""

import mlflow
import argparse
from mlflow.tracking import MlflowClient

def promote_to_production(model_name, version=None, run_id=None):
    """Promote a model to production stage"""
    client = MlflowClient()
    
    if run_id:
        # Get version from run_id
        versions = client.get_latest_versions(model_name)
        for v in versions:
            if v.run_id == run_id:
                version = v.version
                break
    
    if not version:
        # Get latest version
        latest = client.get_latest_versions(model_name, stages=["None", "Staging"])
        if latest:
            version = latest[0].version
        else:
            raise ValueError("No model version found")
    
    # Archive current production
    current_prod = client.get_latest_versions(model_name, stages=["Production"])
    for v in current_prod:
        client.transition_model_version_stage(
            name=model_name,
            version=v.version,
            stage="Archived"
        )
        print(f"Archived version {v.version}")
    
    # Promote new version
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    print(f"Promoted version {version} to Production")
    
    # Set alias
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=version
    )
    print(f"Set alias 'production' to version {version}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--version', type=str, help='Model version to promote')
    parser.add_argument('--run-id', type=str, help='Run ID of the model')
    parser.add_argument('--stage', type=str, default='production',
                       choices=['staging', 'production', 'archived'])
    args = parser.parse_args()
    
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000'))
    
    if args.stage == 'production':
        promote_to_production(args.model_name, args.version, args.run_id)

if __name__ == "__main__":
    main()
