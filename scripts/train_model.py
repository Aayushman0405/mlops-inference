#!/usr/bin/env python3
"""
Advanced training script with hyperparameter tuning and MLflow tracking
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import optuna
import argparse
import logging
from datetime import datetime
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path):
    """Load and preprocess the dataset"""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Basic preprocessing
    df = df.dropna()
    
    # Feature engineering
    df['pickup_hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
    df['pickup_hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)
    df['pickup_day_sin'] = np.sin(2 * np.pi * df['pickup_day'] / 7)
    df['pickup_day_cos'] = np.cos(2 * np.pi * df['pickup_day'] / 7)
    
    # Split features and target
    target_col = 'trip_duration'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y

def objective(trial, X_train, y_train):
    """Optuna objective function"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }
    
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X_train, y_train, cv=3, 
                            scoring='neg_mean_absolute_error')
    return scores.mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--model-name', type=str, default='nyc_taxi_rf')
    parser.add_argument('--experiment-name', type=str, default='nyc_taxi_trip_duration')
    parser.add_argument('--n-trials', type=int, default=20)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--register-model', action='store_true')
    args = parser.parse_args()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000'))
    mlflow.set_experiment(args.experiment_name)
    
    # Load data
    X, y = load_and_preprocess_data(args.data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log dataset info
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X.shape[1])
        
        # Hyperparameter optimization
        logger.info("Starting hyperparameter optimization...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train), 
                      n_trials=args.n_trials)
        
        # Train best model
        best_params = study.best_params
        logger.info(f"Best parameters: {best_params}")
        
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "best_cv_score": study.best_value
        })
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')
        
        # Save feature columns
        with open('feature_columns.txt', 'w') as f:
            for col in X.columns:
                f.write(f"{col}\n")
        mlflow.log_artifact('feature_columns.txt')
        
        # Log model
        if args.register_model:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=args.model_name
            )
            logger.info(f"Model registered as {args.model_name}")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        logger.info(f"Training complete. MAE: {mae:.2f}, R2: {r2:.3f}")

if __name__ == "__main__":
    main()
