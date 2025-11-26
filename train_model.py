"""
Train the latency optimization model
"""

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
from generate_data import load_datasets, generate_latency_data
from sklearn.model_selection import train_test_split

def train_model(X_train, y_train, X_val=None, y_val=None, 
                n_estimators=100, max_depth=15, random_state=42):
    """
    Train a Random Forest model for latency prediction
    
    Parameters:
    -----------
    X_train : numpy array
        Training features
    y_train : numpy array
        Training targets
    X_val : numpy array, optional
        Validation features
    y_val : numpy array, optional
        Validation targets
    n_estimators : int
        Number of trees in the forest
    max_depth : int
        Maximum depth of trees
    random_state : int
        Random seed
    
    Returns:
    --------
    model : RandomForestRegressor
        Trained model
    scaler : StandardScaler
        Fitted scaler
    """
    print("\n" + "=" * 60)
    print("Training Latency Optimization Model")
    print("=" * 60)
    
    # Scale features
    print("\n1. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    
    # Train model
    print(f"2. Training Random Forest model...")
    print(f"   - Number of trees: {n_estimators}")
    print(f"   - Max depth: {max_depth}")
    print(f"   - Training samples: {len(X_train)}")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on training set
    print("\n3. Evaluating on training set...")
    y_train_pred = model.predict(X_train_scaled)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"   - RMSE: {train_rmse:.2f} ms")
    print(f"   - MAE: {train_mae:.2f} ms")
    print(f"   - R² Score: {train_r2:.4f}")
    
    # Evaluate on validation set if provided
    if X_val is not None and y_val is not None:
        print("\n4. Evaluating on validation set...")
        y_val_pred = model.predict(X_val_scaled)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(f"   - RMSE: {val_rmse:.2f} ms")
        print(f"   - MAE: {val_mae:.2f} ms")
        print(f"   - R² Score: {val_r2:.4f}")
    
    return model, scaler

def save_model(model, scaler, output_dir="model"):
    """Save the trained model and scaler"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "latency_model.joblib")
    scaler_path = os.path.join(output_dir, "latency_scaler.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n5. Saving model...")
    print(f"   ✓ Model saved to: {model_path}")
    print(f"   ✓ Scaler saved to: {scaler_path}")
    
    return model_path, scaler_path

def load_model(model_dir="model"):
    """Load the trained model and scaler"""
    model_path = os.path.join(model_dir, "latency_model.joblib")
    scaler_path = os.path.join(model_dir, "latency_scaler.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model files not found in {model_dir}. Train the model first.")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train latency optimization model")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing train.csv and test.csv")
    parser.add_argument("--model-dir", type=str, default="model",
                        help="Directory to save the trained model")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Number of trees in the forest")
    parser.add_argument("--max-depth", type=int, default=15,
                        help="Maximum depth of trees")
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate new data if datasets don't exist")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Load or generate data
    try:
        print("Loading datasets...")
        X_train, X_test, y_train, y_test = load_datasets(args.data_dir)
    except FileNotFoundError:
        if args.generate_data:
            print("Datasets not found. Generating new data...")
            from generate_data import generate_latency_data
            X, y = generate_latency_data(n_samples=5000, random_seed=args.random_seed)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=args.random_seed
            )
            # Save the generated data
            from generate_data import save_datasets
            save_datasets(X_train, X_test, y_train, y_test, args.data_dir)
        else:
            print("Error: Datasets not found. Use --generate-data to create them.")
            exit(1)
    
    # Train model
    model, scaler = train_model(
        X_train, y_train, X_test, y_test,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_seed
    )
    
    # Save model
    save_model(model, scaler, args.model_dir)
    
    print("\n" + "=" * 60)
    print("Model training completed successfully!")
    print("=" * 60)

