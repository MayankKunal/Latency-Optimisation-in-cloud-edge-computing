"""
Test the latency optimization model on new data
"""

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
from train_model import load_model
from generate_data import load_datasets, generate_latency_data

def evaluate_model(model, scaler, X_test, y_test):
    """
    Evaluate the model on test data
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained model
    scaler : StandardScaler
        Fitted scaler
    X_test : numpy array
        Test features
    y_test : numpy array
        Test targets
    
    Returns:
    --------
    dict : Dictionary containing evaluation metrics
    """
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    max_error = np.max(np.abs(y_test - y_pred))
    mean_error = np.mean(y_test - y_pred)
    
    # Calculate percentiles of errors
    errors = np.abs(y_test - y_pred)
    p50_error = np.percentile(errors, 50)
    p95_error = np.percentile(errors, 95)
    p99_error = np.percentile(errors, 99)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'max_error': max_error,
        'mean_error': mean_error,
        'p50_error': p50_error,
        'p95_error': p95_error,
        'p99_error': p99_error,
        'predictions': y_pred,
        'actual': y_test
    }
    
    return metrics

def print_evaluation_report(metrics, dataset_name="Test Dataset"):
    """Print a formatted evaluation report"""
    print("\n" + "=" * 60)
    print(f"Model Evaluation Report - {dataset_name}")
    print("=" * 60)
    
    print("\n📊 Overall Metrics:")
    print(f"   R² Score:        {metrics['r2']:.4f}")
    print(f"   RMSE:            {metrics['rmse']:.2f} ms")
    print(f"   MAE:             {metrics['mae']:.2f} ms")
    print(f"   MAPE:            {metrics['mape']:.2f}%")
    print(f"   Mean Error:      {metrics['mean_error']:.2f} ms")
    print(f"   Max Error:       {metrics['max_error']:.2f} ms")
    
    print("\n📈 Error Distribution:")
    print(f"   Median Error:    {metrics['p50_error']:.2f} ms")
    print(f"   95th Percentile: {metrics['p95_error']:.2f} ms")
    print(f"   99th Percentile: {metrics['p99_error']:.2f} ms")
    
    # Sample predictions
    print("\n🔍 Sample Predictions (first 10):")
    print(f"{'Actual':>12} {'Predicted':>12} {'Error':>12} {'Error %':>12}")
    print("-" * 50)
    for i in range(min(10, len(metrics['actual']))):
        actual = metrics['actual'][i]
        pred = metrics['predictions'][i]
        error = actual - pred
        error_pct = (error / actual) * 100 if actual > 0 else 0
        print(f"{actual:>12.2f} {pred:>12.2f} {error:>12.2f} {error_pct:>11.2f}%")

def save_predictions(metrics, output_path="predictions.csv"):
    """Save predictions to CSV file"""
    df = pd.DataFrame({
        'actual_latency_ms': metrics['actual'],
        'predicted_latency_ms': metrics['predictions'],
        'error_ms': metrics['actual'] - metrics['predictions'],
        'error_percentage': ((metrics['actual'] - metrics['predictions']) / metrics['actual']) * 100
    })
    
    df.to_csv(output_path, index=False)
    print(f"\n💾 Predictions saved to: {output_path}")

def test_on_custom_data(model, scaler, data_path):
    """Test model on custom CSV data"""
    print(f"\nLoading custom test data from: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Check required columns
    required_features = [
        'network_bandwidth',
        'server_load',
        'request_size',
        'distance',
        'cpu_usage',
        'memory_usage',
        'cache_hit_rate',
        'num_concurrent_requests'
    ]
    
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Extract features
    X_test = df[required_features].values
    
    # Check if target exists
    if 'latency_ms' in df.columns:
        y_test = df['latency_ms'].values
        print(f"Found {len(X_test)} samples with ground truth labels")
        metrics = evaluate_model(model, scaler, X_test, y_test)
        print_evaluation_report(metrics, "Custom Dataset")
        return metrics
    else:
        print(f"Found {len(X_test)} samples (no ground truth labels)")
        print("Making predictions only...")
        
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
        
        # Save predictions
        result_df = df.copy()
        result_df['predicted_latency_ms'] = predictions
        output_path = data_path.replace('.csv', '_predictions.csv')
        result_df.to_csv(output_path, index=False)
        print(f"✓ Predictions saved to: {output_path}")
        
        return {'predictions': predictions}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test latency optimization model")
    parser.add_argument("--model-dir", type=str, default="model",
                        help="Directory containing the trained model")
    parser.add_argument("--test-data", type=str, default=None,
                        help="Path to custom test CSV file")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing test.csv (if --test-data not provided)")
    parser.add_argument("--generate-test", action="store_true",
                        help="Generate new test data")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of test samples to generate")
    parser.add_argument("--save-predictions", type=str, default="predictions.csv",
                        help="Path to save predictions CSV")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading trained model...")
    try:
        model, scaler = load_model(args.model_dir)
        print("✓ Model loaded successfully")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using train_model.py")
        exit(1)
    
    # Load or generate test data
    if args.test_data:
        # Test on custom data
        if not os.path.exists(args.test_data):
            print(f"Error: Test data file not found: {args.test_data}")
            exit(1)
        test_on_custom_data(model, scaler, args.test_data)
    elif args.generate_test:
        # Generate new test data
        print(f"Generating {args.n_samples} test samples...")
        X_test, y_test = generate_latency_data(n_samples=args.n_samples, random_seed=999)
        metrics = evaluate_model(model, scaler, X_test, y_test)
        print_evaluation_report(metrics, "Generated Test Dataset")
        save_predictions(metrics, args.save_predictions)
    else:
        # Use existing test data
        try:
            print("Loading test dataset...")
            _, X_test, _, y_test = load_datasets(args.data_dir)
            metrics = evaluate_model(model, scaler, X_test, y_test)
            print_evaluation_report(metrics, "Test Dataset")
            save_predictions(metrics, args.save_predictions)
        except FileNotFoundError:
            print(f"Error: Test data not found in {args.data_dir}")
            print("Use --generate-test to create test data or --test-data to specify a custom file")
            exit(1)
    
    print("\n" + "=" * 60)
    print("Model testing completed!")
    print("=" * 60)

