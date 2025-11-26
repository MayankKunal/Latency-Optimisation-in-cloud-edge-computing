"""
Generate training and testing datasets for latency optimization model
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def generate_latency_data(n_samples=5000, random_seed=42, noise_level=10):
    """
    Generate synthetic training data for latency prediction
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_seed : int
        Random seed for reproducibility
    noise_level : float
        Standard deviation of noise to add
    
    Returns:
    --------
    X : numpy array
        Feature matrix
    y : numpy array
        Target values (latency in ms)
    """
    rng = np.random.RandomState(random_seed)
    
    X = []
    y = []
    
    print(f"Generating {n_samples} samples...")
    
    for i in range(n_samples):
        # Features that affect latency
        network_bandwidth = rng.uniform(1, 100)  # Mbps
        server_load = rng.uniform(0, 1)  # 0-1 scale
        request_size = rng.uniform(0.1, 10)  # MB
        distance = rng.uniform(1, 1000)  # km
        cpu_usage = rng.uniform(0, 1)  # 0-1 scale
        memory_usage = rng.uniform(0, 1)  # 0-1 scale
        cache_hit_rate = rng.uniform(0, 1)  # 0-1 scale
        num_concurrent_requests = rng.uniform(1, 100)
        
        # Simulate latency based on features (with some noise)
        base_latency = (
            50 +  # Base latency (ms)
            (distance * 0.1) +  # Distance penalty
            (request_size * 5) +  # Size penalty
            (server_load * 100) +  # Load penalty
            (100 / max(network_bandwidth, 1)) +  # Bandwidth penalty
            (cpu_usage * 50) +  # CPU penalty
            (memory_usage * 30) -  # Memory penalty
            (cache_hit_rate * 40) +  # Cache benefit
            (num_concurrent_requests * 0.5) +  # Concurrency penalty
            rng.normal(0, noise_level)  # Noise
        )
        
        latency = max(10, base_latency)  # Minimum 10ms
        
        X.append([
            network_bandwidth,
            server_load,
            request_size,
            distance,
            cpu_usage,
            memory_usage,
            cache_hit_rate,
            num_concurrent_requests
        ])
        y.append(latency)
    
    return np.array(X), np.array(y)

def save_datasets(X_train, X_test, y_train, y_test, output_dir="data"):
    """Save training and testing datasets to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature names
    feature_names = [
        'network_bandwidth',
        'server_load',
        'request_size',
        'distance',
        'cpu_usage',
        'memory_usage',
        'cache_hit_rate',
        'num_concurrent_requests'
    ]
    
    # Create DataFrames
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['latency_ms'] = y_train
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['latency_ms'] = y_test
    
    # Save to CSV
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✓ Training data saved to: {train_path}")
    print(f"  - Samples: {len(train_df)}")
    print(f"  - Features: {len(feature_names)}")
    print(f"  - Latency range: {y_train.min():.2f} - {y_train.max():.2f} ms")
    
    print(f"\n✓ Testing data saved to: {test_path}")
    print(f"  - Samples: {len(test_df)}")
    print(f"  - Latency range: {y_test.min():.2f} - {y_test.max():.2f} ms")
    
    return train_path, test_path

def load_datasets(data_dir="data"):
    """Load training and testing datasets from CSV files"""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset files not found in {data_dir}. Run generate_data.py first.")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate features and target
    feature_names = [
        'network_bandwidth',
        'server_load',
        'request_size',
        'distance',
        'cpu_usage',
        'memory_usage',
        'cache_hit_rate',
        'num_concurrent_requests'
    ]
    
    X_train = train_df[feature_names].values
    y_train = train_df['latency_ms'].values
    
    X_test = test_df[feature_names].values
    y_test = test_df['latency_ms'].values
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Configuration
    n_samples = 5000
    test_size = 0.2
    random_seed = 42
    
    print("=" * 60)
    print("Latency Optimization Model - Data Generation")
    print("=" * 60)
    
    # Generate data
    X, y = generate_latency_data(n_samples=n_samples, random_seed=random_seed)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    
    print(f"\nData split:")
    print(f"  - Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  - Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Save datasets
    save_datasets(X_train, X_test, y_train, y_test)
    
    print("\n" + "=" * 60)
    print("Data generation completed successfully!")
    print("=" * 60)

