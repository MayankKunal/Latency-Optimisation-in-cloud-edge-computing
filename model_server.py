# FastAPI Server for Latency Optimization Model
# This server provides endpoints for predicting and optimizing latency

import json
from typing import List, Optional
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime

# -------------------------
# Training Data Generation (Simulated)
# -------------------------

def generate_training_data(n_samples=2000):
    """Generate synthetic training data for latency prediction"""
    rng = np.random.RandomState(42)
    
    X = []
    y = []
    
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
            rng.normal(0, 10)  # Noise
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

# Train model if not exists
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

if not os.path.exists(f"{model_dir}/latency_model.joblib"):
    print("Training latency prediction model...")
    X, y = generate_training_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"Model trained - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
    
    # Save model and scaler
    joblib.dump(model, f"{model_dir}/latency_model.joblib")
    joblib.dump(scaler, f"{model_dir}/latency_scaler.joblib")
    print("Model saved successfully!")
else:
    print("Loading existing model...")

# Load model and scaler
model = joblib.load(f"{model_dir}/latency_model.joblib")
scaler = joblib.load(f"{model_dir}/latency_scaler.joblib")

# -------------------------
# FastAPI Application
# -------------------------

app = FastAPI(
    title="Latency Optimization API",
    description="API for predicting and optimizing latency in distributed systems",
    version="1.0.0"
)

# -------------------------
# Pydantic Models
# -------------------------

class LatencyFeatures(BaseModel):
    """Features for latency prediction"""
    network_bandwidth: float = Field(..., ge=0.1, description="Network bandwidth in Mbps")
    server_load: float = Field(..., ge=0, le=1, description="Server load (0-1)")
    request_size: float = Field(..., ge=0.1, description="Request size in MB")
    distance: float = Field(..., ge=0, description="Distance to server in km")
    cpu_usage: float = Field(..., ge=0, le=1, description="CPU usage (0-1)")
    memory_usage: float = Field(..., ge=0, le=1, description="Memory usage (0-1)")
    cache_hit_rate: float = Field(..., ge=0, le=1, description="Cache hit rate (0-1)")
    num_concurrent_requests: float = Field(..., ge=1, description="Number of concurrent requests")

class LatencyPrediction(BaseModel):
    """Latency prediction response"""
    predicted_latency_ms: float = Field(..., description="Predicted latency in milliseconds")
    confidence_interval_lower: float = Field(..., description="Lower bound of confidence interval")
    confidence_interval_upper: float = Field(..., description="Upper bound of confidence interval")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchLatencyRequest(BaseModel):
    """Batch prediction request"""
    features: List[LatencyFeatures]

class BatchLatencyResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[LatencyPrediction]

class OptimizationRequest(BaseModel):
    """Request for latency optimization"""
    current_features: LatencyFeatures
    optimizable_params: Optional[List[str]] = Field(
        default=None,
        description="Parameters that can be optimized (if None, optimizes all)"
    )
    constraints: Optional[dict] = Field(
        default=None,
        description="Constraints for optimization (e.g., {'network_bandwidth': {'min': 10, 'max': 50}})"
    )

class OptimizationSuggestion(BaseModel):
    """Optimization suggestion"""
    parameter: str
    current_value: float
    suggested_value: float
    expected_latency_improvement_ms: float
    improvement_percentage: float

class OptimizationResponse(BaseModel):
    """Optimization response"""
    current_latency_ms: float
    optimized_latency_ms: float
    latency_reduction_ms: float
    latency_reduction_percentage: float
    suggestions: List[OptimizationSuggestion]
    optimized_features: LatencyFeatures

# -------------------------
# API Endpoints
# -------------------------

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Latency Optimization API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict - Predict latency for given features",
            "predict_batch": "/predict/batch - Predict latency for multiple feature sets",
            "optimize": "/optimize - Get latency optimization suggestions"
        }
    }

@app.post("/predict", response_model=LatencyPrediction)
async def predict_latency(features: LatencyFeatures):
    """
    Predict latency based on system features
    
    - **network_bandwidth**: Network bandwidth in Mbps
    - **server_load**: Server load (0-1)
    - **request_size**: Request size in MB
    - **distance**: Distance to server in km
    - **cpu_usage**: CPU usage (0-1)
    - **memory_usage**: Memory usage (0-1)
    - **cache_hit_rate**: Cache hit rate (0-1)
    - **num_concurrent_requests**: Number of concurrent requests
    """
    try:
        # Prepare features
        feature_array = np.array([[
            features.network_bandwidth,
            features.server_load,
            features.request_size,
            features.distance,
            features.cpu_usage,
            features.memory_usage,
            features.cache_hit_rate,
            features.num_concurrent_requests
        ]])
        
        # Scale features
        feature_scaled = scaler.transform(feature_array)
        
        # Predict
        prediction = model.predict(feature_scaled)[0]
        
        # Get prediction intervals (using tree-based method)
        tree_predictions = [tree.predict(feature_scaled)[0] for tree in model.estimators_]
        std_dev = np.std(tree_predictions)
        
        # Confidence interval (95%)
        lower = max(0, prediction - 1.96 * std_dev)
        upper = prediction + 1.96 * std_dev
        
        return LatencyPrediction(
            predicted_latency_ms=round(prediction, 2),
            confidence_interval_lower=round(lower, 2),
            confidence_interval_upper=round(upper, 2),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchLatencyResponse)
async def predict_latency_batch(request: BatchLatencyRequest):
    """Predict latency for multiple feature sets at once"""
    try:
        predictions = []
        
        for features in request.features:
            feature_array = np.array([[
                features.network_bandwidth,
                features.server_load,
                features.request_size,
                features.distance,
                features.cpu_usage,
                features.memory_usage,
                features.cache_hit_rate,
                features.num_concurrent_requests
            ]])
            
            feature_scaled = scaler.transform(feature_array)
            prediction = model.predict(feature_scaled)[0]
            
            tree_predictions = [tree.predict(feature_scaled)[0] for tree in model.estimators_]
            std_dev = np.std(tree_predictions)
            
            lower = max(0, prediction - 1.96 * std_dev)
            upper = prediction + 1.96 * std_dev
            
            predictions.append(LatencyPrediction(
                predicted_latency_ms=round(prediction, 2),
                confidence_interval_lower=round(lower, 2),
                confidence_interval_upper=round(upper, 2),
                timestamp=datetime.now().isoformat()
            ))
        
        return BatchLatencyResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_latency(request: OptimizationRequest):
    """
    Get suggestions to optimize latency based on current system state
    
    This endpoint analyzes the current features and suggests parameter changes
    that would reduce latency.
    """
    try:
        # Get current latency
        current_features = request.current_features
        feature_array = np.array([[
            current_features.network_bandwidth,
            current_features.server_load,
            current_features.request_size,
            current_features.distance,
            current_features.cpu_usage,
            current_features.memory_usage,
            current_features.cache_hit_rate,
            current_features.num_concurrent_requests
        ]])
        
        feature_scaled = scaler.transform(feature_array)
        current_latency = model.predict(feature_scaled)[0]
        
        # Parameters to optimize
        optimizable = request.optimizable_params or [
            'network_bandwidth', 'server_load', 'request_size', 'distance',
            'cpu_usage', 'memory_usage', 'cache_hit_rate', 'num_concurrent_requests'
        ]
        
        suggestions = []
        optimized_values = {
            'network_bandwidth': current_features.network_bandwidth,
            'server_load': current_features.server_load,
            'request_size': current_features.request_size,
            'distance': current_features.distance,
            'cpu_usage': current_features.cpu_usage,
            'memory_usage': current_features.memory_usage,
            'cache_hit_rate': current_features.cache_hit_rate,
            'num_concurrent_requests': current_features.num_concurrent_requests
        }
        
        # Test optimizations for each parameter
        for param in optimizable:
            if param not in optimized_values:
                continue
            
            current_value = optimized_values[param]
            constraints = request.constraints.get(param, {}) if request.constraints else {}
            
            # Define optimization ranges and directions
            param_configs = {
                'network_bandwidth': {'min': constraints.get('min', current_value * 1.5), 'max': constraints.get('max', 100), 'better': 'higher'},
                'server_load': {'min': constraints.get('min', 0), 'max': constraints.get('max', current_value * 0.7), 'better': 'lower'},
                'request_size': {'min': constraints.get('min', 0.1), 'max': constraints.get('max', current_value * 0.8), 'better': 'lower'},
                'distance': {'min': constraints.get('min', 0), 'max': constraints.get('max', current_value * 0.8), 'better': 'lower'},
                'cpu_usage': {'min': constraints.get('min', 0), 'max': constraints.get('max', current_value * 0.7), 'better': 'lower'},
                'memory_usage': {'min': constraints.get('min', 0), 'max': constraints.get('max', current_value * 0.7), 'better': 'lower'},
                'cache_hit_rate': {'min': constraints.get('min', current_value * 1.2), 'max': constraints.get('max', 1.0), 'better': 'higher'},
                'num_concurrent_requests': {'min': constraints.get('min', 1), 'max': constraints.get('max', current_value * 0.7), 'better': 'lower'}
            }
            
            if param not in param_configs:
                continue
            
            config = param_configs[param]
            test_values = np.linspace(config['min'], config['max'], 10)
            
            best_value = current_value
            best_latency = current_latency
            
            for test_value in test_values:
                test_features = optimized_values.copy()
                test_features[param] = test_value
                
                test_array = np.array([[
                    test_features['network_bandwidth'],
                    test_features['server_load'],
                    test_features['request_size'],
                    test_features['distance'],
                    test_features['cpu_usage'],
                    test_features['memory_usage'],
                    test_features['cache_hit_rate'],
                    test_features['num_concurrent_requests']
                ]])
                
                test_scaled = scaler.transform(test_array)
                test_latency = model.predict(test_scaled)[0]
                
                if test_latency < best_latency:
                    best_latency = test_latency
                    best_value = test_value
            
            if best_latency < current_latency:
                improvement = current_latency - best_latency
                improvement_pct = (improvement / current_latency) * 100
                
                suggestions.append(OptimizationSuggestion(
                    parameter=param,
                    current_value=round(current_value, 3),
                    suggested_value=round(best_value, 3),
                    expected_latency_improvement_ms=round(improvement, 2),
                    improvement_percentage=round(improvement_pct, 2)
                ))
                
                optimized_values[param] = best_value
        
        # Calculate final optimized latency
        final_array = np.array([[
            optimized_values['network_bandwidth'],
            optimized_values['server_load'],
            optimized_values['request_size'],
            optimized_values['distance'],
            optimized_values['cpu_usage'],
            optimized_values['memory_usage'],
            optimized_values['cache_hit_rate'],
            optimized_values['num_concurrent_requests']
        ]])
        
        final_scaled = scaler.transform(final_array)
        optimized_latency = model.predict(final_scaled)[0]
        
        latency_reduction = current_latency - optimized_latency
        reduction_pct = (latency_reduction / current_latency) * 100 if current_latency > 0 else 0
        
        return OptimizationResponse(
            current_latency_ms=round(current_latency, 2),
            optimized_latency_ms=round(optimized_latency, 2),
            latency_reduction_ms=round(latency_reduction, 2),
            latency_reduction_percentage=round(reduction_pct, 2),
            suggestions=suggestions,
            optimized_features=LatencyFeatures(**optimized_values)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

