# Latency Optimization Model - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start Guide](#quick-start-guide)
5. [Detailed Usage](#detailed-usage)
6. [API Documentation](#api-documentation)
7. [Model Architecture](#model-architecture)
8. [Data Format](#data-format)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)
11. [Project Structure](#project-structure)

---

## Project Overview

This project implements a **Latency Optimization Model** using machine learning to predict and optimize latency in distributed computing systems (cloud-edge computing). The system uses a Random Forest Regressor to predict latency based on various system parameters and provides optimization suggestions to minimize latency.

### Key Capabilities

- **Latency Prediction**: Predict latency based on system features
- **Batch Processing**: Process multiple predictions at once
- **Optimization**: Get suggestions to reduce latency
- **RESTful API**: Easy-to-use FastAPI endpoints
- **Model Training**: Train custom models on your data
- **Comprehensive Testing**: Evaluate model performance

---

## Features

### 1. Machine Learning Model
- **Algorithm**: Random Forest Regressor
- **Features**: 8 input features affecting latency
- **Performance**: High accuracy with R² > 0.90
- **Scalability**: Handles large datasets efficiently

### 2. FastAPI Server
- **Interactive Documentation**: Swagger UI at `/docs`
- **Multiple Endpoints**: Predict, batch predict, and optimize
- **Type Safety**: Pydantic models for request/response validation
- **Error Handling**: Comprehensive error messages

### 3. Training Pipeline
- **Data Generation**: Synthetic data generation for testing
- **Model Training**: Automated training with evaluation
- **Model Persistence**: Save and load trained models
- **Hyperparameter Tuning**: Configurable model parameters

### 4. Testing & Evaluation
- **Comprehensive Metrics**: RMSE, MAE, R², MAPE
- **Error Analysis**: Percentile-based error distribution
- **Custom Data Testing**: Test on your own datasets
- **Prediction Export**: Save predictions to CSV

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project

```bash
# If using git
git clone https://github.com/MayankKunal/Latency-Optimisation-in-cloud-edge-computing.git
cd Latency-Optimisation-in-cloud-edge-computing

# Or download and extract the project folder
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or using pip3
python3 -m pip install -r requirements.txt
```

**Required Packages:**
- `fastapi` - Web framework for API
- `uvicorn` - ASGI server
- `scikit-learn` - Machine learning library
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `joblib` - Model serialization
- `pydantic` - Data validation

### Step 3: Verify Installation

```bash
python3 -c "import fastapi, sklearn, numpy, pandas; print('All packages installed successfully!')"
```

---

## Quick Start Guide

### Complete Workflow (5 Steps)

#### Step 1: Generate Training Data

```bash
python3 generate_data.py
```

**What it does:**
- Generates 5000 synthetic samples
- Splits into training (4000) and testing (1000) sets
- Saves to `data/train.csv` and `data/test.csv`

**Output:**
```
✓ Training data saved to: data/train.csv
✓ Testing data saved to: data/test.csv
```

#### Step 2: Train the Model

```bash
python3 train_model.py
```

**What it does:**
- Loads training data
- Trains Random Forest model
- Evaluates on test data
- Saves model to `model/` directory

**Output:**
```
Training Latency Optimization Model
- RMSE: 16.41 ms
- MAE: 12.52 ms
- R² Score: 0.9010
✓ Model saved successfully!
```

#### Step 3: Test the Model

```bash
python3 test_model.py
```

**What it does:**
- Loads trained model
- Tests on test dataset
- Provides evaluation metrics
- Saves predictions to `predictions.csv`

#### Step 4: Start the API Server

```bash
python3 -m uvicorn model_server:app --reload
```

**Server will start at:**
- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`

#### Step 5: Test the API

Open your browser and go to:
```
http://127.0.0.1:8000/docs
```

Use the interactive Swagger UI to test endpoints!

---

## Detailed Usage

### 1. Data Generation

#### Basic Usage
```bash
python3 generate_data.py
```

#### Customize Data Generation

Edit `generate_data.py` to modify:
- Number of samples: Change `n_samples` parameter
- Data distribution: Modify feature ranges
- Latency formula: Adjust the latency calculation
- Noise level: Change random noise amount

**Example: Generate more data**
```python
# In generate_data.py, change:
n_samples = 10000  # Generate 10,000 samples instead of 5,000
```

### 2. Model Training

#### Basic Training
```bash
python3 train_model.py
```

#### Advanced Training Options

```bash
# Train with custom parameters
python3 train_model.py --n-estimators 200 --max-depth 20

# Use custom data directory
python3 train_model.py --data-dir my_data --model-dir my_models

# Generate data if missing
python3 train_model.py --generate-data
```

**Available Options:**
- `--data-dir`: Directory containing datasets (default: `data`)
- `--model-dir`: Directory to save model (default: `model`)
- `--n-estimators`: Number of trees (default: 100)
- `--max-depth`: Maximum tree depth (default: 15)
- `--generate-data`: Generate data if missing
- `--random-seed`: Random seed for reproducibility

**Hyperparameter Tuning Tips:**
- **More trees** (`--n-estimators 200`): Better accuracy, slower training
- **Deeper trees** (`--max-depth 20`): More complex patterns, risk of overfitting
- **Fewer trees** (`--n-estimators 50`): Faster training, may reduce accuracy

### 3. Model Testing

#### Test on Existing Data
```bash
python3 test_model.py
```

#### Test on Custom Data
```bash
# Test on your own CSV file
python3 test_model.py --test-data my_data.csv

# Generate new test data
python3 test_model.py --generate-test --n-samples 500

# Save predictions with custom name
python3 test_model.py --save-predictions my_predictions.csv
```

**Test Data Requirements:**
- CSV file with required feature columns
- Optional: Include `latency_ms` column for evaluation
- See [Data Format](#data-format) section for details

### 4. Running the API Server

#### Basic Server
```bash
python3 -m uvicorn model_server:app --reload
```

#### Production Server
```bash
# Run without auto-reload (for production)
python3 -m uvicorn model_server:app --host 0.0.0.0 --port 8000

# Run with multiple workers
python3 -m uvicorn model_server:app --workers 4
```

**Server Options:**
- `--reload`: Auto-reload on code changes (development)
- `--host`: Host address (default: 127.0.0.1)
- `--port`: Port number (default: 8000)
- `--workers`: Number of worker processes

---

## API Documentation

### Base URL
```
http://127.0.0.1:8000
```

### Interactive Documentation
Visit `http://127.0.0.1:8000/docs` for Swagger UI with interactive testing.

### Endpoints

#### 1. Root Endpoint
**GET** `/`

Get API information and available endpoints.

**Response:**
```json
{
  "message": "Latency Optimization API",
  "version": "1.0.0",
  "endpoints": {
    "predict": "/predict - Predict latency for given features",
    "predict_batch": "/predict/batch - Predict latency for multiple feature sets",
    "optimize": "/optimize - Get latency optimization suggestions"
  }
}
```

#### 2. Predict Latency
**POST** `/predict`

Predict latency for a single set of features.

**Request Body:**
```json
{
  "network_bandwidth": 50.0,
  "server_load": 0.3,
  "request_size": 2.0,
  "distance": 100.0,
  "cpu_usage": 0.4,
  "memory_usage": 0.5,
  "cache_hit_rate": 0.7,
  "num_concurrent_requests": 10
}
```

**Response:**
```json
{
  "predicted_latency_ms": 125.45,
  "confidence_interval_lower": 110.23,
  "confidence_interval_upper": 140.67,
  "timestamp": "2024-01-15T10:30:00"
}
```

**cURL Example:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "network_bandwidth": 50.0,
    "server_load": 0.3,
    "request_size": 2.0,
    "distance": 100.0,
    "cpu_usage": 0.4,
    "memory_usage": 0.5,
    "cache_hit_rate": 0.7,
    "num_concurrent_requests": 10
  }'
```

#### 3. Batch Prediction
**POST** `/predict/batch`

Predict latency for multiple feature sets at once.

**Request Body:**
```json
{
  "features": [
    {
      "network_bandwidth": 50.0,
      "server_load": 0.3,
      "request_size": 2.0,
      "distance": 100.0,
      "cpu_usage": 0.4,
      "memory_usage": 0.5,
      "cache_hit_rate": 0.7,
      "num_concurrent_requests": 10
    },
    {
      "network_bandwidth": 75.0,
      "server_load": 0.2,
      "request_size": 1.5,
      "distance": 50.0,
      "cpu_usage": 0.3,
      "memory_usage": 0.4,
      "cache_hit_rate": 0.8,
      "num_concurrent_requests": 5
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "predicted_latency_ms": 125.45,
      "confidence_interval_lower": 110.23,
      "confidence_interval_upper": 140.67,
      "timestamp": "2024-01-15T10:30:00"
    },
    {
      "predicted_latency_ms": 95.32,
      "confidence_interval_lower": 85.10,
      "confidence_interval_upper": 105.54,
      "timestamp": "2024-01-15T10:30:00"
    }
  ]
}
```

#### 4. Optimize Latency
**POST** `/optimize`

Get suggestions to optimize latency based on current system state.

**Request Body:**
```json
{
  "current_features": {
    "network_bandwidth": 30.0,
    "server_load": 0.8,
    "request_size": 5.0,
    "distance": 500.0,
    "cpu_usage": 0.9,
    "memory_usage": 0.8,
    "cache_hit_rate": 0.3,
    "num_concurrent_requests": 50
  },
  "optimizable_params": ["network_bandwidth", "server_load", "cache_hit_rate"],
  "constraints": {
    "network_bandwidth": {"min": 20, "max": 100},
    "server_load": {"min": 0, "max": 0.5}
  }
}
```

**Response:**
```json
{
  "current_latency_ms": 350.25,
  "optimized_latency_ms": 180.50,
  "latency_reduction_ms": 169.75,
  "latency_reduction_percentage": 48.45,
  "suggestions": [
    {
      "parameter": "network_bandwidth",
      "current_value": 30.0,
      "suggested_value": 80.0,
      "expected_latency_improvement_ms": 120.5,
      "improvement_percentage": 34.4
    },
    {
      "parameter": "server_load",
      "current_value": 0.8,
      "suggested_value": 0.4,
      "expected_latency_improvement_ms": 40.2,
      "improvement_percentage": 11.5
    }
  ],
  "optimized_features": {
    "network_bandwidth": 80.0,
    "server_load": 0.4,
    "request_size": 5.0,
    "distance": 500.0,
    "cpu_usage": 0.9,
    "memory_usage": 0.8,
    "cache_hit_rate": 0.7,
    "num_concurrent_requests": 50
  }
}
```

#### 5. Health Check
**GET** `/health`

Check if the API server is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00"
}
```

---

## Model Architecture

### Algorithm: Random Forest Regressor

**Why Random Forest?**
- Handles non-linear relationships
- Robust to outliers
- Provides feature importance
- Good generalization
- Fast prediction

### Input Features (8 features)

| Feature | Description | Range | Impact on Latency |
|---------|-------------|-------|-------------------|
| `network_bandwidth` | Network bandwidth in Mbps | 1-100 | Higher = Lower latency |
| `server_load` | Server load (0-1) | 0-1 | Higher = Higher latency |
| `request_size` | Request size in MB | 0.1-10 | Larger = Higher latency |
| `distance` | Distance to server in km | 1-1000 | Farther = Higher latency |
| `cpu_usage` | CPU usage (0-1) | 0-1 | Higher = Higher latency |
| `memory_usage` | Memory usage (0-1) | 0-1 | Higher = Higher latency |
| `cache_hit_rate` | Cache hit rate (0-1) | 0-1 | Higher = Lower latency |
| `num_concurrent_requests` | Concurrent requests | 1-100 | More = Higher latency |

### Model Parameters (Default)

- **Number of Trees**: 100
- **Max Depth**: 15
- **Random State**: 42 (for reproducibility)
- **Scaling**: StandardScaler (normalizes features)

### Model Performance

Typical performance metrics:
- **R² Score**: > 0.90 (90% variance explained)
- **RMSE**: ~15-20 ms
- **MAE**: ~12-15 ms
- **MAPE**: ~5-7%

---

## Data Format

### Training/Testing Data Format

CSV file with the following columns:

```csv
network_bandwidth,server_load,request_size,distance,cpu_usage,memory_usage,cache_hit_rate,num_concurrent_requests,latency_ms
50.5,0.3,2.1,100,0.4,0.5,0.7,10,125.23
75.2,0.2,1.5,50,0.3,0.4,0.8,5,95.15
30.0,0.8,5.0,500,0.9,0.8,0.3,50,350.45
```

### Column Descriptions

1. **network_bandwidth** (float): Network bandwidth in Mbps (1-100)
2. **server_load** (float): Server load ratio (0-1)
3. **request_size** (float): Request size in MB (0.1-10)
4. **distance** (float): Distance to server in km (1-1000)
5. **cpu_usage** (float): CPU usage ratio (0-1)
6. **memory_usage** (float): Memory usage ratio (0-1)
7. **cache_hit_rate** (float): Cache hit rate (0-1)
8. **num_concurrent_requests** (float): Number of concurrent requests (1-100)
9. **latency_ms** (float): Target latency in milliseconds (for training)

### Using Your Own Data

1. **Prepare CSV file** with required columns
2. **Split into train/test** (80/20 recommended)
3. **Train model**: `python3 train_model.py --data-dir your_data_dir`
4. **Test model**: `python3 test_model.py --test-data your_test.csv`

---

## Examples

### Example 1: Complete Workflow

```bash
# 1. Generate data
python3 generate_data.py

# 2. Train model
python3 train_model.py --n-estimators 150

# 3. Test model
python3 test_model.py

# 4. Start server
python3 -m uvicorn model_server:app --reload

# 5. Test API (in another terminal)
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "network_bandwidth": 50.0,
    "server_load": 0.3,
    "request_size": 2.0,
    "distance": 100.0,
    "cpu_usage": 0.4,
    "memory_usage": 0.5,
    "cache_hit_rate": 0.7,
    "num_concurrent_requests": 10
  }'
```

### Example 2: Python Client

```python
import requests
import json

# API endpoint
url = "http://127.0.0.1:8000/predict"

# Request data
data = {
    "network_bandwidth": 50.0,
    "server_load": 0.3,
    "request_size": 2.0,
    "distance": 100.0,
    "cpu_usage": 0.4,
    "memory_usage": 0.5,
    "cache_hit_rate": 0.7,
    "num_concurrent_requests": 10
}

# Make request
response = requests.post(url, json=data)
result = response.json()

print(f"Predicted Latency: {result['predicted_latency_ms']} ms")
print(f"Confidence Interval: {result['confidence_interval_lower']} - {result['confidence_interval_upper']} ms")
```

### Example 3: Optimization Request

```python
import requests

url = "http://127.0.0.1:8000/optimize"

data = {
    "current_features": {
        "network_bandwidth": 30.0,
        "server_load": 0.8,
        "request_size": 5.0,
        "distance": 500.0,
        "cpu_usage": 0.9,
        "memory_usage": 0.8,
        "cache_hit_rate": 0.3,
        "num_concurrent_requests": 50
    }
}

response = requests.post(url, json=data)
result = response.json()

print(f"Current Latency: {result['current_latency_ms']} ms")
print(f"Optimized Latency: {result['optimized_latency_ms']} ms")
print(f"Reduction: {result['latency_reduction_percentage']:.2f}%")

for suggestion in result['suggestions']:
    print(f"\n{suggestion['parameter']}:")
    print(f"  Current: {suggestion['current_value']}")
    print(f"  Suggested: {suggestion['suggested_value']}")
    print(f"  Improvement: {suggestion['expected_latency_improvement_ms']:.2f} ms")
```

### Example 4: Batch Processing

```python
import requests

url = "http://127.0.0.1:8000/predict/batch"

data = {
    "features": [
        {
            "network_bandwidth": 50.0,
            "server_load": 0.3,
            "request_size": 2.0,
            "distance": 100.0,
            "cpu_usage": 0.4,
            "memory_usage": 0.5,
            "cache_hit_rate": 0.7,
            "num_concurrent_requests": 10
        },
        {
            "network_bandwidth": 75.0,
            "server_load": 0.2,
            "request_size": 1.5,
            "distance": 50.0,
            "cpu_usage": 0.3,
            "memory_usage": 0.4,
            "cache_hit_rate": 0.8,
            "num_concurrent_requests": 5
        }
    ]
}

response = requests.post(url, json=data)
results = response.json()

for i, pred in enumerate(results['predictions']):
    print(f"Request {i+1}: {pred['predicted_latency_ms']} ms")
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Module Not Found Error

**Error:**
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution:**
```bash
pip install -r requirements.txt
# or
python3 -m pip install -r requirements.txt
```

#### 2. Model Not Found

**Error:**
```
FileNotFoundError: Model files not found
```

**Solution:**
```bash
# Train the model first
python3 train_model.py
```

#### 3. Data Not Found

**Error:**
```
FileNotFoundError: Dataset files not found
```

**Solution:**
```bash
# Generate data first
python3 generate_data.py

# Or use --generate-data flag
python3 train_model.py --generate-data
```

#### 4. Port Already in Use

**Error:**
```
Address already in use
```

**Solution:**
```bash
# Use a different port
python3 -m uvicorn model_server:app --port 8001

# Or kill the process using port 8000
lsof -ti:8000 | xargs kill
```

#### 5. Permission Denied

**Error:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Install packages for user
python3 -m pip install --user -r requirements.txt
```

#### 6. Low Model Accuracy

**Symptoms:**
- Low R² score (< 0.7)
- High RMSE

**Solutions:**
- Increase training data: Generate more samples
- Tune hyperparameters: Try more trees or deeper trees
- Check data quality: Ensure features are correctly formatted
- Feature engineering: Add more relevant features

#### 7. API Not Responding

**Check:**
1. Server is running: `curl http://127.0.0.1:8000/health`
2. Model is loaded: Check server logs
3. Correct endpoint: Verify URL and method
4. Request format: Check JSON structure

---

## Project Structure

```
Latency-Optimisation-in-cloud-edge-computing/
│
├── model_server.py          # FastAPI server with endpoints
├── generate_data.py         # Data generation script
├── train_model.py          # Model training script
├── test_model.py           # Model testing script
├── requirements.txt        # Python dependencies
├── README.md              # Quick start guide
├── DOCUMENTATION.md       # This comprehensive documentation
│
├── data/                  # Generated datasets
│   ├── train.csv         # Training data
│   └── test.csv          # Testing data
│
├── model/                 # Trained models
│   ├── latency_model.joblib    # Trained model
│   └── latency_scaler.joblib   # Feature scaler
│
└── predictions.csv        # Test predictions (generated)
```

### File Descriptions

- **model_server.py**: Main FastAPI application with all endpoints
- **generate_data.py**: Generates synthetic training/testing data
- **train_model.py**: Trains the Random Forest model
- **test_model.py**: Tests model and provides evaluation metrics
- **requirements.txt**: List of required Python packages
- **README.md**: Quick reference guide
- **DOCUMENTATION.md**: Complete documentation (this file)

---

## Additional Resources

### Performance Tips

1. **For Production:**
   - Use `--workers` flag for multiple processes
   - Disable `--reload` in production
   - Use a reverse proxy (nginx) for better performance

2. **For Better Accuracy:**
   - Train on more data (10,000+ samples)
   - Tune hyperparameters
   - Use real-world data instead of synthetic

3. **For Faster Training:**
   - Reduce number of trees
   - Reduce max depth
   - Use fewer samples

### Integration Examples

#### Docker (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Systemd Service (Linux)

Create `/etc/systemd/system/latency-api.service`:
```ini
[Unit]
Description=Latency Optimization API
After=network.target

[Service]
User=your-user
WorkingDirectory=/path/to/project
ExecStart=/usr/bin/python3 -m uvicorn model_server:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## Support and Contact

For issues, questions, or contributions:
- **GitHub Repository**: https://github.com/MayankKunal/Latency-Optimisation-in-cloud-edge-computing
- **Documentation**: See README.md and this file
- **API Docs**: http://127.0.0.1:8000/docs (when server is running)

---

## License

This project is provided as-is for educational and research purposes.

---

## Version History

- **v1.0.0** (Current)
  - Initial release
  - FastAPI server with prediction and optimization endpoints
  - Complete training and testing pipeline
  - Comprehensive documentation

---

**Last Updated**: January 2024

**Author**: Mayank Kumar

**Project**: Latency Optimization in Cloud-Edge Computing

