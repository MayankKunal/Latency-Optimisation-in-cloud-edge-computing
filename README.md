# Latency Optimization Model - Training and Testing Guide

This project provides a complete pipeline for training and testing a latency optimization model using FastAPI.

## Project Structure

```
.
├── model_server.py      # FastAPI server with model endpoints
├── generate_data.py     # Script to generate training/testing datasets
├── train_model.py       # Script to train the model
├── test_model.py        # Script to test the model on new data
├── requirements.txt     # Python dependencies
├── data/                # Generated datasets (created automatically)
│   ├── train.csv
│   └── test.csv
└── model/               # Trained models (created automatically)
    ├── latency_model.joblib
    └── latency_scaler.joblib
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using pip3:
```bash
python3 -m pip install -r requirements.txt
```

## Quick Start

### Step 1: Generate Training and Testing Data

Generate synthetic datasets for training and testing:

```bash
python3 generate_data.py
```

This will create:
- `data/train.csv` - Training dataset (4000 samples by default)
- `data/test.csv` - Testing dataset (1000 samples by default)

**Options:**
- Modify `n_samples` in `generate_data.py` to change the number of samples
- The data is split 80/20 for train/test

### Step 2: Train the Model

Train the latency prediction model:

```bash
python3 train_model.py
```

Or with options:

```bash
python3 train_model.py --n-estimators 200 --max-depth 20
```

**Options:**
- `--data-dir`: Directory containing datasets (default: `data`)
- `--model-dir`: Directory to save model (default: `model`)
- `--n-estimators`: Number of trees (default: 100)
- `--max-depth`: Maximum tree depth (default: 15)
- `--generate-data`: Generate data if datasets don't exist
- `--random-seed`: Random seed (default: 42)

This will:
- Load training data
- Train a Random Forest model
- Evaluate on test data
- Save the model to `model/` directory

### Step 3: Test the Model

Test the trained model on new data:

```bash
python3 test_model.py
```

**Options:**
- `--model-dir`: Directory containing model (default: `model`)
- `--test-data`: Path to custom test CSV file
- `--data-dir`: Use test.csv from data directory
- `--generate-test`: Generate new test data
- `--n-samples`: Number of test samples to generate
- `--save-predictions`: Path to save predictions CSV

**Examples:**

Test on existing test dataset:
```bash
python3 test_model.py
```

Test on custom CSV file:
```bash
python3 test_model.py --test-data my_test_data.csv
```

Generate and test on new data:
```bash
python3 test_model.py --generate-test --n-samples 500
```

### Step 4: Run the FastAPI Server

Start the API server:

```bash
python3 -m uvicorn model_server:app --reload
```

The server will be available at:
- API: `http://127.0.0.1:8000`
- Interactive Docs: `http://127.0.0.1:8000/docs`

## Dataset Format

### Input Features

The model expects the following features:

| Feature | Description | Range |
|---------|-------------|-------|
| `network_bandwidth` | Network bandwidth in Mbps | 1-100 |
| `server_load` | Server load (0-1) | 0-1 |
| `request_size` | Request size in MB | 0.1-10 |
| `distance` | Distance to server in km | 1-1000 |
| `cpu_usage` | CPU usage (0-1) | 0-1 |
| `memory_usage` | Memory usage (0-1) | 0-1 |
| `cache_hit_rate` | Cache hit rate (0-1) | 0-1 |
| `num_concurrent_requests` | Number of concurrent requests | 1-100 |

### Target Variable

- `latency_ms`: Latency in milliseconds (predicted value)

### Example CSV Format

```csv
network_bandwidth,server_load,request_size,distance,cpu_usage,memory_usage,cache_hit_rate,num_concurrent_requests,latency_ms
50.5,0.3,2.1,100,0.4,0.5,0.7,10,85.23
75.2,0.2,1.5,50,0.3,0.4,0.8,5,62.15
```

## API Endpoints

Once the server is running, you can use these endpoints:

### 1. Predict Latency
```bash
POST /predict
```

Request body:
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

### 2. Batch Prediction
```bash
POST /predict/batch
```

### 3. Optimize Latency
```bash
POST /optimize
```

Get suggestions to reduce latency based on current system state.

## Complete Workflow Example

```bash
# 1. Generate data
python3 generate_data.py

# 2. Train model
python3 train_model.py --n-estimators 150 --max-depth 20

# 3. Test model
python3 test_model.py

# 4. Start server
python3 -m uvicorn model_server:app --reload
```

## Custom Data

### Using Your Own Training Data

1. Create a CSV file with the required features and `latency_ms` column
2. Split into train/test manually or use the scripts
3. Update `train_model.py` to load your data

### Using Your Own Test Data

1. Create a CSV file with the required features
2. Optionally include `latency_ms` for evaluation
3. Test using:
```bash
python3 test_model.py --test-data your_data.csv
```

## Model Evaluation Metrics

The test script provides:
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error (in ms)
- **MAE**: Mean Absolute Error (in ms)
- **MAPE**: Mean Absolute Percentage Error
- **Error Distribution**: Percentiles of prediction errors

## Troubleshooting

### Model not found
- Make sure you've run `train_model.py` first
- Check that `model/latency_model.joblib` exists

### Data not found
- Run `generate_data.py` first
- Or use `--generate-data` flag with `train_model.py`

### Import errors
- Install all dependencies: `pip install -r requirements.txt`
- Make sure you're using Python 3.7+

## Advanced Usage

### Hyperparameter Tuning

Edit `train_model.py` to experiment with:
- Different `n_estimators` values
- Different `max_depth` values
- Different random forest parameters

### Custom Data Generation

Modify `generate_data.py` to:
- Change the latency formula
- Add new features
- Adjust noise levels
- Use real data distributions

## License

This project is provided as-is for educational and research purposes.

