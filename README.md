# Boston House Price Prediction API

A production-ready ML API that predicts Boston house prices using a Random Forest model. Includes CI/CD pipeline, Docker deployment, and monitoring with Prometheus & Grafana.

## Features

- **REST API** — Flask-based prediction endpoint
- **Random Forest Model** — Trained on Boston Housing dataset with versioned model artifacts (`.joblib`)
- **Docker** — Containerized deployment with Docker Compose
- **Monitoring** — Prometheus metrics and Grafana dashboards
- **CI/CD** — GitHub Actions pipeline: train → test → build → push to Docker Hub

## Project Structure

```
boston-house-project/
├── app.py              # Flask API with /predict and /metrics endpoints
├── train.py            # Model training with versioning (model_v1.joblib, model_v2.joblib, ...)
├── test_model.py       # Model performance tests (MSE, MAE assertions)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build for the API
├── docker-compose.yml  # API + Prometheus + Grafana stack
├── prometheus.yml      # Prometheus scrape config
├── data/
│   ├── train/          # X_train.csv, y_train.csv
│   ├── test/           # X_test.csv, y_test.csv
│   └── dataset/        # housing.csv (raw data)
├── models/             # Trained models + current_model.txt (active model pointer)
└── .github/workflows/
    └── ml_pipeline.yml # CI/CD pipeline
```

## Prerequisites

- Python 3.10+
- Docker & Docker Compose (for containerized setup)

## Quick Start

### Option 1: Local Development

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model (required before running the API)
python train.py

# Run API
python app.py
# API available at http://localhost:5000
```

### Option 2: Docker Compose (Full Stack)

```bash
# Build and run all services
docker compose up --build

# Services:
# - Housing API:  http://localhost:5000
# - Prometheus:   http://localhost:9090
# - Grafana:      http://localhost:3000 (login: admin/admin)
```

> **Note:** Ensure a trained model exists in `models/` (e.g. `model_v1.joblib`) before running. Run `python train.py` locally first, or the CI pipeline will train on push.

## API Endpoints

| Endpoint   | Method | Description                              |
|------------|--------|------------------------------------------|
| `/`        | GET    | Health check — returns "Boston Housing API Running" |
| `/predict` | POST   | Get house price prediction               |
| `/metrics` | GET    | Prometheus metrics (for scraping)       |

### Example: Get a Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.00632, 18, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98]}'
```

Response:
```json
{"prediction": 24.5}
```

The model expects **13 features** (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT from the Boston Housing dataset).

## Model Training

```bash
python train.py
```

- Trains a `RandomForestRegressor` (200 trees, `random_state=42`) on `data/train/`
- Saves models as `model_v1.joblib`, `model_v2.joblib`, etc. (auto-incremented)
- Updates `models/current_model.txt` to point at the new model

## Testing

```bash
# Run model performance tests
python test_model.py
```

Validates MSE < 18 and MAE < 3.0 on the test set.

## Monitoring

### Prometheus Metrics

The API exposes these metrics at `/metrics`:

| Metric                     | Type    | Description                      |
|----------------------------|---------|----------------------------------|
| `api_requests_total`       | Counter | Total requests by method, endpoint, status |
| `api_request_latency_seconds` | Histogram | Request latency             |
| `model_mse`                | Gauge   | Mean Squared Error on test set   |
| `model_mae`                | Gauge   | Mean Absolute Error on test set  |
| `model_version`            | Gauge   | Current model version number     |

### Grafana Setup

1. Open http://localhost:3000 and log in (`admin` / `admin`)
2. Add data source: **Prometheus** → URL: `http://prometheus:9090` → Save & Test
3. Create dashboards using metrics like `api_requests_total`, `rate(api_request_latency_seconds_sum[5m])`, etc.

<img width="1436" height="704" alt="Screenshot 2026-02-18 at 11 24 20 AM" src="https://github.com/user-attachments/assets/445f87c7-414f-46d5-b82c-3c67fc4cbdf9" />

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ml_pipeline.yml`) runs on push/PR to `main`:

1. **train-and-test** — Installs deps, trains model, runs `test_model.py`
2. **build-and-publish** — Builds Docker image and pushes to Docker Hub

### Required Secrets

Configure these in your GitHub repository (Settings → Secrets and variables → Actions):

- `DOCKER_USERNAME` — Docker Hub username
- `DOCKER_PASSWORD` — Docker Hub password or access token

Image will be published as: `$DOCKER_USERNAME/housing-api:latest`

