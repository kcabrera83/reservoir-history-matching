# Reservoir History Matching

ML-based reservoir production history matching and forecasting system using Gaussian Processes and PyTorch.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| GP Modeling | **GPyTorch** - Gaussian Process regression |
| Deep Learning | **PyTorch** - Neural network models |
| Data Processing | pandas, numpy, joblib |
| Web Server | **FastAPI** + uvicorn |
| Monitoring | prometheus-fastapi-instrumentator |
| Validation | pydantic v2 |
| Visualization | matplotlib, seaborn |

### Key Libraries
- GPyTorch - Scalable Gaussian Process modeling
- PyTorch - Deep learning framework
- FastAPI - Modern async web framework
- pandas / numpy - Data processing

## Overview

This project provides machine learning models for:
- **History Matching**: Gaussian Process regression for predicting oil, water, and gas production rates from reservoir properties
- **Production Forecasting**: PyTorch neural networks for forecasting future production using historical data

## Project Structure

```
reservoir-history-matching/
├── reservoir_history_matching/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── history_matcher.py
│   │   └── production_forecaster.py
│   └── utils/
│       ├── __init__.py
│       └── preprocessor.py
├── outputs/models/
├── templates/
│   └── index.html
├── .github/workflows/
│   └── ci.yml
├── train.py
├── app.py
├── test_api.py
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Running the API

```bash
python app.py
```

The API will start on port 5006.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/health` | Health check |
| GET | `/api/models` | Model information and metrics |
| POST | `/api/predict` | Predict production rates from reservoir properties |
| POST | `/api/forecast` | Forecast future production from historical data |

### Predict Example

```bash
curl -X POST http://localhost:5006/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.20, 500, 3500, 0.30, 100, 3000, 100, 0]}'
```

### Forecast Example

```bash
curl -X POST http://localhost:5006/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"recent_data": [...], "n_steps": 10}'
```

## Running Tests

```bash
python test_api.py
```

## Dashboard

Access the dashboard at `http://localhost:5006` after starting the server.

---

Elaborado por Ing. Kelvin Cabrera
