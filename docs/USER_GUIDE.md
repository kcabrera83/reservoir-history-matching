# User Guide - Reservoir History Matching

## Overview
ML-based reservoir production history matching and forecasting system. Uses GradientBoosting to match reservoir properties to production rates and forecast future production from historical data.

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/kcabrera83/reservoir-history-matching.git
cd reservoir-history-matching
pip install -r requirements.txt
```

### Training Models
```bash
python train.py
```
Generates synthetic reservoir data (50 wells, 100 timesteps), trains history matcher and production forecaster.

### Starting the Server
```bash
python app.py
```
Open http://localhost:5006 in your browser.

## Dashboard Features
- **History Matching**: Predict production rates from reservoir properties
- **Production Forecasting**: Forecast future production from historical data
- **Model Metrics**: View training and testing performance
- **Visualization**: Production curves and forecast plots

## Reservoir Properties

### Input Features (for /api/predict)
| Index | Feature | Unit | Description |
|-------|---------|------|-------------|
| 0 | porosity | fraction | Rock porosity (0-1) |
| 1 | permeability_md | mD | Rock permeability |
| 2 | depth_m | m | Reservoir depth |
| 3 | water_saturation | fraction | Water saturation (0-1) |
| 4 | net_pay_m | m | Net productive thickness |
| 5 | pressure_psi | psi | Reservoir pressure |
| 6 | temperature_c | C | Reservoir temperature |
| 7 | oil_viscosity_cp | cP | Oil viscosity |

### Output Production Rates
| Field | Unit | Description |
|-------|------|-------------|
| oil_rate_bopd | bopd | Oil production rate (barrels oil per day) |
| water_rate_bwpd | bwpd | Water production rate (barrels water per day) |
| gas_rate_mscfd | mscfd | Gas production rate (thousand standard cubic feet per day) |

## API Usage

### Using curl
```bash
# Predict production from reservoir properties
curl -X POST http://localhost:5006/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.20, 500, 3500, 0.30, 100, 3000, 100, 0]}'

# Forecast future production
curl -X POST http://localhost:5006/api/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "recent_data": [
      {"oil_rate_bopd": 350, "water_rate_bwpd": 120, "gas_rate_mscfd": 850},
      {"oil_rate_bopd": 345, "water_rate_bwpd": 125, "gas_rate_mscfd": 840}
    ],
    "n_steps": 10
  }'

# Get model information
curl http://localhost:5006/api/models

# Health check
curl http://localhost:5006/api/health
```

### Using Python
```python
import requests

# Predict production rates
response = requests.post("http://localhost:5006/api/predict", json={
    "features": [0.20, 500, 3500, 0.30, 100, 3000, 100, 0]
})
prediction = response.json()["prediction"]
print(f"Oil: {prediction['oil_rate_bopd']} bopd")
print(f"Water: {prediction['water_rate_bwpd']} bwpd")
print(f"Gas: {prediction['gas_rate_mscfd']} mscfd")

# Forecast future production
historical_data = [
    {"oil_rate_bopd": 350, "water_rate_bwpd": 120, "gas_rate_mscfd": 850},
    {"oil_rate_bopd": 345, "water_rate_bwpd": 125, "gas_rate_mscfd": 840},
]
response = requests.post("http://localhost:5006/api/forecast", json={
    "recent_data": historical_data,
    "n_steps": 10
})
forecast = response.json()["forecast"]
for step in forecast:
    print(f"Oil: {step['oil_rate_bopd']:.1f} bopd")
```

## Models
- **History Matcher**: GradientBoosting regressor mapping reservoir properties to production rates
- **Production Forecaster**: GradientBoosting-based forecaster using lookback windows (alternative to LSTM)
