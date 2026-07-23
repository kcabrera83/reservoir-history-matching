# API Documentation - Reservoir History Matching

## Base URL
```
http://localhost:5006
```

## Endpoints

### GET /
Main dashboard with interactive web interface.

**Response:** HTML page with reservoir analysis panels.

---

### GET /api/health
Service health check with model status.

**Response (200):**
```json
{
  "status": "healthy",
  "models_loaded": {
    "history_matcher": true,
    "production_forecaster": true
  }
}
```

---

### GET /api/models
Model information and metrics.

**Response (200):**
```json
{
  "history_matcher": {
    "type": "gradient_boosting",
    "metrics": {
      "train": {"r2": 0.98, "mae": 15.5, "rmse": 22.3},
      "test": {"r2": 0.95, "mae": 20.1, "rmse": 28.7}
    },
    "targets": ["oil_rate_bopd", "water_rate_bwpd", "gas_rate_mscfd"]
  },
  "production_forecaster": {
    "lookback": 10,
    "metrics": {
      "train": {"r2": 0.97, "mae": 12.3, "rmse": 18.5},
      "test": {"r2": 0.94, "mae": 18.7, "rmse": 25.1}
    },
    "targets": ["oil_rate_bopd", "water_rate_bwpd", "gas_rate_mscfd"]
  }
}
```

---

### POST /api/predict
Predict reservoir production rates from reservoir properties.

**Request:**
```json
{
  "features": [0.20, 500, 3500, 0.30, 100, 3000, 100, 0]
}
```

**Feature Array:**
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | porosity | Rock porosity (fraction) |
| 1 | permeability_md | Permeability (mD) |
| 2 | depth_m | Reservoir depth (m) |
| 3 | water_saturation | Water saturation (fraction) |
| 4 | net_pay_m | Net pay thickness (m) |
| 5 | pressure_psi | Reservoir pressure (psi) |
| 6 | temperature_c | Reservoir temperature (C) |
| 7 | oil_viscosity_cp | Oil viscosity (cP) |

**Response (200):**
```json
{
  "prediction": {
    "oil_rate_bopd": 350.25,
    "water_rate_bwpd": 120.50,
    "gas_rate_mscfd": 850.00
  }
}
```

**Error Response (400):**
```json
{"error": "Missing features in request body"}
```

**Error Response (503):**
```json
{"error": "History matcher not trained"}
```

---

### POST /api/forecast
Forecast future production from historical data.

**Request:**
```json
{
  "recent_data": [
    {"oil_rate_bopd": 350, "water_rate_bwpd": 120, "gas_rate_mscfd": 850},
    {"oil_rate_bopd": 345, "water_rate_bwpd": 125, "gas_rate_mscfd": 840},
    ...
  ],
  "n_steps": 10
}
```

**Response (200):**
```json
{
  "forecast": [
    {"oil_rate_bopd": 340.50, "water_rate_bwpd": 130.20, "gas_rate_mscfd": 830.00},
    {"oil_rate_bopd": 335.80, "water_rate_bwpd": 135.50, "gas_rate_mscfd": 820.00},
    ...
  ]
}
```

**Error Response (400):**
```json
{"error": "Missing recent_data in request body"}
```

**Error Response (503):**
```json
{"error": "Production forecaster not trained"}
```

---

### GET /api/docs
OpenAPI 3.0 self-documentation.

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request - missing or invalid data |
| 503 | Service unavailable - model not trained |
