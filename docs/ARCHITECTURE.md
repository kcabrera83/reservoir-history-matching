# Architecture - Reservoir History Matching

## System Overview
```
                    +-------------------+
                    |   Flask Server    |
                    |   (app.py)        |
                    |   Port 5006       |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
+-------------v-----------+  +-------------v-----------+
| History Matcher          |  | Production Forecaster   |
| (GradientBoosting)       |  | (GradientBoosting)      |
| Properties -> Rates      |  | Lookback -> Forecast    |
+-------------+-----------+  +-------------+-----------+
              |                             |
+-------------v-----------------------------v-----------+
|              ReservoirPreprocessor                     |
|       (Standard Scaling + Serialization)              |
+------------------------+------------------------------+
                         |
               +---------v-----------+
               |  Synthetic Dataset  |
               |  (50 wells x 100ts) |
               +--------------------+
```

## Components

### Data Layer
- **Data Source**: Synthetic reservoir data generator (`ReservoirDataGenerator`) producing 50 wells with 100 timesteps each
- **Well Properties**: Porosity, permeability, depth, water saturation, net pay, pressure, temperature, viscosity
- **Production Data**: Oil rate, water rate, gas rate per timestep
- **Preprocessing**: Standard scaling with persistent preprocessor (pickle serialized)

### Model Layer

#### History Matcher
- **Algorithm**: GradientBoosting regressor
- **Input**: 8 reservoir properties (porosity, permeability, depth, water_saturation, net_pay, pressure, temperature, viscosity)
- **Output**: 3 production rates (oil_rate_bopd, water_rate_bwpd, gas_rate_mscfd)
- **Metrics**: R2, MAE, RMSE (train/test splits)
- **Purpose**: Map static reservoir properties to production performance

#### Production Forecaster
- **Algorithm**: GradientBoosting with sliding window approach
- **Input**: Lookback window of recent production data (default: 10 timesteps)
- **Output**: Future production rates for n_steps ahead
- **Metrics**: R2, MAE, RMSE (train/test splits)
- **Purpose**: Forecast future production from historical trends (alternative to LSTM/RNN)

### API Layer
- **Framework**: Flask
- **Endpoints**: 5 REST endpoints (health, models, predict, forecast, docs)
- **Model Loading**: Eager loading at startup from `outputs/models/`
- **Persistence**: Preprocessor saved as pickle for consistent transformations

### Dashboard Layer
- **Frontend**: Flask + HTML/CSS/JS
- **Features**: Reservoir property input, production prediction, forecast visualization

## Data Flow

### History Matching Flow
1. **Input**: 8 reservoir properties as numeric array
2. **Preprocessing**: `ReservoirPreprocessor` applies standard scaling
3. **Prediction**: HistoryMatcher predicts 3 production rates simultaneously
4. **Response**: Oil, water, and gas rates in standard units

### Forecasting Flow
1. **Input**: Recent production history (array of records) + n_steps
2. **Preprocessing**: Format into lookback windows
3. **Prediction**: ProductionForecaster generates multi-step forecasts
4. **Response**: Array of predicted production rates for each future timestep

## Training Pipeline
1. Generate synthetic reservoir data (50 wells, 100 timesteps)
2. Generate history matching dataset (properties -> rates)
3. Train HistoryMatcher with 80/20 split
4. Generate forecast dataset with lookback windows
5. Train ProductionForecaster with 80/20 split
6. Save models and preprocessor to `outputs/models/`
7. Print metrics for both models

## File Structure
```
reservoir-history-matching/
├── reservoir_history_matching/
│   ├── data_generator.py       # Synthetic reservoir data
│   ├── models/
│   │   ├── history_matcher.py  # Properties -> Production
│   │   └── production_forecaster.py # Lookback -> Forecast
│   └── utils/
│       └── preprocessor.py     # Scaling + persistence
├── outputs/models/             # Trained models + preprocessor
├── templates/index.html        # Dashboard
├── app.py                      # Flask server
├── train.py                    # Training pipeline
├── test_api.py                 # API tests
├── setup.py                    # Package setup
└── .github/workflows/ci.yml   # CI/CD
```
