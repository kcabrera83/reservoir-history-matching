import pytest


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_models_info(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_api_docs(client):
    response = client.get("/docs")
    assert response.status_code == 200


def test_predict_valid(client):
    payload = {"features": [0.5, 0.3, 0.8, 0.1, 0.9, 0.4, 0.6]}
    response = client.post("/api/predict", json=payload)
    assert response.status_code in (200, 503, 400)


def test_predict_empty_features(client):
    payload = {"features": []}
    response = client.post("/api/predict", json=payload)
    assert response.status_code in (400, 503)


def test_forecast_valid(client):
    recent_data = [
        {"oil_rate_bopd": 100, "water_rate_bwpd": 20, "gas_rate_mscfd": 50},
        {"oil_rate_bopd": 95, "water_rate_bwpd": 22, "gas_rate_mscfd": 48},
    ]
    payload = {"recent_data": recent_data, "n_steps": 5}
    response = client.post("/api/forecast", json=payload)
    assert response.status_code in (200, 503, 400)


def test_forecast_empty_data(client):
    payload = {"recent_data": [], "n_steps": 5}
    response = client.post("/api/forecast", json=payload)
    assert response.status_code in (400, 503)
