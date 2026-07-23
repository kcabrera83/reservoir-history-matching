import sys
sys.path.insert(0, '.')
from app import app

from fastapi.testclient import TestClient

client = TestClient(app)
passed = 0
failed = 0


def test(name, condition, detail=''):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} {detail}")
        failed += 1


def test_health():
    print("\nTest: GET /api/health")
    r = client.get("/api/health")
    data = r.json()
    test("Status 200", r.status_code == 200)
    test("Status healthy", data.get('status') == 'healthy')
    test("Models loaded key", 'models_loaded' in data)


def test_models():
    print("\nTest: GET /api/models")
    r = client.get("/api/models")
    data = r.json()
    test("Status 200", r.status_code == 200)
    test("History matcher info", 'history_matcher' in data)
    test("Forecaster info", 'production_forecaster' in data)


def test_predict():
    print("\nTest: POST /api/predict")
    features = [0.20, 500, 3500, 0.30, 100, 3000, 100, 0]
    r = client.post("/api/predict", json={'features': features})
    data = r.json()
    test("Status 200", r.status_code == 200)
    test("Has prediction", 'prediction' in data)
    if 'prediction' in data:
        p = data['prediction']
        test("Has oil_rate_bopd", 'oil_rate_bopd' in p)
        test("Has water_rate_bwpd", 'water_rate_bwpd' in p)
        test("Has gas_rate_mscfd", 'gas_rate_mscfd' in p)
        test("Oil rate positive", p.get('oil_rate_bopd', 0) >= 0)


def test_predict_missing():
    print("\nTest: POST /api/predict (missing features)")
    r = client.post("/api/predict", json={})
    test("Returns 422 for missing features", r.status_code in (400, 422))


def test_forecast():
    print("\nTest: POST /api/forecast")
    recent = []
    for i in range(10):
        recent.append({
            'oil_rate_bopd': 100 * (0.985 ** i),
            'water_rate_bwpd': 20 + i * 2,
            'gas_rate_mscfd': 500 * (0.985 ** i),
            'pressure_psi': 3500 - 50 * i,
        })
    r = client.post("/api/forecast", json={'recent_data': recent, 'n_steps': 5})
    data = r.json()
    test("Status 200", r.status_code == 200)
    test("Has forecast", 'forecast' in data)
    if 'forecast' in data:
        test("Forecast is list", isinstance(data['forecast'], list))
        test("Forecast length correct", len(data['forecast']) == 5)


def test_forecast_missing():
    print("\nTest: POST /api/forecast (missing data)")
    r = client.post("/api/forecast", json={})
    test("Returns 422 for missing data", r.status_code in (400, 422))


if __name__ == '__main__':
    print("=" * 50)
    print("RESERVOIR HISTORY MATCHING - API TESTS")
    print("=" * 50)

    test_health()
    test_models()
    test_predict()
    test_predict_missing()
    test_forecast()
    test_forecast_missing()

    print("\n" + "=" * 50)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 50)
    sys.exit(0 if failed == 0 else 1)
