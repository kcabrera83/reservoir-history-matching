import requests
import json
import sys

BASE_URL = 'http://127.0.0.1:5006'
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
    try:
        r = requests.get(f'{BASE_URL}/api/health', timeout=5)
        data = r.json()
        test("Status 200", r.status_code == 200)
        test("Status healthy", data.get('status') == 'healthy')
        test("Models loaded key", 'models_loaded' in data)
    except Exception as e:
        test("Health endpoint reachable", False, str(e))


def test_models():
    print("\nTest: GET /api/models")
    try:
        r = requests.get(f'{BASE_URL}/api/models', timeout=5)
        data = r.json()
        test("Status 200", r.status_code == 200)
        test("History matcher info", 'history_matcher' in data)
        test("Forecaster info", 'production_forecaster' in data)
    except Exception as e:
        test("Models endpoint reachable", False, str(e))


def test_index():
    print("\nTest: GET /")
    try:
        r = requests.get(f'{BASE_URL}/', timeout=5)
        test("Status 200", r.status_code == 200)
        test("Contains dashboard title", 'Reservoir History Matching' in r.text)
        test("Contains Kelvin Cabrera", 'Kelvin Cabrera' in r.text)
    except Exception as e:
        test("Index endpoint reachable", False, str(e))


def test_predict():
    print("\nTest: POST /api/predict")
    features = [0.20, 500, 3500, 0.30, 100, 3000, 100, 0]
    try:
        r = requests.post(f'{BASE_URL}/api/predict',
                          json={'features': features}, timeout=10)
        data = r.json()
        test("Status 200", r.status_code == 200)
        test("Has prediction", 'prediction' in data)
        if 'prediction' in data:
            p = data['prediction']
            test("Has oil_rate_bopd", 'oil_rate_bopd' in p)
            test("Has water_rate_bwpd", 'water_rate_bwpd' in p)
            test("Has gas_rate_mscfd", 'gas_rate_mscfd' in p)
            test("Oil rate positive", p.get('oil_rate_bopd', 0) >= 0)
    except Exception as e:
        test("Predict endpoint reachable", False, str(e))


def test_predict_missing():
    print("\nTest: POST /api/predict (missing features)")
    try:
        r = requests.post(f'{BASE_URL}/api/predict',
                          json={}, timeout=5)
        test("Returns 400 for missing features", r.status_code == 400)
    except Exception as e:
        test("Missing features handling", False, str(e))


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
    try:
        r = requests.post(f'{BASE_URL}/api/forecast',
                          json={'recent_data': recent, 'n_steps': 5}, timeout=10)
        data = r.json()
        test("Status 200", r.status_code == 200)
        test("Has forecast", 'forecast' in data)
        if 'forecast' in data:
            test("Forecast is list", isinstance(data['forecast'], list))
            test("Forecast length correct", len(data['forecast']) == 5)
    except Exception as e:
        test("Forecast endpoint reachable", False, str(e))


def test_forecast_missing():
    print("\nTest: POST /api/forecast (missing data)")
    try:
        r = requests.post(f'{BASE_URL}/api/forecast',
                          json={}, timeout=5)
        test("Returns 400 for missing data", r.status_code == 400)
    except Exception as e:
        test("Missing data handling", False, str(e))


if __name__ == '__main__':
    print("=" * 50)
    print("RESERVOIR HISTORY MATCHING - API TESTS")
    print("=" * 50)

    test_index()
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
