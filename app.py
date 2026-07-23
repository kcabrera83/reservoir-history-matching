import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

from reservoir_history_matching.models.history_matcher import HistoryMatcher
from reservoir_history_matching.models.production_forecaster import ProductionForecaster
from reservoir_history_matching.utils.preprocessor import ReservoirPreprocessor

app = Flask(__name__)
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'models')

matcher = None
forecaster = None
preprocessor = None


def load_models():
    global matcher, forecaster, preprocessor
    matcher_path = os.path.join(MODEL_DIR, 'history_matcher.pkl')
    forecaster_path = os.path.join(MODEL_DIR, 'production_forecaster.pkl')
    preprocessor_path = os.path.join(MODEL_DIR, 'preprocessor.pkl')

    if os.path.exists(matcher_path):
        matcher = HistoryMatcher.load(matcher_path)
    if os.path.exists(forecaster_path):
        forecaster = ProductionForecaster.load(forecaster_path)
    if os.path.exists(preprocessor_path):
        preprocessor = ReservoirPreprocessor.load(preprocessor_path)


load_models()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'history_matcher': matcher is not None and matcher.is_trained,
            'production_forecaster': forecaster is not None and forecaster.is_trained,
        }
    })


@app.route('/api/models', methods=['GET'])
def models_info():
    info = {}
    if matcher and matcher.is_trained:
        info['history_matcher'] = {
            'type': matcher.model_type,
            'metrics': matcher.metrics,
            'targets': matcher.targets,
        }
    if forecaster and forecaster.is_trained:
        info['production_forecaster'] = {
            'lookback': forecaster.lookback,
            'metrics': forecaster.metrics,
            'targets': forecaster.targets,
        }
    return jsonify(info)


@app.route('/api/predict', methods=['POST'])
def predict():
    if not matcher or not matcher.is_trained:
        return jsonify({'error': 'History matcher not trained'}), 503
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'Missing features in request body'}), 400
    try:
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = preprocessor.transform(features)
        prediction = matcher.predict(features_scaled)
        result = prediction.iloc[0].to_dict()
        return jsonify({
            'prediction': {
                'oil_rate_bopd': round(float(result['oil_rate_bopd']), 2),
                'water_rate_bwpd': round(float(result['water_rate_bwpd']), 2),
                'gas_rate_mscfd': round(float(result['gas_rate_mscfd']), 2),
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/forecast', methods=['POST'])
def forecast():
    if not forecaster or not forecaster.is_trained:
        return jsonify({'error': 'Production forecaster not trained'}), 503
    data = request.get_json()
    if not data or 'recent_data' not in data:
        return jsonify({'error': 'Missing recent_data in request body'}), 400
    try:
        recent = pd.DataFrame(data['recent_data'])
        n_steps = data.get('n_steps', 10)
        forecasts = forecaster.forecast_future(recent, n_steps=n_steps)
        result = forecasts.to_dict(orient='records')
        for r in result:
            for k in r:
                r[k] = round(float(r[k]), 2)
        return jsonify({'forecast': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/docs')
def api_docs():
    return jsonify({
        "openapi": "3.0.0",
        "info": {"title": "Reservoir History Matching - Reservoir History", "version": "1.0.0"},
        "paths": {
            "/": {"get": {"summary": "Main dashboard"}},
            "/api/health": {"get": {"summary": "Service health check"}},
            "/api/models": {"get": {"summary": "Information about trained models"}},
            "/api/predict": {"post": {"summary": "Predict reservoir production (oil, water, gas rates)"}},
            "/api/forecast": {"post": {"summary": "Future production forecast"}},
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)
