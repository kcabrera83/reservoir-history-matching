import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import os


class ProductionForecaster:
    def __init__(self, lookback=10):
        self.lookback = lookback
        self.models = {}
        self.targets = ['target_oil', 'target_water', 'target_gas']
        self.metrics = {}
        self.is_trained = False

    def train(self, X_train, y_train, X_test=None, y_test=None):
        print("Training ProductionForecaster...")
        for target in self.targets:
            print(f"  Training model for {target}...")
            model = GradientBoostingRegressor(
                n_estimators=150, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            model.fit(X_train, y_train[target])
            self.models[target] = model

        self.is_trained = True
        self.metrics['train'] = self._evaluate(X_train, y_train)
        if X_test is not None and y_test is not None:
            self.metrics['test'] = self._evaluate(X_test, y_test)

        print("Training complete.")
        for split, m in self.metrics.items():
            print(f"  {split}: R2={m['r2']:.4f}, MAE={m['mae']:.4f}")
        return self.metrics

    def _evaluate(self, X, y):
        preds = self.predict(X)
        r2_scores = []
        for t in self.targets:
            r2_scores.append(r2_score(y[t].values, preds[t].values))
        avg_r2 = np.mean(r2_scores)
        all_true = y.values.flatten()
        all_pred = preds.values.flatten()
        return {
            'r2': float(avg_r2),
            'mae': float(mean_absolute_error(all_true, all_pred)),
            'rmse': float(np.sqrt(mean_squared_error(all_true, all_pred))),
        }

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        result = {}
        for target in self.targets:
            result[target] = self.models[target].predict(X)
        return pd.DataFrame(result)

    def _build_features_from_raw(self, raw_df):
        series_cols = ['oil_rate_bopd', 'water_rate_bwpd', 'gas_rate_mscfd', 'pressure_psi']
        features = {}
        for col in series_cols:
            values = raw_df[col].values
            for lag in range(1, self.lookback + 1):
                features[f'{col}_lag{lag}'] = values[-lag] if lag <= len(values) else 0.0
        features['porosity'] = raw_df['porosity'].values[-1] if 'porosity' in raw_df.columns else 0.2
        features['permeability_md'] = raw_df['permeability_md'].values[-1] if 'permeability_md' in raw_df.columns else 500.0
        features['timestep'] = len(raw_df)
        return pd.DataFrame([features])

    def forecast_future(self, raw_recent_data, n_steps=10):
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        forecasts = []
        history = raw_recent_data.copy()
        for step in range(n_steps):
            features = self._build_features_from_raw(history)
            pred = self.predict(features)
            forecasts.append(pred.iloc[0].to_dict())
            new_row = {
                'oil_rate_bopd': pred.iloc[0]['target_oil'],
                'water_rate_bwpd': pred.iloc[0]['target_water'],
                'gas_rate_mscfd': pred.iloc[0]['target_gas'],
                'pressure_psi': history['pressure_psi'].values[-1] * 0.998 if 'pressure_psi' in history.columns else 0,
                'porosity': history['porosity'].values[-1] if 'porosity' in history.columns else 0.2,
                'permeability_md': history['permeability_md'].values[-1] if 'permeability_md' in history.columns else 500.0,
            }
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True).iloc[-self.lookback:]
        result = pd.DataFrame(forecasts)
        result.columns = ['target_oil', 'target_water', 'target_gas']
        return result

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
