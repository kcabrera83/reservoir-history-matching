import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import os


class HistoryMatcher:
    def __init__(self, model_type='gradient_boosting'):
        self.model_type = model_type
        self.models = {}
        self.targets = ['oil_rate_bopd', 'water_rate_bwpd', 'gas_rate_mscfd']
        self.metrics = {}
        self.is_trained = False

    def _create_model(self):
        if self.model_type == 'gradient_boosting':
            base = GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
        elif self.model_type == 'random_forest':
            base = RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_split=5,
                random_state=42, n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        return MultiOutputRegressor(base)

    def train(self, X_train, y_train, X_test=None, y_test=None):
        print(f"Training HistoryMatcher with {self.model_type}...")
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        self.is_trained = True

        self.metrics['train'] = self._evaluate(X_train, y_train)
        if X_test is not None and y_test is not None:
            self.metrics['test'] = self._evaluate(X_test, y_test)

        print("Training complete.")
        for split, m in self.metrics.items():
            print(f"  {split}: R2={m['r2']:.4f}, MAE={m['mae']:.4f}")
        return self.metrics

    def _evaluate(self, X, y):
        preds = self.model.predict(X)
        if isinstance(y, pd.DataFrame):
            y = y.values
        return {
            'r2': float(r2_score(y, preds)),
            'mae': float(mean_absolute_error(y, preds)),
            'rmse': float(np.sqrt(mean_squared_error(y, preds))),
        }

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        preds = self.model.predict(X)
        return pd.DataFrame(preds, columns=self.targets)

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
