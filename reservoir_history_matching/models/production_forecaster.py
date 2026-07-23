import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import os


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ProductionForecaster:
    def __init__(self, lookback=10):
        self.lookback = lookback
        self.gp_models = {}
        self.likelihoods = {}
        self.targets = ['target_oil', 'target_water', 'target_gas']
        self.metrics = {}
        self.is_trained = False

    def _train_single_gp(self, X_train, y_train, n_iter=100, lr=0.1):
        train_x = torch.tensor(X_train, dtype=torch.float32)
        train_y = torch.tensor(y_train, dtype=torch.float32)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(n_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        return model, likelihood

    def train(self, X_train, y_train, X_test=None, y_test=None):
        print("Training ProductionForecaster (GPyTorch Gaussian Process)...")
        for target in self.targets:
            print(f"  Training GP for {target}...")
            model, likelihood = self._train_single_gp(X_train, y_train[target].values)
            self.gp_models[target] = model
            self.likelihoods[target] = likelihood

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
        test_x = torch.tensor(np.array(X, dtype=np.float32))
        result = {}
        for target in self.targets:
            model = self.gp_models[target]
            likelihood = self.likelihoods[target]
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = likelihood(model(test_x))
                result[target] = pred.mean.numpy()
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
        data = {
            'lookback': self.lookback,
            'targets': self.targets,
            'metrics': self.metrics,
            'is_trained': self.is_trained,
            'gp_models': self.gp_models,
            'likelihoods': self.likelihoods,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = ProductionForecaster(lookback=data['lookback'])
        obj.targets = data['targets']
        obj.metrics = data['metrics']
        obj.is_trained = data['is_trained']
        obj.gp_models = data['gp_models']
        obj.likelihoods = data['likelihoods']
        return obj
