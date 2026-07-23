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


class HistoryMatcher:
    def __init__(self, model_type='gp'):
        self.model_type = model_type
        self.gp_models = {}
        self.likelihoods = {}
        self.targets = ['oil_rate_bopd', 'water_rate_bwpd', 'gas_rate_mscfd']
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
        print("Training HistoryMatcher (GPyTorch Gaussian Process)...")
        if isinstance(y_train, pd.DataFrame):
            y_arr = y_train.values
        else:
            y_arr = np.array(y_train)

        for i, target in enumerate(self.targets):
            print(f"  Training GP for {target}...")
            model, likelihood = self._train_single_gp(X_train, y_arr[:, i])
            self.gp_models[target] = model
            self.likelihoods[target] = likelihood

        self.is_trained = True
        self.metrics['train'] = self._evaluate(X_train, y_train)
        if X_test is not None and y_test is not None:
            self.metrics['test'] = self._evaluate(X_test, y_test)

        print("Training complete.")
        for split, m in self.metrics.items():
            print(f"  {split}: R2={m['r2']:.4f}, MAE={m['mae']:.4f}")

    def _evaluate(self, X, y):
        preds = self.predict(X)
        if isinstance(y, pd.DataFrame):
            y = y.values
        return {
            'r2': float(r2_score(y, preds.values)),
            'mae': float(mean_absolute_error(y, preds.values)),
            'rmse': float(np.sqrt(mean_squared_error(y, preds.values))),
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
        return pd.DataFrame(result, columns=self.targets)

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {
            'model_type': self.model_type,
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
        obj = HistoryMatcher(model_type=data['model_type'])
        obj.targets = data['targets']
        obj.metrics = data['metrics']
        obj.is_trained = data['is_trained']
        obj.gp_models = data['gp_models']
        obj.likelihoods = data['likelihoods']
        return obj
