import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os


class ReservoirPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.target_columns = []
        self.fitted = False

    def fit_transform(self, X, y=None):
        self.feature_columns = list(X.columns) if hasattr(X, 'columns') else []
        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True
        return X_scaled

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Preprocessor has not been fitted yet.")
        return self.scaler.transform(X)

    def inverse_transform(self, X_scaled):
        if not self.fitted:
            raise ValueError("Preprocessor has not been fitted yet.")
        return self.scaler.inverse_transform(X_scaled)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
