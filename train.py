import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from reservoir_history_matching.data_generator import ReservoirDataGenerator
from reservoir_history_matching.utils.preprocessor import ReservoirPreprocessor
from reservoir_history_matching.models.history_matcher import HistoryMatcher
from reservoir_history_matching.models.production_forecaster import ProductionForecaster

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("RESERVOIR HISTORY MATCHING - MODEL TRAINING")
    print("=" * 60)

    print("\n[1/5] Generating synthetic reservoir data...")
    generator = ReservoirDataGenerator(n_wells=50, n_timesteps=100, seed=42)
    well_props, production = generator.generate_dataset()
    print(f"  Generated {len(well_props)} wells, {len(production)} production records")

    print("\n[2/5] Preparing history matching dataset...")
    X_hist, y_hist = generator.generate_matching_dataset(production)
    X_hist_train, X_hist_test, y_hist_train, y_hist_test = train_test_split(
        X_hist, y_hist, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_hist_train)}, Test: {len(X_hist_test)}")

    print("\n[3/5] Training HistoryMatcher (GradientBoosting)...")
    preprocessor = ReservoirPreprocessor()
    X_hist_train_scaled = preprocessor.fit_transform(X_hist_train)
    X_hist_test_scaled = preprocessor.transform(X_hist_test)

    matcher = HistoryMatcher(model_type='gradient_boosting')
    matcher.train(X_hist_train_scaled, y_hist_train, X_hist_test_scaled, y_hist_test)
    matcher.save(os.path.join(OUTPUT_DIR, 'history_matcher.pkl'))
    preprocessor.save(os.path.join(OUTPUT_DIR, 'preprocessor.pkl'))
    print(f"  Saved history matcher to {OUTPUT_DIR}")

    print("\n[4/5] Preparing forecast dataset...")
    X_fore, y_fore = generator.generate_forecast_dataset(production, lookback=10)
    X_fore_train, X_fore_test, y_fore_train, y_fore_test = train_test_split(
        X_fore, y_fore, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_fore_train)}, Test: {len(X_fore_test)}")

    print("\n[5/5] Training ProductionForecaster...")
    forecaster = ProductionForecaster(lookback=10)
    forecaster.train(X_fore_train, y_fore_train, X_fore_test, y_fore_test)
    forecaster.save(os.path.join(OUTPUT_DIR, 'production_forecaster.pkl'))
    print(f"  Saved forecaster to {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Models saved in: {OUTPUT_DIR}")
    print("=" * 60)

    print("\nHistory Matcher Metrics:")
    for split, m in matcher.metrics.items():
        print(f"  {split}: R2={m['r2']:.4f}, MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}")

    print("\nProduction Forecaster Metrics:")
    for split, m in forecaster.metrics.items():
        print(f"  {split}: R2={m['r2']:.4f}, MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}")

    return matcher, forecaster


if __name__ == '__main__':
    main()
