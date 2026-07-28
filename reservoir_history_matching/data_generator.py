import numpy as np
import pandas as pd


class ReservoirDataGenerator:
    def __init__(self, n_wells=50, n_timesteps=100, seed=2024):
        self.n_wells = n_wells
        self.n_timesteps = n_timesteps
        self.seed = seed
        np.random.seed(seed)

    def generate_well_properties(self):
        data = {
            'well_id': [f'W-{i:03d}' for i in range(self.n_wells)],
            'x_location': np.random.uniform(0, 5000, self.n_wells),
            'y_location': np.random.uniform(0, 5000, self.n_wells),
            'porosity': np.random.uniform(0.05, 0.35, self.n_wells),
            'permeability_md': np.random.uniform(1, 2000, self.n_wells),
            'initial_pressure_psi': np.random.uniform(2000, 5000, self.n_wells),
            'water_saturation': np.random.uniform(0.1, 0.6, self.n_wells),
            'net_pay_thickness_ft': np.random.uniform(10, 200, self.n_wells),
            'reservoir_temp_f': np.random.uniform(150, 300, self.n_wells),
            'oil_viscosity_cp': np.random.uniform(0.5, 10.0, self.n_wells),
        }
        return pd.DataFrame(data)

    def generate_production_history(self, well_props):
        records = []
        for _, well in well_props.iterrows():
            base_oil_rate = (
                well['permeability_md'] * well['net_pay_thickness_ft']
                * well['porosity'] * 0.001
            )
            for t in range(self.n_timesteps):
                decline = np.exp(-0.015 * t)
                noise_oil = np.random.normal(0, base_oil_rate * 0.05)
                noise_water = np.random.normal(0, 5)
                noise_gas = np.random.normal(0, base_oil_rate * 0.1)

                oil_rate = max(0, base_oil_rate * decline + noise_oil)
                water_cut = min(0.95, 0.1 + 0.005 * t + np.random.normal(0, 0.02))
                water_rate = oil_rate * water_cut / max(1 - water_cut, 0.01)
                gas_oil_ratio = 500 + 2 * t + np.random.normal(0, 20)
                gas_rate = oil_rate * gas_oil_ratio

                pressure = well['initial_pressure_psi'] * (1 - 0.005 * t) + np.random.normal(0, 10)

                injection_rate = base_oil_rate * 0.3 * (1 + 0.01 * t) if t > 10 else 0

                records.append({
                    'well_id': well['well_id'],
                    'timestep': t,
                    'oil_rate_bopd': round(oil_rate, 2),
                    'water_rate_bwpd': round(max(0, water_rate), 2),
                    'gas_rate_mscfd': round(max(0, gas_rate), 2),
                    'oil_cumulative_bbl': 0,
                    'water_cumulative_bbl': 0,
                    'gas_cumulative_mscf': 0,
                    'pressure_psi': round(pressure, 2),
                    'water_injection_bwpd': round(injection_rate, 2),
                    'porosity': well['porosity'],
                    'permeability_md': well['permeability_md'],
                    'initial_pressure_psi': well['initial_pressure_psi'],
                    'water_saturation': well['water_saturation'],
                    'net_pay_thickness_ft': well['net_pay_thickness_ft'],
                    'x_location': well['x_location'],
                    'y_location': well['y_location'],
                })

        df = pd.DataFrame(records)
        cum_oil = df.groupby('well_id')['oil_rate_bopd'].cumsum() * 1
        cum_water = df.groupby('well_id')['water_rate_bwpd'].cumsum() * 1
        cum_gas = df.groupby('well_id')['gas_rate_mscfd'].cumsum() * 1
        df['oil_cumulative_bbl'] = cum_oil.round(2)
        df['water_cumulative_bbl'] = cum_water.round(2)
        df['gas_cumulative_mscf'] = cum_gas.round(2)
        return df

    def generate_dataset(self):
        well_props = self.generate_well_properties()
        production = self.generate_production_history(well_props)
        return well_props, production

    def generate_matching_dataset(self, production_df):
        feature_cols = [
            'porosity', 'permeability_md', 'initial_pressure_psi',
            'water_saturation', 'net_pay_thickness_ft',
            'pressure_psi', 'water_injection_bwpd', 'timestep'
        ]
        target_cols = ['oil_rate_bopd', 'water_rate_bwpd', 'gas_rate_mscfd']
        X = production_df[feature_cols].copy()
        y = production_df[target_cols].copy()
        return X, y

    def generate_forecast_dataset(self, production_df, lookback=10):
        records = []
        grouped = production_df.groupby('well_id')
        for well_id, group in grouped:
            group = group.sort_values('timestep').reset_index(drop=True)
            for i in range(lookback, len(group)):
                past = group.iloc[i - lookback:i]
                future = group.iloc[i]
                features = {}
                for col in ['oil_rate_bopd', 'water_rate_bwpd', 'gas_rate_mscfd', 'pressure_psi']:
                    for lag in range(1, lookback + 1):
                        features[f'{col}_lag{lag}'] = past.iloc[-lag][col]
                features['porosity'] = future['porosity']
                features['permeability_md'] = future['permeability_md']
                features['timestep'] = future['timestep']
                features['target_oil'] = future['oil_rate_bopd']
                features['target_water'] = future['water_rate_bwpd']
                features['target_gas'] = future['gas_rate_mscfd']
                records.append(features)
        df = pd.DataFrame(records)
        target_cols = ['target_oil', 'target_water', 'target_gas']
        X = df.drop(columns=target_cols)
        y = df[target_cols]
        return X, y
