import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Reservoir History Matching", layout="wide")
st.title("Reservoir History Matching")
st.markdown("Match reservoir history and forecast production.")

import joblib, numpy as np
d = Path(__file__).parent / 'outputs' / 'models'
models = {'oil': joblib.load(d / 'oil_production_model.pkl'), 'water': joblib.load(d / 'water_production_model.pkl'), 'gas': joblib.load(d / 'gas_production_model.pkl')}

st.sidebar.header("Input Parameters")
porosity = st.sidebar.slider('Porosity', 5, 35, 20)
perm = st.sidebar.slider('Perm', 0, 1000, 500)
thickness = st.sidebar.slider('Thickness', 10, 500, 255)
pressure = st.sidebar.slider('Pressure', 1000, 10000, 5500)
sw = st.sidebar.slider('Sw', 10, 90, 50)
kv_kh = st.sidebar.slider('Kv Kh', 0, 1, 0)
compress = st.sidebar.slider('Compress', 1, 100, 50)
viscosity = st.sidebar.slider('Viscosity', 0, 100, 50)
ff = st.sidebar.slider('Ff', 1, 3, 2)
drainage = st.sidebar.slider('Drainage', 40, 2000, 1020)

if st.sidebar.button("Run"):
    try:
        x = np.array([[porosity, perm, thickness, pressure, sw, kv_kh, compress, viscosity, ff, drainage]])
        cols = st.columns(3)
        for i, (k, m) in enumerate(models.items()):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            if 'label_encoder' in m:
                val = m['label_encoder'].inverse_transform(p)[0]
            else:
                val = f'{p[0]:.2f}'
            cols[i].metric(k.title(), val)
    except Exception as e:
        st.error(str(e))