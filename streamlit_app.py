import streamlit as st
import joblib
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Reservoir History Matching", layout="wide")
st.title("Reservoir History Matching")
st.markdown("Match reservoir history and forecast production rates.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: joblib.load(d / v) for k, v in [("oil", "oil_production_model.pkl"), ("water", "water_production_model.pkl"), ("gas", "gas_production_model.pkl")]}

models = load_models()

st.sidebar.header("Input Parameters")
porosity_pct = st.sidebar.slider("Porosity Pct", 5, 35, 20)
permeability_md = st.sidebar.slider("Permeability Md", 0, 1000, 500)
net_thickness_ft = st.sidebar.slider("Net Thickness Ft", 10, 500, 255)
reservoir_pressure_psi = st.sidebar.slider("Reservoir Pressure Psi", 1000, 10000, 5500)
water_saturation_pct = st.sidebar.slider("Water Saturation Pct", 10, 90, 50)
kv_kh_ratio = st.sidebar.slider("Kv Kh Ratio", 0, 1, 0)
compressibility_1e6_psi = st.sidebar.slider("Compressibility 1E6 Psi", 1, 100, 50)
viscosity_cp = st.sidebar.slider("Viscosity Cp", 0, 100, 50)
formation_factor_bbl_stb = st.sidebar.slider("Formation Factor Bbl Stb", 1, 3, 2)
drainage_area_acres = st.sidebar.slider("Drainage Area Acres", 40, 2000, 1020)

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[porosity_pct, permeability_md, net_thickness_ft, reservoir_pressure_psi, water_saturation_pct, kv_kh_ratio, compressibility_1e6_psi, viscosity_cp, formation_factor_bbl_stb, drainage_area_acres]])
        m = models["oil"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Oil", result if isinstance(result, str) else f"{result:.4f}")
        m = models["water"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Water", result if isinstance(result, str) else f"{result:.4f}")
        m = models["gas"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Gas", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")

