import streamlit as st
import joblib, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Reservoir History Matching", layout="wide")
st.title("Reservoir History Matching")
st.markdown("History matching & production forecasting")

@st.cache_resource
def load_models():
    base = Path(__file__).parent / 'outputs' / 'models'
    return {'oil': joblib.load(base / 'oil_production_model.pkl'), 'gas': joblib.load(base / 'gas_production_model.pkl')}

models = load_models()

def predict(name, x):
    m = models[name]
    if isinstance(m, dict):
        X = m['scaler'].transform(x)
        p = m['model'].predict(X)
        if 'label_encoder' in m:
            return m['label_encoder'].inverse_transform(p)[0]
        return float(p[0])
    return float(m.predict(x)[0])

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader('Parameters')
    poro = st.slider('Poro', 5, 35, 20)
    perm = st.slider('Perm', 0, 1000, 500)
    thick = st.slider('Thick', 10, 500, 255)
    pres = st.slider('Pres', 1000, 10000, 5500)
    sw = st.slider('Sw', 10, 90, 50)
    kv_kh = st.slider('Kv Kh', 0, 1, 0)
    comp = st.slider('Comp', 1, 100, 50)
    visc = st.slider('Visc', 0, 100, 50)
    ff = st.slider('Ff', 1, 3, 2)
    area = st.slider('Area', 40, 2000, 1020)
    run = st.button('Run Prediction', use_container_width=True)

with col2:
    if run:
        x = np.array([[poro, perm, thick, pres, sw, kv_kh, comp, visc, ff, area]])
        results = {}
        results['oil'] = predict('oil', x)
        results['gas'] = predict('gas', x)
        st.subheader('Results')
        rcols = st.columns(len(results))
        for i, (k, v) in enumerate(results.items()):
            label = k.replace('_', ' ').title()
            if isinstance(v, str):
                rcols[i].metric(label, v)
            else:
                rcols[i].metric(label, f'{v:.2f}')
        # Plot
        fig, ax = plt.subplots()
        names = [k.replace('_',' ').title() for k in results]
        vals = [float(v) if isinstance(v, (int,float,str)) and str(v).replace('.','').replace('-','').isdigit() else 0 for v in results.values()]
        if any(v != 0 for v in vals):
            ax.bar(names, vals, color=['#0077B6','#00B4D8','#90E0EF'])
            ax.set_ylabel('Value')
            st.pyplot(fig)