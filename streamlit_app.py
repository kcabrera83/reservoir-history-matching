
import streamlit as st
import numpy as np
import joblib
import os, sys
from dataclasses import dataclass, field
from typing import List, Optional

st.set_page_config(page_title="Reservoir History Matching", layout="wide")
st.title("Reservoir History Matching")
st.caption("Assisted history matching with production data assimilation")

@dataclass
class ModelContainer:
    model: object
    scaler: object
    features: List[str]
    target: str

def load_artifacts(artifact_dir: str = "outputs/models") -> dict:
    cache = {}
    for fname in os.listdir(artifact_dir):
        if fname.endswith(".pkl"):
            path = os.path.join(artifact_dir, fname)
            data = joblib.load(path)
            cache[fname.replace(".pkl", "")] = ModelContainer(
                model=data["model"],
                scaler=data.get("scaler"),
                features=data.get("feature_names", []),
                target=data.get("target_name", "target"),
            )
    return cache

artifacts = load_artifacts()

tab_names = list(artifacts.keys()) if artifacts else ["predict"]
tabs = st.tabs(tab_names)

for i, (name, container) in enumerate(artifacts.items()):
    with tabs[i]:
        st.subheader(f"{name} inference")
        cols = st.columns(len(container.features) if container.features else 1)
        inputs = []
        for j, feat in enumerate(container.features if container.features else ["feature_0"]):
            with cols[j % len(cols)]:
                val = st.number_input(f"{feat}", value=0.0, key=f"{name}_{feat}")
                inputs.append(val)
        if st.button("Predict", key=f"btn_{name}"):
            X = np.array(inputs).reshape(1, -1)
            if container.scaler:
                X = container.scaler.transform(X)
            pred = container.model.predict(X)[0]
            st.success(f"Prediction: {pred:.4f}")
