import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "concrete_strength_pipeline.joblib"


class ConcreteFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        total_binder = (
            X_out["cement_kg_m3"]
            + X_out["blast_furnace_slag_kg_m3"]
            + X_out["fly_ash_kg_m3"]
        )
        X_out["total_binder_kg_m3"] = total_binder
        X_out["water_cement_ratio"] = np.where(
            X_out["cement_kg_m3"] > 0,
            X_out["water_kg_m3"] / X_out["cement_kg_m3"],
            0,
        )
        X_out["water_binder_ratio"] = np.where(
            total_binder > 0,
            X_out["water_kg_m3"] / total_binder,
            0,
        )
        X_out["superplasticizer_binder_ratio"] = np.where(
            total_binder > 0,
            X_out["superplasticizer_kg_m3"] / total_binder,
            0,
        )
        return X_out


sys.modules["__main__"].ConcreteFeatureEngineer = ConcreteFeatureEngineer


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


pipeline = load_model()

st.title("Concrete Compressive Strength Predictor")
st.write("Adjust the concrete mix proportions to estimate compressive strength.")

col1, col2 = st.columns(2)

with col1:
    cement = st.number_input("Cement (kg/m³)", 100.0, 600.0, 320.0, step=5.0)
    slag = st.number_input("Blast Furnace Slag (kg/m³)", 0.0, 400.0, 100.0, step=5.0)
    fly_ash = st.number_input("Fly Ash (kg/m³)", 0.0, 300.0, 0.0, step=5.0)
    water = st.number_input("Water (kg/m³)", 100.0, 260.0, 165.0, step=2.0)

with col2:
    superplasticizer = st.number_input("Superplasticizer (kg/m³)", 0.0, 35.0, 6.0, step=0.5)
    coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", 700.0, 1200.0, 1020.0, step=10.0)
    fine_agg = st.number_input("Fine Aggregate (kg/m³)", 500.0, 1000.0, 780.0, step=10.0)
    age = st.selectbox("Curing Age (Days)", [1, 3, 7, 14, 28, 56, 90, 180, 365])

input_df = pd.DataFrame([{
    "cement_kg_m3": cement,
    "blast_furnace_slag_kg_m3": slag,
    "fly_ash_kg_m3": fly_ash,
    "water_kg_m3": water,
    "superplasticizer_kg_m3": superplasticizer,
    "coarse_aggregate_kg_m3": coarse_agg,
    "fine_aggregate_kg_m3": fine_agg,
    "age_days": age,
}])

pred_mpa = float(pipeline.predict(input_df)[0])

st.divider()
st.metric("Predicted Compressive Strength", f"{pred_mpa:.2f} MPa")

if pred_mpa >= 45.2:
    st.success("PASS — predicted strength exceeds the 40 MPa target with margin.")
elif pred_mpa > 40.0:
    st.warning("BORDERLINE — above 40 MPa, but still within model uncertainty.")
else:
    st.error("FAIL — predicted below 40 MPa.")
