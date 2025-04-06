import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
df = pd.read_csv("Cleaned_Autodock_Results.csv")

st.title("🧬 Binding Affinity Predictor")

pair = st.selectbox("Choose Protein-Ligand Pair", df["PROTEIN-LIGAND"].unique())
features = df[df["PROTEIN-LIGAND"] == pair].drop(columns=["binding energy"])

if st.button("Predict"):
    pred = model.predict(features)[0]
    st.success(f"Predicted Binding Affinity: {pred:.2f} kcal/mol")
