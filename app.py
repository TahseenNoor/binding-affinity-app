import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
df = pd.read_csv("Cleaned_Autodock_Results.csv")

st.title("🧬 Binding Affinity Predictor")

pair = st.selectbox("Choose Protein-Ligand Pair", df["PROTEIN-LIGAND"].unique())
# Define the exact columns used during training (no strings or NaNs)
features = df[df['PROTEIN-LIGAND'] == selected_pair][[
    'binding energy', 'cluster RMSD', 'reference RMSD',
    'ligand efficiency', 'Internal energy', 
    'vdw hb desolve energy', 'Electrostatic energy',
    'Total internal', 'Torsional energy', 'unbound energy'
]]

# Optionally fill NaNs with 0 or mean
features = features.fillna(0)
pred = model.predict(features)[0]


if st.button("Predict"):
    pred = model.predict(features)[0]
    st.success(f"Predicted Binding Affinity: {pred:.2f} kcal/mol")
