import streamlit as st
import pandas as pd
import joblib

# Load your dataset and model
df = pd.read_csv("Cleaned_Autodock_Results.csv")
model = joblib.load("model.pkl")

# Streamlit App Title
st.title("🔬 Binding Affinity Predictor")

# Dropdown to select protein-ligand pair
selected_pair = st.selectbox("Choose Protein-Ligand Pair", df['PROTEIN-LIGAND'].unique())

# Predict button
if st.button("Predict"):
    try:
        row = df[df['PROTEIN-LIGAND'] == selected_pair]

        features = row[[
            'binding energy', 'cluster RMSD', 'reference RMSD',
            'ligand efficiency', 'Internal energy', 
            'vdw hb desolve energy', 'Electrostatic energy',
            'Total internal', 'Torsional energy', 'unbound energy'
        ]]

        features = features.fillna(0)

        pred = model.predict(features)[0]

        st.success(f"🧪 Predicted Binding Affinity: {pred:.2f} kcal/mol")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
