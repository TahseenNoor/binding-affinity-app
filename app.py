import streamlit as st
import pandas as pd
import joblib

# Load your dataset and model
df = pd.read_csv("Cleaned_Autodock_Results.csv")
model = joblib.load("binding_model_train.pkl")


# App title
st.title("🔬 Binding Affinity Predictor")

# Dropdown for protein-ligand pair
selected_pair = st.selectbox("Choose Protein-Ligand Pair", df['PROTEIN-LIGAND'].unique())

# Predict button
if st.button("Predict"):
    try:
        row = df[df['PROTEIN-LIGAND'] == selected_pair]

       features = row[[ 
                'Electrostatic energy', 
                'Torsional energy', 
                'vdw hb desolve energy', 
                'Intermol energy'
        ]]


        features = features.fillna(0)

        pred = model.predict(features)[0]
        st.success(f"🧪 Predicted Binding Affinity: {pred:.2f} kcal/mol")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
