# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
energy_model = joblib.load('model.pkl')  # this is your docking energy model
descriptor_model = joblib.load('descriptor_model.pkl')  # descriptor model
combined_model = joblib.load('model_with_importance.pkl')  # combined model

# Load data
energy_data = pd.read_csv('Cleaned_Autodock_Results.csv')
descriptor_data = pd.read_csv('descriptors final.csv')

# Streamlit App
st.set_page_config(page_title="Afferaze Binding Predictor", page_icon="🧪", layout="centered")

# Custom Title
st.markdown(
    """
    <h1 style='text-align: center; font-family: Garamond; font-size: 60px; color: #3C486B;'>AFFERAZE</h1>
    <p style='text-align: center; font-family: Garamond; font-size: 20px;'>Predict binding affinity using <b>energy values</b> or <b>molecular descriptors</b> 🎯</p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Choose mode
mode = st.radio("Choose Prediction Mode:", 
                ("🧪 Use Docking Energy Values", 
                 "🌿 Use Molecular Descriptors", 
                 "🚀 Combined Input (Descriptors + Energy Values)", 
                 "🛠️ Manual Input (Energy Only, Any Names)", 
                 "🔮 Magic Mode (Any Known Ligand)")
)

st.markdown("### 🔍 Select Protein and Enter Ligand")

protein_choice = st.text_input("Choose a Protein", "TNF")
ligand_choice = st.text_input("Enter Ligand Name", "ZONOROL")

if st.button("🎯 Predict Binding Affinity (from Dataset)"):
    try:
        if mode == "🧪 Use Docking Energy Values":
            match = energy_data[(energy_data['Protein'] == protein_choice) & (energy_data['Ligand'] == ligand_choice)]
            if not match.empty:
                X = match[['Binding Energy (kcal/mol)', 'Inhibition Constant (Ki) (nM)']]
                pred = energy_model.predict(X)
                st.success(f"Predicted Binding Affinity: {pred[0]:.2f}")
            else:
                st.error("Ligand not found in Energy Dataset.")

        elif mode == "🌿 Use Molecular Descriptors":
            match = descriptor_data[(descriptor_data['Protein'] == protein_choice) & (descriptor_data['Ligand'] == ligand_choice)]
            if not match.empty:
                X = match.drop(columns=['Protein', 'Ligand'])
                pred = descriptor_model.predict(X)
                st.success(f"Predicted Binding Affinity: {pred[0]:.2f}")
            else:
                st.error("Ligand not found in Descriptor Dataset.")

        elif mode == "🚀 Combined Input (Descriptors + Energy Values)":
            energy_match = energy_data[(energy_data['Protein'] == protein_choice) & (energy_data['Ligand'] == ligand_choice)]
            descriptor_match = descriptor_data[(descriptor_data['Protein'] == protein_choice) & (descriptor_data['Ligand'] == ligand_choice)]
            if not energy_match.empty and not descriptor_match.empty:
                energy_X = energy_match[['Binding Energy (kcal/mol)', 'Inhibition Constant (Ki) (nM)']]
                descriptor_X = descriptor_match.drop(columns=['Protein', 'Ligand'])
                combined_X = pd.concat([energy_X.reset_index(drop=True), descriptor_X.reset_index(drop=True)], axis=1)
                pred = combined_model.predict(combined_X)
                st.success(f"Predicted Binding Affinity: {pred[0]:.2f}")
            else:
                st.error("Ligand not found in both datasets.")

        elif mode == "🛠️ Manual Input (Energy Only, Any Names)":
            be = st.number_input("Enter Binding Energy (kcal/mol)", value=-7.0)
            ki = st.number_input("Enter Inhibition Constant (Ki) (nM)", value=200.0)
            manual_input = np.array([[be, ki]])
            pred = energy_model.predict(manual_input)
            st.success(f"Predicted Binding Affinity: {pred[0]:.2f}")

        elif mode == "🔮 Magic Mode (Any Known Ligand)":
            # Magic mode: First try Energy dataset, if not found try Descriptors
            match = energy_data[(energy_data['Ligand'] == ligand_choice)]
            if not match.empty:
                X = match[['Binding Energy (kcal/mol)', 'Inhibition Constant (Ki) (nM)']]
                pred = energy_model.predict(X)
                st.success(f"Predicted Binding Affinity (from Energy): {pred[0]:.2f}")
            else:
                match = descriptor_data[(descriptor_data['Ligand'] == ligand_choice)]
                if not match.empty:
                    X = match.drop(columns=['Protein', 'Ligand'])
                    pred = descriptor_model.predict(X)
                    st.success(f"Predicted Binding Affinity (from Descriptors): {pred[0]:.2f}")
                else:
                    st.error("Ligand not found anywhere 😢.")

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")

# Footer
st.markdown("---")
st.caption("Made with ❤️ by Afferaze Team")
