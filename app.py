import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your models
energy_model = joblib.load('energy_model.pkl')  # Replace with your real model paths
descriptor_model = joblib.load('descriptor_model.pkl')
combined_model = joblib.load('combined_model.pkl')

# Load your dataset
dataset = pd.read_csv('https://github.com/TahseenNoor/binding-affinity-app/raw/refs/heads/main/descriptors%20final.csv')

# Basic Styling
st.markdown(
    """
    <h1 style='text-align: center; font-family: Garamond, serif; font-size: 60px; color: #4B0082;'>
        AFFERAZE
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("")  # Small space
st.markdown("<h4 style='text-align: center; font-weight: normal;'>Predict binding affinity using <b>energy values</b> or <b>molecular descriptors</b> 🧪</h4>", unsafe_allow_html=True)
st.write("---")

# Sidebar Mode Selection
st.subheader("Choose Prediction Mode:")
mode = st.radio(
    "",
    ("🧪 Use Docking Energy Values", 
     "🌿 Use Molecular Descriptors", 
     "🚀 Combined Input (Descriptors + Energy Values)", 
     "🛠 Manual Input (Energy Only, Any Names)", 
     "✨ Magic Mode (Any Known Ligand)")
)

# Protein and Ligand Input
st.markdown("### 🔎 Select Protein and Enter Ligand")
protein_name = st.text_input("Choose a Protein", "TNF")
ligand_name = st.text_input("Enter Ligand Name", "ZONOROL")

# Fetch row from dataset
selected_row = dataset[
    (dataset['Protein'] == protein_name.upper()) & 
    (dataset['Ligand'] == ligand_name.upper())
]

if selected_row.empty:
    st.warning("Ligand not found in dataset. You can still use Manual Input or Magic Mode.")

# Prediction based on mode
if mode == "🧪 Use Docking Energy Values":
    if not selected_row.empty:
        energy_features = selected_row[['Vina Score', 'AD4 Score', 'Ad4 Affinity', 'RFScore V3']].values
        prediction = energy_model.predict(energy_features)
        st.success(f"Predicted Binding Affinity: {prediction[0]:.2f}")
    else:
        st.error("Docking energy values not found for this ligand.")

elif mode == "🌿 Use Molecular Descriptors":
    if not selected_row.empty:
        descriptor_features = selected_row.drop(columns=['Protein', 'Ligand', 'Vina Score', 'AD4 Score', 'Ad4 Affinity', 'RFScore V3']).values
        prediction = descriptor_model.predict(descriptor_features)
        st.success(f"Predicted Binding Affinity: {prediction[0]:.2f}")
    else:
        st.error("Molecular descriptors not found for this ligand.")

elif mode == "🚀 Combined Input (Descriptors + Energy Values)":
    if not selected_row.empty:
        combined_features = selected_row.drop(columns=['Protein', 'Ligand']).values
        prediction = combined_model.predict(combined_features)
        st.success(f"Predicted Binding Affinity: {prediction[0]:.2f}")
    else:
        st.error("Combined input features not available for this ligand.")

elif mode == "🛠 Manual Input (Energy Only, Any Names)":
    vina = st.number_input("Vina Score", value=-7.0)
    ad4 = st.number_input("AD4 Score", value=-6.5)
    ad4_affinity = st.number_input("AD4 Affinity", value=-7.2)
    rfscore = st.number_input("RFScore V3", value=-6.8)
    
    manual_features = np.array([[vina, ad4, ad4_affinity, rfscore]])
    prediction = energy_model.predict(manual_features)
    st.success(f"Predicted Binding Affinity: {prediction[0]:.2f}")

elif mode == "✨ Magic Mode (Any Known Ligand)":
    st.info("Searching with any partial or complete ligand name...")
    matched_rows = dataset[dataset['Ligand'].str.contains(ligand_name.upper(), na=False)]
    
    if not matched_rows.empty:
        st.write(f"Found {len(matched_rows)} matching ligands:")
        st.dataframe(matched_rows[['Protein', 'Ligand']])
        
        idx = st.selectbox("Select the ligand row to predict:", matched_rows.index)
        selected = matched_rows.loc[[idx]]
        
        combined_features = selected.drop(columns=['Protein', 'Ligand']).values
        prediction = combined_model.predict(combined_features)
        st.success(f"Predicted Binding Affinity: {prediction[0]:.2f}")
    else:
        st.error("No ligands matched. Try manual mode instead!")

