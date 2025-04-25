import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# ------------------------ LOAD MODELS AND DATA ------------------------
rf_model = joblib.load("rf_model_final.sav")
svm_model = joblib.load("svm_model_final.sav")
combined_model = joblib.load("combined_model_final.sav")
manual_model = joblib.load("manual_model_final.sav")

descriptors_df = None
pharma_df = None

try:
    descriptors_df = pd.read_csv("descriptors final.csv")
    pharma_df = pd.read_csv("pharmokinetics final.csv")
    st.success("✅ ADMET data loaded successfully!")
except Exception as e:
    st.error(f"❌ Could not load ADMET data. Error: {e}")

# ------------------------ UI SETUP ------------------------
st.title("🧠 Ligand Activity Predictor + ADMET Profiler")
st.sidebar.header("Select Prediction Mode")
mode = st.sidebar.selectbox("Choose a mode:", [
    "🔬 Use Docking Energy Values",
    "🧪 Use Molecular Descriptors",
    "🧬 Combined Input",
    "🛠️ Manual Input",
    "🪄 Magic Mode"
])

# ------------------------ MODE FUNCTIONS ------------------------
def display_admet_profile(ligand_input):
    if pharma_df is not None:
        lig_match = pharma_df[pharma_df['Name'].str.lower() == ligand_input.lower()]
        if not lig_match.empty:
            st.markdown("### 🧪 ADMET Profile")
            st.dataframe(lig_match)
        else:
            st.info("ℹ️ No ADMET profile found for this ligand.")

# ------------------------ PREDICTION MODES ------------------------
if mode == "🔬 Use Docking Energy Values":
    st.header("🔬 Predict Using Docking Energies")
    vdw = st.number_input("Van der Waals Energy", value=0.0)
    ele = st.number_input("Electrostatic Energy", value=0.0)
    total = st.number_input("Total Docking Energy", value=0.0)
    ligand_input = st.text_input("Ligand Name (for ADMET)")

    if st.button("Predict"):
        pred = rf_model.predict([[vdw, ele, total]])[0]
        st.success(f"Prediction: {'Active' if pred==1 else 'Inactive'}")
        if ligand_input:
            display_admet_profile(ligand_input)

elif mode == "🧪 Use Molecular Descriptors":
    st.header("🧪 Predict Using Molecular Descriptors")
    ligand_input = st.selectbox("Choose a Ligand", descriptors_df['Name'].unique())

    if st.button("Predict"):
        desc_row = descriptors_df[descriptors_df['Name'] == ligand_input].drop('Name', axis=1)
        pred = svm_model.predict(desc_row)[0]
        st.success(f"Prediction: {'Active' if pred==1 else 'Inactive'}")
        display_admet_profile(ligand_input)

elif mode == "🧬 Combined Input":
    st.header("🧬 Predict Using Combined Input")
    ligand_input = st.selectbox("Choose a Ligand", descriptors_df['Name'].unique())
    vdw = st.number_input("Van der Waals Energy", value=0.0)
    ele = st.number_input("Electrostatic Energy", value=0.0)
    total = st.number_input("Total Docking Energy", value=0.0)

    if st.button("Predict"):
        desc_row = descriptors_df[descriptors_df['Name'] == ligand_input].drop('Name', axis=1)
        combined_input = np.hstack([desc_row.values[0], [vdw, ele, total]])
        pred = combined_model.predict([combined_input])[0]
        st.success(f"Prediction: {'Active' if pred==1 else 'Inactive'}")
        display_admet_profile(ligand_input)

elif mode == "🛠️ Manual Input":
    st.header("🛠️ Predict Using Manual Input of All Features")
    feature_names = joblib.load("feature_names_final.sav")
    inputs = []
    for feat in feature_names:
        val = st.number_input(feat, value=0.0)
        inputs.append(val)
    ligand_input = st.text_input("Ligand Name (optional, for ADMET)")

    if st.button("Predict"):
        pred = manual_model.predict([inputs])[0]
        st.success(f"Prediction: {'Active' if pred==1 else 'Inactive'}")
        if ligand_input:
            display_admet_profile(ligand_input)

elif mode == "🪄 Magic Mode":
    st.header("🪄 Magic Mode - Auto Predict")
    uploaded_file = st.file_uploader("Upload a CSV file with Docking Energies and Descriptors")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        ligand_input = st.text_input("Ligand Name (optional, for ADMET)")

        if st.button("Run Magic Prediction"):
            try:
                if set(feature_names).issubset(df.columns):
                    X = df[feature_names]
                    preds = manual_model.predict(X)
                    df['Prediction'] = ["Active" if p == 1 else "Inactive" for p in preds]
                    st.success("✅ Prediction Complete!")
                    st.dataframe(df)
                else:
                    st.error("❌ Required features missing in uploaded file.")

                if ligand_input:
                    display_admet_profile(ligand_input)

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
