import streamlit as st
import pandas as pd
import joblib
import base64
from difflib import get_close_matches

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(
    page_title="AFFERAZE",
    layout="wide",
    page_icon="🧬"
)

# ------------------------ BACKGROUND IMAGE ------------------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

try:
    img_base64 = get_base64("image.png")
except:
    img_base64 = ""

st.markdown(f"""
    <style>
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Garamond', serif;
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-attachment: fixed;
        height: 100vh;
        overflow-y: scroll;
    }}
    [data-testid="stAppViewContainer"] {{
        background-color: rgba(255, 255, 255, 0.88);
        padding: 2rem;
        border-radius: 15px;
        align-items: center;
    }}
    .prediction-highlight {{
        background-color: #eee;
        padding: 1rem;
        border-left: 5px solid #6a5acd;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c2c2c;
    }}
    </style>
""", unsafe_allow_html=True)

# ------------------------ LOAD DATA ------------------------
df = pd.read_csv("Cleaned_Autodock_Results.csv")
df['PROTEIN-LIGAND'] = df['PROTEIN-LIGAND'].astype(str).str.strip().str.lower()

valid_proteins = ["stat", "ace", "mmp3", "tnf", "tlr4", "cyp27b1"]
df['PROTEIN'] = df['PROTEIN-LIGAND'].apply(lambda x: x.split('-')[0])
df['LIGAND'] = df['PROTEIN-LIGAND'].apply(lambda x: x.split('-')[1])
df = df[df['PROTEIN'].isin(valid_proteins)]

# Load descriptor CSV
desc_df = pd.read_csv("pharmokinetics final.csv")
desc_df.columns = desc_df.columns.str.strip().str.lower()
desc_df.rename(columns={"compounds": "ligand"}, inplace=True)
desc_df['ligand'] = desc_df['ligand'].astype(str).str.strip().str.lower()

# Merge energy and descriptor data
energy_df = df.copy()
energy_df['LIGAND'] = energy_df['LIGAND'].astype(str).str.strip().str.lower()
energy_df.rename(columns={"binding energy": "Binding Affinity"}, inplace=True)

try:
    combined_df = pd.merge(energy_df, desc_df, left_on="LIGAND", right_on="ligand")
except:
    combined_df = None

# ------------------------ LOAD MODELS ------------------------
energy_model = joblib.load("model_with_importance .pkl")
descriptor_model = joblib.load("descriptor_model .pkl")
combined_model = joblib.load("combined_model .pkl")

# ------------------------ HEADER ------------------------
st.markdown("# 🧬 AFFERAZE")
st.markdown("Predict binding affinity using **energy values** or **molecular descriptors** 💊")
st.markdown("---")

# ------------------------ MODE SELECTOR ------------------------
mode = st.radio("Choose Prediction Mode:", [
    "🔬 Use Docking Energy Values",
    "🧪 Use Molecular Descriptors",
    "🧬 Combined Input (Descriptors + Energy Values)"
])

if mode == "🔬 Use Docking Energy Values":
    st.markdown("### 🔍 Select Protein and Enter Ligand")
    protein_input = st.selectbox("Choose a Protein", ["STAT", "ACE", "MMP3", "TNF", "TLR4", "CYP27B1"])
    ligand_input = st.text_input("Enter Ligand Name:")

    if st.button("🔬 Predict Binding Affinity"):
        protein = protein_input.lower().strip()
        ligand = ligand_input.lower().strip()
        key = f"{protein}-{ligand}"

        match = df[df['PROTEIN-LIGAND'] == key]
        if not match.empty:
            features = match[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']]
            features = features[energy_model.feature_names_in_]
            pred = energy_model.predict(features)[0]
            st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{pred:.2f} kcal/mol</b></div>", unsafe_allow_html=True)
        else:
            st.error("❌ No exact match found.")

elif mode == "🧪 Use Molecular Descriptors":
    st.markdown("### 🧪 Enter Descriptor Values")
    input_vals = []
    for feature in descriptor_model.feature_names_in_:
        val = st.number_input(feature)
        input_vals.append(val)

    if st.button("🧪 Predict via Descriptors"):
        df_input = pd.DataFrame([input_vals], columns=descriptor_model.feature_names_in_)
        pred = descriptor_model.predict(df_input)[0]
        st.markdown(f"<div class='prediction-highlight'>🧪 Predicted Binding Affinity: <b>{pred:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

elif mode == "🧬 Combined Input (Descriptors + Energy Values)":
    st.markdown("### 🧬 Enter All Values")
    energy_inputs = []
    for col in ['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']:
        energy_inputs.append(st.number_input(col))

    descriptor_inputs = []
    for feature in [f for f in combined_model.feature_names_in_ if f not in ['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']]:
        descriptor_inputs.append(st.number_input(feature))

    total_input = energy_inputs + descriptor_inputs

    if st.button("🧬 Predict via Combined Model"):
        df_comb = pd.DataFrame([total_input], columns=combined_model.feature_names_in_)
        pred = combined_model.predict(df_comb)[0]
        st.markdown(f"<div class='prediction-highlight'>📊 Combined Prediction: <b>{pred:.2f} kcal/mol</b></div>", unsafe_allow_html=True)
