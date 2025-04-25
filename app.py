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

# ------------------------ LOAD BACKGROUND IMAGE ------------------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64("image.png")

# ------------------------ CUSTOM CSS ------------------------
st.markdown(f"""
    <style>
    html, body {{
        font-family: 'Garamond', serif;
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    [data-testid="stAppViewContainer"] {{
        background-color: rgba(255, 255, 255, 0.88);
        padding: 2rem;
        border-radius: 15px;
    }}
    h1, h2, h3 {{
        color: #2c2c2c;
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
    .suggestion-card {{
        background-color: #f8f8ff;
        padding: 1rem;
        border-left: 4px solid #6a5acd;
        border-radius: 10px;
        margin-top: 20px;
        color: black;
        overflow-x: auto;
    }}
    .stButton>button {{
        background-color: #6a5acd;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }}
    .stButton>button:hover {{
        background-color: #5a4bc7;
    }}
    </style>
""", unsafe_allow_html=True)

# ------------------------ LOAD MODELS AND DATA ------------------------
df = pd.read_csv("Cleaned_Autodock_Results.csv")
df['PROTEIN-LIGAND'] = df['PROTEIN-LIGAND'].astype(str).str.strip().str.lower()

valid_proteins = ["stat", "ace", "mmp3", "tnf", "tlr4", "cyp27b1"]
df['PROTEIN'] = df['PROTEIN-LIGAND'].apply(lambda x: x.split('-')[0])
df['LIGAND'] = df['PROTEIN-LIGAND'].apply(lambda x: x.split('-')[1])
df = df[df['PROTEIN'].isin(valid_proteins)]

energy_model = joblib.load("model_with_importance.pkl")
descriptor_model = joblib.load("descriptor_model.pkl")
combined_model = joblib.load("combined_model.pkl")  # Ensure this exists

# ------------------------ HEADER ------------------------
st.markdown("<h1 style='text-align: center;'>🧬 AFFERAZE</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Predict Binding Affinity using AI Models (Energy + Descriptors)</h3>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------ MODE SELECTOR ------------------------
mode = st.radio("Choose Prediction Mode:", [
    "🔬 Use Docking Energy Values",
    "🧪 Use Molecular Descriptors",
    "🧬 Combined Input (Descriptors + Energy Values)",
    "🛠️ Manual Input (Energy Only, Any Names)",
    "🪄 Magic Mode (Any Known Ligand)"
])

# ------------------------ ENERGY MODE ------------------------
if mode == "🔬 Use Docking Energy Values":
    st.markdown("### 🔍 Select Protein and Enter Ligand")
    protein_input = st.selectbox("Choose a Protein", ["STAT", "ACE", "MMP3", "TNF", "TLR4", "CYP27B1"])
    ligand_input = st.text_input("Enter Ligand Name:")

    if st.button("🔬 Predict Binding Affinity (from Dataset)"):
        if protein_input and ligand_input:
            key = f"{protein_input.strip().lower()}-{ligand_input.strip().lower()}"
            row = df[df['PROTEIN-LIGAND'] == key]

            if not row.empty:
                features = row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)
                pred = energy_model.predict(features)[0]
                st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{pred:.2f} kcal/mol</b></div>", unsafe_allow_html=True)
            else:
                st.error("No exact match found.")
                matches = get_close_matches(key, df['PROTEIN-LIGAND'].tolist(), n=3, cutoff=0.6)
                if matches:
                    st.markdown("### 💡 Did you mean:")
                    for m in matches:
                        st.markdown(f"- `{m}`")
        else:
            st.error("Please enter both Protein and Ligand.")

# ------------------------ DESCRIPTOR MODE ------------------------
elif mode == "🧪 Use Molecular Descriptors":
    st.markdown("### 🧪 Enter Descriptor Values")
    d1 = st.number_input("Descriptor 1 (e.g., MolWt)")
    d2 = st.number_input("Descriptor 2 (e.g., LogP)")
    d3 = st.number_input("Descriptor 3 (e.g., TPSA)")

    if st.button("🧪 Predict via Descriptors"):
        features = pd.DataFrame([[d1, d2, d3]], columns=['Descriptor1', 'Descriptor2', 'Descriptor3'])
        pred = descriptor_model.predict(features)[0]
        st.markdown(f"<div class='prediction-highlight'>🧪 Predicted Binding Affinity: <b>{pred:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

# ------------------------ COMBINED MODE ------------------------
elif mode == "🧬 Combined Input (Descriptors + Energy Values)":
    st.markdown("### 🧬 Enter Descriptors + Energy Values")
    d1 = st.number_input("Descriptor 1")
    d2 = st.number_input("Descriptor 2")
    d3 = st.number_input("Descriptor 3")
    e1 = st.number_input("Electrostatic Energy")
    e2 = st.number_input("Torsional Energy")
    e3 = st.number_input("vdw hb desolve Energy")
    e4 = st.number_input("Intermol Energy")

    if st.button("🧬 Predict via Combined Model"):
        all_features = pd.DataFrame([[d1, d2, d3, e1, e2, e3, e4]], columns=[
            'Descriptor1', 'Descriptor2', 'Descriptor3',
            'Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy'])
        pred = combined_model.predict(all_features)[0]
        st.markdown(f"<div class='prediction-highlight'>🔗 Combined Prediction: <b>{pred:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

# ------------------------ MANUAL MODE ------------------------
elif mode == "🛠️ Manual Input (Energy Only, Any Names)":
    st.markdown("### 🛠️ Enter Energy Values")
    e1 = st.number_input("Electrostatic Energy")
    e2 = st.number_input("Torsional Energy")
    e3 = st.number_input("vdw hb desolve Energy")
    e4 = st.number_input("Intermol Energy")

    if st.button("⚙️ Predict Binding Affinity"):
        features = pd.DataFrame([[e1, e2, e3, e4]], columns=[
            'Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy'])
        pred = energy_model.predict(features)[0]
        st.markdown(f"<div class='prediction-highlight'>🛠️ Manual Prediction: <b>{pred:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

# ------------------------ MAGIC MODE ------------------------
elif mode == "🪄 Magic Mode (Any Known Ligand)":
    st.markdown("### 🪄 Enter Protein and Ligand Name")
    protein_input = st.selectbox("Choose a Protein", ["STAT", "ACE", "MMP3", "TNF", "TLR4", "CYP27B1"])
    ligand_input = st.text_input("Enter Ligand Name:")

    if st.button("🪄 Predict Anyway!"):
        key = f"{protein_input.strip().lower()}-{ligand_input.strip().lower()}"
        match = df[df['PROTEIN-LIGAND'] == key]

        if not match.empty:
            features = match[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)
            pred = energy_model.predict(features)[0]
            st.markdown(f"<div class='prediction-highlight'>✨ Magic Prediction (Exact): <b>{pred:.2f} kcal/mol</b></div>", unsafe_allow_html=True)
        else:
            ligand_match = df[df['LIGAND'] == ligand_input.strip().lower()]
            if not ligand_match.empty:
                row = ligand_match.iloc[0]
                features = row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0).to_frame().T
                pred = energy_model.predict(features)[0]
                st.markdown(f"""
                    <div class='prediction-highlight'>
                        🔮 Used `{row['PROTEIN-LIGAND']}` instead<br>
                        ✨ Magic Prediction (Closest): <b>{pred:.2f} kcal/mol</b>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("🫥 Couldn't find any match for that ligand.")
