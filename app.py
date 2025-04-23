import streamlit as st
import pandas as pd
import joblib
import base64
import difflib

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
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Palatino Linotype', serif;
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
    
    h1, h2, h3, h4 {{
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
df['PROTEIN-LIGAND'] = df['PROTEIN-LIGAND'].str.strip().str.lower()
energy_model = joblib.load("model_with_importance.pkl")
descriptor_model = joblib.load("descriptor_model.pkl")

# ------------------------ HEADER ------------------------
st.markdown("# 🧬 AFFERAZE")
st.markdown("Predict binding affinity using **energy values** or **molecular descriptors** 💊")
st.markdown("---")

# ------------------------ MODE SELECTOR ------------------------
mode = st.radio("Choose Prediction Mode:", [
    "🔬 Use Docking Energy Values",
    "🧪 Use Molecular Descriptors",
    "🧬 Combined Input (Descriptors + Energy Values)",
    "🛠️ Manual Input (Energy Only, Any Names)"
])

# ------------------------ ENERGY MODE ------------------------
if mode == "🔬 Use Docking Energy Values":
    st.markdown("### 🔍 Enter Protein and Ligand Names")

    protein_filter = st.text_input("🔍 Filter Protein List (optional)").strip().lower()
    unique_proteins = sorted(set(p.split("-")[0] for p in df['PROTEIN-LIGAND']))
    if protein_filter:
        filtered_proteins = [p for p in unique_proteins if protein_filter in p]
    else:
        filtered_proteins = unique_proteins

    protein_input = st.selectbox("Select Protein", filtered_proteins)
    ligand_input = st.text_input("Enter Ligand Name").strip().lower()

    if st.button("🔬 Predict Binding Affinity (from Dataset)"):
        protein_input = protein_input.strip().lower()
        ligand_input = ligand_input.strip().lower()
        combined_input = f"{protein_input}-{ligand_input}".strip()

        st.write(f"🔍 Looking for pair: `{combined_input}`")
        matching_row = df[df['PROTEIN-LIGAND'] == combined_input]

        if not matching_row.empty:
            features = matching_row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)
            prediction = energy_model.predict(features)[0]
            st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)
        else:
            st.error("❌ No exact match found.")
            # Suggest closest matches
            close_matches = difflib.get_close_matches(combined_input, df['PROTEIN-LIGAND'].tolist(), n=5, cutoff=0.6)
            if close_matches:
                st.markdown("### 🔎 Did you mean one of these?")
                for match in close_matches:
                    st.markdown(f"- `{match}`")

# ------------------------ OTHER MODES (same as before — unchanged) ------------------------
# Descriptor mode, Combined mode, Manual mode
# Leave them as-is unless you want improvements there too.
