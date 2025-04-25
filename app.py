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
df['PROTEIN-LIGAND'] = df['PROTEIN-LIGAND'].astype(str).str.strip().str.lower()

valid_proteins = ["stat", "ace", "mmp3", "tnf", "tlr4", "cyp27b1"]
df['PROTEIN'] = df['PROTEIN-LIGAND'].apply(lambda x: x.split('-')[0])
df['LIGAND'] = df['PROTEIN-LIGAND'].apply(lambda x: x.split('-')[1])
df = df[df['PROTEIN'].isin(valid_proteins)]

energy_model = joblib.load("model_with_importance.pkl")
descriptor_model = joblib.load("descriptor_model.pkl")

# Load descriptors and ADMET data
try:
    pharma_df = pd.read_csv("pharmokinetics final.csv")
    st.success("✅ ADMET data loaded successfully!")
except Exception as e:
    st.error(f"❌ Could not load ADMET data. Error: {e}")
    pharma_df = None

# ------------------------ HEADER ------------------------
st.markdown("# 🧬 AFFERAZE")
st.markdown("Predict binding affinity using **energy values** or **molecular descriptors** 💊")
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
            protein_input = protein_input.strip().lower()
            ligand_input = ligand_input.strip().lower()
            combined_input = f"{protein_input}-{ligand_input}"

            st.write(f"🔍 Looking for pair: {combined_input}")
            matching_row = df[df['PROTEIN-LIGAND'] == combined_input]

            if not matching_row.empty:
                features = matching_row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)
                prediction = energy_model.predict(features)[0]
                st.markdown(f"### 🧬 Best Matched Pair: {combined_input}")
                st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

                # Show ADMET if available
                if pharma_df is not None:
                    lig_match = pharma_df[pharma_df['Name'].str.lower() == ligand_input]
                    if not lig_match.empty:
                        st.markdown("### 🧪 ADMET Profile")
                        st.dataframe(lig_match)
                    else:
                        st.info("ℹ️ No ADMET profile found for this ligand.")

            else:
                st.error("❌ No exact match found.")
                potential_matches = get_close_matches(combined_input, df['PROTEIN-LIGAND'].tolist(), n=3, cutoff=0.6)
                if potential_matches:
                    st.markdown("### 💡 Did you mean:")
                    for match in potential_matches:
                        st.markdown(f"- {match}")
        else:
            st.error("Please enter both a Protein and Ligand name.")

# ------------------------ DESCRIPTOR MODE ------------------------
elif mode == "🧪 Use Molecular Descriptors":
    st.markdown("### 🧪 Enter Descriptor Values")
    d1 = st.number_input("Descriptor 1 (e.g., MolWt)")
    d2 = st.number_input("Descriptor 2 (e.g., LogP)")
    d3 = st.number_input("Descriptor 3 (e.g., TPSA)")

    if st.button("🧪 Predict via Descriptors"):
        features = pd.DataFrame([[d1, d2, d3]], columns=['Descriptor1', 'Descriptor2', 'Descriptor3'])
        prediction = descriptor_model.predict(features)[0]
        st.markdown(f"<div class='prediction-highlight'>🧪 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

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
        prediction = energy_model.predict(all_features)[0]
        st.markdown(f"<div class='prediction-highlight'>🔗 Combined Prediction: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

# ------------------------ MANUAL MODE ------------------------
elif mode == "🛠️ Manual Input (Energy Only, Any Names)":
    st.markdown("### 🛠️ Enter Energy Values for Any Protein-Ligand")
    e1 = st.number_input("Electrostatic Energy")
    e2 = st.number_input("Torsional Energy")
    e3 = st.number_input("vdw hb desolve Energy")
    e4 = st.number_input("Intermol Energy")

    if st.button("⚙️ Predict Binding Affinity"):
        features = pd.DataFrame([[e1, e2, e3, e4]], columns=['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy'])
        prediction = energy_model.predict(features)[0]
        st.markdown(f"<div class='prediction-highlight'>🛠️ Manual Prediction: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

# ------------------------ MAGIC MODE ------------------------
elif mode == "🪄 Magic Mode (Any Known Ligand)":
    st.markdown("### 🪄 Enter Any Ligand + Protein Name (flexible match)")
    protein_input = st.selectbox("Choose a Protein", ["STAT", "ACE", "MMP3", "TNF", "TLR4", "CYP27B1"])
    ligand_input = st.text_input("Enter Ligand Name:")

    if st.button("🪄 Predict Anyway!"):
        protein_input = protein_input.strip().lower()
        ligand_input = ligand_input.strip().lower()
        query_key = f"{protein_input}-{ligand_input}"

        # Try exact match first
        exact = df[df['PROTEIN-LIGAND'] == query_key]
        if not exact.empty:
            features = exact[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)
            prediction = energy_model.predict(features)[0]
            st.markdown(f"<div class='prediction-highlight'>✨ Magic Prediction (Exact): <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

            # ADMET display
            if pharma_df is not None:
                lig_match = pharma_df[pharma_df['Name'].str.lower() == ligand_input]
                if not lig_match.empty:
                    st.markdown("### 🧪 ADMET Profile")
                    st.dataframe(lig_match)
                else:
                    st.info("ℹ️ No ADMET profile found for this ligand.")

        else:
            st.warning("Exact match not found — searching by ligand only ⚡️")
            ligand_match = df[df['LIGAND'] == ligand_input]

            if not ligand_match.empty:
                row = ligand_match.iloc[0]
                features = row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0).to_frame().T
                prediction = energy_model.predict(features)[0]
                st.markdown(f"""
                    <div class='prediction-highlight'>
                        🔮 Used {row['PROTEIN-LIGAND']} instead<br>
                        ✨ Magic Prediction (Closest): <b>{prediction:.2f} kcal/mol</b>
                    </div>
                """, unsafe_allow_html=True)

                # ADMET for fallback
                if pharma_df is not None:
                    lig_match = pharma_df[pharma_df['Name'].str.lower() == ligand_input]
                    if not lig_match.empty:
                        st.markdown("### 🧪 ADMET Profile")
                        st.dataframe(lig_match)
                    else:
                        st.info("ℹ️ No ADMET profile found for this ligand.")

            else:
                st.error("🫥 Couldn't find any match for that ligand either.")
                # -------------------- UNIVERSAL ADMET DISPLAY --------------------
            if pharma_df is not None and ligand_input:
                admet_match = pharma_df[pharma_df['Name'].str.lower() == ligand_input.lower()]
                if not admet_match.empty:
                    st.markdown("### 🧪 ADMET Profile")
                    st.dataframe(admet_match)
                else:
                    st.info("ℹ️ No ADMET profile found for this ligand.")

