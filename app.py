import streamlit as st
import pandas as pd
import joblib
import base64
from difflib import get_close_matches

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(
    page_title="AFFERAZE - Binding Affinity Predictor",
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

# ------------------------ HEADER ------------------------
st.title("🧬 AFFERAZE")
st.markdown("### Predict binding affinity using **energy values** or **molecular descriptors** 💊")
st.markdown("---")

# ------------------------ ADMET Analysis Section ------------------------

st.markdown("---")
st.markdown("<h2 style='text-align: center;'>📊 ADMET Analysis (Absorption, Distribution, Metabolism, Excretion, Toxicity)</h2>", unsafe_allow_html=True)

try:
    # Load your local files
    descriptors_df = pd.read_csv("descriptors final.csv")
    pharma_df = pd.read_csv("pharmokinetics final.csv")
    
    # Merge them based on the 'Name' column (or whichever column matches)
    admet_df = pd.merge(descriptors_df, pharma_df, on="Name", how="inner")
    
    # Display the ADMET table inside an expander
    with st.expander("🔍 View ADMET Data Table"):
        st.dataframe(admet_df, use_container_width=True)
        
except Exception as e:
    st.error(f"❌ Could not load ADMET data. Error: {e}")



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

            st.write(f"🔍 Looking for pair: `{combined_input}`")
            matching_row = df[df['PROTEIN-LIGAND'] == combined_input]

            if not matching_row.empty:
                features = matching_row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)
                prediction = energy_model.predict(features)[0]
                st.markdown(f"### 🧬 Best Matched Pair: `{combined_input}`")
                st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)
            else:
                st.error("❌ No exact match found.")
                potential_matches = get_close_matches(combined_input, df['PROTEIN-LIGAND'].tolist(), n=3, cutoff=0.6)
                if potential_matches:
                    st.markdown("### 💡 Did you mean...")
                    for match in potential_matches:
                        st.write(f"- {match}")
