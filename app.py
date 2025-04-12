import streamlit as st
import pandas as pd
import joblib
import base64
from PIL import Image
import os

# Load model
data = pd.read_csv("data.csv")
model = joblib.load("model.pkl")

# Background styling
def set_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set background
set_bg_from_local("bg.png")

# Sidebar and logo
st.sidebar.image("logo.png", width=120)

# Title and Description
st.markdown("""
    <h1 style='text-align: left;'>Description</h1>
    <p style='text-align: justify;'>
    This tool predicts binding affinity between a target and compound using ML models. Designed for drug discovery researchers. Styled with biotech vibes.
    </p>
    """, unsafe_allow_html=True)

st.markdown("""
    Predicting the binding affinity between genes and compounds is crucial 🔬 for drug discovery and precision medicine,
    as it helps identify which compounds may effectively target specific genes 🧬. Typically, a threshold binding affinity value—
    often expressed as a dissociation constant (Kd) or binding free energy (ΔG)—is used to evaluate the strength of interaction 💥.
    A high affinity (e.g., Kd <1μM) indicates strong binding and potential therapeutic value 💊, while a low affinity (e.g., Kd > 10μM)
    may suggest weak or non-specific interactions 🚫. If the predicted affinity is above the threshold (weaker binding), the compound
    may need optimization through structural modification or may be discarded from further testing 🧪. Conversely, if the affinity is
    below the threshold (stronger binding), the compound can be prioritized for in vitro or in vivo validation 🌱. This predictive
    step accelerates the drug development process ⏩ and reduces the cost of experimental screening 🧫, making it a key tool in
    computational biology and cheminformatics 💻.
    """, unsafe_allow_html=True)

# Dropdown for protein-ligand pair
pair = st.selectbox("Choose a Protein-Ligand Pair", data['Pair'])

# Prediction logic
if st.button("🔮 Predict Binding Affinity"):
    row = data[data['Pair'] == pair]
    if not row.empty:
        features = row.iloc[:, 2:].values
        prediction = model.predict(features)[0]
        st.success("✅ Prediction Result")
        st.info(f"🧬 **Predicted Binding Affinity:** {prediction:.2f} kcal/mol")
        st.subheader("🧠 AI Suggestion:")
        st.write("You may consider further lead optimization if affinity is above threshold.")
    else:
        st.warning("No data found for selected pair.")

# ADMET Analysis
with st.expander("ADMET Analysis"):
    tab1, tab2 = st.tabs([
        "Molecule Properties and Druglikeness",
        "Pharmacokinetic Profile and Toxicity Prediction"
    ])

    with tab1:
        st.subheader("Molecule Properties and Druglikeness")
        try:
            df1 = pd.read_excel("descriptors final.xlsx")
            st.dataframe(df1, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load descriptors file: {e}")

    with tab2:
        st.subheader("Pharmacokinetic Profile and Toxicity Prediction")
        try:
            df2 = pd.read_excel("pharmokinetics final.xlsx")
            st.dataframe(df2, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load pharmacokinetics file: {e}")
