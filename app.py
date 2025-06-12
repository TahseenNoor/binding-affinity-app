import streamlit as st
import pandas as pd
import joblib
import base64
import os
from difflib import get_close_matches

# ---------------------------- PAGE CONFIG ----------------------------
st.set_page_config(
    page_title="AFFERAZE",
    layout="wide",
    page_icon="🧬"
)

# ---------------------------- LOAD BACKGROUND IMAGE ----------------------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64("image.png")

# ---------------------------- CUSTOM CSS ----------------------------
st.markdown(f"""
    <style>
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Garamond', serif;
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Binding Affinity Predictor")

# ---------------------------- MODEL LOADING ----------------------------
model_path = os.path.join(os.path.dirname(__file__), "combined_model.pkl")
energy_model = joblib.load(model_path)

# ---------------------------- FILE UPLOAD ----------------------------
st.header("📤 Upload Your Files")

desc_file = st.file_uploader("Upload Descriptors CSV", type=["csv"])
energy_file = st.file_uploader("Upload AutoDock Results CSV", type=["csv"])

if desc_file and energy_file:
    try:
        desc_df = pd.read_csv(desc_file)
        energy_df = pd.read_csv(energy_file)

        # Rename ligand name column in descriptors
        if "Name" in desc_df.columns:
            name_col = "Name"
        elif "compounds" in desc_df.columns:
            name_col = "compounds"
        else:
            raise ValueError("❌ Could not find a column with ligand names in descriptor dataset.")

        desc_df.rename(columns={name_col: "Ligand"}, inplace=True)

        # Rename ligand column in energy
        if "LIGAND" in energy_df.columns:
            ligand_col = "LIGAND"
        elif "PROTEIN-LIGAND" in energy_df.columns:
            ligand_col = "PROTEIN-LIGAND"
        else:
            raise ValueError("❌ Could not find ligand name column in docking results.")

        energy_df.rename(columns={ligand_col: "Ligand"}, inplace=True)

        # Rename binding energy if necessary
        if "Binding Affinity" not in energy_df.columns and "binding energy" in energy_df.columns:
            energy_df.rename(columns={"binding energy": "Binding Affinity"}, inplace=True)

        # Keep necessary columns
        energy_cols = ['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']
        descriptor_cols = ['molecular weight', 'logp', 'rotatable bonds', 'acceptor', 'donor', 'surface area', 'molar refractivity']
        all_features = energy_cols + descriptor_cols

        energy_df = energy_df[['Ligand'] + energy_cols + ['Binding Affinity']]
        desc_df = desc_df[['Ligand'] + descriptor_cols]

        # Merge
        merged_df = pd.merge(energy_df, desc_df, on="Ligand", how="inner")

        st.success(f"✅ Merged {len(merged_df)} compounds successfully!")

        st.dataframe(merged_df)

        if st.button("🔮 Predict Binding Affinity"):
            X = merged_df[all_features]
            preds = energy_model.predict(X)
            merged_df["Predicted Affinity"] = preds
            st.subheader("📈 Results")
            st.dataframe(merged_df[["Ligand", "Binding Affinity", "Predicted Affinity"]])

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("👆 Please upload both descriptor and docking result CSV files.")
