import streamlit as st
import pandas as pd
import joblib
import os
import re

# ------------------------ CONFIG ------------------------
st.set_page_config(page_title="Binding Affinity Predictor", layout="wide")
st.title("🔬 Binding Affinity Predictor")
st.markdown("Upload descriptor and docking energy data to predict binding affinity.")

# ------------------------ LOAD MODELS ------------------------
@st.cache_resource
def load_models():
    descriptor_model = joblib.load("model_with_importance.pkl")
    energy_model = joblib.load("combined_model.pkl")
    return descriptor_model, energy_model

descriptor_model, energy_model = load_models()

# ------------------------ FILE UPLOAD ------------------------
desc_file = st.file_uploader("📁 Upload descriptor file (CSV)", type="csv")
energy_file = st.file_uploader("📁 Upload docking energy file (CSV)", type="csv")

if desc_file and energy_file:
    try:
        # Read data
        desc_df = pd.read_csv(desc_file)
        energy_df = pd.read_csv(energy_file)

        # Clean descriptor data
        if "compounds" not in desc_df.columns:
            st.error("Descriptor file must contain a 'compounds' column.")
        else:
            desc_df.rename(columns={"compounds": "Ligand"}, inplace=True)

            # Clean energy data
            if "PROTEIN-LIGAND" not in energy_df.columns:
                st.error("Energy file must contain a 'PROTEIN-LIGAND' column.")
            else:
                # Extract ligand name
                energy_df["Ligand"] = energy_df["PROTEIN-LIGAND"].apply(lambda x: re.split("[-_]", x)[-1].strip().lower())
                desc_df["Ligand"] = desc_df["Ligand"].str.strip().str.lower()

                # Select relevant energy features
                energy_features = ["Electrostatic energy", "Torsional energy", "vdw hb desolve energy", "Intermol energy"]
                missing_cols = [col for col in energy_features if col not in energy_df.columns]

                if missing_cols:
                    st.error(f"Missing energy columns: {missing_cols}")
                else:
                    # Merge datasets
                    merged_df = pd.merge(desc_df, energy_df[["Ligand"] + energy_features], on="Ligand", how="inner")

                    # Predict
                    desc_features = desc_df.columns.drop("Ligand").tolist()
                    energy_feats = energy_features

                    desc_preds = descriptor_model.predict(merged_df[desc_features])
                    energy_preds = energy_model.predict(merged_df[energy_feats])

                    merged_df["Prediction (Descriptors)"] = desc_preds
                    merged_df["Prediction (Energies)"] = energy_preds
                    merged_df["Average Prediction"] = merged_df[["Prediction (Descriptors)", "Prediction (Energies)"]].mean(axis=1)

                    st.success("✅ Predictions generated successfully!")
                    st.dataframe(merged_df[["Ligand", "Average Prediction"] + desc_features + energy_feats])

                    csv = merged_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

else:
    st.warning("Please upload both descriptor and energy CSV files.")
