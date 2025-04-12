import streamlit as st
import pandas as pd

st.set_page_config(page_title="Binding Affinity Predictor", layout="wide")

st.title("🔬 Binding Affinity Predictor Web App")

# Tabs for navigation
tabs = st.tabs(["Home", "Prediction", "ADMET Analysis", "About"])

# --- HOME TAB ---
with tabs[0]:
    st.header("Welcome!")
    st.write(
        """
        This web application allows you to analyze molecules for their potential binding affinity and ADMET properties.
        Upload your molecular data or explore the inbuilt datasets to get started!
        """
    )
    st.image("https://images.unsplash.com/photo-1581091870622-2c1f1f07d1d0", use_column_width=True)

# --- PREDICTION TAB ---
with tabs[1]:
    st.header("🧪 Predict Binding Affinity")
    uploaded_file = st.file_uploader("Upload your molecular descriptor CSV file", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.subheader("Preview of Uploaded Data")
            st.dataframe(data.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a CSV file to proceed.")

# --- ADMET ANALYSIS TAB ---
with tabs[2]:
    st.header("🧬 ADMET Analysis")

    # Molecule Properties and Druglikeness
    st.subheader("Molecule Properties and Druglikeness")
    try:
        descriptors_url = "https://github.com/TahseenNoor/binding-affinity-app/raw/refs/heads/main/descriptors%20final.csv"
        descriptors_df = pd.read_csv(descriptors_url)
        st.dataframe(descriptors_df)
    except Exception as e:
        st.error("Error loading descriptors data.")

    # Pharmacokinetics and Toxicity Prediction
    st.subheader("Pharmacokinetic Profile and Toxicity Prediction")
    try:
        pharmo_url = "https://github.com/TahseenNoor/binding-affinity-app/raw/refs/heads/main/pharmokinetics%20final.csv"
        pharmo_df = pd.read_csv(pharmo_url)
        st.dataframe(pharmo_df)
    except Exception as e:
        st.error("Error loading pharmacokinetics data.")

# --- ABOUT TAB ---
with tabs[3]:
    st.header("ℹ️ About This App")
    st.markdown(
        """
        This application was developed as a part of a research-oriented project to assist in the drug discovery pipeline by evaluating molecular properties, 
        predicting their binding affinity, and analyzing ADMET profiles.
        """
    )
    st.markdown("**Developer:** Tahseen Noor")
