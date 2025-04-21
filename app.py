import streamlit as st
import pandas as pd
import joblib
import base64

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(page_title="AFFERAZE", layout="wide", page_icon="🧬")

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
    }}
    </style>
""", unsafe_allow_html=True)

# ------------------------ LOAD MODELS AND DATA ------------------------
df = pd.read_csv("Cleaned_Autodock_Results.csv")
energy_model = joblib.load("model_with_importance.pkl")

# ------------------------ HEADER ------------------------
st.title("🧬 AFFERAZE")
st.markdown("Manually enter **Protein** and **Ligand** names to predict binding affinity.")

# ------------------------ INPUT FIELDS ------------------------
protein_input = st.text_input("Enter Protein Name")
ligand_input = st.text_input("Enter Ligand Name")

if st.button("🔍 Predict Binding Affinity"):
    if not protein_input or not ligand_input:
        st.warning("Please enter both Protein and Ligand names.")
    else:
        pair_name = f"{protein_input.strip()} - {ligand_input.strip()}"
        row = df[df['PROTEIN-LIGAND'].str.strip() == pair_name]

        if row.empty:
            st.error(f"❌ The combination '{pair_name}' was not found in the dataset.")
        else:
            try:
                features = row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)
                prediction = energy_model.predict(features)[0]

                st.success(f"✅ Match found: {pair_name}")
                st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

                # Feature Importance
                if hasattr(energy_model, 'feature_importances_'):
                    importances = energy_model.feature_importances_
                    feature_names = features.columns
                    feature_impact = dict(zip(feature_names, importances))
                    feature_df = pd.DataFrame(list(feature_impact.items()), columns=['Feature', 'Importance'])

                    st.markdown("### 📊 Feature Importance")
                    st.dataframe(feature_df.style.format({"Importance": "{:.3f}"}), use_container_width=True)
                    st.bar_chart(feature_df.set_index("Feature"))

            except Exception as e:
                st.error(f"Something went wrong: {e}")
