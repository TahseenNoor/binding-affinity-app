import streamlit as st
import pandas as pd
import joblib
import base64

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
}}

[data-testid="stAppViewContainer"] {{
    background-color: rgba(255, 255, 255, 0.88);
    padding: 2rem;
    border-radius: 15px;
    max-height: 100vh;
    overflow-y: auto;
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
}}
</style>
""", unsafe_allow_html=True)

# ------------------------ LOAD MODEL AND DATA ------------------------
df = pd.read_csv("Cleaned_Autodock_Results.csv")
model = joblib.load("model_with_importance.pkl")

# ------------------------ ENCODE FAKE NAMES ------------------------
df = df.copy()
df["Custom Name"] = [f"🧬 Protein {chr(65+i)} + Ligand {chr(88+i)}" for i in range(len(df))]

# ------------------------ HEADER ------------------------
st.markdown("# 🧬 AFFERAZE")
st.markdown("Predict binding affinity between proteins and ligands using ML 💊")
st.markdown("---")

# ------------------------ MODE SELECTOR ------------------------
mode = st.radio("Choose Prediction Mode:", ["🔎 Select from Dataset", "🧪 Enter Custom Energy Values"])

# ------------------------ SELECT FROM EXISTING DATA ------------------------
if mode == "🔎 Select from Dataset":
    selection_dict = dict(zip(df["Custom Name"], df["PROTEIN-LIGAND"]))
    selected_label = st.selectbox("Choose a Protein-Ligand Pair", list(selection_dict.keys()))

    if st.button("🔬 Predict Binding Affinity (from Dataset)"):
        try:
            selected_real_name = selection_dict[selected_label]
            row = df[df['PROTEIN-LIGAND'] == selected_real_name]
            features = row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)
            prediction = model.predict(features)[0]

            st.markdown(f"### 🧬 Real Compound: `{selected_real_name}`")
            st.markdown(
                f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>",
                unsafe_allow_html=True
            )

            # Feature Importance Table
            importances = model.feature_importances_
            feature_df = pd.DataFrame({
                'Feature': features.columns,
                'Importance': importances
            }).sort_values(by="Importance", ascending=False)

            st.markdown("<div class='suggestion-card'><h4>🧠 Feature Importance:</h4></div>", unsafe_allow_html=True)
            st.dataframe(feature_df)

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ------------------------ CUSTOM INPUT SECTION ------------------------
else:
    st.markdown("### ✍️ Enter Custom Values")
    compound_name = st.text_input("Enter Compound Name (optional)", "")
    electro = st.number_input("Electrostatic energy", value=0.0)
    torsional = st.number_input("Torsional energy", value=0.0)
    vdw = st.number_input("VDW + HB + Desolvation energy", value=0.0)
    intermol = st.number_input("Intermolecular energy", value=0.0)

    if st.button("🔮 Predict Binding Affinity (Custom Input)"):
        features = pd.DataFrame([[electro, torsional, vdw, intermol]],
                                columns=['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy'])
        prediction = model.predict(features)[0]

        if compound_name.strip():
            st.markdown(f"### 🧬 Custom Compound: `{compound_name.strip()}`")

        st.markdown(
            f"<div class='prediction-highlight'>📊 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>",
            unsafe_allow_html=True
        )

        importances = model.feature_importances_
        feature_df = pd.DataFrame({
            'Feature': features.columns,
            'Importance': importances
        }).sort_values(by="Importance", ascending=False)

        st.markdown("### 📌 Feature Importance (Model Weights):")
        st.dataframe(feature_df)

# ------------------------ FOOTER ------------------------
st.markdown("---")
st.caption("🧠 Powered by Machine Learning | Created with ❤️ for biotech research.")
