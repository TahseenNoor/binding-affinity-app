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

# ------------------------ LOAD BACKGROUND IMAGE & CONVERT TO BASE64 ------------------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64("image.png")

# ------------------------ CUSTOM CSS WITH EMBEDDED IMAGE ------------------------
st.markdown(f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    font-family: 'Palatino Linotype', serif;
    background-image: url("data:image/png;base64,{img_base64}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
    color: black !important;
}}

[data-testid="stAppViewContainer"] {{
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 15px;
    display: flex;
    flex-direction: column;
    align-items: center;
}}

h1, h2, h3, h4 {{
    color: #2c2c2c;
    font-family: 'Palatino Linotype', serif;
}}

.stButton>button {{
    background-color: #6a5acd;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.6em 1.5em;
    border: none;
}}
.stButton>button:hover {{
    background-color: #836fff;
    transform: scale(1.02);
}}

.suggestion-card {{
    background-color: #f8f8ff;
    padding: 1rem;
    border-left: 4px solid #6a5acd;
    border-radius: 10px;
    margin-top: 20px;
    color: black;
    box-shadow: 0 0 10px rgba(0,0,0,0.08);
}}

[data-testid="stMetric"] {{
    background-color: #fff !important;
    border-radius: 12px;
    padding: 10px;
}}

.prediction-highlight {{
    background-color: #eee;
    padding: 1rem;
    border-left: 5px solid #6a5acd;
    border-radius: 10px;
    margin: 1rem 0;
    font-size: 1.2rem;
    font-weight: bold;
    color: #2c2c2c;
}}

.content-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 3rem;
    width: 100%;
}}

.result-container {{
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
}}
</style>
""", unsafe_allow_html=True)

# ------------------------ LOAD MODEL AND DATA ------------------------
df = pd.read_csv("Cleaned_Autodock_Results.csv")
model = joblib.load("model_with_importance.pkl")

# ------------------------ HEADER ------------------------
st.markdown("# 🧬 Binding Affinity Predictor")
st.markdown("Welcome to Binding Affinity Predictor, the next-generation tool designed to accelerate drug discovery and enhance precision medicine. 🌍 In today’s fast-paced biotech world, understanding the interaction between proteins and compounds is critical to finding effective therapies. This AI-powered platform uses state-of-the-art machine learning models to predict the binding affinity between target proteins and various ligands, offering significant value to researchers, clinicians, and pharmaceutical companies working toward new drug development. 🧬💊.")
st.markdown("---")

# ------------------------ DESCRIPTION ------------------------
st.markdown("### Description:")
st.write("This tool predicts binding affinity between a target and compound using ML models. Designed for drug discovery researchers. Styled with biotech vibes.")

# ------------------------ INPUT SECTION ------------------------
st.markdown("---")
selected_pair = st.selectbox("Choose a Protein-Ligand Pair", df['PROTEIN-LIGAND'].unique())

# ------------------------ PREDICTION ------------------------
if st.button("🔬 Predict Binding Affinity"):
    try:
        row = df[df['PROTEIN-LIGAND'] == selected_pair]
        features = row[[ 'Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy' ]].fillna(0)

        prediction = model.predict(features)[0]
        st.markdown("### ✅ Prediction Result")
        st.markdown(
            f"<div class='prediction-highlight'>🧬 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>",
            unsafe_allow_html=True
        )

        importances = model.feature_importances_
        feature_names = features.columns
        feature_impact = dict(zip(feature_names, importances))

        st.markdown("<div class='suggestion-card'><h4>🧠 AI Suggestion:</h4>", unsafe_allow_html=True)
        for feat, score in feature_impact.items():
            st.markdown(f"<p>- <b>{feat}</b> is important in predicting the binding affinity. Adjust it for better results.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

         # Show feature importance in a simple table format
        st.markdown("### 🧠 Feature Importance:")
        feature_df = pd.DataFrame(list(feature_impact.items()), columns=['Feature', 'Importance'])
        st.dataframe(feature_df.sort_values(by='Importance', ascending=False))

        # Bar chart for feature importance
        st.markdown("### 📊 Feature Importance Visualization:")
        st.bar_chart(feature_df.set_index('Feature')['Importance'])


    except Exception as e:
        st.error(f"Something went wrong: {e}")
