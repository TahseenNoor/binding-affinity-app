import streamlit as st
import pandas as pd
import joblib

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(
    page_title="Binding Affinity Predictor + ADMET Analysis",
    layout="wide",
    page_icon="🧬"
)

# ------------------------ CUSTOM CSS ------------------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Palatino Linotype', serif;
    background: linear-gradient(to right, rgba(255,255,255,0.9), rgba(242,242,242,0.9)),
                url("https://i.imgur.com/VcGu1xR.jpeg");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    color: black !important;
}

h1, h2, h3, h4 {
    color: #2c2c2c;
    font-family: 'Palatino Linotype', serif;
}

.stButton>button {
    background-color: #6a5acd;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.6em 1.5em;
    border: none;
}
.stButton>button:hover {
    background-color: #836fff;
    transform: scale(1.02);
}

.suggestion-card {
    background-color: #f8f8ff;
    padding: 1rem;
    border-left: 4px solid #6a5acd;
    border-radius: 10px;
    margin-top: 20px;
    color: black;
    box-shadow: 0 0 10px rgba(0,0,0,0.08);
}

[data-testid="stMetric"] {
    background-color: #fff !important;
    border-radius: 12px;
    padding: 10px;
}

.prediction-highlight {
    background-color: #eee;
    padding: 1rem;
    border-left: 5px solid #6a5acd;
    border-radius: 10px;
    margin: 1rem 0;
    font-size: 1.2rem;
    font-weight: bold;
    color: #2c2c2c;
}
</style>
""", unsafe_allow_html=True)

# ------------------------ LOAD MODEL AND DATA ------------------------
df = pd.read_csv("Cleaned_Autodock_Results.csv")
model = joblib.load("model_with_importance.pkl")
descriptor_df = pd.read_excel("descriptors final.xlsx")
admet_df = pd.read_excel("pharmokinetics final.xlsx")

# ------------------------ LAYOUT ------------------------
tab1, tab2 = st.tabs(["\ud83d\udd2c Binding Affinity Prediction", "\ud83d\udc8a ADMET Analysis"])

# ------------------------ TAB 1: Binding Affinity ------------------------
with tab1:
    st.markdown("# 🧬 Binding Affinity Predictor")
    st.markdown("This AI-powered tool predicts binding affinity between a target protein and a compound. Optimized for drug discovery research and enhanced with biotech visual aesthetics.")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3004/3004496.png", width=80)
        selected_pair = st.selectbox("Choose a Protein-Ligand Pair", df['PROTEIN-LIGAND'].unique())

        if st.button("\ud83d\udd2c Predict Binding Affinity"):
            try:
                row = df[df['PROTEIN-LIGAND'] == selected_pair]
                features = row[[
                    'Electrostatic energy',
                    'Torsional energy',
                    'vdw hb desolve energy',
                    'Intermol energy'
                ]].fillna(0)

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
                sorted_feats = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)
                for feat, score in sorted_feats:
                    st.markdown(f"<p>- <b>{feat}</b> is highly influential. Try minimizing it to improve binding.</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Something went wrong: {e}")

    with col2:
        st.markdown("### Description")
        st.write("This tool predicts binding affinity between a target and compound using ML models. Designed for drug discovery researchers. Styled with biotech vibes.")
        st.markdown("---")
        st.markdown("""
        Predicting the binding affinity between genes and compounds is crucial 🧬 for drug discovery and precision medicine...
        """)

# ------------------------ TAB 2: ADMET ------------------------
with tab2:
    st.markdown("# 💊 ADMET Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌿 Molecule Properties and Druglikeness")
        selected_desc = st.selectbox("Select Compound (Descriptors)", descriptor_df.iloc[:, 0].unique())
        desc_row = descriptor_df[descriptor_df.iloc[:, 0] == selected_desc]

        if not desc_row.empty:
            for col in descriptor_df.columns[1:]:
                st.markdown(f"**{col}:** {desc_row.iloc[0][col]}")

    with col2:
        st.markdown("### 💉 Pharmacokinetic Profile and Toxicity Prediction")
        selected_cmp = st.selectbox("Select Compound (Pharmacokinetics)", admet_df['Compound'].unique())
        admet_row = admet_df[admet_df['Compound'] == selected_cmp]

        if not admet_row.empty:
            for col in admet_df.columns[1:]:
                val = admet_row.iloc[0][col]
                st.markdown(f"**{col}:** {val}")
        else:
            st.warning("Compound not found in ADMET dataset.")










