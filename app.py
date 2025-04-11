import streamlit as st
import pandas as pd
import joblib

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(
    page_title="Binding Affinity Predictor",
    layout="wide",
    page_icon="🧬"
)

# ------------------------ CUSTOM CSS ------------------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #001a00;
    background-image: 
        linear-gradient(135deg, rgba(0, 255, 128, 0.05) 1px, transparent 1px),
        linear-gradient(45deg, rgba(0, 255, 128, 0.05) 1px, transparent 1px);
    background-size: 50px 50px;
    color: white !important;
}

section.main > div {
    padding-top: 2rem;
}

.stButton>button {
    background-color: #00994d;
    color: white;
    border-radius: 12px;
    padding: 0.5em 1.5em;
    border: none;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #00cc66;
    transform: scale(1.03);
}

.suggestion-card {
    background-color: rgba(0, 0, 0, 0.4);
    padding: 1rem;
    border-left: 5px solid #33ff99;
    border-radius: 12px;
    margin-top: 20px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ------------------------ LOAD MODEL AND DATA ------------------------
df = pd.read_csv("Cleaned_Autodock_Results.csv")
model = joblib.load("model_with_importance.pkl")

# ------------------------ HEADER ------------------------
st.markdown("## 🧬 Binding Affinity Predictor")
st.markdown("This AI-powered tool predicts binding affinity between a target protein and a compound. Optimized for drug discovery research and enhanced with biotech visual aesthetics.")
st.markdown("---")

# ------------------------ LAYOUT ------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004496.png", width=80)
    selected_pair = st.selectbox("Choose a Protein-Ligand Pair", df['PROTEIN-LIGAND'].unique())
    if st.button("🔬 Predict Binding Affinity"):
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
            st.success(f"🧬 Predicted Binding Affinity: **{prediction:.2f} kcal/mol**")

            # Feature importance
            importances = model.feature_importances_
            feature_names = features.columns
            feature_impact = dict(zip(feature_names, importances))

            # AI suggestion card
            st.markdown("""
                <div class='suggestion-card'>
                    <h4>🧠 AI Suggestion:</h4>
            """, unsafe_allow_html=True)

            sorted_feats = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)
            for feat, score in sorted_feats:
                st.markdown(f"<p>- <b>{feat}</b> is highly influential. Try minimizing it to improve binding.</p>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Something went wrong: {e}")

with col2:
    st.markdown("### Description")
    st.write("This tool predicts binding affinity between a target and compound using ML models. "
             "Designed for drug discovery researchers. Styled with biotech vibes.")


