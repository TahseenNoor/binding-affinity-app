import streamlit as st
import pandas as pd
import joblib

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="Binding Affinity Predictor", layout="wide", page_icon="🧬")

# ---------------------- CUSTOM BIOTECH STYLING ----------------------
st.markdown("""
<style>
/* Biotech background grid */
.stApp {
    background: linear-gradient(135deg, #004d00, #003300);
    background-image: radial-gradient(circle, rgba(0,255,128,0.1) 1px, transparent 1px),
                      radial-gradient(circle, rgba(0,255,128,0.1) 1px, transparent 1px);
    background-size: 40px 40px;
    background-position: 0 0, 20px 20px;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Buttons */
.stButton>button {
    background-color: #007a33;
    color: white;
    border-radius: 10px;
    padding: 0.5em 1.5em;
    border: none;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #005f26;
    transform: scale(1.05);
}

/* Success card */
.stAlert.success {
    background-color: rgba(0, 255, 100, 0.1);
    border-left: 4px solid #00e676;
    color: #b9ffcc;
}

/* Suggestion card */
.suggestion-card {
    background-color: rgba(0, 0, 0, 0.4);
    padding: 1rem;
    border-left: 5px solid #33ff99;
    border-radius: 12px;
    margin-top: 20px;
    color: white;
}

h4 {
    color: #66ffcc;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- LOAD MODEL + DATA ----------------------
df = pd.read_csv("Cleaned_Autodock_Results.csv")
model = joblib.load("model_with_importance.pkl")

# ---------------------- HEADER ----------------------
st.markdown("## 🧬 Binding Affinity Predictor + Smart Suggestions")
st.markdown("---")

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### Choose a Protein-Ligand Pair")
    selected_pair = st.selectbox("Select Pair", df['PROTEIN-LIGAND'].unique())
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

            # Feature importance suggestions
            importances = model.feature_importances_
            feature_names = features.columns
            sorted_feats = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

            st.markdown("### 💡 AI Suggestion:")
            for feat, _ in sorted_feats:
                st.markdown(f"""
                    <div class='suggestion-card'>
                        <h4>🧠 Tip:</h4>
                        <p><b>{feat}</b> is highly influential. Try minimizing it to potentially improve binding.</p>
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Something went wrong: {e}")

with col2:
    st.markdown("### Description")
    st.write("This AI-powered tool predicts binding affinity between a target protein and a compound. "
             "Optimized for drug discovery research and enhanced with visual design aesthetics.")

