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
/* Biotech-inspired background */
.stApp {
    background: linear-gradient(135deg, #004d00, #003300);
    background-image: radial-gradient(circle, rgba(0,255,128,0.1) 1px, transparent 1px),
                      radial-gradient(circle, rgba(0,255,128,0.1) 1px, transparent 1px);
    background-size: 40px 40px;
    background-position: 0 0, 20px 20px;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* General button and block styling */
.stButton>button {
    background-color: #007a33;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.5em 1.5em;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #005f26;
    transform: scale(1.05);
}

/* Custom card for suggestions */
.suggestion-card {
    background-color: rgba(0, 0, 0, 0.4);
    padding: 1rem;
    border-left: 5px solid #33ff99;
    border-radius: 12px;
    margin-top: 20px;
    color: white;
}

/* Custom green success */
.stAlert.success {
    background-color: rgba(0, 255, 100, 0.1);
    border-left: 4px solid #00e676;
    color: #b9ffcc;
}

h4 {
    color: #66ffcc;
}
</style>
""", unsafe_allow_html=True)



# ------------------------ LOAD DATA ------------------------
df = pd.read_csv("Cleaned_Autodock_Results.csv")
model = joblib.load("model_with_importance.pkl")

# ------------------------ HEADER ------------------------
st.markdown("## 🧬 Binding Affinity Predictor + Smart Suggestions")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004496.png", width=80)
    st.markdown("### Choose a Protein-Ligand Pair")
    selected_pair = st.selectbox("Select Pair", df['PROTEIN-LIGAND'].unique())

with col2:
    st.markdown("### Description")
    st.write("This AI-powered tool predicts binding affinity between a target protein and a compound. "
             "Optimized for drug discovery research and enhanced with visual design aesthetics.")    

# ------------------------ PREDICTION ------------------------
if st.button("🔍 Predict Binding Affinity"):
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
        st.success(f"🧪 Predicted Binding Affinity: **{prediction:.2f} kcal/mol**")

        # Feature importance
        importances = model.feature_importances_
        feature_names = features.columns
        feature_impact = dict(zip(feature_names, importances))

        # AI Suggestions
        st.markdown("""
            <div class='suggestion-card'>
                <h4>🧠 AI Suggestion:</h4>
        """, unsafe_allow_html=True)

        sorted_feats = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)
        for feat, score in sorted_feats:
            st.markdown(f"<p>- <strong>{feat}</strong> is highly influential. Consider minimizing it.</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
