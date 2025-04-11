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
        body {
            background-color: #0f0f0f;
            color: #e0e0e0;
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #0f0f0f;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .css-1v0mbdj, .stButton>button {
            background-color: #222;
            color: white;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1a1a1a;
            transform: scale(1.03);
        }
        .suggestion-card {
            background-color: #1c1c1c;
            padding: 1rem;
            border-left: 5px solid #06d6a0;
            margin-top: 20px;
            border-radius: 10px;
        }
        .dna-icon {
            height: 50px;
            margin-right: 10px;
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
