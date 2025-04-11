import streamlit as st
import pandas as pd
import joblib

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(
    page_title="Binding Affinity Predictor",
    layout="wide",
    page_icon="🧬"
)

st.markdown("""
    <style>
        .main {
            background-image: url('https://wallpaperaccess.com/full/1923025.jpg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        body {
            color: #e0e0e0;
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            background-color: #007f5f;
            color: white;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #00684a;
            transform: scale(1.03);
        }
        .prediction-box {
            background: linear-gradient(90deg, #00ffc3, #00b8ff);
            padding: 1rem;
            font-size: 18px;
            font-weight: bold;
            color: #000000;
            border-radius: 12px;
            box-shadow: 0 0 12px rgba(0, 255, 204, 0.5);
            margin-bottom: 1rem;
        }
        .suggestion-card {
            background: linear-gradient(135deg, rgba(0,255,128,0.15), rgba(0,200,100,0.2));
            padding: 1rem;
            border-left: 5px solid #00ffcc;
            border-radius: 12px;
            margin-top: 20px;
            color: #e6ffe6;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)


# ------------------------ HEADER ------------------------
st.markdown("## 🧬 Binding Affinity Predictor")
st.markdown("---")

# ------------------------ LOAD DATA & MODEL ------------------------
df = pd.read_csv("Cleaned_Autodock_Results.csv")
model = joblib.load("model_with_importance.pkl")

# ------------------------ LAYOUT ------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004496.png", width=80)
    st.markdown("### Choose a Protein-Ligand Pair")
    selected_pair = st.selectbox("Select Pair", df['PROTEIN-LIGAND'].unique())
    predict_button = st.button("🔬 Predict Binding Affinity")

with col2:
    st.markdown("### Description")
    st.write("This AI-powered tool predicts binding affinity between a target protein and a compound. "
             "Optimized for drug discovery research and enhanced with visual design aesthetics.")

# ------------------------ PREDICTION OUTPUT ------------------------
if predict_button:
    try:
        row = df[df['PROTEIN-LIGAND'] == selected_pair]
        features = row[[
            'Electrostatic energy',
            'Torsional energy',
            'vdw hb desolve energy',
            'Intermol energy'
        ]].fillna(0)

        prediction = model.predict(features)[0]

        # Bright prediction box
        st.markdown(f"""
            <div class="prediction-box">
                🧬 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b>
            </div>
        """, unsafe_allow_html=True)

        # AI Suggestions
        importances = model.feature_importances_
        feature_names = features.columns
        feature_impact = dict(zip(feature_names, importances))

        st.markdown("""
            <div class='suggestion-card'>
                <h4>🧠 AI Suggestion:</h4>
        """, unsafe_allow_html=True)

        sorted_feats = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)
        for feat, score in sorted_feats:
            st.markdown(f"<p>• <b>{feat}</b> is highly influential. Try minimizing it to potentially improve binding.</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Something went wrong: {e}")



