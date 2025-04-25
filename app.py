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
    font-family: 'Garamond', serif;
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
}}

h1, h2, h3, h4 {{
    color: #2c2c2c;
    font-family: 'Palatino Linotyp', serif;
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
</style>
""", unsafe_allow_html=True)

# ------------------------ LOAD MODEL AND DATA ------------------------
df = pd.read_csv("Cleaned_Autodock_Results.csv")
model = joblib.load("model_with_importance.pkl")

# Load ADMET data
admet_df = pd.read_csv("pharmokinetics final.csv", encoding='ISO-8859-1')
admet_df['Compound'] = admet_df['Compound'].astype(str).str.strip().str.upper()

# Add ligand column for matching
df['Ligand'] = df['PROTEIN-LIGAND'].apply(lambda x: x.split('-')[-1].strip().upper())

# ------------------------ HEADER ------------------------
st.markdown("# 🧬 AFFERAZE")
st.markdown("This AI-powered tool predicts binding affinity between a target protein and a compound. Optimized for drug discovery research and enhanced with biotech visual aesthetics.")
st.markdown("---")

# ------------------------ LAYOUT ------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004496.png", width=80)
    selected_pair = st.selectbox("Choose a Protein-Ligand Pair", df['PROTEIN-LIGAND'].unique())
    ligand_name = selected_pair.split('-')[-1].strip().upper()

    st.markdown("### 🧪 ADMET Profile Lookup (Auto)")
    matches = admet_df[admet_df['Compound'].str.contains(ligand_name[:5], na=False)]
    if not matches.empty:
        st.dataframe(matches.head())
    else:
        st.info("No matching ADMET data found.")

    st.markdown("#### 🔍 Debug Info")
    st.write(f"Looking for ADMET data with compound name: {ligand_name}")
    st.write("Top 5 matching candidates from ADMET dataset above.")

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
            st.markdown(
                f"<div class='prediction-highlight'>🧬 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>",
                unsafe_allow_html=True
            )

            # Show full ADMET if available
            ligand_admet = admet_df[admet_df['Compound'] == ligand_name]
            if not ligand_admet.empty:
                st.markdown("### 🧪 Full ADMET Profile")
                st.dataframe(ligand_admet.drop(columns=["Compound"]), use_container_width=True)

            # Feature importance
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
    st.write("This tool predicts binding affinity between a target and compound using ML models. "
             "Designed for drug discovery researchers. Styled with biotech vibes.")
    
    st.markdown("---")
    st.markdown("""
    Predicting the binding affinity between genes and compounds is crucial 🧬 for drug discovery and precision medicine,
    as it helps identify which compounds may effectively target specific genes 🧪. Typically, a threshold binding affinity
    value—often expressed as a dissociation constant (Kd) or binding free energy (ΔG)—is used to evaluate the strength of
    interaction 💥. A high affinity (e.g., Kd < 1μM) indicates strong binding and potential therapeutic value 💊,
    while a low affinity (e.g., Kd > 10μM) may suggest weak or non-specific interactions 🚫. If the predicted affinity is above the threshold
    (weaker binding), the compound may need optimization through structural modification or may be discarded from further testing 🔧.
    Conversely, if the affinity is below the threshold (stronger binding), the compound can be prioritized for in vitro or in vivo validation 🧪.
    This predictive step accelerates the drug development process ⏩ and reduces the cost of experimental screening 📉, making it a key tool
    in computational biology and cheminformatics 🖥️.
    """)
