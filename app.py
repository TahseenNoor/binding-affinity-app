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
    height: 100vh;
    overflow-y: scroll;
}}

[data-testid="stAppViewContainer"] {{
    background-color: rgba(255, 255, 255, 0.88);
    padding: 2rem;
    border-radius: 15px;
    align-items: center;
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
    overflow-x: auto;
}}

.stButton>button {{
    background-color: #6a5acd;
    color: white;
    padding: 0.5rem 1.5rem;
    border-radius: 5px;
    border: none;
    cursor: pointer;
}}

.stButton>button:hover {{
    background-color: #5a4bc7;
}}
</style>
""", unsafe_allow_html=True)

# ------------------------ LOAD MODEL AND DATA ------------------------
df = pd.read_csv("Cleaned_Autodock_Results.csv")
model = joblib.load("model_with_importance.pkl")

# ------------------------ Generate Anonymous Mapping ------------------------
anon_map = {}
reverse_map = {}
for i, real_name in enumerate(df['PROTEIN-LIGAND']):
    anon_name = f"🧬 Protein {chr(65 + i)} + Ligand {chr(88 + (i % 3))}"
    anon_map[anon_name] = real_name
    reverse_map[real_name] = anon_name
df['Anon Name'] = df['PROTEIN-LIGAND'].map(reverse_map)

# ------------------------ HEADER ------------------------
st.markdown("# 🧬 AFFERAZE")
st.markdown("Predict binding affinity between proteins and ligands using ML 💊")
st.markdown("---")

# ------------------------ MODE SELECTOR ------------------------
mode = st.radio("Choose Prediction Mode:", ["🔎 Select from Dataset", "🧪 Enter Custom Energy Values"])

# ------------------------ SELECT FROM EXISTING DATA ------------------------
if mode == "🔎 Select from Dataset":
    selected_anon = st.selectbox("Choose a Protein-Ligand Pair", df['Anon Name'].unique())

    if st.button("🔬 Predict Binding Affinity (from Dataset)"):
        try:
            real_name = anon_map[selected_anon]
            row = df[df['PROTEIN-LIGAND'] == real_name]
            features = row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)
            prediction = model.predict(features)[0]

            st.markdown(f"### 🧬 Real Pair: `{real_name}`")
            st.markdown(
                f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>",
                unsafe_allow_html=True
            )

            # Feature importance
            importances = model.feature_importances_
            feature_names = features.columns
            feature_impact = dict(zip(feature_names, importances))
            feature_df = pd.DataFrame(list(feature_impact.items()), columns=['Feature', 'Importance'])

            st.markdown("### 📊 Feature Importance Table")
            st.dataframe(feature_df.style.format({"Importance": "{:.3f}"}), use_container_width=True)

            st.markdown("### 📈 Feature Importance Chart")
            st.bar_chart(feature_df.set_index("Feature"))

            # AI Suggestion Card (added)
            st.markdown("<div class='suggestion-card'><h4>🧠 AI Suggestion:</h4>", unsafe_allow_html=True)
            for feat, score in feature_impact.items():
                st.markdown(f"<p>- <b>{feat}</b> is important in predicting the binding affinity. Adjust it for better results.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

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
        feature_impact = dict(zip(features.columns, importances))
        feature_df = pd.DataFrame(list(feature_impact.items()), columns=['Feature', 'Importance'])

        st.markdown("### 📊 Feature Importance Table")
        st.dataframe(feature_df.style.format({"Importance": "{:.3f}"}), use_container_width=True)

        st.markdown("### 📈 Feature Importance Chart")
        st.bar_chart(feature_df.set_index("Feature"))

        # AI Suggestion Card (added)
        st.markdown("<div class='suggestion-card'><h4>🧠 AI Suggestion:</h4>", unsafe_allow_html=True)
        for feat, score in feature_impact.items():
            st.markdown(f"<p>- <b>{feat}</b> is important in predicting the binding affinity. Adjust it for better results.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------ FOOTER ------------------------
st.markdown("---")
st.caption("🧠 Powered by Machine Learning | Created with ❤️ for biotech research.")
