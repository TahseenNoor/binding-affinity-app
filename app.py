import streamlit as st
import pandas as pd
import joblib
import base64
from fuzzywuzzy import process

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

# ------------------------ LOAD MODELS AND DATA ------------------------
df = pd.read_csv("Cleaned_Autodock_Results.csv")
energy_model = joblib.load("model_with_importance.pkl")
descriptor_model = joblib.load("descriptor_model.pkl")

# ------------------------ HEADER ------------------------
st.markdown("# 🧬 AFFERAZE")
st.markdown("Predict binding affinity using **energy values** or **molecular descriptors** 💊")
st.markdown("---")

# ------------------------ MODE SELECTOR ------------------------
mode = st.radio("Choose Prediction Mode:", [
    "🔬 Use Docking Energy Values",
    "🧪 Use Molecular Descriptors",
    "🧬 Combined Input (Descriptors + Energy Values)"
])

# ------------------------ ENERGY MODE ------------------------
if mode == "🔬 Use Docking Energy Values":
    st.markdown("### 🔍 Enter Protein and Ligand Names")
    protein_input = st.text_input("Protein Name")
    ligand_input = st.text_input("Ligand Name")

    if st.button("🔬 Predict Binding Affinity"):
        combined_input = f"{protein_input.strip()}_{ligand_input.strip()}"
        best_match, score = process.extractOne(combined_input, df['PROTEIN-LIGAND'])

        if score > 60:
            row = df[df['PROTEIN-LIGAND'] == best_match]
            features = row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)
            prediction = energy_model.predict(features)[0]

            st.markdown(f"### 🧬 Matched Pair: `{best_match}`")
            st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

            if hasattr(energy_model, 'feature_importances_'):
                importances = energy_model.feature_importances_
                feature_df = pd.DataFrame({
                    'Feature': features.columns,
                    'Importance': importances
                })
                st.markdown("### 📊 Feature Importance Table")
                st.dataframe(feature_df.style.format({"Importance": "{:.3f}"}), use_container_width=True)
                st.markdown("### 📈 Feature Importance Chart")
                st.bar_chart(feature_df.set_index("Feature"))

                st.markdown("<div class='suggestion-card'><h4>🧠 AI Suggestion:</h4>", unsafe_allow_html=True)
                for _, row in feature_df.iterrows():
                    st.markdown(f"<p>- <b>{row['Feature']}</b> plays a key role. Modifying this could improve outcomes.</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("No close match found for that protein-ligand combination.")

# ------------------------ DESCRIPTOR MODE ------------------------
elif mode == "🧪 Use Molecular Descriptors":
    st.markdown("### ✍️ Enter Molecular Descriptor Values")
    prot_name = st.text_input("Protein Name (optional)", "")
    lig_name = st.text_input("Ligand Name", "")

    mw = st.number_input("Molecular Weight", value=0.0)
    mr = st.number_input("Molar Refractivity", value=0.0)
    logp = st.number_input("LogP (octanol-water)", value=0.0)
    acc = st.number_input("Number of H-Bond Acceptors", value=0.0)

    if st.button("🔮 Predict Binding Affinity (Descriptors)"):
        features = pd.DataFrame([[mr, mw, acc, logp]], columns=['molar refractivity', 'molecular weight', 'acceptor', 'logp'])
        prediction = descriptor_model.predict(features)[0]

        if lig_name.strip():
            st.markdown(f"### 🧬 Input Ligand: `{lig_name}`")
        if prot_name.strip():
            st.markdown(f"### 🧬 Input Protein: `{prot_name}`")

        st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

        if hasattr(descriptor_model, 'feature_importances_'):
            importances = descriptor_model.feature_importances_
            feature_df = pd.DataFrame({
                'Feature': features.columns,
                'Importance': importances
            })

            st.markdown("### 📊 Feature Importance Table")
            st.dataframe(feature_df.style.format({"Importance": "{:.3f}"}), use_container_width=True)
            st.markdown("### 📈 Feature Importance Chart")
            st.bar_chart(feature_df.set_index("Feature"))

            st.markdown("<div class='suggestion-card'><h4>🧠 AI Suggestion:</h4>", unsafe_allow_html=True)
            for _, row in feature_df.iterrows():
                st.markdown(f"<p>- <b>{row['Feature']}</b> influences binding predictions. Check its value for optimization.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ------------------------ COMBINED MODE ------------------------
elif mode == "🧬 Combined Input (Descriptors + Energy Values)":
    st.markdown("### 🔬 Enter Values")
    protein_input = st.text_input("Protein Name")
    ligand_input = st.text_input("Ligand Name")
    mw = st.number_input("Molecular Weight", value=0.0)
    mr = st.number_input("Molar Refractivity", value=0.0)
    logp = st.number_input("LogP", value=0.0)
    acc = st.number_input("Number of H-Bond Acceptors", value=0.0)

    if st.button("🔬 Predict Combined Binding Affinity"):
        combined_input = f"{protein_input.strip()}_{ligand_input.strip()}"
        best_match, score = process.extractOne(combined_input, df['PROTEIN-LIGAND'])

        if score > 60:
            row = df[df['PROTEIN-LIGAND'] == best_match]
            energy_features = row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)

            combined_features = pd.DataFrame([[mr, mw, acc, logp] + energy_features.values[0].tolist()],
                                             columns=['molar refractivity', 'molecular weight', 'acceptor', 'logp'] + energy_features.columns.tolist())

            combined_model = energy_model
            prediction = combined_model.predict(combined_features)[0]

            st.markdown(f"### 🧬 Matched Pair: `{best_match}`")
            st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)
        else:
            st.warning("Could not find a close match for the entered names.")

# ------------------------ FOOTER ------------------------
st.markdown("---")
st.caption("🧠 Powered by Machine Learning | Created with ❤️ for biotech research.")
