import streamlit as st
import pandas as pd
import joblib
import base64
import numpy as np

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

# ------------------------ LOAD DATA & MODELS ------------------------
df = pd.read_csv("Cleaned_Autodock_Results.csv")
energy_model = joblib.load("model_with_importance.pkl")
descriptor_model = joblib.load("descriptor_model.pkl")
combined_model = joblib.load("combined_model.pkl")  # <- load your combined model

# Mapping
anon_map = {}
reverse_map = {}
for i, real_name in enumerate(df['PROTEIN-LIGAND']):
    anon_name = f"🧬 Protein {chr(65 + i)} + Ligand {chr(88 + (i % 3))}"
    anon_map[anon_name] = real_name
    reverse_map[real_name] = anon_name
df['Anon Name'] = df['PROTEIN-LIGAND'].map(reverse_map)

# ------------------------ FEATURE STATS FOR DYNAMIC SUGGESTIONS ------------------------
energy_cols = ['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']
descriptor_cols = ['molar refractivity', 'molecular weight', 'acceptor', 'logp']
combined_cols = descriptor_cols + energy_cols

all_stats = df[combined_cols].describe()

# ------------------------ AI SUGGESTION GENERATOR ------------------------
def generate_dynamic_suggestions(input_row, feature_importance, mode='energy'):
    tone_options = [
        "Technical", "Encouraging", "Casual", "Experimental"
    ]
    motivational_quotes = [
        "🧠 Keep optimizing — small changes lead to big gains!",
        "🚀 You’re one tweak away from breakthrough results.",
        "🔬 The chemistry is in your hands.",
        "💡 Science is about iteration. You’re doing great!"
    ]

    output = ""
    tone = np.random.choice(tone_options)
    quote = np.random.choice(motivational_quotes)

    impact_sorted = sorted(feature_importance.items(), key=lambda x: -x[1])
    top_feats = [f for f, _ in impact_sorted[:3]]

    output += f"<div class='suggestion-card'><h4>🧠 AI Suggestions ({tone} Mode):</h4>"

    for feat in top_feats:
        val = input_row[feat].values[0]
        mean = all_stats.at[feat, 'mean']
        q1 = all_stats.at[feat, '25%']
        q3 = all_stats.at[feat, '75%']

        if val > q3:
            phrasing = f"- <b>{feat}</b> is relatively high. Consider reducing it to enhance binding efficiency."
        elif val < q1:
            phrasing = f"- <b>{feat}</b> is quite low. This may weaken the interaction. Explore increasing it slightly."
        else:
            phrasing = f"- <b>{feat}</b> is within a moderate range. Keep monitoring its influence."

        output += f"<p>{phrasing}</p>"

    output += f"<p><i>{quote}</i></p></div>"
    return output

# ------------------------ HEADER ------------------------
st.markdown("# 🧬 AFFERAZE")
st.markdown("Predict binding affinity using **energy values**, **molecular descriptors**, or **both** 💊")
st.markdown("---")

# ------------------------ MODE SELECTOR ------------------------
mode = st.radio("Choose Prediction Mode:", [
    "🔬 Use Docking Energy Values",
    "🧪 Use Molecular Descriptors",
    "🧬 Combined Input (Descriptors + Energy Values)"
])

# ------------------------ ENERGY MODE ------------------------
if mode == "🔬 Use Docking Energy Values":
    st.markdown("### 🔍 Select a Protein-Ligand Pair")
    selected_name = st.selectbox("Choose a Protein-Ligand Pair", df['Anon Name'].unique())

    if st.button("🔬 Predict Binding Affinity"):
        real_name = anon_map[selected_name]
        row = df[df['PROTEIN-LIGAND'] == real_name]
        features = row[energy_cols].fillna(0)
        prediction = energy_model.predict(features)[0]

        st.markdown(f"### 🧬 Real Pair: `{real_name}`")
        st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

        importances = energy_model.feature_importances_
        feature_impact = dict(zip(features.columns, importances))

        st.markdown("### 📊 Feature Importance")
        st.dataframe(pd.DataFrame(feature_impact.items(), columns=['Feature', 'Importance']).style.format({"Importance": "{:.3f}"}))
        st.bar_chart(pd.DataFrame(feature_impact, index=["Importance"]).T)

        st.markdown(generate_dynamic_suggestions(features, feature_impact), unsafe_allow_html=True)

# ------------------------ DESCRIPTOR MODE ------------------------
elif mode == "🧪 Use Molecular Descriptors":
    st.markdown("### ✍️ Enter Molecular Descriptors")
    prot_name = st.text_input("Protein Name (optional)", "")
    lig_name = st.text_input("Ligand Name", "")
    mw = st.number_input("Molecular Weight", value=0.0)
    mr = st.number_input("Molar Refractivity", value=0.0)
    logp = st.number_input("LogP (octanol-water)", value=0.0)
    acc = st.number_input("Number of H-Bond Acceptors", value=0.0)

    if st.button("🔮 Predict Binding Affinity"):
        features = pd.DataFrame([[mr, mw, acc, logp]], columns=descriptor_cols)
        prediction = descriptor_model.predict(features)[0]

        st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

        importances = descriptor_model.feature_importances_
        feature_impact = dict(zip(features.columns, importances))

        st.dataframe(pd.DataFrame(feature_impact.items(), columns=['Feature', 'Importance']).style.format({"Importance": "{:.3f}"}))
        st.bar_chart(pd.DataFrame(feature_impact, index=["Importance"]).T)

        st.markdown(generate_dynamic_suggestions(features, feature_impact), unsafe_allow_html=True)

# ------------------------ COMBINED MODE ------------------------
elif mode == "🧬 Combined Input (Descriptors + Energy Values)":
    st.markdown("### 🧪 Enter All Features")
    mr = st.number_input("Molar Refractivity", value=0.0)
    mw = st.number_input("Molecular Weight", value=0.0)
    acc = st.number_input("H-Bond Acceptors", value=0.0)
    logp = st.number_input("LogP", value=0.0)
    electro = st.number_input("Electrostatic Energy", value=0.0)
    torsion = st.number_input("Torsional Energy", value=0.0)
    vdw = st.number_input("VDW/HB/Desolvation Energy", value=0.0)
    intermol = st.number_input("Intermolecular Energy", value=0.0)

    if st.button("🔬 Predict Combined Binding Affinity"):
        row = pd.DataFrame([[mr, mw, acc, logp, electro, torsion, vdw, intermol]], columns=combined_cols)
        prediction = combined_model.predict(row)[0]

        st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

        importances = combined_model.feature_importances_
        feature_impact = dict(zip(row.columns, importances))

        st.dataframe(pd.DataFrame(feature_impact.items(), columns=['Feature', 'Importance']).style.format({"Importance": "{:.3f}"}))
        st.bar_chart(pd.DataFrame(feature_impact, index=["Importance"]).T)

        st.markdown(generate_dynamic_suggestions(row, feature_impact), unsafe_allow_html=True)

# ------------------------ FOOTER ------------------------
st.markdown("---")
st.caption("🧠 Powered by Machine Learning | Created with ❤️ for biotech research.")
