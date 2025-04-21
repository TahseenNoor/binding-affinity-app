import streamlit as st
import pandas as pd
import joblib
import base64
import random

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

# Generate mapping (keeping for now, might not need later)
anon_map = {}
reverse_map = {}
for i, real_name in enumerate(df['PROTEIN-LIGAND']):
    anon_name = f"🧬 Protein {chr(65 + i)} + Ligand {chr(88 + (i % 3))}"
    anon_map[anon_name] = real_name
    reverse_map[real_name] = anon_name
df['Anon Name'] = df['PROTEIN-LIGAND'].map(reverse_map)

# ------------------------ AI Suggestion Function ------------------------
def generate_ai_suggestions(feature_impact, energy_values=None, descriptors=None):
    suggestions = []

    # Dynamic thresholds and advice based on feature values
    for feat, score in feature_impact.items():
        # Threshold-based suggestions for energy-based features
        if feat == 'Electrostatic energy':
            if energy_values is not None and energy_values['Electrostatic energy'] < -3:
                suggestions.append("⚡ High electrostatic energy may reduce binding affinity. Consider adjusting this to improve binding.")
            else:
                suggestions.append("⚡ Electrostatic energy seems optimal. Further optimization may not yield significant results.")
        
        if feat == 'Torsional energy':
            if energy_values is not None and energy_values['Torsional energy'] < -2:
                suggestions.append("🔄 Low torsional energy is ideal for efficient binding. You’re on the right track!")
            else:
                suggestions.append("🔄 Torsional energy could be reduced further for better binding performance.")

        if feat == 'vdw hb desolve energy':
            if energy_values is not None and energy_values['vdw hb desolve energy'] > -4:
                suggestions.append("🔬 High van der Waals energy could affect binding negatively. Try lowering this for more favorable results.")
            else:
                suggestions.append("🔬 Your van der Waals energy is well-balanced. This can lead to a stronger binding affinity.")

        if feat == 'Intermol energy':
            if energy_values is not None and energy_values['Intermol energy'] < -5:
                suggestions.append("⚡ Very low intermolecular energy is excellent for binding. You may be very close to optimal.")
            else:
                suggestions.append("⚡ Try reducing the intermolecular energy to see an improvement in the binding affinity.")

        # Threshold-based suggestions for descriptor features
        if feat == 'molar refractivity':
            if descriptors is not None and descriptors['molar refractivity'] > 80:
                suggestions.append("🧪 High molar refractivity may increase hydrophobic interactions, which can lead to a stronger binding.")
            else:
                suggestions.append("🧪 Molar refractivity seems to be in the optimal range, but further exploration might help.")

        if feat == 'molecular weight':
            if descriptors is not None and descriptors['molecular weight'] > 500:
                suggestions.append("⚖️ High molecular weight can sometimes lead to increased binding strength, but may also cause steric hindrance.")
            else:
                suggestions.append("⚖️ Your molecular weight seems optimal. Larger molecules may show better interactions.")

        if feat == 'acceptor':
            if descriptors is not None and descriptors['acceptor'] > 3:
                suggestions.append("🧬 More hydrogen bond acceptors can potentially increase binding strength. Consider optimizing this feature.")
            else:
                suggestions.append("🧬 Low number of H-Bond acceptors may result in weaker binding. Adding more could improve results.")

        if feat == 'logp':
            if descriptors is not None and descriptors['logp'] > 2:
                suggestions.append("🔍 High LogP value indicates good hydrophobic interactions, but make sure it doesn't cause toxicity or solubility issues.")
            else:
                suggestions.append("🔍 A lower LogP value suggests better solubility but might reduce binding strength. Check if increasing this improves results.")
    
    # Add a randomized motivational quote or insight to make it more engaging
    motivational_quotes = [
        "🚀 You're very close to optimizing your binding affinity! Keep adjusting and experimenting.",
        "🔬 Small adjustments can lead to big changes in binding performance. Keep exploring!",
        "🧠 AI suggests that further fine-tuning of key parameters could yield even better results. Keep going!",
        "⚡ High impact features identified! Consider changing them for more optimized binding predictions."
    ]
    
    suggestions.append(random.choice(motivational_quotes))
    
    return suggestions

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
    st.markdown("### 🔍 Select or Enter Energy-Based Values")
    selected_name = st.selectbox("Choose a Protein-Ligand Pair", df['Anon Name'].unique())

    if st.button("🔬 Predict Binding Affinity (from Dataset)"):
        try:
            real_name = anon_map[selected_name]
            row = df[df['PROTEIN-LIGAND'] == real_name]
            energy_features = row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)
            prediction = energy_model.predict(energy_features)[0]

            st.markdown(f"### 🧬 Real Pair: `{real_name}`")
            st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

            # Feature importance
            if hasattr(energy_model, 'feature_importances_'):
                importances = energy_model.feature_importances_
                feature_names = energy_features.columns
                feature_impact = dict(zip(feature_names, importances))

                st.markdown("### 📊 Feature Importance Table")
                st.dataframe(pd.DataFrame(list(feature_impact.items()), columns=['Feature', 'Importance']).style.format({"Importance": "{:.3f}"}), use_container_width=True)
                st.markdown("### 📈 Feature Importance Chart")
                st.bar_chart(pd.DataFrame(list(feature_impact.items()), columns=['Feature', 'Importance']).set_index("Feature"))

                # AI Suggestions
                st.markdown("<div class='suggestion-card'><h4>🧠 AI Suggestion:</h4>", unsafe_allow_html=True)
                suggestions = generate_ai_suggestions(feature_impact, energy_values=energy_features.iloc[0].to_dict())
                for suggestion in suggestions:
                    st.markdown(f"<p>- {suggestion}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Something went wrong: {e}")

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

        importances = descriptor_model.feature_importances_
        feature_impact = dict(zip(features.columns, importances))

        st.markdown("### 📊 Feature Importance Table")
        st.dataframe(pd.DataFrame(list(feature_impact.items()), columns=['Feature', 'Importance']).style.format({"Importance": "{:.3f}"}), use_container_width=True)
        st.markdown("### 📈 Feature Importance Chart")
        st.bar_chart(pd.DataFrame(list(feature_impact.items()), columns=['Feature', 'Importance']).set_index("Feature"))

        # AI Suggestions
        st.markdown("<div class='suggestion-card'><h4>🧠 AI Suggestion:</h4>", unsafe_allow_html=True)
        suggestions = generate_ai_suggestions(feature_impact, descriptors=features.iloc[0].to_dict())
        for suggestion in suggestions:
            st.markdown(f"<p>- {suggestion}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------ FOOTER ------------------------
st.markdown("---")
st.caption("🧠 Powered by Machine Learning | Created with ❤️ for biotech research.")
