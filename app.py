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
            features = row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)
            prediction = energy_model.predict(features)[0]

            st.markdown(f"### 🧬 Real Pair: `{real_name}`")
            st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

            # Feature importance
            if hasattr(energy_model, 'feature_importances_'):
                importances = energy_model.feature_importances_
                feature_names = features.columns
                feature_impact = dict(zip(feature_names, importances))
                feature_df = pd.DataFrame(list(feature_impact.items()), columns=['Feature', 'Importance'])

                st.markdown("### 📊 Feature Importance Table")
                st.dataframe(feature_df.style.format({"Importance": "{:.3f}"}), use_container_width=True)
                st.markdown("### 📈 Feature Importance Chart")
                st.bar_chart(feature_df.set_index("Feature"))

                # AI Suggestion
                st.markdown("<div class='suggestion-card'><h4>🧠 AI Suggestion:</h4>", unsafe_allow_html=True)
                for feat, score in feature_impact.items():
                    st.markdown(f"<p>- <b>{feat}</b> is important in predicting the binding affinity. Adjust it for better results.</p>", unsafe_allow_html=True)
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

        # Feature importance
        if hasattr(descriptor_model, 'feature_importances_'):
            importances = descriptor_model.feature_importances_
            feature_impact = dict(zip(features.columns, importances))
            feature_df = pd.DataFrame(list(feature_impact.items()), columns=['Feature', 'Importance'])

            st.markdown("### 📊 Feature Importance Table")
            st.dataframe(feature_df.style.format({"Importance": "{:.3f}"}), use_container_width=True)
            st.markdown("### 📈 Feature Importance Chart")
            st.bar_chart(feature_df.set_index("Feature"))

            # AI Suggestion
            st.markdown("<div class='suggestion-card'><h4>🧠 AI Suggestion:</h4>", unsafe_allow_html=True)
            for feat, score in feature_impact.items():
                st.markdown(f"<p>- <b>{feat}</b> influences binding predictions. Check its value for optimization.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ------------------------ COMBINED MODE ------------------------
elif mode == "🧬 Combined Input (Descriptors + Energy Values)":
    st.markdown("### 🔬 Enter Energy Values and Molecular Descriptors")

    # Energy values inputs
    selected_name = st.selectbox("Choose a Protein-Ligand Pair", df['Anon Name'].unique())
    # Enter molecular descriptor values
    mw = st.number_input("Molecular Weight", value=0.0)
    mr = st.number_input("Molar Refractivity", value=0.0)
    logp = st.number_input("LogP", value=0.0)
    acc = st.number_input("Number of H-Bond Acceptors", value=0.0)

    if st.button("🔬 Predict Combined Binding Affinity"):
        try:
            real_name = anon_map[selected_name]
            row = df[df['PROTEIN-LIGAND'] == real_name]
            energy_features = row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)

            # Combine energy values and descriptor values
            combined_features = pd.DataFrame([[mr, mw, acc, logp] + energy_features.values[0].tolist()],
                                            columns=['molar refractivity', 'molecular weight', 'acceptor', 'logp'] + energy_features.columns.tolist())

            # Predict using combined model (assuming you have a combined model)
            combined_model = energy_model  # If you don't have a separate combined model, use an existing one
            prediction = combined_model.predict(combined_features)[0]

            st.markdown(f"### Predicted Binding Affinity: {prediction:.2f} kcal/mol")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ------------------------ FOOTER ------------------------
st.markdown("---")
st.caption("🧠 Powered by Machine Learning | Created with ❤️ for biotech research.")
