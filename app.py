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
    
    # User input for Protein and Ligand Names
    protein_input = st.text_input("Enter Protein Name:")
    ligand_input = st.text_input("Enter Ligand Name:")

    if st.button("🔬 Predict Binding Affinity (from Dataset)"):
        if protein_input and ligand_input:
            # Combine protein and ligand into a single string
            combined_input = f"{protein_input.strip()} + {ligand_input.strip()}"
            
            # Check if the 'PROTEIN-LIGAND' column exists and contains the input pair
            if 'PROTEIN-LIGAND' in df.columns:
                # Direct match in the 'PROTEIN-LIGAND' column
                matching_row = df[df['PROTEIN-LIGAND'] == combined_input]
                
                if not matching_row.empty:
                    st.write(f"Best matched protein-ligand pair: `{combined_input}`")
                    
                    # Fetch data and make prediction
                    features = matching_row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)
                    prediction = energy_model.predict(features)[0]
                    
                    # Display prediction and result
                    st.markdown(f"### 🧬 Best Matched Pair: `{combined_input}`")
                    st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)
                    
                    # Feature importance (if available)
                    if hasattr(energy_model, 'feature_importances_'):
                        importances = energy_model.feature_importances_
                        feature_names = features.columns
                        feature_impact = dict(zip(feature_names, importances))
                        feature_df = pd.DataFrame(list(feature_impact.items()), columns=['Feature', 'Importance'])
                        
                        st.markdown("### 📊 Feature Importance Table")
                        st.dataframe(feature_df.style.format({"Importance": "{:.3f}"}), use_container_width=True)
                        
                else:
                    st.error(f"No match found for the protein-ligand pair: `{combined_input}`. Please check your inputs.")
            else:
                st.error("No valid Protein-Ligand data found in the dataset. Please check the dataset.")
        else:
            st.error("Please enter both a Protein and Ligand name.")

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

# ------------------------ COMBINED MODE ------------------------
elif mode == "🧬 Combined Input (Descriptors + Energy Values)":
    st.markdown("### 🔬 Enter Energy Values and Molecular Descriptors")

    # Energy values inputs
    protein_input = st.text_input("Enter Protein Name:")
    ligand_input = st.text_input("Enter Ligand Name:")

    # Molecular descriptor values
    mw = st.number_input("Molecular Weight", value=0.0)
    mr = st.number_input("Molar Refractivity", value=0.0)
    logp = st.number_input("LogP", value=0.0)
    acc = st.number_input("Number of H-Bond Acceptors", value=0.0)

    if st.button("🔬 Predict Combined Binding Affinity"):
        if protein_input and ligand_input:
            combined_input = f"{protein_input.strip()} + {ligand_input.strip()}"
            
            # Direct match in the 'PROTEIN-LIGAND' column
            matching_row = df[df['PROTEIN-LIGAND'] == combined_input]
            
            if not matching_row.empty:
                # Fetch data and make prediction
                features = matching_row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)
                prediction = energy_model.predict(features)[0]
                
                # Display prediction and result
                st.markdown(f"### 🧬 Best Matched Pair: `{combined_input}`")
                st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)
            else:
                st.error(f"No match found for the protein-ligand pair: `{combined_input}`. Please check your inputs.")

# ------------------------ FOOTER ------------------------
st.markdown("---")
st.caption("🧠 Powered by Machine Learning | Created with ❤️ for biotech research.")
