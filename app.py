import streamlit as st
import pandas as pd
import joblib

# Load dataset and model
df = pd.read_csv("Cleaned_Autodock_Results.csv")
model = joblib.load("model.pkl")

# App title
st.title("🔬 Binding Affinity Predictor + Smart Suggestions")

# Dropdown for protein-ligand pair
selected_pair = st.selectbox("Choose Protein-Ligand Pair", df['PROTEIN-LIGAND'].unique())

# Predict button
if st.button("Predict"):
    try:
        row = df[df['PROTEIN-LIGAND'] == selected_pair]

        # Features used in model
        features = row[[
            'Electrostatic energy',
            'Torsional energy',
            'vdw hb desolve energy',
            'Intermol energy'
        ]].fillna(0)

        # Prediction
        prediction = model.predict(features)[0]
        st.success(f"🧪 Predicted Binding Affinity: {prediction:.2f} kcal/mol")

        # Feature importance (based on training model)
        importances = model.feature_importances_
        feature_names = features.columns
        feature_impact = dict(zip(feature_names, importances))

        # Display AI-powered suggestions
        st.markdown("### 💡 Suggestions to Improve Binding Affinity:")
        sorted_feats = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)
        for feat, score in sorted_feats:
            st.write(f"- **{feat}** is highly influential. Try minimizing it to potentially improve binding.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
