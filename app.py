# ------------------------ MODE SELECTOR ------------------------
mode = st.radio("Choose Prediction Mode:", [
    "🔬 Use Docking Energy Values",
    "🧪 Use Molecular Descriptors",
    "🧬 Combined Input (Descriptors + Energy Values)",
    "🛠️ Manual Input (Energy Only, Any Names)"  # 👈 NEW MODE ADDED HERE
])

# ------------------------ ENERGY MODE ------------------------
if mode == "🔬 Use Docking Energy Values":
    # [Your existing logic... unchanged]

# ------------------------ DESCRIPTOR MODE ------------------------
elif mode == "🧪 Use Molecular Descriptors":
    # [Your existing logic... unchanged]

# ------------------------ COMBINED MODE ------------------------
elif mode == "🧬 Combined Input (Descriptors + Energy Values)":
    # [Your existing logic... unchanged]

# ------------------------ NEW MODE: Manual Energy Input ------------------------
elif mode == "🛠️ Manual Input (Energy Only, Any Names)":
    st.markdown("### 🧪 Enter Any Protein-Ligand Names + Energy Values")

    prot_name = st.text_input("Enter Protein Name")
    lig_name = st.text_input("Enter Ligand Name")

    st.markdown("#### ⚡ Energy-Based Features")
    elec = st.number_input("Electrostatic energy", value=0.0)
    tors = st.number_input("Torsional energy", value=0.0)
    vdw = st.number_input("vdw hb desolve energy", value=0.0)
    inter = st.number_input("Intermol energy", value=0.0)

    if st.button("🎯 Predict Binding Affinity"):
        energy_features = pd.DataFrame([[
            elec, tors, vdw, inter
        ]], columns=[
            'Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy'
        ])

        prediction = energy_model.predict(energy_features)[0]

        st.markdown(f"### 🧬 Pair: `{prot_name}` + `{lig_name}`")
        st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)
