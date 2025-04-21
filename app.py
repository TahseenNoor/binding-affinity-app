if mode == "🔬 Use Docking Energy Values":
    st.markdown("### 🔍 Select from Dataset or Enter Custom Values")
    
    use_custom = st.checkbox("Enter custom values instead of selecting from dataset")

    if use_custom:
        prot_name = st.text_input("Protein Name")
        lig_name = st.text_input("Ligand Name")

        electro = st.number_input("Electrostatic Energy", value=0.0)
        torsion = st.number_input("Torsional Energy", value=0.0)
        vdw = st.number_input("VDW/HB/Desolvation Energy", value=0.0)
        intermol = st.number_input("Intermolecular Energy", value=0.0)

        if st.button("🔬 Predict Binding Affinity (Custom Energy Input)"):
            features = pd.DataFrame([[electro, torsion, vdw, intermol]],
                                    columns=['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy'])
            prediction = energy_model.predict(features)[0]

            if prot_name or lig_name:
                st.markdown(f"### 🧬 Input Pair: `{prot_name} - {lig_name}`")
            st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

            if hasattr(energy_model, 'feature_importances_'):
                importances = energy_model.feature_importances_
                feature_impact = dict(zip(features.columns, importances))
                feature_df = pd.DataFrame(list(feature_impact.items()), columns=['Feature', 'Importance'])

                st.markdown("### 📊 Feature Importance Table")
                st.dataframe(feature_df.style.format({"Importance": "{:.3f}"}), use_container_width=True)
                st.markdown("### 📈 Feature Importance Chart")
                st.bar_chart(feature_df.set_index("Feature"))

                st.markdown("<div class='suggestion-card'><h4>🧠 AI Suggestion:</h4>", unsafe_allow_html=True)
                for feat, score in feature_impact.items():
                    st.markdown(f"<p>- <b>{feat}</b> is key in affinity prediction. Try optimizing this!</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    else:
        selected_name = st.selectbox("Choose a Protein-Ligand Pair", df['Anon Name'].unique())

        if st.button("🔬 Predict Binding Affinity (from Dataset)"):
            try:
                real_name = anon_map[selected_name]
                row = df[df['PROTEIN-LIGAND'] == real_name]
                features = row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)
                prediction = energy_model.predict(features)[0]

                st.markdown(f"### 🧬 Real Pair: `{real_name}`")
                st.markdown(f"<div class='prediction-highlight'>📉 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>", unsafe_allow_html=True)

                if hasattr(energy_model, 'feature_importances_'):
                    importances = energy_model.feature_importances_
                    feature_names = features.columns
                    feature_impact = dict(zip(feature_names, importances))
                    feature_df = pd.DataFrame(list(feature_impact.items()), columns=['Feature', 'Importance'])

                    st.markdown("### 📊 Feature Importance Table")
                    st.dataframe(feature_df.style.format({"Importance": "{:.3f}"}), use_container_width=True)
                    st.markdown("### 📈 Feature Importance Chart")
                    st.bar_chart(feature_df.set_index("Feature"))

                    st.markdown("<div class='suggestion-card'><h4>🧠 AI Suggestion:</h4>", unsafe_allow_html=True)
                    for feat, score in feature_impact.items():
                        st.markdown(f"<p>- <b>{feat}</b> is important in prediction. Tweak accordingly!</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
