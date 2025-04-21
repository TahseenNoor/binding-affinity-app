if mode == "🔬 Use Docking Energy Values":
    st.markdown("### 🔍 Enter Protein and Ligand Names")
    
    # User input for Protein and Ligand Names
    protein_input = st.text_input("Enter Protein Name:")
    ligand_input = st.text_input("Enter Ligand Name:")

    if st.button("🔬 Predict Binding Affinity (from Dataset)"):
        if protein_input and ligand_input:
            # Generate the combined Protein-Ligand name from inputs
            combined_input = f"{protein_input.strip()} + {ligand_input.strip()}"
            
            # Check if the 'PROTEIN-LIGAND' column exists and is not empty
            if 'PROTEIN-LIGAND' in df.columns and df['PROTEIN-LIGAND'].notnull().any():
                # Fuzzy matching to find best match
                best_match, score = process.extractOne(combined_input, df['PROTEIN-LIGAND'])
                
                if score >= 80:
                    st.write(f"Best matched protein-ligand pair: {best_match} with score: {score}")
                    
                    # Fetch data and make prediction
                    row = df[df['PROTEIN-LIGAND'] == best_match]
                    features = row[['Electrostatic energy', 'Torsional energy', 'vdw hb desolve energy', 'Intermol energy']].fillna(0)
                    prediction = energy_model.predict(features)[0]
                    
                    # Display prediction and result
                    st.markdown(f"### 🧬 Best Matched Pair: `{best_match}`")
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
                else:
                    st.error("The protein-ligand names you entered do not match any data with a good enough score. Please check your inputs.")
            else:
                st.error("No valid Protein-Ligand data found in the dataset. Please check the dataset.")
        else:
            st.error("Please enter both a Protein and Ligand name.")
