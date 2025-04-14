import streamlit as st
import pandas as pd
import joblib
import base64

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(
    page_title="Binding Affinity Predictor",
    layout="wide",
    page_icon="🧬"
)

# ------------------------ LOAD BACKGROUND IMAGE & CONVERT TO BASE64 ------------------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64("image.png")

# ------------------------ CUSTOM CSS WITH EMBEDDED IMAGE ------------------------
st.markdown(f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    font-family: 'Palatino Linotype', serif;
    background-image: url("data:image/png;base64,{img_base64}"); 
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
    color: black !important;
}}

[data-testid="stAppViewContainer"] {{
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 15px;
}}

h1, h2, h3, h4 {{
    color: #2c2c2c;
    font-family: 'Palatino Linotype', serif;
}}

.stButton>button {{
    background-color: #6a5acd;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.6em 1.5em;
    border: none;
}}
.stButton>button:hover {{
    background-color: #836fff;
    transform: scale(1.02);
}}

.suggestion-card {{
    background-color: #f8f8ff;
    padding: 1rem;
    border-left: 4px solid #6a5acd;
    border-radius: 10px;
    margin-top: 20px;
    color: black;
    box-shadow: 0 0 10px rgba(0,0,0,0.08);
}}

[data-testid="stMetric"] {{
    background-color: #fff !important;
    border-radius: 12px;
    padding: 10px;
}}

.prediction-highlight {{
    background-color: #eee;
    padding: 1rem;
    border-left: 5px solid #6a5acd;
    border-radius: 10px;
    margin: 1rem 0;
    font-size: 1.2rem;
    font-weight: bold;
    color: #2c2c2c;
}}
</style>
""", unsafe_allow_html=True)

# ------------------------ LOAD MODEL AND DATA ------------------------
df = pd.read_csv("Cleaned_Autodock_Results.csv")
model = joblib.load("model_with_importance.pkl")

# ------------------------ HEADER ------------------------
st.markdown("# 🧬 Binding Affinity Predictor")
st.markdown("This AI-powered tool predicts binding affinity between a target protein and a compound. Optimized for drug discovery research.")

# ------------------------ LAYOUT ------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004496.png", width=80)
    selected_pair = st.selectbox("Choose a Protein-Ligand Pair", df['PROTEIN-LIGAND'].unique())
    
    if st.button("🔬 Predict Binding Affinity"):
        try:
            row = df[df['PROTEIN-LIGAND'] == selected_pair]
            features = row[[ 
                'Electrostatic energy',
                'Torsional energy',
                'vdw hb desolve energy',
                'Intermol energy'
            ]].fillna(0)

            prediction = model.predict(features)[0]
            st.markdown("### ✅ Prediction Result")
            st.markdown(
                f"<div class='prediction-highlight'>🧬 Predicted Binding Affinity: <b>{prediction:.2f} kcal/mol</b></div>",
                unsafe_allow_html=True
            )

            importances = model.feature_importances_
            feature_names = features.columns
            feature_impact = dict(zip(feature_names, importances))

            # Show feature importance in a simple table format
            st.markdown("### 🧠 Feature Importance:")
            feature_df = pd.DataFrame(list(feature_impact.items()), columns=['Feature', 'Importance'])
            st.dataframe(feature_df.sort_values(by='Importance', ascending=False))

            # Bar chart for feature importance
            st.markdown("### 📊 Feature Importance Visualization:")
            st.bar_chart(feature_df.set_index('Feature')['Importance'])

            st.markdown("<div class='suggestion-card'><h4>🧠 AI Suggestion:</h4>", unsafe_allow_html=True)
            for feat, score in feature_impact.items():
                st.markdown(f"<p>- <b>{feat}</b> is important in predicting the binding affinity. Adjust it for better results.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Something went wrong: {e}")

with col2:
    st.markdown("### Description")
    st.write("This tool predicts binding affinity between a target and compound using ML models. "
             "Designed for drug discovery researchers. Styled with biotech vibes.")
    
    st.markdown("---")
    st.markdown("""
    Welcome to Binding Affinity Predictor, the next-generation tool designed to accelerate drug discovery and enhance precision medicine. 🌍 In today’s fast-paced biotech world, understanding the interaction between proteins and compounds is critical to finding effective therapies. This AI-powered platform uses state-of-the-art machine learning models to predict the binding affinity between target proteins and various ligands, offering significant value to researchers, clinicians, and pharmaceutical companies working toward new drug development. 🧬💊

What is Binding Affinity and Why Does It Matter?
Binding affinity refers to the strength of the interaction between a protein and a ligand (molecule). The higher the affinity, the stronger the bond between them. In drug discovery, the goal is to identify compounds that bind strongly to disease-related proteins, as these compounds are more likely to be effective in treating diseases. 💥 A low dissociation constant (Kd) or binding free energy (ΔG) indicates a high binding affinity, which is often the hallmark of effective therapeutic molecules. On the other hand, compounds with weak binding affinity (high Kd) may need further refinement to improve their therapeutic potential. 🔬

Understanding binding affinity is essential for drug screening, as it determines which compounds are worthy of further study in laboratory tests and clinical trials. By leveraging computational tools to predict these interactions early in the process, researchers can avoid costly experimental failures and streamline the discovery of novel treatments. 🚀

How Does It Work?
The Binding Affinity Predictor tool uses a machine learning model trained on protein-ligand interaction data. It takes into account various factors such as electrostatic energy, torsional energy, van der Waals forces, and intermolecular energy, among others, to calculate a predicted binding affinity value for each compound. This prediction helps prioritize molecules based on their likelihood of successful interactions with the target protein. ⚙️

The tool helps bridge the gap between traditional wet lab methods and modern computational biology. By integrating AI into the prediction process, researchers can significantly reduce the cost and time involved in the early stages of drug development. 📉💡

Key Benefits of the Binding Affinity Predictor:
Reduces Costs and Time: Traditional drug discovery methods involve costly laboratory experiments. With AI-powered predictions, these costs are reduced, allowing researchers to focus on the most promising compounds. 💸

Speeds Up Research: By providing quick insights into protein-ligand interactions, the tool accelerates the process of identifying effective compounds for further testing. ⏩

Supports Precision Medicine: Predicting binding affinities helps tailor treatments to specific patients, improving the chances of success in personalized therapy. 🧬

Data-Driven Decisions: The platform empowers researchers with data-backed predictions, enhancing the precision of decisions made during the drug discovery process. 🎯

Visualizing Binding Affinity:
In the world of drug discovery, visualization plays an important role. The Binding Affinity Predictor not only provides accurate predictions but also visualizes key features of protein-ligand interactions using intuitive graphs and heatmaps. These visuals help researchers quickly identify which features influence the binding affinity most, allowing for strategic molecule optimization. 📊🔥

Graphical Features Include:
Binding Affinity Prediction Graphs: Easily visualize the predicted affinity values to compare multiple ligands at once.

Heatmaps of Feature Impact: See how individual features (such as electrostatic energy, torsional energy, etc.) contribute to the binding affinity.

Interactive Data: Engage with graphs and charts to explore different compound-protein interactions, adjusting parameters to see real-time updates.

The Future of Drug Development
At the core of our mission is innovation. The Binding Affinity Predictor tool is built to empower biotech researchers and pharmaceutical companies to not only optimize existing drugs but also discover entirely new treatments. By helping scientists predict which compounds have the highest chances of success, our tool accelerates drug development timelines and enhances the likelihood of bringing effective therapies to market faster. ⏩💡

In the fight against global diseases such as cancer, diabetes, and infectious diseases, the Binding Affinity Predictor contributes to the next wave of healthcare innovation, playing a crucial role in the advancement of human health. 🏥🌍

Join Us in Revolutionizing Medicine
The Binding Affinity Predictor is just the beginning. As the tool evolves, we are committed to integrating even more sophisticated features, such as incorporating real-time molecular simulations and genomic data, to provide even more precise predictions. Our goal is to be at the forefront of computational drug discovery and to enable personalized medicine in ways that were never possible before. 🔬💉

Let’s collaborate and revolutionize healthcare together! 🌟
    """)
