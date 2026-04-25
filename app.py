import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import io

# --- Page Branding ---
st.set_page_config(page_title="NovaClass AI | Stellar Observatory", layout="wide", page_icon="🔭")

# Custom CSS to make it look like a high-end dashboard
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e445e; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_research_data():
    csv_data = """source_id,parallax,phot_g_mean_mag,bp_rp,spectral_type
    1,10.66,0.03,0.00,A,2,1.80,0.12,-0.03,B,3,1.21,0.20,1.24,K,4,0.38,1.01,-0.11,O
    5,0.38,0.73,1.44,M,6,2.50,1.88,1.60,M,7,0.38,1.31,1.02,G,8,0.38,2.02,-0.20,B
    9,2.50,2.15,0.48,F,10,2.50,2.62,1.38,K,11,1.21,2.71,-0.16,B,12,10.66,3.01,0.61,G
    13,2.50,3.52,0.92,G,14,1.80,3.75,-0.13,B,15,10.66,4.10,1.85,M,16,2.50,4.25,1.04,K"""
    df = pd.read_csv(io.StringIO(csv_data))
    df['abs_mag'] = df['phot_g_mean_mag'] - 5 * np.log10(1/(df['parallax']/1000)) + 5
    return df

df = load_research_data()

# --- Advanced Training ---
X = df[['abs_mag', 'bp_rp']]
le = LabelEncoder()
y = le.fit_transform(df['spectral_type'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# We use more neurons for a "deeper" look
model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
model.fit(X_scaled, y)

# --- Layout ---
st.title("🔭 NovaClass AI: Deep Space Classifier")
st.write("Leveraging Multi-Layer Perceptrons to categorize Gaia DR3 Stellar Observations.")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.header("🎛️ Sensor Inputs")
    input_color = st.slider("Color Index (Bp-Rp)", -0.5, 4.0, 0.5, help="Difference between blue and red light.")
    input_mag = st.slider("Absolute Magnitude", -8.0, 15.0, 4.8, help="How bright the star actually is.")
    
with col2:
    st.header("🧠 AI Inference")
    new_sample = scaler.transform([[input_mag, input_color]])
    
    # Get probabilities for all classes!
    probs = model.predict_proba(new_sample)[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx] * 100
    pred_class = le.inverse_transform([pred_idx])[0]
    
    st.metric("Predicted Class", f"Type {pred_class}", delta=f"{confidence:.1f}% Confidence")
    
    # Progress bars for top 3 candidates
    st.write("**Classification Confidence:**")
    top_indices = np.argsort(probs)[-3:][::-1]
    for i in top_indices:
        st.write(f"Class {le.inverse_transform([i])[0]}")
        st.progress(float(probs[i]))

with col3:
    st.header("📊 Gaia H-R Diagram")
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e2130')
    
    # Plot background stars
    ax.scatter(df['bp_rp'], df['abs_mag'], c='white', alpha=0.1, s=20)
    
    # Plot the user star with a "glow" effect
    ax.scatter(input_color, input_mag, c='#00f2ff', s=300, edgecolors='white', zorder=5)
    ax.scatter(input_color, input_mag, c='#00f2ff', s=1000, alpha=0.2, zorder=4) # Glow
    
    ax.set_xlabel("Color Index (Temperature Proxy)", color='white')
    ax.set_ylabel("Absolute Magnitude (Luminosity)", color='white')
    ax.tick_params(colors='white')
    ax.invert_yaxis()
    st.pyplot(fig)

# --- Interactive Research Table ---
with st.expander("📂 View Underlying Gaia Research Catalog"):
    st.dataframe(df.style.background_gradient(subset=['abs_mag'], cmap='viridis'), use_container_width=True)