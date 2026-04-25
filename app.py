import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
import plotly.graph_objects as go
import io

# --- Page Branding ---
st.set_page_config(
    page_title="NovaClass | Stellar Observatory", 
    layout="wide", 
    page_icon="🔭",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a sleek, modern dark UI
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    
    /* Style the metric container */
    [data-testid="stMetricValue"] { color: #00f2ff !important; font-size: 2.5rem !important;}
    [data-testid="stMetricDelta"] { color: #00e676 !important; }
    
    /* Headers */
    h1, h2, h3 { color: #ffffff !important; font-weight: 300 !important; }
    
    /* Expander styling */
    .streamlit-expanderHeader { background-color: #1e2538 !important; border-radius: 5px; }
    
    /* Progress bar colors */
    .stProgress > div > div > div > div { background-color: #00f2ff; }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_research_data():
    # Fixed the CSV string so rows separate correctly
    csv_data = """source_id,parallax,phot_g_mean_mag,bp_rp,spectral_type
1,10.66,0.03,0.00,A
2,1.80,0.12,-0.03,B
3,1.21,0.20,1.24,K
4,0.38,1.01,-0.11,O
5,0.38,0.73,1.44,M
6,2.50,1.88,1.60,M
7,0.38,1.31,1.02,G
8,0.38,2.02,-0.20,B
9,2.50,2.15,0.48,F
10,2.50,2.62,1.38,K
11,1.21,2.71,-0.16,B
12,10.66,3.01,0.61,G
13,2.50,3.52,0.92,G
14,1.80,3.75,-0.13,B
15,10.66,4.10,1.85,M
16,2.50,4.25,1.04,K"""
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

# --- Header ---
st.title("NovaClass: Deep Space Classifier")
st.markdown("<p style='color: #8b9bb4; font-size: 1.1rem; margin-bottom: 2rem;'>Leveraging Multi-Layer Perceptrons to categorize Gaia DR3 Stellar Observations.</p>", unsafe_allow_html=True)


# --- Layout ---
col1, col2, col3 = st.columns([1, 1, 1.8], gap="large")

with col1:
    st.subheader(" Sensor Inputs")
    st.write("Adjust the stellar parameters below:")
    input_color = st.slider("Color Index (Bp-Rp)", -0.5, 4.0, 0.5, help="Difference between blue and red light. Higher is redder/cooler.")
    input_mag = st.slider("Absolute Magnitude", -8.0, 15.0, 4.8, help="How intrinsically bright the star is. Lower numbers are brighter.")

with col2:
    st.subheader("AI Inference")
    # Suppress sklearn warning by passing a DataFrame with valid feature names
    input_df = pd.DataFrame([[input_mag, input_color]], columns=['abs_mag', 'bp_rp'])
    new_sample = scaler.transform(input_df)

    # Get probabilities for all classes
    probs = model.predict_proba(new_sample)[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx] * 100
    pred_class = le.inverse_transform([pred_idx])[0]

    # Modern Metric Display
    st.metric(label="Predicted Spectral Class", value=f"Type {pred_class}", delta=f"{confidence:.1f}% Confidence")
    st.divider()

    # Progress bars for top 3 candidates
    st.markdown("**Probability Breakdown:**")
    top_indices = np.argsort(probs)[-3:][::-1]
    for i in top_indices:
        class_name = le.inverse_transform([i])[0]
        prob_val = float(probs[i])
        st.write(f"Class **{class_name}** ({prob_val*100:.1f}%)")
        st.progress(prob_val)

with col3:
    st.subheader(" Gaia H-R Diagram")
    
    # Modern Plotly Interactive Graph
    fig = go.Figure()
    
    # Plot background stars
    fig.add_trace(go.Scatter(
        x=df['bp_rp'], y=df['abs_mag'],
        mode='markers',
        name='Catalog Stars',
        hoverinfo='skip'
    ))
    
    # Plot the user star with a glow effect
    fig.add_trace(go.Scatter(
        x=[input_color], y=[input_mag],
        mode='markers',
        marker=dict(
            color='#00f2ff', 
            size=18, 
            line=dict(color='white', width=2),
            symbol='star'
        ),
        name='Your Star',
        hovertemplate="<b>Your Star</b><br>Color (Bp-Rp): %{x}<br>Abs Mag: %{y}<extra></extra>"
    ))

    # Formatting the Plotly chart
    fig.update_yaxes(autorange="reversed", title="Absolute Magnitude (Luminosity)") # H-R diagram convention
    fig.update_xaxes(title="Color Index (Temperature Proxy)")
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- Interactive Research Table ---
st.write("---")
with st.expander("📂 View Underlying Gaia Research Catalog"):
    # Apply a nice gradient to the dataframe
    styled_df = df.style.background_gradient(subset=['abs_mag'], cmap='Blues').format({"parallax": "{:.2f}", "phot_g_mean_mag": "{:.2f}", "abs_mag": "{:.2f}"})
    st.dataframe(styled_df, use_container_width=True)
