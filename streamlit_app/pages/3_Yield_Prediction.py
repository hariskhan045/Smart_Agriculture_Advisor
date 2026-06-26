"""
3_Yield_Prediction.py – Streamlit page
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

st.set_page_config(page_title="Yield Prediction", page_icon="📈", layout="wide")

MODEL_DIR   = os.path.join(os.path.dirname(__file__), '../../saved_models/')
DATASET_DIR = os.path.join(os.path.dirname(__file__), '../../datasets/')

@st.cache_resource
def load_models():
    model     = joblib.load(MODEL_DIR + 'crop_yield_model.pkl')
    scaler    = joblib.load(MODEL_DIR + 'yield_scaler.pkl')
    le_crop   = joblib.load(MODEL_DIR + 'yield_le_crop.pkl')
    le_season = joblib.load(MODEL_DIR + 'yield_le_season.pkl')
    le_state  = joblib.load(MODEL_DIR + 'yield_le_state.pkl')
    return model, scaler, le_crop, le_season, le_state

@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_DIR + 'crop_yield.csv')

st.markdown("## 📈 Crop Yield Prediction System")
st.markdown("Enter agricultural parameters to predict expected crop yield.")
st.divider()

model, scaler, le_crop, le_season, le_state = load_models()
df = load_dataset()

CROPS   = sorted(le_crop.classes_.tolist())
SEASONS = sorted(le_season.classes_.tolist())
STATES  = sorted(le_state.classes_.tolist())

# ── Sidebar inputs ────────────────────────────────────────────
with st.sidebar:
    st.header("🌾 Crop Parameters")
    crop      = st.selectbox("Crop",   CROPS,   index=CROPS.index('Rice') if 'Rice' in CROPS else 0)
    crop_year = st.number_input("Crop Year", min_value=1997, max_value=2030, value=2020)
    season    = st.selectbox("Season", SEASONS)
    state     = st.selectbox("State",  STATES)
    st.markdown("---")
    area       = st.number_input("Area (Hectares)",        min_value=1.0,   max_value=1e7,  value=10000.0,  step=100.0)
    rainfall   = st.number_input("Annual Rainfall (mm)",   min_value=10.0,  max_value=3000.0, value=900.0, step=10.0)
    fertilizer = st.number_input("Fertilizer Used (kg)",   min_value=0.0,   max_value=1e8,  value=1000000.0, format="%.0f")
    pesticide  = st.number_input("Pesticide Used (kg)",    min_value=0.0,   max_value=1e7,  value=2000.0,   format="%.0f")
    predict_btn = st.button("🚀 Predict Yield", type="primary", use_container_width=True)

if predict_btn:
    crop_enc   = le_crop.transform([crop])[0]
    season_enc = le_season.transform([season])[0]
    state_enc  = le_state.transform([state])[0]
    X = np.array([[crop_enc, crop_year, season_enc, state_enc,
                   area, rainfall, fertilizer, pesticide]])
    X_sc = scaler.transform(X)
    predicted_yield = float(model.predict(X_sc)[0])
    total_production = predicted_yield * area / 1000  # approximate tonnes

    # Colour-code
    if predicted_yield > 3.0:
        color, label = '#2E7D32', 'Excellent'
    elif predicted_yield > 1.5:
        color, label = '#F57F17', 'Good'
    else:
        color, label = '#C62828', 'Below Average'

    st.markdown(f"""
    <div style="background:{color}18;border:2px solid {color};
                border-radius:14px;padding:28px;margin-bottom:24px;">
      <h2 style="color:{color};margin:0">📈 Predicted Yield: <b>{predicted_yield:.4f} Tonnes/Ha</b></h2>
      <p style="font-size:1.05rem;margin-top:10px">
        <b>Performance:</b> {label} &nbsp;|&nbsp;
        <b>Est. Total Production:</b> {total_production:,.0f} Tonnes &nbsp;|&nbsp;
        <b>Crop:</b> {crop} &nbsp;|&nbsp; <b>State:</b> {state}
      </p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Historical Yield for Selected Crop")
        crop_hist = df[df['Crop'] == crop].groupby('Crop_Year')['Yield'].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=crop_hist['Crop_Year'], y=crop_hist['Yield'],
            mode='lines+markers', name='Historical Avg',
            line=dict(color='#2d9e47', width=2)
        ))
        fig.add_hline(y=predicted_yield, line_dash='dash', line_color=color,
                      annotation_text=f"Predicted: {predicted_yield:.3f}",
                      annotation_position="top right")
        fig.update_layout(
            title=f'{crop} Yield Trend', height=320,
            template='plotly_white', xaxis_title='Year', yaxis_title='Yield (T/Ha)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🗺️ State-wise Average Yield (Selected Crop)")
        state_yield = df[df['Crop'] == crop].groupby('State')['Yield'].mean().sort_values(
            ascending=False).reset_index()
        state_yield.columns = ['State', 'Avg Yield']
        fig2 = px.bar(state_yield, x='State', y='Avg Yield',
                      color='Avg Yield', color_continuous_scale='Greens',
                      title=f'{crop} – State-wise Average Yield')
        fig2.update_layout(height=320, template='plotly_white',
                           xaxis=dict(tickangle=45))
        st.plotly_chart(fig2, use_container_width=True)

    # Input summary
    st.subheader("📋 Input Summary")
    in_df = pd.DataFrame({
        'Parameter': ['Crop', 'Year', 'Season', 'State', 'Area (Ha)',
                      'Rainfall (mm)', 'Fertilizer (kg)', 'Pesticide (kg)',
                      'Predicted Yield', 'Est. Production'],
        'Value': [crop, crop_year, season, state, f'{area:,.0f}',
                  f'{rainfall:.0f}', f'{fertilizer:,.0f}', f'{pesticide:,.0f}',
                  f'{predicted_yield:.4f} T/Ha', f'{total_production:,.0f} T']
    })
    st.dataframe(in_df, hide_index=True, use_container_width=True)

else:
    st.info("👈 Select crop parameters in the sidebar, then click **Predict Yield**.")

    # ── Overview charts ──────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌾 Top 15 Crops by Average Yield")
        top15 = df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(15)
        fig = px.bar(x=top15.values, y=top15.index, orientation='h',
                     color=top15.values, color_continuous_scale='Greens',
                     labels={'x': 'Avg Yield (T/Ha)', 'y': 'Crop'})
        fig.update_layout(height=400, template='plotly_white', coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📅 Yearly Yield Trend (All Crops)")
        yearly = df.groupby('Crop_Year')['Yield'].mean().reset_index()
        fig2 = go.Figure(go.Scatter(
            x=yearly['Crop_Year'], y=yearly['Yield'],
            mode='lines+markers', fill='tozeroy',
            fillcolor='rgba(45,158,71,0.15)',
            line=dict(color='#2d9e47', width=2)
        ))
        fig2.update_layout(
            title='Average Crop Yield Over Time (India)',
            height=400, template='plotly_white',
            xaxis_title='Year', yaxis_title='Yield (T/Ha)'
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🏆 State-wise Total Production")
    state_prod = df.groupby('State')['Production'].sum().sort_values(
        ascending=False).reset_index()
    fig3 = px.bar(state_prod, x='State', y='Production',
                  color='Production', color_continuous_scale='Viridis',
                  title='Total Agricultural Production by State')
    fig3.update_layout(height=380, template='plotly_white',
                       xaxis=dict(tickangle=45))
    st.plotly_chart(fig3, use_container_width=True)
