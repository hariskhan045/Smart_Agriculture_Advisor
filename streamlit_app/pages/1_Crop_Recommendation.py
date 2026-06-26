"""
1_Crop_Recommendation.py – Streamlit page
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

st.set_page_config(page_title="Crop Recommendation", page_icon="🌱", layout="wide")

MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../saved_models/')

@st.cache_resource
def load_models():
    model  = joblib.load(MODEL_DIR + 'crop_recommendation_model.pkl')
    scaler = joblib.load(MODEL_DIR + 'crop_rec_scaler.pkl')
    le     = joblib.load(MODEL_DIR + 'crop_rec_label_encoder.pkl')
    return model, scaler, le

CROP_INFO = {
    'rice':        ('🌾', '#4CAF50', 'Paddy / Kharif', 'Requires flooded fields; tropical climate.'),
    'maize':       ('🌽', '#FFC107', 'Kharif / Rabi',  'Versatile grain; moderate water needs.'),
    'chickpea':    ('🫘', '#8D6E63', 'Rabi',           'Legume that fixes nitrogen; drought-tolerant.'),
    'kidneybeans': ('🫘', '#D32F2F', 'Kharif',         'High-protein bean; warm humid conditions.'),
    'pigeonpeas':  ('🌿', '#388E3C', 'Kharif',         'Drought-resistant; popular in tropical India.'),
    'mothbeans':   ('🌿', '#795548', 'Kharif',         'Extremely drought-tolerant legume.'),
    'mungbean':    ('🫘', '#66BB6A', 'Kharif',         'Fast-growing; improves soil health.'),
    'blackgram':   ('🫘', '#212121', 'Kharif',         'High-protein lentil; grows in hot climates.'),
    'lentil':      ('🫘', '#FF7043', 'Rabi',           'Cool-season legume rich in protein.'),
    'pomegranate': ('🍎', '#E91E63', 'Perennial',      'Drought-tolerant; saline-tolerant.'),
    'banana':      ('🍌', '#FFEB3B', 'Perennial',      'Tropical crop; heavy water requirement.'),
    'mango':       ('🥭', '#FF8F00', 'Perennial',      'Tropical fruit tree; long growing cycle.'),
    'grapes':      ('🍇', '#7B1FA2', 'Perennial',      'Temperate fruit; needs well-drained soil.'),
    'watermelon':  ('🍉', '#F44336', 'Kharif',         'High water; warm sandy loam soil.'),
    'muskmelon':   ('🍈', '#FFA726', 'Kharif',         'Warm; well-drained loamy soils.'),
    'apple':       ('🍎', '#EF5350', 'Perennial',      'Temperate; cold stratification needed.'),
    'orange':      ('🍊', '#FB8C00', 'Perennial',      'Subtropical; moderately frost-sensitive.'),
    'papaya':      ('🍈', '#FF7043', 'Perennial',      'Tropical; fast fruiting in 9–11 months.'),
    'coconut':     ('🥥', '#795548', 'Perennial',      'Coastal tropics; high humidity.'),
    'cotton':      ('🌿', '#E0E0E0', 'Kharif',         'Fibre crop; black soil is ideal.'),
    'jute':        ('🌿', '#8D6E63', 'Kharif',         'Fibre crop; warm humid river deltas.'),
    'coffee':      ('☕', '#4E342E', 'Perennial',      'Tropical highlands; shade-grown.'),
}

st.markdown("## 🌱 Crop Recommendation System")
st.markdown("Enter soil and climate parameters to get the AI-recommended crop.")
st.divider()

model, scaler, le = load_models()

# ── Sidebar inputs ────────────────────────────────────────────
with st.sidebar:
    st.header("🧪 Soil & Climate Parameters")
    N    = st.slider("Nitrogen (N) – kg/ha",         0, 140,  90,  help="Ratio of Nitrogen in soil")
    P    = st.slider("Phosphorous (P) – kg/ha",      5, 145,  42,  help="Ratio of Phosphorous in soil")
    K    = st.slider("Potassium (K) – kg/ha",        5, 205,  43,  help="Ratio of Potassium in soil")
    temp = st.slider("Temperature (°C)",             8.0, 43.0, 20.0, step=0.5)
    hum  = st.slider("Humidity (%)",                14.0, 99.0, 82.0, step=0.5)
    ph   = st.slider("Soil pH",                     3.5, 9.5,  6.5,  step=0.1)
    rain = st.slider("Rainfall (mm)",               20.0, 299.0, 202.0, step=1.0)
    predict_btn = st.button("🚀 Recommend Crop", type="primary", use_container_width=True)

# ── Prediction ───────────────────────────────────────────────
if predict_btn:
    X = np.array([[N, P, K, temp, hum, ph, rain]])
    X_sc = scaler.transform(X)
    pred = model.predict(X_sc)[0]
    proba_arr = model.predict_proba(X_sc)[0]
    classes   = model.classes_
    top5 = sorted(zip(classes, proba_arr), key=lambda x: -x[1])[:5]
    confidence = top5[0][1] * 100

    info = CROP_INFO.get(pred, ('🌿', '#2d9e47', '—', '—'))
    icon, color, season, note = info

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{color}22,{color}11);
                border:2px solid {color};border-radius:14px;padding:28px;margin-bottom:24px;">
      <h2 style="color:{color};margin:0">{icon} Recommended Crop: <b>{pred.title()}</b></h2>
      <p style="font-size:1.05rem;margin-top:10px">
        <b>Confidence:</b> {confidence:.1f}% &nbsp;|&nbsp;
        <b>Season:</b> {season} &nbsp;|&nbsp;
        <b>Note:</b> {note}
      </p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Top 5 Crop Probabilities")
        names = [c.title() for c, _ in top5]
        probs = [p * 100 for _, p in top5]
        fig = go.Figure(go.Bar(
            x=probs, y=names, orientation='h',
            marker_color=[color] + ['#90CAF9'] * 4,
            text=[f'{p:.1f}%' for p in probs],
            textposition='outside'
        ))
        fig.update_layout(
            xaxis=dict(range=[0, 110], title='Confidence (%)'),
            yaxis=dict(autorange='reversed'),
            height=260, margin=dict(l=10, r=40, t=10, b=10),
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📋 Input Parameter Summary")
        params = {'Parameter': ['Nitrogen', 'Phosphorous', 'Potassium',
                                'Temperature', 'Humidity', 'pH', 'Rainfall'],
                  'Value': [N, P, K, temp, hum, ph, rain],
                  'Unit':  ['kg/ha', 'kg/ha', 'kg/ha', '°C', '%', '', 'mm']}
        df_p = pd.DataFrame(params)
        st.dataframe(df_p, hide_index=True, use_container_width=True)

    # ── NPK Gauge charts ─────────────────────────────────────
    st.subheader("🧮 NPK Nutrient Gauges")
    g1, g2, g3 = st.columns(3)
    for col_g, val, label, max_v in [
        (g1, N, 'Nitrogen (N)', 140),
        (g2, P, 'Phosphorous (P)', 145),
        (g3, K, 'Potassium (K)', 205)
    ]:
        fig_g = go.Figure(go.Indicator(
            mode='gauge+number',
            value=val,
            title={'text': label},
            gauge={
                'axis': {'range': [0, max_v]},
                'bar': {'color': '#2d9e47'},
                'steps': [
                    {'range': [0, max_v * 0.33], 'color': '#ffe082'},
                    {'range': [max_v * 0.33, max_v * 0.66], 'color': '#a5d6a7'},
                    {'range': [max_v * 0.66, max_v], 'color': '#388e3c'}
                ]
            }
        ))
        fig_g.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
        col_g.plotly_chart(fig_g, use_container_width=True)

else:
    st.info("👈 Adjust the soil and climate sliders in the sidebar, then click **Recommend Crop**.")

    # ── Sample averages per crop ──────────────────────────────
    st.subheader("📖 Crop Quick Reference")
    df_cr = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                     '../../datasets/Crop_recommendation.csv'))
    agg = df_cr.groupby('label')[['N', 'P', 'K', 'temperature', 'humidity',
                                   'ph', 'rainfall']].mean().round(1)
    st.dataframe(agg, use_container_width=True)
