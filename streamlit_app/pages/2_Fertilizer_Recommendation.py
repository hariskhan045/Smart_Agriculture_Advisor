"""
2_Fertilizer_Recommendation.py – Streamlit page
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

st.set_page_config(page_title="Fertilizer Recommendation", page_icon="🧪", layout="wide")

MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../saved_models/')

@st.cache_resource
def load_models():
    model   = joblib.load(MODEL_DIR + 'fertilizer_recommendation_model.pkl')
    scaler  = joblib.load(MODEL_DIR + 'fertilizer_scaler.pkl')
    le_soil = joblib.load(MODEL_DIR + 'fertilizer_le_soil.pkl')
    le_crop = joblib.load(MODEL_DIR + 'fertilizer_le_crop.pkl')
    le_fert = joblib.load(MODEL_DIR + 'fertilizer_le_fert.pkl')
    return model, scaler, le_soil, le_crop, le_fert

FERTILIZER_INFO = {
    'Urea':       ('#1565C0', '46-0-0', 'High-nitrogen fertilizer; promotes vegetative growth. '
                              'Apply in split doses for best uptake.'),
    'DAP':        ('#2E7D32', '18-46-0', 'Di-Ammonium Phosphate; starter fertilizer. '
                              'Excellent for root development and seed germination.'),
    'MOP':        ('#6A1B9A', '0-0-60', 'Muriate of Potash; improves drought resistance, '
                              'fruit quality, and disease immunity.'),
    'SSP':        ('#E65100', '0-16-0', 'Single Super Phosphate; also supplies calcium and sulfur. '
                              'Ideal for legumes and oilseeds.'),
    '28-28':      ('#00838F', '28-28-0', 'Balanced N-P fertilizer for crops requiring equal '
                              'nitrogen and phosphorous at planting.'),
    '14-35-14':   ('#AD1457', '14-35-14', 'High-phosphorus blend; excellent for transplanted '
                              'crops and initial root establishment.'),
    '20-20':      ('#558B2F', '20-20-0', 'Balanced starter fertilizer; widely used in mixed '
                              'cropping systems.'),
    '10-26-26':   ('#4527A0', '10-26-26', 'Low-nitrogen, high P-K formula; ideal for mature '
                              'crops and fruit-bearing plants.'),
}

st.markdown("## 🧪 Fertilizer Recommendation System")
st.markdown("Input soil conditions and crop type to receive the optimal fertilizer recommendation.")
st.divider()

model, scaler, le_soil, le_crop, le_fert = load_models()

SOIL_TYPES = le_soil.classes_.tolist()
CROP_TYPES = le_crop.classes_.tolist()

# ── Sidebar inputs ────────────────────────────────────────────
with st.sidebar:
    st.header("🌍 Field Parameters")
    soil_type = st.selectbox("Soil Type",  SOIL_TYPES)
    crop_type = st.selectbox("Crop Type",  CROP_TYPES)
    st.markdown("---")
    temperature = st.slider("Temperature (°C)", 15.0, 45.0, 26.0, step=0.5)
    humidity    = st.slider("Humidity (%)",     20.0, 90.0, 55.0, step=0.5)
    moisture    = st.slider("Soil Moisture (%)",20.0, 80.0, 45.0, step=0.5)
    st.markdown("---")
    st.subheader("NPK Values")
    nitrogen    = st.slider("Nitrogen  (kg/ha)", 0, 100, 37)
    potassium   = st.slider("Potassium (kg/ha)", 0, 100, 0)
    phosphorous = st.slider("Phosphorous (kg/ha)", 0, 100, 0)
    predict_btn = st.button("🚀 Recommend Fertilizer", type="primary", use_container_width=True)

if predict_btn:
    soil_enc = le_soil.transform([soil_type])[0]
    crop_enc = le_crop.transform([crop_type])[0]
    X = np.array([[temperature, humidity, moisture, nitrogen,
                   potassium, phosphorous, soil_enc, crop_enc]])
    X_sc = scaler.transform(X)
    pred = model.predict(X_sc)[0]
    proba_arr = model.predict_proba(X_sc)[0]
    classes   = model.classes_

    info = FERTILIZER_INFO.get(pred, ('#2d9e47', '—', 'General purpose fertilizer.'))
    color, npk, desc = info
    confidence = max(proba_arr) * 100

    st.markdown(f"""
    <div style="background:{color}18;border:2px solid {color};
                border-radius:14px;padding:28px;margin-bottom:24px;">
      <h2 style="color:{color};margin:0">🧪 Recommended Fertilizer: <b>{pred}</b></h2>
      <p style="font-size:1.05rem;margin-top:10px">
        <b>NPK Ratio:</b> {npk} &nbsp;|&nbsp; <b>Confidence:</b> {confidence:.1f}%
      </p>
      <p style="color:#444;font-size:.95rem">{desc}</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Fertilizer Probabilities")
        sorted_probs = sorted(zip(classes, proba_arr), key=lambda x: -x[1])
        names = [c for c, _ in sorted_probs]
        probs = [p * 100 for _, p in sorted_probs]
        fig = px.bar(x=probs, y=names, orientation='h',
                     labels={'x': 'Confidence (%)', 'y': 'Fertilizer'},
                     color=probs, color_continuous_scale='Greens',
                     text=[f'{p:.1f}%' for p in probs])
        fig.update_layout(height=300, template='plotly_white',
                          margin=dict(l=10, r=40, t=10, b=10),
                          coloraxis_showscale=False,
                          yaxis=dict(autorange='reversed'))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📋 NPK Deficiency Analysis")
        deficiency = {}
        if nitrogen < 20:    deficiency['Nitrogen'] = 'Low – consider N-rich fertilizer'
        elif nitrogen > 70:  deficiency['Nitrogen'] = 'High – reduce N application'
        else:                deficiency['Nitrogen'] = 'Optimal'
        if phosphorous < 20: deficiency['Phosphorous'] = 'Low – consider P-rich fertilizer'
        elif phosphorous > 70: deficiency['Phosphorous'] = 'High – reduce P application'
        else:                deficiency['Phosphorous'] = 'Optimal'
        if potassium < 20:   deficiency['Potassium'] = 'Low – consider K-rich fertilizer'
        elif potassium > 70: deficiency['Potassium'] = 'High – reduce K application'
        else:                deficiency['Potassium'] = 'Optimal'

        for nutrient, status in deficiency.items():
            emoji = '✅' if 'Optimal' in status else ('⬇️' if 'Low' in status else '⬆️')
            color_s = 'green' if 'Optimal' in status else 'red'
            st.markdown(f"**{nutrient}:** <span style='color:{color_s}'>{emoji} {status}</span>",
                        unsafe_allow_html=True)

        # Radar chart
        fig_r = go.Figure(go.Scatterpolar(
            r=[nitrogen, potassium, phosphorous, humidity, moisture, nitrogen],
            theta=['Nitrogen', 'Potassium', 'Phosphorous', 'Humidity', 'Moisture', 'Nitrogen'],
            fill='toself', line_color=color
        ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=260, margin=dict(l=20, r=20, t=20, b=20),
            template='plotly_white'
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # Fertilizer guide table
    st.subheader("📖 Fertilizer Quick Reference")
    guide = []
    for name, (_, npk_r, descr) in FERTILIZER_INFO.items():
        guide.append({'Fertilizer': name, 'NPK Ratio': npk_r, 'Best Use': descr[:80] + '…'})
    st.dataframe(pd.DataFrame(guide), hide_index=True, use_container_width=True)

else:
    st.info("👈 Select soil type, crop type, and NPK values in the sidebar, then click **Recommend Fertilizer**.")

    st.subheader("🔍 About Fertilizers")
    data = []
    for name, (color_h, npk_r, descr) in FERTILIZER_INFO.items():
        data.append({'Fertilizer': name, 'NPK Ratio': npk_r, 'Description': descr})
    st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
