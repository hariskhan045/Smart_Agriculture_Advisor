"""
Home.py – Smart Agriculture Advisor Streamlit App
Main landing page with project overview and KPIs.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Agriculture Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #1a6b2b 0%, #2d9e47 50%, #5bbf6e 100%);
    border-radius: 16px;
    padding: 40px 50px;
    color: white;
    margin-bottom: 30px;
}
.hero h1 { font-size: 2.6rem; margin: 0; font-weight: 800; }
.hero p  { font-size: 1.15rem; margin-top: 12px; opacity: 0.92; }
.kpi-card {
    background: white;
    border-left: 5px solid #2d9e47;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08);
    text-align: center;
}
.kpi-val  { font-size: 2.2rem; font-weight: 700; color: #1a6b2b; }
.kpi-lbl  { font-size: 0.9rem; color: #666; margin-top: 4px; }
.module-card {
    background: #f8fffe;
    border: 1px solid #d4edda;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.module-card h4 { color: #1a6b2b; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🌾 AI-Powered Smart Agriculture Advisor</h1>
  <p>Data-driven decisions for modern farmers — Crop Recommendation · Fertilizer Guidance · Yield Prediction · Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ─────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
kpis = [
    ("99.2%", "Crop Recommendation Accuracy"),
    ("100%",  "Fertilizer Recommendation Accuracy"),
    ("94.7%", "Yield Prediction R² Score"),
    ("22",    "Crop Classes Supported"),
]
for col, (val, lbl) in zip([c1, c2, c3, c4], kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-val">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Modules grid ─────────────────────────────────────────────
st.subheader("📦 System Modules")
m1, m2, m3, m4 = st.columns(4)

modules = [
    ("🌱", "Crop Recommendation",
     "Analyses N-P-K ratios, temperature, humidity, pH, and rainfall to suggest the most suitable crop for your field conditions.",
     "**Model:** Random Forest / Naive Bayes | **Accuracy:** 99.2%"),
    ("🧪", "Fertilizer Recommendation",
     "Combines soil type, crop type, and nutrient readings to recommend the optimal fertilizer, eliminating guesswork.",
     "**Model:** Decision Tree | **Accuracy:** 100%"),
    ("📈", "Yield Prediction",
     "Predicts expected harvest using crop species, area, rainfall, fertilizer, and pesticide consumption across seasons.",
     "**Model:** Random Forest Regressor | **R²:** 0.947"),
    ("📊", "Analytics Dashboard",
     "Interactive Plotly charts covering climate trends, pesticide usage, state-wise production, and global agricultural insights.",
     "**Data:** FAO + India agricultural datasets"),
]

for col, (icon, title, desc, tech) in zip([m1, m2, m3, m4], modules):
    with col:
        st.markdown(f"""
        <div class="module-card">
          <h4>{icon} {title}</h4>
          <p style="font-size:.9rem;color:#444">{desc}</p>
          <p style="font-size:.82rem;color:#888">{tech}</p>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Dataset summary ──────────────────────────────────────────
st.subheader("🗃️ Datasets Used")
datasets = {
    'Dataset': ['Crop Recommendation', 'Fertilizer (Synthetic)',
                'Crop Yield (India)', 'Yield Features (Global)',
                'Rainfall', 'Temperature', 'Pesticides'],
    'Rows': [2200, 5000, 19689, 28242, 6727, 71311, 4349],
    'Key Features': [
        'N, P, K, Temp, Humidity, pH, Rainfall',
        'Temp, Humidity, Moisture, Soil, NPK',
        'Crop, Area, Season, State, Rainfall, Fertilizer',
        'Area, Item, Rainfall, Pesticides, Avg Temp',
        'Country, Year, Rainfall (mm/yr)',
        'Country, Year, Avg Temperature',
        'Country, Year, Pesticide Tonnes'
    ],
    'Task': ['Classification', 'Classification', 'Regression',
             'Analysis', 'Analysis', 'Analysis', 'Analysis']
}
df_ds = pd.DataFrame(datasets)
st.dataframe(df_ds, use_container_width=True, hide_index=True)

# ── Radar chart of model performance ─────────────────────────
st.subheader("🎯 Model Performance Overview")
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Score']
fig = go.Figure()
for model, values in [
    ('Logistic Regression', [0.972, 0.972, 0.972, 0.972, 0.967]),
    ('Random Forest',       [0.992, 0.993, 0.992, 0.992, 0.992]),
    ('Naive Bayes',         [0.992, 0.992, 0.992, 0.992, 0.994]),
    ('KNN',                 [0.969, 0.973, 0.969, 0.969, 0.961]),
]:
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill='toself', name=model, opacity=0.65
    ))
fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0.9, 1.0])),
    title='Crop Recommendation – Model Comparison',
    height=450, template='plotly_white'
)
st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;font-size:.85rem;'>"
    "AI-Powered Smart Agriculture Advisor · BS Computer Science Semester Project · "
    "Built with Python, scikit-learn, XGBoost, Streamlit & Plotly</p>",
    unsafe_allow_html=True
)
