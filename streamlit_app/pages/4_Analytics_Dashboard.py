"""
4_Analytics_Dashboard.py – Streamlit page
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")
DATASET_DIR = os.path.join(os.path.dirname(__file__), '../../datasets/')

@st.cache_data
def load_all():
    rain = pd.read_csv(DATASET_DIR + 'rainfall.csv')
    rain.columns = rain.columns.str.strip()
    rain['average_rain_fall_mm_per_year'] = pd.to_numeric(
        rain['average_rain_fall_mm_per_year'], errors='coerce')
    rain.dropna(inplace=True)

    temp = pd.read_csv(DATASET_DIR + 'temp.csv')
    temp.dropna(inplace=True)

    pest = pd.read_csv(DATASET_DIR + 'pesticides.csv')
    pest.columns = pest.columns.str.strip()
    pest.dropna(inplace=True)

    yield_df = pd.read_csv(DATASET_DIR + 'yield_df.csv')
    yield_df.dropna(inplace=True)

    crop_yield = pd.read_csv(DATASET_DIR + 'crop_yield.csv')
    crop_yield.dropna(inplace=True)

    crop_rec = pd.read_csv(DATASET_DIR + 'Crop_recommendation.csv')

    return rain, temp, pest, yield_df, crop_yield, crop_rec

rain, temp, pest, yield_df, crop_yield, crop_rec = load_all()

# ── Header ────────────────────────────────────────────────────
st.markdown("## 📊 Agricultural Analytics Dashboard")
st.markdown("Interactive insights across climate, production, pesticides, and crop nutrition data.")
st.divider()

# ── Tab layout ────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌧️ Climate Trends",
    "🧪 Pesticide Analysis",
    "🌾 Crop Production",
    "🔬 Crop Nutrition",
    "📈 Global Yield"
])

# ────────────────────────────────────────────────────────────
# TAB 1: CLIMATE TRENDS
# ────────────────────────────────────────────────────────────
with tab1:
    st.subheader("🌧️ Climate Trend Analysis")

    col1, col2 = st.columns(2)

    with col1:
        global_rain = rain.groupby('Year')['average_rain_fall_mm_per_year'].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=global_rain['Year'], y=global_rain['average_rain_fall_mm_per_year'],
            mode='lines', fill='tozeroy',
            fillcolor='rgba(33,150,243,0.15)',
            line=dict(color='#1976D2', width=2),
            name='Global Avg Rainfall'
        ))
        fig.update_layout(title='Global Average Rainfall (mm/year)',
                          template='plotly_white', height=360,
                          xaxis_title='Year', yaxis_title='Rainfall (mm)')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Global average annual rainfall trend from FAO dataset. "
                   "Fluctuations reflect ENSO cycles and shifting monsoon patterns.")

    with col2:
        global_temp = temp.groupby('year')['avg_temp'].mean().reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=global_temp['year'], y=global_temp['avg_temp'],
            mode='lines', fill='tozeroy',
            fillcolor='rgba(244,67,54,0.12)',
            line=dict(color='#D32F2F', width=2),
            name='Global Avg Temp'
        ))
        fig2.update_layout(title='Global Average Temperature (°C)',
                           template='plotly_white', height=360,
                           xaxis_title='Year', yaxis_title='Temp (°C)')
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Rising global temperatures since the industrial era, "
                   "with acceleration after the 1980s, consistent with IPCC findings.")

    # Top 10 highest rainfall countries
    top_rain = rain.groupby('Area')['average_rain_fall_mm_per_year'].mean().sort_values(
        ascending=False).head(10).reset_index()
    fig3 = px.bar(top_rain, x='Area', y='average_rain_fall_mm_per_year',
                  color='average_rain_fall_mm_per_year', color_continuous_scale='Blues',
                  title='Top 10 Countries by Average Annual Rainfall',
                  labels={'average_rain_fall_mm_per_year': 'Rainfall (mm/yr)', 'Area': 'Country'})
    fig3.update_layout(template='plotly_white', height=360, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Tropical and equatorial nations record the highest precipitation, "
               "supporting year-round agriculture without supplemental irrigation.")

    # Temperature distribution by country (box – top 15 countries by data volume)
    top15_countries = temp['country'].value_counts().head(15).index
    temp_sub = temp[temp['country'].isin(top15_countries)]
    fig4 = px.box(temp_sub, x='country', y='avg_temp',
                  color='country',
                  title='Temperature Distribution (Top 15 Countries by Data Volume)',
                  labels={'avg_temp': 'Avg Temp (°C)'})
    fig4.update_layout(template='plotly_white', height=400, showlegend=False,
                       xaxis=dict(tickangle=30))
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Box plots reveal the median and variability of annual temperatures. "
               "Countries closer to the equator exhibit higher medians and smaller ranges.")


# ────────────────────────────────────────────────────────────
# TAB 2: PESTICIDE ANALYSIS
# ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🧪 Pesticide Usage Analysis")

    col1, col2 = st.columns(2)
    with col1:
        pest_yearly = pest.groupby('Year')['Value'].sum().reset_index()
        fig = px.bar(pest_yearly, x='Year', y='Value',
                     color='Value', color_continuous_scale='Oranges',
                     title='Global Annual Pesticide Usage (Tonnes Active Ingredients)',
                     labels={'Value': 'Pesticide (tonnes AI)'})
        fig.update_layout(template='plotly_white', height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Global pesticide use increased sharply from the 1990s, "
                   "driven by expansion of intensive commercial agriculture.")

    with col2:
        top_pest = pest.groupby('Area')['Value'].sum().sort_values(
            ascending=False).head(10).reset_index()
        fig2 = px.pie(top_pest, names='Area', values='Value',
                      title='Top 10 Countries – Total Pesticide Usage Share',
                      color_discrete_sequence=px.colors.sequential.Reds_r)
        fig2.update_layout(height=360)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("A small number of high-output agricultural economies dominate "
                   "global pesticide consumption, reflecting intensive farming systems.")

    # Pesticide trend by top 5 countries
    top5_pest = pest.groupby('Area')['Value'].sum().sort_values(
        ascending=False).head(5).index
    pest_sub = pest[pest['Area'].isin(top5_pest)]
    pest_trend = pest_sub.groupby(['Year', 'Area'])['Value'].sum().reset_index()
    fig3 = px.line(pest_trend, x='Year', y='Value', color='Area',
                   title='Pesticide Usage Trend – Top 5 Countries',
                   labels={'Value': 'Pesticide (tonnes AI)'})
    fig3.update_layout(template='plotly_white', height=380)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Individual country trends reveal different phases of agricultural "
               "intensification. Some countries show peak-and-decline patterns "
               "reflecting integrated pest management adoption.")


# ────────────────────────────────────────────────────────────
# TAB 3: CROP PRODUCTION (India)
# ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🌾 Indian Crop Production Analytics")

    col1, col2 = st.columns(2)
    with col1:
        top_crop_prod = crop_yield.groupby('Crop')['Production'].sum().sort_values(
            ascending=False).head(15).reset_index()
        fig = px.bar(top_crop_prod, x='Production', y='Crop', orientation='h',
                     color='Production', color_continuous_scale='Greens',
                     title='Top 15 Crops by Total Production',
                     labels={'Production': 'Total Production (Tonnes)'})
        fig.update_layout(template='plotly_white', height=420,
                          yaxis=dict(autorange='reversed'), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Sugarcane, rice, and wheat dominate Indian agricultural output "
                   "by volume, reflecting the staple food demand of 1.4 billion people.")

    with col2:
        top_state_prod = crop_yield.groupby('State')['Production'].sum().sort_values(
            ascending=False).head(15).reset_index()
        fig2 = px.bar(top_state_prod, x='State', y='Production',
                      color='Production', color_continuous_scale='Viridis',
                      title='Top 15 States by Total Production')
        fig2.update_layout(template='plotly_white', height=420,
                           xaxis=dict(tickangle=45), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("States like Uttar Pradesh and Maharashtra lead in absolute production "
                   "volumes, driven by large arable land areas and irrigation availability.")

    # Season-wise production
    season_prod = crop_yield.groupby('Season')['Production'].sum().reset_index()
    season_prod['Season'] = season_prod['Season'].str.strip()
    fig3 = px.pie(season_prod, names='Season', values='Production',
                  title='Production Share by Season',
                  color_discrete_sequence=px.colors.qualitative.Set2)
    col3, _ = st.columns([1, 1])
    with col3:
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Kharif (monsoon) season contributes the highest share of India's "
                   "total agricultural output, followed by Rabi (winter) crops.")

    # Yield over years for selected crop
    selected_crop = st.selectbox("Select Crop to Explore", sorted(crop_yield['Crop'].unique()))
    crop_sub = crop_yield[crop_yield['Crop'] == selected_crop]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=crop_sub['Crop_Year'], y=crop_sub['Yield'],
        mode='markers', marker=dict(color='#2d9e47', size=5, opacity=0.5),
        name='Individual Records'
    ))
    yearly_avg = crop_sub.groupby('Crop_Year')['Yield'].mean()
    fig4.add_trace(go.Scatter(
        x=yearly_avg.index, y=yearly_avg.values,
        mode='lines+markers', line=dict(color='#D32F2F', width=2),
        name='Annual Average'
    ))
    fig4.update_layout(
        title=f'{selected_crop} – Yield Trend Over Years',
        template='plotly_white', height=380,
        xaxis_title='Year', yaxis_title='Yield (T/Ha)'
    )
    st.plotly_chart(fig4, use_container_width=True)


# ────────────────────────────────────────────────────────────
# TAB 4: CROP NUTRITION (Crop Recommendation Dataset)
# ────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🔬 Soil Nutrient & Climate Analysis by Crop")

    col1, col2 = st.columns(2)
    with col1:
        # Average NPK per crop
        npk_avg = crop_rec.groupby('label')[['N', 'P', 'K']].mean().reset_index()
        fig = px.bar(npk_avg, x='label', y=['N', 'P', 'K'],
                     barmode='group', title='Average NPK by Crop',
                     labels={'label': 'Crop', 'value': 'kg/ha', 'variable': 'Nutrient'},
                     color_discrete_sequence=['#1976D2', '#388E3C', '#F57F17'])
        fig.update_layout(template='plotly_white', height=380,
                          xaxis=dict(tickangle=45))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("N, P, K requirements vary widely across crops. Legumes typically "
                   "need less N (nitrogen fixation), while cereals demand more.")

    with col2:
        # Correlation heatmap
        import numpy as np
        num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        corr = crop_rec[num_cols].corr()
        fig2 = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                         title='Feature Correlation Heatmap', zmin=-1, zmax=1)
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Correlation matrix shows feature inter-dependencies. "
                   "Most soil features are weakly correlated, indicating high information value individually.")

    # pH distribution by crop
    fig3 = px.violin(crop_rec, x='label', y='ph', box=True,
                     title='pH Requirement Distribution by Crop',
                     labels={'label': 'Crop', 'ph': 'Soil pH'})
    fig3.update_layout(template='plotly_white', height=400,
                       xaxis=dict(tickangle=45))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("pH tolerance bands differ markedly between crops. Coffee prefers acidic "
               "soils, while chickpea thrives in near-neutral conditions.")


# ────────────────────────────────────────────────────────────
# TAB 5: GLOBAL YIELD (yield_df)
# ────────────────────────────────────────────────────────────
with tab5:
    st.subheader("📈 Global Crop Yield & Climate Correlation")

    col1, col2 = st.columns(2)
    with col1:
        # Global yield trend
        global_yield = yield_df.groupby('Year')['hg/ha_yield'].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=global_yield['Year'], y=global_yield['hg/ha_yield'],
            mode='lines+markers', fill='tozeroy',
            fillcolor='rgba(45,158,71,0.15)',
            line=dict(color='#2d9e47', width=2)
        ))
        fig.update_layout(title='Global Average Crop Yield (hg/ha) Trend',
                          template='plotly_white', height=360,
                          xaxis_title='Year', yaxis_title='Yield (hg/ha)')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The green revolution and subsequent agricultural technology adoption "
                   "drove sustained yield improvements across global cropping systems.")

    with col2:
        # Rainfall vs Yield scatter
        fig2 = px.scatter(yield_df.sample(min(3000, len(yield_df)), random_state=42),
                          x='average_rain_fall_mm_per_year', y='hg/ha_yield',
                          color='avg_temp', color_continuous_scale='RdYlGn',
                          opacity=0.5,
                          title='Rainfall vs Yield (coloured by Temperature)',
                          labels={
                              'average_rain_fall_mm_per_year': 'Annual Rainfall (mm)',
                              'hg/ha_yield': 'Yield (hg/ha)',
                              'avg_temp': 'Avg Temp (°C)'
                          })
        fig2.update_layout(template='plotly_white', height=360)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("An optimal rainfall-temperature combination exists for peak yields. "
                   "Extreme values in either direction depress agricultural productivity.")

    # Top 10 countries by avg yield
    top_yield = yield_df.groupby('Area')['hg/ha_yield'].mean().sort_values(
        ascending=False).head(10).reset_index()
    fig3 = px.bar(top_yield, x='Area', y='hg/ha_yield',
                  color='hg/ha_yield', color_continuous_scale='Greens',
                  title='Top 10 Countries by Average Crop Yield',
                  labels={'hg/ha_yield': 'Avg Yield (hg/ha)', 'Area': 'Country'})
    fig3.update_layout(template='plotly_white', height=360, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("High-efficiency agricultural economies achieve extraordinary yields per "
               "hectare through genetic improvement, irrigation, and precision farming.")

    # Pesticide vs Yield
    fig4 = px.scatter(yield_df.sample(min(2000, len(yield_df)), random_state=1),
                      x='pesticides_tonnes', y='hg/ha_yield',
                      trendline='ols', opacity=0.4,
                      title='Pesticide Usage vs Crop Yield',
                      labels={'pesticides_tonnes': 'Pesticides (tonnes)',
                              'hg/ha_yield': 'Yield (hg/ha)'})
    fig4.update_layout(template='plotly_white', height=360)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Moderate pesticide application correlates with higher yields, "
               "but diminishing returns and environmental costs emerge at high doses.")
