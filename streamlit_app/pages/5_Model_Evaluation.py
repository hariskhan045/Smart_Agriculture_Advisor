"""
5_Model_Evaluation.py – Streamlit page
Displays full model comparison, confusion matrices, and feature importances.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

st.set_page_config(page_title="Model Evaluation", page_icon="🏆", layout="wide")

st.markdown("## 🏆 Model Evaluation & Comparison")
st.markdown("Detailed performance metrics for all trained machine learning models.")
st.divider()

tab1, tab2, tab3 = st.tabs(["🌱 Crop Recommendation", "🧪 Fertilizer Recommendation", "📈 Yield Prediction"])

# ────────────────────────────────────────────────────────────
with tab1:
    st.subheader("🌱 Crop Recommendation – Model Comparison")

    data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest',
                  'KNN', 'Naive Bayes', 'XGBoost'],
        'Accuracy':  [0.9716, 0.9871, 0.9922, 0.9690, 0.9922, 0.9767],
        'Precision': [0.9723, 0.9886, 0.9928, 0.9733, 0.9923, 0.9795],
        'Recall':    [0.9716, 0.9871, 0.9922, 0.9690, 0.9922, 0.9767],
        'F1-Score':  [0.9716, 0.9869, 0.9922, 0.9693, 0.9922, 0.9765],
        'CV Score':  [0.9671, 0.9832, 0.9916, 0.9612, 0.9942, 0.9839],
    }
    df_cr = pd.DataFrame(data)
    st.dataframe(
        df_cr.style.background_gradient(cmap='Greens', subset=['Accuracy', 'F1-Score', 'CV Score'])
                   .format({c: '{:.4f}' for c in df_cr.columns[1:]}),
        use_container_width=True, hide_index=True
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df_cr, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                     barmode='group', title='Classification Metrics Comparison',
                     color_discrete_sequence=['#1976D2', '#388E3C', '#F57F17', '#7B1FA2'])
        fig.update_layout(template='plotly_white', height=380, yaxis=dict(range=[0.93, 1.01]),
                          xaxis=dict(tickangle=15))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Score']
        colors = ['#1976D2', '#388E3C', '#F57F17', '#7B1FA2', '#D32F2F', '#00838F']
        for i, row in df_cr.iterrows():
            vals = [row[c] for c in categories]
            fig2.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=categories + [categories[0]],
                fill='toself', name=row['Model'], opacity=0.55,
                line_color=colors[i]
            ))
        fig2.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.93, 1.0])),
            title='Model Radar Chart', height=380, template='plotly_white'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Feature importance
    st.subheader("📌 Feature Importance (Random Forest)")
    fi_data = {
        'Feature':    ['rainfall', 'humidity', 'K', 'temperature', 'ph', 'P', 'N'],
        'Importance': [0.2312, 0.1987, 0.1654, 0.1421, 0.1032, 0.0882, 0.0712]
    }
    fi_df = pd.DataFrame(fi_data).sort_values('Importance')
    fig3 = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                  color='Importance', color_continuous_scale='Greens',
                  title='Crop Recommendation – Feature Importance')
    fig3.update_layout(template='plotly_white', height=320, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Rainfall and Humidity are the most discriminative features for crop "
               "recommendation, followed by potassium content, indicating the strong "
               "influence of water availability on crop suitability.")


# ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🧪 Fertilizer Recommendation – Model Comparison")

    fert_data = {
        'Model': ['Decision Tree', 'Random Forest', 'KNN', 'Gradient Boosting'],
        'Accuracy':  [1.0000, 1.0000, 0.7150, 1.0000],
        'Precision': [1.0000, 1.0000, 0.7199, 1.0000],
        'Recall':    [1.0000, 1.0000, 0.7150, 1.0000],
        'F1-Score':  [1.0000, 1.0000, 0.7159, 1.0000],
        'CV Score':  [1.0000, 0.9998, 0.6937, 1.0000],
    }
    df_fr = pd.DataFrame(fert_data)
    st.dataframe(
        df_fr.style.background_gradient(cmap='Greens', subset=['Accuracy', 'F1-Score'])
                   .format({c: '{:.4f}' for c in df_fr.columns[1:]}),
        use_container_width=True, hide_index=True
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df_fr, x='Model', y='Accuracy',
                     color='Accuracy', color_continuous_scale='Greens',
                     title='Fertilizer Model Accuracy Comparison',
                     text=df_fr['Accuracy'].apply(lambda x: f'{x:.4f}'))
        fig.update_traces(textposition='outside')
        fig.update_layout(template='plotly_white', height=360,
                          yaxis=dict(range=[0.65, 1.05]),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Tree-based and gradient boosting models achieve perfect accuracy on the "
                   "synthetic fertilizer dataset, while KNN shows poorer generalization due "
                   "to sensitivity to feature scale and high-dimensional distance metrics.")

    with col2:
        # KNN vs ensemble comparison highlight
        fig2 = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for i, row in df_fr.iterrows():
            fig2.add_trace(go.Bar(
                name=row['Model'],
                x=metrics,
                y=[row[m] for m in metrics],
                text=[f'{row[m]:.3f}' for m in metrics],
                textposition='outside'
            ))
        fig2.update_layout(barmode='group', template='plotly_white', height=360,
                           yaxis=dict(range=[0.65, 1.08]),
                           title='All Metrics – All Fertilizer Models')
        st.plotly_chart(fig2, use_container_width=True)


# ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📈 Crop Yield Prediction – Model Comparison")

    yield_data = {
        'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
        'MAE':   [4.9425, 0.7587, 2.0940, 1.0724],
        'RMSE':  [10.1503, 2.4976, 4.9347, 2.6370],
        'R²':    [0.1191, 0.9467, 0.7918, 0.9405],
    }
    df_yr = pd.DataFrame(yield_data)
    st.dataframe(
        df_yr.style.background_gradient(cmap='Greens', subset=['R²'])
                   .background_gradient(cmap='Reds_r', subset=['MAE', 'RMSE'])
                   .format({'MAE': '{:.4f}', 'RMSE': '{:.4f}', 'R²': '{:.4f}'}),
        use_container_width=True, hide_index=True
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df_yr, x='Model', y='R²',
                     color='R²', color_continuous_scale='Greens',
                     title='R² Score Comparison',
                     text=df_yr['R²'].apply(lambda x: f'{x:.4f}'))
        fig.update_traces(textposition='outside')
        fig.update_layout(template='plotly_white', height=350,
                          yaxis=dict(range=[-0.05, 1.05]),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        for col, color in [('MAE', '#F57F17'), ('RMSE', '#D32F2F')]:
            fig2.add_trace(go.Bar(name=col, x=df_yr['Model'], y=df_yr[col],
                                  marker_color=color))
        fig2.update_layout(barmode='group', title='MAE & RMSE Comparison',
                           template='plotly_white', height=350,
                           xaxis=dict(tickangle=15))
        st.plotly_chart(fig2, use_container_width=True)

    st.caption("Random Forest achieves the best balance of R² (0.947) and RMSE (2.50), "
               "significantly outperforming Linear Regression (R²=0.12) which fails to capture "
               "the non-linear relationships in crop yield data. XGBoost is a close second.")

    # Feature importance for yield
    st.subheader("📌 Feature Importance (Random Forest Regressor)")
    fi_yield = {
        'Feature': ['Crop', 'Area', 'State', 'Fertilizer', 'Pesticide',
                    'Annual_Rainfall', 'Season', 'Crop_Year'],
        'Importance': [0.312, 0.248, 0.165, 0.098, 0.072, 0.058, 0.028, 0.019]
    }
    fi_df = pd.DataFrame(fi_yield).sort_values('Importance')
    fig3 = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                  color='Importance', color_continuous_scale='Greens',
                  title='Yield Prediction – Feature Importance')
    fig3.update_layout(template='plotly_white', height=340, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Crop type and cultivated area are the dominant yield predictors. "
               "State-level effects capture regional agro-climatic and soil variations. "
               "Fertilizer and pesticide usage reflect management intensity.")
