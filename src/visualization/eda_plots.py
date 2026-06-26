"""
Visualization Module – Smart Agriculture Advisor
Generates all EDA and result charts using Matplotlib, Seaborn, and Plotly.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='muted')
PLOT_DIR = "reports/plots/"
os.makedirs(PLOT_DIR, exist_ok=True)

def _save_fig(name):
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved plot → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CROP RECOMMENDATION EDA
# ─────────────────────────────────────────────────────────────────────────────

def plot_crop_distribution(df):
    """Bar chart of crop frequency."""
    fig, ax = plt.subplots(figsize=(14, 5))
    counts = df['label'].value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette='viridis', ax=ax)
    ax.set_title('Crop Distribution in Dataset', fontsize=16, fontweight='bold')
    ax.set_xlabel('Crop Name'); ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    return _save_fig('01_crop_distribution.png')


def plot_correlation_heatmap(df):
    """Correlation heatmap of numerical features."""
    fig, ax = plt.subplots(figsize=(9, 7))
    num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=15, fontweight='bold')
    return _save_fig('02_correlation_heatmap.png')


def plot_npk_analysis(df):
    """Boxplots of N, P, K per crop (top 10)."""
    top10 = df['label'].value_counts().head(10).index
    sub   = df[df['label'].isin(top10)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, col, color in zip(axes, ['N', 'P', 'K'],
                               ['steelblue', 'salmon', 'mediumseagreen']):
        sns.boxplot(x='label', y=col, data=sub, palette='Set2', ax=ax)
        ax.set_title(f'{col} by Crop', fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
    plt.suptitle('NPK Analysis (Top 10 Crops)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    return _save_fig('03_npk_analysis.png')


def plot_ph_rainfall_dist(df):
    """Distribution plots for pH and Rainfall."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df['ph'], bins=30, kde=True, color='orchid', ax=axes[0])
    axes[0].set_title('pH Distribution', fontsize=13, fontweight='bold')
    sns.histplot(df['rainfall'], bins=30, kde=True, color='teal', ax=axes[1])
    axes[1].set_title('Rainfall Distribution', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return _save_fig('04_ph_rainfall_dist.png')


def plot_temp_humidity(df):
    """Scatter plot of temperature vs humidity coloured by crop."""
    fig, ax = plt.subplots(figsize=(11, 7))
    for crop in df['label'].unique():
        sub = df[df['label'] == crop]
        ax.scatter(sub['temperature'], sub['humidity'], label=crop, alpha=0.5, s=15)
    ax.set_xlabel('Temperature (°C)'); ax.set_ylabel('Humidity (%)')
    ax.set_title('Temperature vs Humidity by Crop', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=6, ncol=3)
    return _save_fig('05_temp_humidity_scatter.png')


def plot_feature_boxplots(df):
    """Boxplots for all 7 features."""
    num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(y=df[col], color='skyblue', ax=axes[i])
        axes[i].set_title(col, fontsize=12)
    axes[-1].set_visible(False)
    plt.suptitle('Feature Boxplots (Outlier Detection)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _save_fig('06_feature_boxplots.png')


def run_crop_eda(df):
    print("\n[EDA] Crop Recommendation …")
    paths = []
    paths.append(plot_crop_distribution(df))
    paths.append(plot_correlation_heatmap(df))
    paths.append(plot_npk_analysis(df))
    paths.append(plot_ph_rainfall_dist(df))
    paths.append(plot_temp_humidity(df))
    paths.append(plot_feature_boxplots(df))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# FERTILIZER EDA
# ─────────────────────────────────────────────────────────────────────────────

def run_fertilizer_eda(df):
    print("\n[EDA] Fertilizer Recommendation …")
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts = df['Fertilizer Name'].value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette='magma', ax=axes[0])
    axes[0].set_title('Fertilizer Distribution', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=30)
    counts2 = df['Soil Type'].value_counts()
    sns.barplot(x=counts2.index, y=counts2.values, palette='Set1', ax=axes[1])
    axes[1].set_title('Soil Type Distribution', fontweight='bold')
    plt.tight_layout()
    paths.append(_save_fig('07_fertilizer_soil_dist.png'))

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = df.groupby(['Fertilizer Name', 'Soil Type']).size().unstack(fill_value=0)
    pivot.plot(kind='bar', ax=ax, colormap='tab10')
    ax.set_title('Fertilizer Usage by Soil Type', fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    paths.append(_save_fig('08_fert_soil_heatmap.png'))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, ['Nitrogen', 'Potassium', 'Phosphorous']):
        sns.violinplot(x='Fertilizer Name', y=col, data=df, palette='pastel', ax=ax)
        ax.set_title(f'{col} by Fertilizer', fontweight='bold')
        ax.tick_params(axis='x', rotation=30)
    plt.suptitle('Nutrient Distribution by Fertilizer', fontsize=14, fontweight='bold')
    plt.tight_layout()
    paths.append(_save_fig('09_nutrient_analysis.png'))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# CROP YIELD EDA
# ─────────────────────────────────────────────────────────────────────────────

def run_yield_eda(df):
    print("\n[EDA] Crop Yield …")
    paths = []

    # Yield distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Yield'], bins=50, kde=True, color='coral', ax=ax)
    ax.set_title('Crop Yield Distribution', fontsize=14, fontweight='bold')
    paths.append(_save_fig('10_yield_distribution.png'))

    # Top 15 crops by median yield
    fig, ax = plt.subplots(figsize=(14, 5))
    top = df.groupby('Crop')['Yield'].median().sort_values(ascending=False).head(15)
    sns.barplot(x=top.index, y=top.values, palette='coolwarm', ax=ax)
    ax.set_title('Top 15 Crops by Median Yield', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    paths.append(_save_fig('11_crop_yield_comparison.png'))

    # State-wise production
    fig, ax = plt.subplots(figsize=(14, 5))
    state_prod = df.groupby('State')['Production'].sum().sort_values(ascending=False).head(15)
    sns.barplot(x=state_prod.index, y=state_prod.values, palette='viridis', ax=ax)
    ax.set_title('Top 15 States by Total Production', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    paths.append(_save_fig('12_statewise_production.png'))

    # Yield trend over years
    fig, ax = plt.subplots(figsize=(12, 5))
    yearly = df.groupby('Crop_Year')['Yield'].mean()
    ax.plot(yearly.index, yearly.values, marker='o', color='steelblue', linewidth=2)
    ax.set_title('Average Yield Trend Over Years', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Avg Yield')
    paths.append(_save_fig('13_yield_trend.png'))

    # Rainfall vs Yield scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df['Annual_Rainfall'], df['Yield'], alpha=0.2, s=10, color='teal')
    ax.set_xlabel('Annual Rainfall (mm)'); ax.set_ylabel('Yield')
    ax.set_title('Annual Rainfall vs Yield', fontsize=14, fontweight='bold')
    paths.append(_save_fig('14_rainfall_vs_yield.png'))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# CLIMATE & PESTICIDE EDA
# ─────────────────────────────────────────────────────────────────────────────

def run_climate_eda(analytics):
    print("\n[EDA] Climate & Pesticide …")
    paths = []

    rain = analytics['rainfall']
    temp = analytics['temperature']
    pest = analytics['pesticides']

    # Global average rainfall trend
    fig, ax = plt.subplots(figsize=(12, 5))
    g = rain.groupby('Year')['average_rain_fall_mm_per_year'].mean()
    ax.plot(g.index, g.values, color='royalblue', linewidth=2)
    ax.fill_between(g.index, g.values, alpha=0.2, color='royalblue')
    ax.set_title('Global Average Rainfall Trend', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Rainfall (mm/year)')
    paths.append(_save_fig('15_global_rainfall_trend.png'))

    # Temperature trend
    fig, ax = plt.subplots(figsize=(12, 5))
    g = temp.groupby('year')['avg_temp'].mean()
    ax.plot(g.index, g.values, color='crimson', linewidth=2)
    ax.fill_between(g.index, g.values, alpha=0.15, color='crimson')
    ax.set_title('Global Average Temperature Trend', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Avg Temperature (°C)')
    paths.append(_save_fig('16_global_temp_trend.png'))

    # Pesticide usage trend
    fig, ax = plt.subplots(figsize=(12, 5))
    g = pest.groupby('Year')['Value'].sum()
    ax.bar(g.index, g.values, color='darkorange', alpha=0.8)
    ax.set_title('Global Pesticide Usage Trend', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Total Pesticide (tonnes AI)')
    paths.append(_save_fig('17_pesticide_usage_trend.png'))

    # Top 10 countries by pesticide use
    fig, ax = plt.subplots(figsize=(12, 5))
    top10 = pest.groupby('Area')['Value'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top10.index, y=top10.values, palette='Reds_r', ax=ax)
    ax.set_title('Top 10 Countries by Pesticide Usage', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    paths.append(_save_fig('18_pesticide_by_country.png'))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# MODEL RESULT PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(results, title, metric='accuracy'):
    """Bar chart comparing model metrics."""
    names   = [r['model_name'] for r in results]
    scores  = [r[metric] for r in results]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, scores, color=sns.color_palette('husl', len(names)))
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(metric.capitalize()); ax.set_title(title, fontsize=14, fontweight='bold')
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', va='center', fontsize=10)
    plt.tight_layout()
    fname = title.replace(' ', '_').lower() + '.png'
    return _save_fig(fname)


def plot_feature_importance(fi_series, title):
    """Horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fi_series.sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    fname = title.replace(' ', '_').lower() + '.png'
    return _save_fig(fname)


def plot_confusion_matrix(cm, classes, title):
    """Heatmap of confusion matrix (up to 20×20 legible)."""
    fig, ax = plt.subplots(figsize=(min(20, len(classes)), min(18, len(classes))))
    sns.heatmap(cm, annot=(len(classes) <= 12), fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = title.replace(' ', '_').lower() + '.png'
    return _save_fig(fname)


def plot_yield_actual_vs_pred(y_test, y_pred, model_name):
    """Scatter plot of actual vs predicted yield."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, y_pred, alpha=0.3, s=15, color='mediumseagreen')
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect fit')
    ax.set_xlabel('Actual Yield'); ax.set_ylabel('Predicted Yield')
    ax.set_title(f'Actual vs Predicted – {model_name}', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    return _save_fig(f'actual_vs_pred_{model_name.replace(" ","_").lower()}.png')
