# 🌾 AI-Powered Smart Agriculture Advisor

> **BS Computer Science – Semester Data Science Project**

An intelligent, end-to-end machine-learning platform that helps farmers make data-driven decisions through Crop Recommendation, Fertilizer Recommendation, Crop Yield Prediction, and an interactive Agricultural Analytics Dashboard.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Modules](#modules)
4. [Dataset Summary](#dataset-summary)
5. [ML Results](#ml-results)
6. [Project Structure](#project-structure)
7. [Installation & Setup](#installation--setup)
8. [Running the Application](#running-the-application)
9. [Technologies Used](#technologies-used)

---

## Project Overview

Traditional farming relies on generational knowledge and intuition. This system replaces guesswork with data science:

| Challenge | Our Solution |
|-----------|-------------|
| Wrong crop selection | ML-based Crop Recommendation (99.2% accuracy) |
| Over/under fertilization | AI Fertilizer Advisor (100% accuracy) |
| Unpredictable harvests | Yield Prediction (R² = 0.947) |
| No data visibility | Interactive Analytics Dashboard |

---

## System Architecture

```
User Input (Streamlit UI)
        │
        ▼
┌─────────────────────────────────────────────┐
│           Streamlit Web Application          │
│  Home │ Crop Rec │ Fertilizer │ Yield │ EDA  │
└────────────────┬────────────────────────────┘
                 │
        ┌────────▼────────┐
        │  saved_models/   │  ← joblib .pkl files
        │  (RF, DT, GBR…) │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  src/ modules   │
        │ preprocessing   │
        │    models       │
        │ visualization   │
        │    utils        │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │   datasets/      │
        │  7 CSV files     │
        └─────────────────┘
```

---

## Modules

### Module 1 – Crop Recommendation
- **Input:** N, P, K, Temperature, Humidity, pH, Rainfall
- **Output:** Recommended crop + confidence scores (top-5)
- **Best Model:** Naive Bayes / Random Forest (tied at 99.22%)
- **Features:** Gauge charts, probability bars, crop reference table

### Module 2 – Fertilizer Recommendation
- **Input:** Soil Type, Crop Type, N, P, K, Temperature, Humidity, Moisture
- **Output:** Recommended fertilizer + NPK deficiency analysis
- **Best Model:** Decision Tree / Gradient Boosting (100%)
- **Features:** NPK radar chart, fertilizer info cards

### Module 3 – Crop Yield Prediction
- **Input:** Crop, Year, Season, State, Area, Rainfall, Fertilizer, Pesticide
- **Output:** Predicted yield (Tonnes/Ha) + total production estimate
- **Best Model:** Random Forest Regressor (R² = 0.9467)
- **Features:** Historical trend overlay, state-wise comparison

### Module 4 – Analytics Dashboard
- Climate trends (rainfall & temperature, 1849–present)
- Pesticide usage (global & country-level)
- Indian crop production by state and season
- Soil nutrient profiles per crop
- Global yield vs rainfall/temperature correlation

---

## Dataset Summary

| Dataset | File | Rows | Purpose |
|---------|------|------|---------|
| Crop Recommendation | Crop_recommendation.csv | 2,200 | Classification |
| Fertilizer (Synthetic) | Generated | 5,000 | Classification |
| Crop Yield (India) | crop_yield.csv | 19,689 | Regression |
| Yield Features (Global) | yield_df.csv | 28,242 | Analysis |
| Rainfall | rainfall.csv | 6,727 | Climate EDA |
| Temperature | temp.csv | 71,311 | Climate EDA |
| Pesticides | pesticides.csv | 4,349 | Trend EDA |

---

## ML Results

### Crop Recommendation (Classification)

| Model | Accuracy | Precision | Recall | F1-Score | CV Score |
|-------|----------|-----------|--------|----------|----------|
| Logistic Regression | 97.16% | 97.23% | 97.16% | 97.16% | 96.71% |
| Decision Tree | 98.71% | 98.86% | 98.71% | 98.69% | 98.32% |
| **Random Forest** | **99.22%** | **99.28%** | **99.22%** | **99.22%** | **99.16%** |
| KNN | 96.90% | 97.33% | 96.90% | 96.93% | 96.12% |
| **Naive Bayes** | **99.22%** | **99.23%** | **99.22%** | **99.22%** | **99.42%** |
| XGBoost | 97.67% | 97.95% | 97.67% | 97.65% | 98.39% |

### Fertilizer Recommendation (Classification)

| Model | Accuracy | F1-Score | CV Score |
|-------|----------|----------|----------|
| **Decision Tree** | **100%** | **100%** | **100%** |
| **Random Forest** | **100%** | **100%** | **99.98%** |
| KNN | 71.50% | 71.59% | 69.37% |
| **Gradient Boosting** | **100%** | **100%** | **100%** |

### Crop Yield Prediction (Regression)

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | 4.9425 | 10.1503 | 0.1191 |
| **Random Forest** | **0.7587** | **2.4976** | **0.9467** |
| Gradient Boosting | 2.0940 | 4.9347 | 0.7918 |
| XGBoost | 1.0724 | 2.6370 | 0.9405 |

---

## Project Structure

```
Smart_Agriculture_Advisor/
│
├── datasets/                        # All CSV data files
│   ├── Crop_recommendation.csv
│   ├── crop_yield.csv
│   ├── pesticides.csv
│   ├── rainfall.csv
│   ├── temp.csv
│   ├── yield.csv
│   └── yield_df.csv
│
├── notebooks/
│   └── Smart_Agriculture_Advisor.ipynb   # Complete EDA + ML notebook
│
├── src/
│   ├── preprocessing/
│   │   └── data_preprocessor.py     # All dataset loading & cleaning
│   ├── models/
│   │   └── train_models.py          # Train, evaluate, compare models
│   ├── visualization/
│   │   └── eda_plots.py             # All EDA & result charts
│   └── utils/
│       └── helpers.py               # Prediction helpers & utilities
│
├── saved_models/                    # Trained .pkl model files
│   ├── crop_recommendation_model.pkl
│   ├── crop_rec_scaler.pkl
│   ├── crop_rec_label_encoder.pkl
│   ├── fertilizer_recommendation_model.pkl
│   ├── fertilizer_scaler.pkl
│   ├── fertilizer_le_soil.pkl
│   ├── fertilizer_le_crop.pkl
│   ├── fertilizer_le_fert.pkl
│   ├── crop_yield_model.pkl
│   ├── yield_scaler.pkl
│   ├── yield_le_crop.pkl
│   ├── yield_le_season.pkl
│   └── yield_le_state.pkl
│
├── streamlit_app/
│   ├── Home.py                      # Landing page
│   └── pages/
│       ├── 1_Crop_Recommendation.py
│       ├── 2_Fertilizer_Recommendation.py
│       ├── 3_Yield_Prediction.py
│       ├── 4_Analytics_Dashboard.py
│       └── 5_Model_Evaluation.py
│
├── reports/
│   ├── plots/                       # 18+ EDA & result plots
│   └── Smart_Agriculture_Report.md  # University project report
│
├── main.py                          # Full pipeline runner
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip

### Step 1: Clone / Extract the Project
```bash
cd Smart_Agriculture_Advisor
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train Models (generates saved_models/)
```bash
python main.py
```
> This runs the full pipeline: preprocessing → EDA → training → evaluation → saving.
> Runtime: ~3–5 minutes depending on hardware.

### Step 4: Launch Streamlit App
```bash
streamlit run streamlit_app/Home.py
```

Then open your browser at `http://localhost:8501`

---

## Running the Application

| Page | URL Path | Description |
|------|----------|-------------|
| Home | `/` | Project overview, KPIs, dataset summary |
| Crop Recommendation | `/Crop_Recommendation` | Predict best crop |
| Fertilizer Recommendation | `/Fertilizer_Recommendation` | Recommend fertilizer |
| Yield Prediction | `/Yield_Prediction` | Predict harvest yield |
| Analytics Dashboard | `/Analytics_Dashboard` | Interactive charts |
| Model Evaluation | `/Model_Evaluation` | Metrics & comparisons |

---

## Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| ML Framework | scikit-learn 1.2+, XGBoost 1.7+ |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web App | Streamlit |
| Model Persistence | joblib |
| Notebook | Jupyter |

---

*AI-Powered Smart Agriculture Advisor · BS Computer Science Semester Project*
