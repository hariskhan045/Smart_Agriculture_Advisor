"""
Data Preprocessing Module
Smart Agriculture Advisor
Handles cleaning, encoding, scaling, and feature engineering for all datasets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = "datasets/"

# ─────────────────────────────────────────────────────────────────────────────
# CROP RECOMMENDATION PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_crop_recommendation():
    """Load and preprocess the Crop Recommendation dataset."""
    df = pd.read_csv(BASE_PATH + "Crop_recommendation.csv")

    print(f"[CropRec] Raw shape: {df.shape}")

    # 1. Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[CropRec] Dropped {before - len(df)} duplicates.")

    # 2. Missing value check
    missing = df.isnull().sum()
    print(f"[CropRec] Missing values:\n{missing[missing > 0]}")
    df.dropna(inplace=True)

    # 3. Outlier removal (IQR on numeric columns)
    numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    for col in numeric_cols:
        Q1 = df[col].quantile(0.01)
        Q3 = df[col].quantile(0.99)
        df = df[(df[col] >= Q1) & (df[col] <= Q3)]
    print(f"[CropRec] After outlier removal: {df.shape}")

    # 4. Label encode target
    le = LabelEncoder()
    df['crop_encoded'] = le.fit_transform(df['label'])

    # 5. Feature & target split
    X = df[numeric_cols]
    y = df['label']

    # 6. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return {
        'df': df,
        'X': X, 'y': y,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'label_encoder': le,
        'feature_names': numeric_cols,
        'target_classes': le.classes_.tolist()
    }


# ─────────────────────────────────────────────────────────────────────────────
# FERTILIZER RECOMMENDATION – BUILT FROM CROP_YIELD.CSV
# ─────────────────────────────────────────────────────────────────────────────

FERTILIZER_RULES = {
    'Arecanut': 'Urea', 'Arhar/Tur': 'DAP', 'Bajra': 'Urea',
    'Banana': 'MOP', 'Castor seed': 'SSP', 'Coconut': 'MOP',
    'Cotton(lint)': 'Urea', 'Dry chillies': 'DAP', 'Garlic': 'Urea',
    'Ginger': 'SSP', 'Gram': 'DAP', 'Groundnut': 'SSP',
    'Horse-gram': 'DAP', 'Jowar': 'Urea', 'Jute': 'Urea',
    'Linseed': 'SSP', 'Maize': 'Urea', 'Masoor': 'DAP',
    'Mesta': 'Urea', 'Moong(Green Gram)': 'DAP', 'Moth': 'DAP',
    'Niger seed': 'SSP', 'Onion': 'Urea', 'Other  Rabi pulses': 'DAP',
    'Other Cereals': 'Urea', 'Other Kharif pulses': 'DAP',
    'Other Summer Pulses': 'DAP', 'Potato': 'MOP', 'Ragi': 'Urea',
    'Rapeseed &Mustard': 'DAP', 'Rice': 'Urea', 'Safflower': 'SSP',
    'Sesamum': 'SSP', 'Small millets': 'Urea', 'Soyabean': 'DAP',
    'Sugarcane': 'Urea', 'Sunflower': 'DAP', 'Sweet potato': 'MOP',
    'Tapioca': 'MOP', 'Tobacco': 'Urea', 'Turmeric': 'MOP',
    'Urad': 'DAP', 'Wheat': 'Urea',
}

SOIL_RULES = {
    'Sandy': {'Urea': 'DAP', 'DAP': 'DAP', 'SSP': 'SSP', 'MOP': 'MOP'},
    'Loamy': {'Urea': 'Urea', 'DAP': 'Urea', 'SSP': 'SSP', 'MOP': 'MOP'},
    'Clay': {'Urea': '20-20', 'DAP': 'DAP', 'SSP': '14-35-14', 'MOP': 'MOP'},
    'Black': {'Urea': 'Urea', 'DAP': 'DAP', 'SSP': 'SSP', 'MOP': 'MOP'},
    'Red': {'Urea': 'Urea', 'DAP': '28-28', 'SSP': 'SSP', 'MOP': 'MOP'},
}

FERTILIZER_FEATURES = [
    'Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
    'Nitrogen', 'Potassium', 'Phosphorous'
]

SOIL_TYPES  = ['Sandy', 'Loamy', 'Clay', 'Black', 'Red']
FERT_NAMES  = ['Urea', 'DAP', 'MOP', 'SSP', '28-28', '14-35-14', '20-20', '10-26-26']
CROP_TYPES  = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley',
               'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']

def _generate_fertilizer_dataset(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    soil  = rng.choice(SOIL_TYPES, n)
    crop  = rng.choice(CROP_TYPES, n)
    temp  = rng.uniform(15, 45, n).round(1)
    hum   = rng.uniform(20, 90, n).round(1)
    moist = rng.uniform(20, 80, n).round(1)
    nitro = rng.integers(0, 100, n).astype(float)
    pots  = rng.integers(0, 100, n).astype(float)
    phos  = rng.integers(0, 100, n).astype(float)

    fert = []
    for i in range(n):
        base = 'Urea'
        if nitro[i] < 30:
            base = 'DAP'
        elif phos[i] < 30:
            base = 'SSP'
        elif pots[i] < 30:
            base = 'MOP'
        adjusted = SOIL_RULES.get(soil[i], {}).get(base, base)
        fert.append(adjusted)

    return pd.DataFrame({
        'Temperature': temp, 'Humidity': hum, 'Moisture': moist,
        'Soil Type': soil, 'Crop Type': crop,
        'Nitrogen': nitro, 'Potassium': pots, 'Phosphorous': phos,
        'Fertilizer Name': fert
    })


def load_fertilizer_recommendation():
    """Generate and preprocess a Fertilizer Recommendation dataset."""
    df = _generate_fertilizer_dataset()
    print(f"[Fertilizer] Synthetic dataset shape: {df.shape}")

    le_soil  = LabelEncoder()
    le_crop  = LabelEncoder()
    le_fert  = LabelEncoder()

    df['Soil_encoded'] = le_soil.fit_transform(df['Soil Type'])
    df['Crop_encoded'] = le_crop.fit_transform(df['Crop Type'])
    df['Fert_encoded'] = le_fert.fit_transform(df['Fertilizer Name'])

    num_features = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    cat_features = ['Soil_encoded', 'Crop_encoded']
    all_features = num_features + cat_features

    X = df[all_features]
    y = df['Fertilizer Name']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return {
        'df': df,
        'X': X, 'y': y,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'le_soil': le_soil, 'le_crop': le_crop, 'le_fert': le_fert,
        'feature_names': all_features,
        'target_classes': le_fert.classes_.tolist(),
        'soil_types': SOIL_TYPES, 'crop_types': CROP_TYPES
    }


# ─────────────────────────────────────────────────────────────────────────────
# CROP YIELD PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def load_crop_yield():
    """Load and preprocess the Crop Yield dataset."""
    df = pd.read_csv(BASE_PATH + "crop_yield.csv")
    print(f"[CropYield] Raw shape: {df.shape}")

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print(f"[CropYield] After dedup/dropna: {df.shape}")

    # Cap extreme yield outliers (above 99th percentile)
    cap = df['Yield'].quantile(0.99)
    df = df[df['Yield'] <= cap]
    df = df[df['Yield'] > 0]
    print(f"[CropYield] After yield capping: {df.shape}")

    le_crop   = LabelEncoder()
    le_season = LabelEncoder()
    le_state  = LabelEncoder()

    df['Crop_enc']   = le_crop.fit_transform(df['Crop'])
    df['Season_enc'] = le_season.fit_transform(df['Season'].str.strip())
    df['State_enc']  = le_state.fit_transform(df['State'])

    feature_cols = ['Crop_enc', 'Crop_Year', 'Season_enc', 'State_enc',
                    'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    target_col   = 'Yield'

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return {
        'df': df,
        'X': X, 'y': y,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'le_crop': le_crop, 'le_season': le_season, 'le_state': le_state,
        'feature_names': feature_cols,
        'crops': le_crop.classes_.tolist(),
        'seasons': le_season.classes_.tolist(),
        'states': le_state.classes_.tolist()
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS DATASETS
# ─────────────────────────────────────────────────────────────────────────────

def load_analytics_datasets():
    """Load rainfall, temperature, pesticide data for dashboard."""
    rain = pd.read_csv(BASE_PATH + "rainfall.csv")
    rain.columns = rain.columns.str.strip()

    temp = pd.read_csv(BASE_PATH + "temp.csv")

    pest = pd.read_csv(BASE_PATH + "pesticides.csv")
    pest.columns = pest.columns.str.strip()

    yield_df = pd.read_csv(BASE_PATH + "yield_df.csv")

    rain['average_rain_fall_mm_per_year'] = pd.to_numeric(
        rain['average_rain_fall_mm_per_year'], errors='coerce')
    rain.dropna(inplace=True)
    temp.dropna(inplace=True)
    pest.dropna(inplace=True)
    yield_df.dropna(inplace=True)

    return {'rainfall': rain, 'temperature': temp,
            'pesticides': pest, 'yield_df': yield_df}
