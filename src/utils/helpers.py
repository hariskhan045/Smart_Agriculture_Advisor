"""
Utility Functions – Smart Agriculture Advisor
"""
import os, sys
import pandas as pd
import numpy as np
import joblib


def load_model(filename, base_dir="saved_models/"):
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}\nRun main.py first to train and save models.")
    return joblib.load(path)


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """Predict recommended crop from soil/climate inputs."""
    model   = load_model('crop_recommendation_model.pkl')
    scaler  = load_model('crop_rec_scaler.pkl')
    le      = load_model('crop_rec_label_encoder.pkl')
    X = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = None
    if hasattr(model, 'predict_proba'):
        proba_arr = model.predict_proba(X_scaled)[0]
        classes   = le.classes_
        proba = dict(zip(classes, (proba_arr * 100).round(2)))
    return pred, proba


def predict_fertilizer(temperature, humidity, moisture, soil_type, crop_type,
                        nitrogen, potassium, phosphorous):
    """Predict recommended fertilizer."""
    model    = load_model('fertilizer_recommendation_model.pkl')
    scaler   = load_model('fertilizer_scaler.pkl')
    le_soil  = load_model('fertilizer_le_soil.pkl')
    le_crop  = load_model('fertilizer_le_crop.pkl')

    soil_enc = le_soil.transform([soil_type])[0]
    crop_enc = le_crop.transform([crop_type])[0]
    X = np.array([[temperature, humidity, moisture, nitrogen, potassium,
                   phosphorous, soil_enc, crop_enc]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = None
    if hasattr(model, 'predict_proba'):
        proba_arr = model.predict_proba(X_scaled)[0]
        classes   = model.classes_
        proba = dict(zip(classes, (proba_arr * 100).round(2)))
    return pred, proba


def predict_yield(crop, crop_year, season, state, area, rainfall, fertilizer, pesticide):
    """Predict crop yield."""
    model     = load_model('crop_yield_model.pkl')
    scaler    = load_model('yield_scaler.pkl')
    le_crop   = load_model('yield_le_crop.pkl')
    le_season = load_model('yield_le_season.pkl')
    le_state  = load_model('yield_le_state.pkl')

    crop_enc   = le_crop.transform([crop])[0]
    season_enc = le_season.transform([season])[0]
    state_enc  = le_state.transform([state])[0]
    X = np.array([[crop_enc, crop_year, season_enc, state_enc,
                   area, rainfall, fertilizer, pesticide]])
    X_scaled = scaler.transform(X)
    return float(model.predict(X_scaled)[0])


def print_section(title):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def results_to_df(results, metric_keys):
    rows = []
    for r in results:
        row = {'Model': r['model_name']}
        for k in metric_keys:
            row[k.upper()] = round(r.get(k, np.nan), 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index('Model')
