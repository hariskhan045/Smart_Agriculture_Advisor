"""
ML Models Module – Smart Agriculture Advisor
Trains, evaluates, and saves models for:
  • Crop Recommendation (Classification)
  • Fertilizer Recommendation (Classification)
  • Crop Yield Prediction (Regression)
"""

import os, time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier, XGBRegressor

SAVED_DIR = "saved_models/"
os.makedirs(SAVED_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save(obj, name):
    path = os.path.join(SAVED_DIR, name)
    joblib.dump(obj, path)
    print(f"  ✓ Saved → {path}")
    return path

def _load(name):
    return joblib.load(os.path.join(SAVED_DIR, name))


def _eval_classifier(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    print(f"\n── {name} ──")
    print(f"  Accuracy : {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    return {'model_name': name, 'model': model,
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'y_pred': y_pred, 'confusion_matrix': cm}


def _eval_regressor(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    print(f"\n── {name} ──")
    print(f"  MAE: {mae:.4f}  MSE: {mse:.4f}  RMSE: {rmse:.4f}  R²: {r2:.4f}")
    return {'model_name': name, 'model': model,
            'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2,
            'y_pred': y_pred}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 – CROP RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────

def train_crop_recommendation(data):
    print("\n" + "="*60)
    print("MODULE 1: CROP RECOMMENDATION MODELS")
    print("="*60)

    Xtr, Xte = data['X_train_scaled'], data['X_test_scaled']
    ytr, yte = data['y_train'], data['y_test']

    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN':                 KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes':         GaussianNB(),
        'XGBoost':             XGBClassifier(n_estimators=100, random_state=42,
                                             eval_metric='mlogloss')
    }

    # Encode labels for XGBoost
    from sklearn.preprocessing import LabelEncoder as _LE
    _le_xgb = _LE()
    ytr_enc = _le_xgb.fit_transform(ytr)
    yte_enc = _le_xgb.transform(yte)

    results = []
    for name, clf in classifiers.items():
        t0 = time.time()
        if name == 'XGBoost':
            clf.fit(Xtr, ytr_enc)
            y_pred_enc = clf.predict(Xte)
            y_pred = _le_xgb.inverse_transform(y_pred_enc)
            elapsed = time.time() - t0
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            acc  = accuracy_score(yte, y_pred)
            prec = precision_score(yte, y_pred, average='weighted', zero_division=0)
            rec  = recall_score(yte, y_pred, average='weighted', zero_division=0)
            f1   = f1_score(yte, y_pred, average='weighted', zero_division=0)
            cm   = confusion_matrix(yte, y_pred)
            print(f"\n── {name} ──")
            print(f"  Accuracy : {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
            cv = cross_val_score(clf, Xtr, ytr_enc, cv=5, scoring='accuracy')
            print(f"  CV Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")
            results.append({'model_name': name, 'model': clf,
                            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                            'y_pred': y_pred, 'confusion_matrix': cm,
                            'train_time': round(elapsed, 3), 'cv_mean': cv.mean(), 'cv_std': cv.std()})
            continue
        clf.fit(Xtr, ytr)
        elapsed = time.time() - t0
        res = _eval_classifier(name, clf, Xte, yte)
        res['train_time'] = round(elapsed, 3)
        cv = cross_val_score(clf, Xtr, ytr, cv=5, scoring='accuracy')
        res['cv_mean'] = cv.mean()
        res['cv_std']  = cv.std()
        print(f"  CV Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")
        results.append(res)

    # Best model by F1
    best = max(results, key=lambda r: r['f1'])
    print(f"\n★ Best Crop Model: {best['model_name']} (F1={best['f1']:.4f})")

    # Hyperparameter tuning on Random Forest
    print("\n[Tuning] Random Forest GridSearchCV …")
    param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20],
                  'min_samples_split': [2, 5]}
    gs = GridSearchCV(RandomForestClassifier(random_state=42),
                      param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    gs.fit(Xtr, ytr)
    tuned_rf = gs.best_estimator_
    tuned_res = _eval_classifier('RF_Tuned', tuned_rf, Xte, yte)
    print(f"  Best params: {gs.best_params_}")

    # Feature importance
    fi = pd.Series(tuned_rf.feature_importances_,
                   index=data['feature_names']).sort_values(ascending=False)

    # Save best model + scaler
    final_model = tuned_rf if tuned_res['f1'] >= best['f1'] else best['model']
    _save(final_model,         'crop_recommendation_model.pkl')
    _save(data['scaler'],      'crop_rec_scaler.pkl')
    _save(data['label_encoder'], 'crop_rec_label_encoder.pkl')

    return {
        'results': results,
        'best': best,
        'tuned_result': tuned_res,
        'feature_importance': fi,
        'final_model': final_model
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 – FERTILIZER RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────

def train_fertilizer_recommendation(data):
    print("\n" + "="*60)
    print("MODULE 2: FERTILIZER RECOMMENDATION MODELS")
    print("="*60)

    Xtr, Xte = data['X_train_scaled'], data['X_test_scaled']
    ytr, yte = data['y_train'], data['y_test']

    classifiers = {
        'Decision Tree':     DecisionTreeClassifier(max_depth=12, random_state=42),
        'Random Forest':     RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN':               KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = []
    for name, clf in classifiers.items():
        clf.fit(Xtr, ytr)
        res = _eval_classifier(name, clf, Xte, yte)
        cv = cross_val_score(clf, Xtr, ytr, cv=5, scoring='accuracy')
        res['cv_mean'] = cv.mean(); res['cv_std'] = cv.std()
        print(f"  CV Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")
        results.append(res)

    best = max(results, key=lambda r: r['f1'])
    print(f"\n★ Best Fertilizer Model: {best['model_name']} (F1={best['f1']:.4f})")

    _save(best['model'],   'fertilizer_recommendation_model.pkl')
    _save(data['scaler'],  'fertilizer_scaler.pkl')
    _save(data['le_soil'], 'fertilizer_le_soil.pkl')
    _save(data['le_crop'], 'fertilizer_le_crop.pkl')
    _save(data['le_fert'], 'fertilizer_le_fert.pkl')

    return {'results': results, 'best': best}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 – CROP YIELD PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def train_crop_yield(data):
    print("\n" + "="*60)
    print("MODULE 3: CROP YIELD PREDICTION MODELS")
    print("="*60)

    Xtr, Xte = data['X_train_scaled'], data['X_test_scaled']
    ytr, yte = data['y_train'], data['y_test']

    regressors = {
        'Linear Regression':      LinearRegression(),
        'Random Forest':          RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting':      GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost':                XGBRegressor(n_estimators=100, random_state=42),
    }

    results = []
    for name, reg in regressors.items():
        reg.fit(Xtr, ytr)
        res = _eval_regressor(name, reg, Xte, yte)
        results.append(res)

    best = max(results, key=lambda r: r['r2'])
    print(f"\n★ Best Yield Model: {best['model_name']} (R²={best['r2']:.4f})")

    # Feature importance from best tree-based model
    tree_models = ['Random Forest', 'Gradient Boosting', 'XGBoost']
    fi_res = [r for r in results if r['model_name'] in tree_models]
    if fi_res:
        fi_best = max(fi_res, key=lambda r: r['r2'])
        fi = pd.Series(fi_best['model'].feature_importances_,
                       index=data['feature_names']).sort_values(ascending=False)
    else:
        fi = None

    _save(best['model'],    'crop_yield_model.pkl')
    _save(data['scaler'],   'yield_scaler.pkl')
    _save(data['le_crop'],  'yield_le_crop.pkl')
    _save(data['le_season'],'yield_le_season.pkl')
    _save(data['le_state'], 'yield_le_state.pkl')

    return {'results': results, 'best': best, 'feature_importance': fi}
