"""
main.py – Smart Agriculture Advisor
End-to-end pipeline: Preprocess → EDA → Train → Evaluate → Save
Run from the project root: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.data_preprocessor import (
    load_crop_recommendation,
    load_fertilizer_recommendation,
    load_crop_yield,
    load_analytics_datasets
)
from src.models.train_models import (
    train_crop_recommendation,
    train_fertilizer_recommendation,
    train_crop_yield
)
from src.visualization.eda_plots import (
    run_crop_eda, run_fertilizer_eda, run_yield_eda, run_climate_eda,
    plot_model_comparison, plot_feature_importance,
    plot_confusion_matrix, plot_yield_actual_vs_pred
)
from src.utils.helpers import print_section, results_to_df
import pandas as pd

os.makedirs("reports/plots", exist_ok=True)


def main():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║   AI-Powered Smart Agriculture Advisor – Full Pipeline  ║")
    print("╚" + "═"*58 + "╝\n")

    # ── 1. LOAD & PREPROCESS ────────────────────────────────────────
    print_section("1. DATA PREPROCESSING")
    crop_data   = load_crop_recommendation()
    fert_data   = load_fertilizer_recommendation()
    yield_data  = load_crop_yield()
    analytics   = load_analytics_datasets()

    # ── 2. EDA ──────────────────────────────────────────────────────
    print_section("2. EXPLORATORY DATA ANALYSIS")
    run_crop_eda(crop_data['df'])
    run_fertilizer_eda(fert_data['df'])
    run_yield_eda(yield_data['df'])
    run_climate_eda(analytics)

    # ── 3. TRAIN CROP RECOMMENDATION ────────────────────────────────
    crop_res = train_crop_recommendation(crop_data)
    plot_model_comparison(crop_res['results'],
                          'Crop_Recommendation_Model_Comparison', metric='accuracy')
    plot_feature_importance(crop_res['feature_importance'],
                            'Crop_Recommendation_Feature_Importance')
    best_cm = crop_res['tuned_result']['confusion_matrix']
    classes = sorted(crop_data['y_test'].unique())
    plot_confusion_matrix(best_cm, classes, 'Crop_Recommendation_Confusion_Matrix')

    print("\n── Crop Recommendation Summary ──")
    df_cr = results_to_df(crop_res['results'],
                          ['accuracy', 'precision', 'recall', 'f1'])
    print(df_cr.to_string())

    # ── 4. TRAIN FERTILIZER RECOMMENDATION ──────────────────────────
    fert_res = train_fertilizer_recommendation(fert_data)
    plot_model_comparison(fert_res['results'],
                          'Fertilizer_Recommendation_Model_Comparison', metric='accuracy')

    print("\n── Fertilizer Recommendation Summary ──")
    df_fr = results_to_df(fert_res['results'],
                          ['accuracy', 'precision', 'recall', 'f1'])
    print(df_fr.to_string())

    # ── 5. TRAIN CROP YIELD PREDICTION ──────────────────────────────
    yield_res = train_crop_yield(yield_data)
    if yield_res['feature_importance'] is not None:
        plot_feature_importance(yield_res['feature_importance'],
                                'Yield_Prediction_Feature_Importance')
    best_yield_res = yield_res['best']
    plot_yield_actual_vs_pred(
        yield_data['y_test'], best_yield_res['y_pred'], best_yield_res['model_name']
    )
    plot_model_comparison(yield_res['results'],
                          'Yield_Prediction_Model_Comparison', metric='r2')

    print("\n── Yield Prediction Summary ──")
    df_yr = results_to_df(yield_res['results'], ['mae', 'rmse', 'r2'])
    print(df_yr.to_string())

    # ── 6. FINAL SUMMARY ────────────────────────────────────────────
    print_section("6. PIPELINE COMPLETE")
    print(f"  ✓ Saved models → saved_models/")
    print(f"  ✓ EDA plots    → reports/plots/")
    print(f"\n  Best Models:")
    print(f"    Crop Recommendation : {crop_res['best']['model_name']} "
          f"(Acc={crop_res['best']['accuracy']:.4f})")
    print(f"    Fertilizer Recom.   : {fert_res['best']['model_name']} "
          f"(Acc={fert_res['best']['accuracy']:.4f})")
    print(f"    Yield Prediction    : {yield_res['best']['model_name']} "
          f"(R²={yield_res['best']['r2']:.4f})")
    print(f"\n  ► Run Streamlit app : streamlit run streamlit_app/Home.py\n")


if __name__ == "__main__":
    main()
