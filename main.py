from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from dataclasses import asdict
from joblib import dump
from pathlib import Path

from src.data import generate_synthetic, split_and_impute
from src.features import transform_features, FeatureBundle
from src.models import train_for_degree, select_best, ModelResult
from src.evaluate import (
    save_histograms, save_scatter_vs_target, save_corr_heatmap,
    save_residual_plot, save_pred_vs_true, save_learning_curve,
    results_table_to_markdown
)
from src.importance import save_coeff_importance
from src.predict import make_new_samples

def log(msg: str):
    print(f"[INFO] {msg}")

def ensure_dir(dir_path: str):
    """Memastikan direktori ada, jika tidak buat"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def main():
    # Konfigurasi
    FIG_DIR = "figures"
    MODEL_DIR = "models"
    DEGREES = [1, 2, 3]  # Derajat polinomial yang akan dicoba
    
    # Buat direktori jika belum ada
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Generate synthetic data
    print("[1/6] Membuat dataset sintetis...")
    df = generate_synthetic(n_samples=400)
    
    # 2. EDA & Visualisasi
    print("[2/6] Melakukan analisis eksplorasi...")
    save_histograms(df, "Harga", FIG_DIR)
    save_scatter_vs_target(df, "Harga", FIG_DIR)
    save_corr_heatmap(df, FIG_DIR)
    
    # 3. Split data
    print("[3/6] Memisahkan data...")
    splits = split_and_impute(df)
    
    all_results = []
    
    # 4. Training model untuk setiap derajat polinomial
    for degree in DEGREES:
        print(f"\n[4/6] Melatih model (degree={degree})...")
        
        # Transformasi fitur
        fb = transform_features(splits.X_train, splits.X_test, degree)
        
        # Training model
        results = train_for_degree(
            fb.X_train, fb.X_test, 
            splits.y_train, splits.y_test, 
            degree=degree
        )
        
        # Simpan model terbaik
        best = select_best(results)
        all_results.extend(results)
        
        # Visualisasi
        y_pred = best.estimator.predict(fb.X_test)
        save_residual_plot(splits.y_test, y_pred, FIG_DIR, f"deg{degree}")
        save_pred_vs_true(splits.y_test, y_pred, FIG_DIR, f"deg{degree}")
        save_learning_curve(best.estimator, fb.X_train, splits.y_train, FIG_DIR, f"deg{degree}")
        
        # Simpan koefisien penting
        if hasattr(best.estimator, 'coef_'):
            save_coeff_importance(best.estimator, 
                               splits.X_train.columns, 
                               FIG_DIR, 
                               f"deg{degree}")
    
    # 5. Simpan hasil evaluasi
    print("\n[5/6] Menyimpan hasil...")
    results_table = []
    for r in all_results:
        results_table.append({
            "Model": f"{r.name} (deg={r.degree})",
            **r.test_scores,
            **{f"best_{k}": v for k, v in r.best_params.items()}
        })
    
    # Tampilkan hasil
    print("\n" + "="*50)
    print("Hasil Evaluasi Model")
    print("="*50)
    print(results_table_to_markdown(results_table))
    
    # 6. Contoh prediksi
    print("\n[6/6] Membuat prediksi contoh...")
    new_data = make_new_samples(3)
    best_model = select_best(all_results)
    
    # Transformasi fitur untuk data baru
    fb = transform_features(splits.X_train, new_data, best_model.degree)
    predictions = best_model.estimator.predict(fb.X_test)
    
    print("\nContoh Prediksi:")
    for i, (_, row) in enumerate(new_data.iterrows()):
        print(f"\nData {i+1}:")
        for col, val in row.items():
            print(f"  {col}: {val:.2f}")
        print(f"  Prediksi Harga: ${predictions[i]:.2f}")
    
    print("\nProses selesai!")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
