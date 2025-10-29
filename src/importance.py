from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union

def get_feature_names(pipeline, feature_names: List[str]) -> List[str]:
    """Mendapatkan nama fitur setelah transformasi PolynomialFeatures"""
    if hasattr(pipeline, 'named_steps'):
        if 'poly' in pipeline.named_steps:
            poly = pipeline.named_steps['poly']
            return poly.get_feature_names_out(feature_names).tolist()
    return feature_names

def save_coeff_importance(estimator, feature_names: Union[List[str], np.ndarray], 
                         outdir: str, name: str, topk: int = 20):
    """
    Menyimpan visualisasi koefisien penting dari model
    
    Args:
        estimator: Model yang sudah di-fit
        feature_names: Nama-nama fitur asli atau pipeline
        outdir: Direktori penyimpanan
        name: Nama file output
        topk: Jumlah koefisien teratas yang akan ditampilkan
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Handle pipeline
    if hasattr(estimator, 'named_steps'):
        feature_names = get_feature_names(estimator, feature_names)
        # Dapatkan estimator akhir dari pipeline
        estimator = estimator.named_steps.get('ridge', estimator.named_steps.get('lasso', estimator))
    
    if not hasattr(estimator, "coef_"):
        print(f"[WARNING] Model {name} tidak memiliki koefisien untuk divisualisasikan")
        return
    
    coefs = estimator.coef_.ravel()
    
    # Pastikan panjang feature_names sesuai dengan jumlah koefisien
    if len(feature_names) != len(coefs):
        feature_names = [f"feature_{i}" for i in range(len(coefs))]
    
    # Ambil top-k koefisien terbesar berdasarkan nilai absolut
    idx = np.argsort(np.abs(coefs))[::-1][:min(topk, len(coefs))]
    
    # Buat plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(idx)), coefs[idx], color='skyblue')
    
    # Tambahkan label nilai di atas setiap bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Atur label sumbu x
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    labels = [str(feature_names[i]) for i in idx]
    plt.xticks(range(len(idx)), labels, rotation=45, ha='right')
    
    plt.title(f"Top-{len(idx)} Koefisien Penting - {name}")
    plt.ylabel('Nilai Koefisien')
    plt.tight_layout()
    
    # Simpan gambar
    output_path = os.path.join(outdir, f"coeff_importance_{name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Koefisien penting disimpan di {output_path}")
