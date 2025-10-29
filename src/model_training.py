import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Melatih model Random Forest untuk prediksi harga properti
    
    Parameters:
    X_train (pandas.DataFrame): Data latih (fitur)
    y_train (pandas.Series): Target latih
    n_estimators (int): Jumlah pohon dalam random forest
    random_state (int): Random state untuk reproduktibilitas
    
    Returns:
    RandomForestRegressor: Model yang telah dilatih
    """
    # Inisialisasi model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1  # Menggunakan semua core CPU yang tersedia
    )
    
    # Melatih model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Mengevaluasi performa model
    
    Parameters:
    model: Model yang telah dilatih
    X_test (pandas.DataFrame): Data uji (fitur)
    y_test (pandas.Series): Target uji
    
    Returns:
    dict: Dictionary berisi metrik evaluasi
    """
    # Melakukan prediksi
    y_pred = model.predict(X_test)
    
    # Menghitung metrik evaluasi
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def save_model(model, filename):
    """
    Menyimpan model ke file
    
    Parameters:
    model: Model yang akan disimpan
    filename (str): Nama file untuk menyimpan model
    """
    # Membuat direktori models jika belum ada
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Menyimpan model
    joblib.dump(model, filename)
    print(f"Model berhasil disimpan di {filename}")

def load_model(filename):
    """
    Memuat model dari file
    
    Parameters:
    filename (str): Path ke file model
    
    Returns:
    Model yang telah dimuat
    """
    return joblib.load(filename)

if __name__ == "__main__":
    # Contoh penggunaan
    from data_processing import load_data, preprocess_data, split_data
    
    # Memuat dan memproses data
    df = load_data("../data/property_data.csv")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Melatih model
    print("Melatih model...")
    model = train_model(X_train, y_train)
    
    # Mengevaluasi model
    metrics = evaluate_model(model, X_test, y_test)
    print("\nHasil Evaluasi Model:")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R²: {metrics['r2']:.4f}")
    
    # Menyimpan model
    save_model(model, "../models/random_forest_model.joblib")
