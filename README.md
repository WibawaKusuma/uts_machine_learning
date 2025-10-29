# Analisis dan Prediksi Harga Properti

Proyek ini merupakan sistem analisis dan prediksi harga properti yang menggunakan berbagai teknik machine learning, termasuk regresi linier, Ridge, dan Lasso dengan polynomial features. Proyek ini mencakup analisis data eksplorasi, visualisasi, dan evaluasi model yang komprehensif.

## 🚀 Fitur Utama

- **Data Sintetis**: Pembuatan dataset sintetis dengan karakteristik realistis
- **Analisis Eksplorasi Data (EDA)**: Visualisasi lengkap termasuk histogram, scatter plot, dan heatmap korelasi
- **Pemrosesan Data Otomatis**: Penanganan missing value dan transformasi fitur
- **Multiple Model**: Perbandingan berbagai model regresi (Linear, Ridge, Lasso)
- **Polynomial Features**: Dukungan untuk non-linearitas dengan polynomial features
- **Evaluasi Komprehensif**: Berbagai metrik evaluasi (MSE, RMSE, MAE, MAPE, R²)
- **Visualisasi**: Grafik residual, learning curve, dan feature importance
- **Reset Otomatis**: Script untuk mengembalikan proyek ke kondisi awal

## 📁 Struktur Proyek

```
machine-learning-new/
├── data/                    # Folder untuk menyimpan dataset
├── figures/                 # Visualisasi dan grafik analisis
│   ├── corr_heatmap.png     # Peta panas korelasi
│   ├── hist_*.png           # Histogram fitur
│   ├── learning_curve_*.png # Kurva pembelajaran
│   ├── pred_vs_true_*.png   # Plot prediksi vs aktual
│   └── residuals_*.png      # Plot residual
├── models/                  # Model yang sudah dilatih dan metrik
│   └── metrics.json         # Hasil evaluasi model
├── src/                     # Kode sumber
│   ├── data.py             # Pembuatan dan pemrosesan data
│   ├── features.py         # Transformasi fitur
│   ├── models.py           # Definisi dan pelatihan model
│   ├── evaluate.py         # Evaluasi dan visualisasi metrik
│   ├── importance.py       # Analisis feature importance
│   └── predict.py          # Utilitas prediksi
├── main.py                 # Script utama
├── reset.py                # Reset proyek
├── requirements.txt        # Dependensi
└── README.md               # Dokumentasi
```

## 🛠️ Instalasi

1. **Persyaratan Sistem**
   - Python 3.8+
   - pip (package manager)
   - virtualenv (disarankan)

2. **Setup Awal**
   ```bash
   # Clone repositori (jika menggunakan git)
   # git clone [URL_REPOSITORY]
   # cd machine-learning-new

   # Buat dan aktifkan virtual environment
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # MacOS/Linux
   # source venv/bin/activate
   ```

3. **Instal Dependensi**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Proyek**
   ```bash
   python main.py
   ```

5. **Reset Proyek** (jika diperlukan)
   ```bash
   python reset.py
   ```
   ```

## 🚀 Penggunaan

1. **Jalankan proyek**
   ```bash
   python main.py
   ```
   
   Ini akan:
   - Memproses data
   - Melatih model
   - Menyimpan model terlatih
   - Membuat visualisasi

2. **Reset proyek** (jika diperlukan)
   ```bash
   python reset.py
   ```
   Akan menghapus:
   - Model yang telah dilatih
   - File visualisasi
   - File cache dan temporary

## 📊 Dataset

Dataset berisi informasi properti dengan fitur-fitur berikut:
- `LuasTanah_m2`: Luas tanah dalam meter persegi
- `LuasBangunan_m2`: Luas bangunan dalam meter persegi
- `JmlKamarTidur`: Jumlah kamar tidur
- `UmurBangunan_thn`: Umur bangunan dalam tahun
- `JarakKePusat_km`: Jarak ke pusat kota dalam kilometer
- `Harga_juta`: Harga properti dalam juta rupiah (target variabel)

## 🤖 Model

Menggunakan algoritma **Random Forest Regressor** dengan metrik evaluasi:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## 📝 Catatan

- Pastikan Python 3.8+ terinstall
- Gunakan virtual environment untuk menghindari konflik dependensi
- File `reset.py` akan menghapus semua file yang dihasilkan oleh program

## Instalasi

1. Clone repositori ini
2. Buat environment virtual (opsional tapi disarankan):
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependensi:
   ```
   pip install -r requirements.txt
   ```

## Penggunaan

1. Letakkan file dataset di folder `data/`
2. Jalankan script utama:
   ```
   python main.py
   ```

## Dataset

Dataset berisi informasi properti dengan fitur-fitur berikut:
- LuasTanah_m2: Luas tanah dalam meter persegi
- LuasBangunan_m2: Luas bangunan dalam meter persegi
- JmlKamarTidur: Jumlah kamar tidur
- UmurBangunan_thn: Umur bangunan dalam tahun
- JarakKePusat_km: Jarak ke pusat kota dalam kilometer
- Harga_juta: Harga properti dalam juta rupiah (target variabel)

## Model

Model yang digunakan adalah Random Forest Regressor untuk memprediksi harga properti berdasarkan fitur-fitur yang tersedia.
