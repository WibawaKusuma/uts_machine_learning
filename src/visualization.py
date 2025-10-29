import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Mengatur gaya plot
sns.set_style('whitegrid')
plt.style.use('seaborn-v0_8')  # Gaya default yang kompatibel
# Menonaktifkan peringatan
warnings.filterwarnings('ignore')
# Memastikan tidak ada figure yang terbuka sebelumnya
plt.close('all')
# Mengatur ukuran default figure
matplotlib.rcParams['figure.figsize'] = [12, 8]
matplotlib.rcParams['figure.dpi'] = 100

def plot_correlation_matrix(df, save_path=None):
    """
    Membuat visualisasi matriks korelasi
    
    Parameters:
    df (pandas.DataFrame): Dataframe yang akan divisualisasikan
    save_path (str, optional): Path untuk menyimpan gambar
    """
    plt.figure()
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Buat heatmap
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                fmt=".2f",
                mask=mask,
                square=True,
                linewidths=0.5)
    
    plt.title('Matriks Korelasi Fitur', pad=20)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_feature_distributions(df, save_dir=None):
    """
    Membuat visualisasi distribusi untuk setiap fitur
    
    Parameters:
    df (pandas.DataFrame): Dataframe yang akan divisualisasikan
    save_dir (str, optional): Direktori untuk menyimpan gambar
    """
    # Pilih hanya kolom numerik
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Hitung jumlah baris dan kolom untuk subplot
    n_cols = 2
    n_rows = (len(numeric_cols) + 1) // n_cols
    
    # Buat figure dengan subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    # Ratakan axes jika hanya ada satu baris
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot setiap fitur
    for idx, col in enumerate(numeric_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        
        # Plot histogram dengan KDE
        sns.histplot(data=df, x=col, kde=True, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'Distribusi {col}', pad=10)
        axes[row, col_idx].set_xlabel('')
        
        # Rotasi label x jika diperlukan
        for tick in axes[row, col_idx].get_xticklabels():
            tick.set_rotation(45)
    
    # Sembunyikan subplot yang tidak terpakai
    for idx in range(len(numeric_cols), n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        fig.delaxes(axes[row, col_idx])
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/distributions.png", bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Membuat visualisasi feature importance dari model
    
    Parameters:
    model: Model yang telah dilatih
    feature_names (list): Daftar nama fitur
    save_path (str, optional): Path untuk menyimpan gambar
    """
    # Mendapatkan feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Membuat plot
    plt.figure()
    
    # Buat bar plot
    bars = plt.barh(range(len(importances)), importances[indices], align='center', color='skyblue')
    
    # Tambahkan nilai di ujung bar
    for idx, (value, bar) in enumerate(zip(importances[indices], bars)):
        width = bar.get_width()
        plt.text(width + 0.01, idx, f'{value:.3f}', va='center')
    
    # Atur label dan judul
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score')
    plt.title('Feature Importance', pad=20)
    
    # Atur margin dan layout
    plt.margins(y=0.1)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    from data_processing import load_data
    
    # Memuat data
    df = load_data("../data/property_data.csv")
    
    # Membuat visualisasi
    print("Membuat visualisasi...")
    
    # Matriks korelasi
    plot_correlation_matrix(df, save_path="../figures/correlation_matrix.png")
    
    # Distribusi fitur
    plot_feature_distributions(df, save_dir="../figures/distributions")
    
    print("Visualisasi berhasil disimpan di folder figures/")
