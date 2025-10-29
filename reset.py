import os
import shutil
import sys
from pathlib import Path

def print_header():
    print("\n=== Reset Proyek Machine Learning ===")
    print("Menghapus file yang dihasilkan dan mereset ke kondisi awal\n")

def remove_path(path, is_dir=False):
    """Hapus file atau direktori dengan aman"""
    try:
        if is_dir and os.path.isdir(path):
            shutil.rmtree(path)
            print(f"✓ Dihapus direktori: {path}")
        elif os.path.isfile(path):
            os.remove(path)
            print(f"✓ Dihapus file: {path}")
        return True
    except Exception as e:
        print(f"✗ Gagal menghapus {path}: {str(e)}")
        return False

def main():
    print_header()
    
    # Daftar file dan folder yang akan dihapus
    to_remove = [
        # Model
        "models/random_forest_model.joblib",
        
        # Folder figures
        "figures",
        
        # File log dan cache
        "*.log",
        "*.tmp",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        
        # Folder cache
        "__pycache__",
        "src/__pycache__",
        "models/__pycache__",
        
        # Lain-lain
        ".coverage",
        "htmlcov/",
        ".pytest_cache/",
        ".mypy_cache/"
    ]
    
    # Hapus file dan folder
    for item in to_remove:
        # Handle wildcards
        if '*' in item or '?' in item:
            import glob
            for f in glob.glob(item, recursive=True):
                remove_path(f, os.path.isdir(f))
        else:
            remove_path(item, item.endswith('/') or os.path.isdir(item))
    
    # Buat ulang struktur folder yang diperlukan
    required_dirs = ["data", "figures", "models", "src"]
    for dir_name in required_dirs:
        try:
            os.makedirs(dir_name, exist_ok=True)
            print(f"✓ Direktori {dir_name} sudah ada/dibuat")
        except Exception as e:
            print(f"✗ Gagal membuat direktori {dir_name}: {str(e)}")
    
    print("\n=== Reset Selesai ===")
    print("Proyek telah direset ke kondisi awal.")
    print("\nUntuk menjalankan proyek kembali, gunakan perintah:")
    print("python main.py")

if __name__ == "__main__":
    main()
