"""
============================================================================
SCRIPT IMPUTASI KHUSUS — PREDIKSI HIPERTENSI
Script terpisah ini hanya melakukan imputasi pada 'dataset_hipertensi_prepared.csv'
tanpa mengulang kembali proses data preparation dari awal.
============================================================================
"""

import pandas as pd
from sklearn.impute import SimpleImputer

# 1. Tentukan path file input dan output
PATH_INPUT = "dataset_hipertensi_prepared.csv"
PATH_OUTPUT = "dataset_hipertensi_imputed.csv"

print("--- MEMULAI PROSES IMPUTASI SAJA ---")

try:
    # 2. Membaca dataset yang belum diimputasi
    df = pd.read_csv(PATH_INPUT)
    print(f"Berhasil membaca {PATH_INPUT}")
    print(f"Ukuran data: {df.shape[0]} baris, {df.shape[1]} kolom")
    
    # Menampilkan jumlah missing value sebelum imputasi
    print(f"\nJumlah nilai kosong sebelum imputasi:\n{df.isna().sum().to_string()}")
    
    # 3. Definisikan pengelompokkan kolom untuk strategi imputasi
    # Kolom numerik diimputasi dengan Median
    num_cols = ["age", "bmi"]  
    
    # Kolom kategorikal/biner diimputasi dengan Most Frequent (Modus)
    cat_cols = [
        "is_female", "is_smoker", "has_diabetes", 
        "has_high_cholesterol", "sleep_disturbance", "sleep_quality"
    ]
    
    # 4. Inisialisasi Imputer
    imp_num = SimpleImputer(strategy="median")
    imp_cat = SimpleImputer(strategy="most_frequent")
    
    # 5. Eksekusi Imputasi pada dataframe kopi
    df_imputed = df.copy()
    
    # Pastikan kolom-kolom tersebut ada di dataset sebelum diimputasi
    num_cols_present = [col for col in num_cols if col in df.columns]
    cat_cols_present = [col for col in cat_cols if col in df.columns]
    
    if num_cols_present:
        df_imputed[num_cols_present] = imp_num.fit_transform(df[num_cols_present])
    if cat_cols_present:
        df_imputed[cat_cols_present] = imp_cat.fit_transform(df[cat_cols_present])
        
    # 6. Simpan hasil ke file CSV baru
    df_imputed.to_csv(PATH_OUTPUT, index=False)
    
    print("\n" + "="*50)
    print(f"PROSES SELESAI!")
    print(f"Sisa nilai kosong setelah imputasi: {df_imputed.isna().sum().sum()} sel")
    print(f"File hasil imputasi disimpan ke: {PATH_OUTPUT}")
    print("="*50)

except FileNotFoundError:
    print(f"\n[ERROR] Berkas '{PATH_INPUT}' tidak ditemukan.")
    print("Pastikan Anda menjalankan script ini di folder yang sama dengan berkas tersebut.")