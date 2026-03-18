import pandas as pd

print("--- MEMULAI PEMBERSIHAN TAHAP AKHIR (SANITASI DATA) ---")

# 1. Load Data yang 'katanya' sudah bersih
df = pd.read_csv('dataset_hipertensi_clean.csv')
jumlah_awal = len(df)
print(f"Jumlah Data Awal: {jumlah_awal} responden")

# 2. Cek Berapa Banyak yang Umurnya Ngaco (di atas 100 tahun)
# Kode 998/999 di IFLS = Missing/Don't Know
data_sampah = df[df['age'] > 100]
print(f"Ditemukan {len(data_sampah)} responden dengan usia > 100 tahun (Kode 998/999).")

# 3. Hapus Data Sampah
df_final = df[df['age'] <= 100]

# 4. Simpan (Timpa file lama agar script lain tidak perlu diubah)
df_final.to_csv('dataset_hipertensi_clean.csv', index=False)

print("\n--- HASIL ---")
print(f"Data Dibuang : {len(data_sampah)} baris")
print(f"Sisa Data    : {len(df_final)} baris")
print("File 'dataset_hipertensi_clean.csv' telah diperbarui.")
print("Sekarang data Anda sudah steril dari umur 998 tahun.")