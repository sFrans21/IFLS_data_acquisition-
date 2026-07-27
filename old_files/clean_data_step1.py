import pandas as pd
import numpy as np

# Load Data Mentah
print("--- MEMULAI DATA CLEANING TAHAP 1 ---")
df = pd.read_csv('master_dataset_raw_final.csv')
print(f"Populasi Awal: {len(df)} responden")

# ---------------------------------------------------------
# 1. MEMBERSIHKAN TARGET (TENSI) & FISIK
# ---------------------------------------------------------
# Aturan: Jika Tensi atau BMI kosong, data tidak berguna untuk training.
# Drop baris yang targetnya NaN.
df_clean = df.dropna(subset=['bp_systolic', 'bp_diastolic', 'bmi'])
print(f"Setelah drop fisik kosong: {len(df_clean)} responden")

# ---------------------------------------------------------
# 2. IMPUTASI NOL (ZERO IMPUTATION)
# ---------------------------------------------------------
# Variabel ini kosong artinya "Tidak melakukan"
# Mie Instan
df_clean['freq_instant_noodle'] = df_clean['freq_instant_noodle'].fillna(0)

# Aktivitas Fisik (ak02, ak05, ak07)
# Jika kosong, asumsi 0 hari
cols_act = ['ak02', 'ak05', 'ak07']
for col in cols_act:
    df_clean[col] = df_clean[col].fillna(0)

# ---------------------------------------------------------
# 3. ENCODING VARIABEL KATEGORIKAL (MAPPING)
# ---------------------------------------------------------
# a. Merokok (km01a): 1=Ya, 3=Tidak -> Jadi 1/0
#    Isi NaN dengan 3 (Tidak) dulu
df_clean['km01a'] = df_clean['km01a'].fillna(3)
df_clean['is_smoker'] = df_clean['km01a'].apply(lambda x: 1 if x == 1 else 0)

# b. Diabetes (is_diabetes): 1=Ya, 3=Tidak -> Jadi 1/0
#    Bersihkan kode 8/9 -> 3 (Tidak)
df_clean['is_diabetes'] = df_clean['is_diabetes'].replace({8: 3, 9: 3}).fillna(3)
df_clean['has_diabetes'] = df_clean['is_diabetes'].apply(lambda x: 1 if x == 1 else 0)

# c. Jenis Kelamin (sex): 1=Pria, 3=Wanita -> Jadi 0=Pria, 1=Wanita
df_clean['is_female'] = df_clean['sex'].apply(lambda x: 1 if x == 3 else 0)

# ---------------------------------------------------------
# 4. REKAYASA FITUR ORANG TUA (GENETIK)
# ---------------------------------------------------------
# Logika: Jika kolom mengandung huruf 'B', 'C', 'D', dst -> Risiko Tinggi (1)
# Jika 'A' (Sehat) atau NaN -> Risiko Rendah (0)

def cek_risiko_ortu(val):
    if pd.isna(val): return 0
    val_str = str(val).upper()
    # Jika mengandung indikator sakit (B-Z), return 1. Jika cuma A, return 0.
    # Kita anggap semua kode selain 'A' adalah indikasi masalah kesehatan/kematian
    if 'A' in val_str and len(val_str) == 1:
        return 0 # Sehat total
    return 1 # Ada riwayat sakit/meninggal

# Terapkan ke Ayah & Ibu
df_clean['risk_father'] = df_clean['father_health'].apply(cek_risiko_ortu)
df_clean['risk_mother'] = df_clean['mother_health'].apply(cek_risiko_ortu)

# Gabung jadi satu skor genetik (0, 1, atau 2)
df_clean['genetic_risk_score'] = df_clean['risk_father'] + df_clean['risk_mother']

# ---------------------------------------------------------
# 5. PEMBUATAN LABEL TARGET (HIPERTENSI)
# ---------------------------------------------------------
# Definisi Hipertensi: Sistolik >= 140 ATAU Diastolik >= 90
# (Panduan JNC-7 / WHO)
df_clean['label_hypertension'] = ((df_clean['bp_systolic'] >= 140) | 
                                  (df_clean['bp_diastolic'] >= 90)).astype(int)

# ---------------------------------------------------------
# FINALISASI
# ---------------------------------------------------------
# Pilih kolom final untuk Machine Learning
final_cols = [
    # Fitur
    'age', 'is_female', 'bmi', 'waist_cm',  # Fisik
    'is_smoker', 'freq_instant_noodle',     # Gaya Hidup
    'ak02', 'ak05', 'ak07',                 # Aktivitas
    'has_diabetes', 'genetic_risk_score',   # Riwayat
    'ps_A', 'ps_B', 'ps_C', 'ps_E', 'ps_F', # Stres
    # Target (Raw & Binary)
    'bp_systolic', 'bp_diastolic', 'label_hypertension' 
]

# Simpan
df_final = df_clean[final_cols]
output_file = 'dataset_hipertensi_clean.csv'
df_final.to_csv(output_file, index=False)

print("\n--- DATA CLEANING SELESAI ---")
print(f"Dimensi Final: {df_final.shape}")
print(f"Proporsi Hipertensi: {df_final['label_hypertension'].mean():.2%}")
print(f"File tersimpan: {output_file}")
print("Siap untuk Pemodelan!")