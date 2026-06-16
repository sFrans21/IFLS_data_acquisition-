# import pandas as pd

# # Memuat dataset
# df = pd.read_csv('master_dataset_raw_final.csv')

# # Inspeksi data dengan info()
# print("--- Data Info ---")
# df.info()

# # Inspeksi data dengan head()
# print("\n--- Data Head ---")
# print(df.head())











# import pandas as pd
# import numpy as np

# df = pd.read_csv('master_dataset_raw_final.csv')

# # 1. Struktur Data
# print(f"Jumlah Baris: {df.shape[0]}, Jumlah Kolom: {df.shape[1]}")

# # 2. Missing Values
# print("\n--- Persentase Missing Values ---")
# missing_pct = df.isnull().mean() * 100
# print(missing_pct[missing_pct > 0].sort_values(ascending=False))

# # 3. Distribusi Target Awal (Meski belum dibersihkan)
# # Asumsi sementara untuk observasi: bp_systolic >= 140 atau bp_diastolic >= 90
# df_valid_bp = df.dropna(subset=['bp_systolic', 'bp_diastolic'])
# df_valid_bp['label_hypertension'] = ((df_valid_bp['bp_systolic'] >= 140) | (df_valid_bp['bp_diastolic'] >= 90)).astype(int)
# print("\n--- Distribusi Target Sementara ---")
# print(df_valid_bp['label_hypertension'].value_counts(normalize=True) * 100)

# # 4. Statistik Deskriptif Numerik Penting
# print("\n--- Statistik Deskriptif ---")
# print(df[['age', 'bmi', 'bp_systolic', 'bp_diastolic']].describe())



# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# df = pd.read_csv('master_dataset_raw_final.csv')
# missing_pct = df.isnull().mean() * 100

# plt.figure(figsize=(10,6))
# missing_pct[missing_pct > 0].sort_values().plot(kind='barh', color='coral')
# plt.title('Persentase Nilai Hilang per Fitur')
# plt.xlabel('Persentase (%)')
# plt.tight_layout()
# plt.savefig('missing_values.png')

# plt.figure(figsize=(6,4))
# sns.boxplot(x=df['age'], color='skyblue')
# plt.title('Distribusi Usia (Deteksi Anomali)')
# plt.tight_layout()
# plt.savefig('age_outliers.png')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Memuat dataset
df = pd.read_csv('master_dataset_raw_final.csv')

# 1. Menghitung Dimensi
print(f"Jumlah Baris: {df.shape[0]}, Jumlah Kolom: {df.shape[1]}")

# 2. Observasi Nilai Hilang (Missing Values)
missing_pct = df.isnull().mean() * 100
print(missing_pct[missing_pct > 0].sort_values(ascending=False))

# Simpan Grafik Nilai Hilang
plt.figure(figsize=(10,6))
missing_pct[missing_pct > 0].sort_values().plot(kind='barh', color='coral')
plt.title('Persentase Nilai Hilang per Fitur')
plt.xlabel('Persentase (%)')
plt.tight_layout()
plt.savefig('missing_values.png')

# 3. Deteksi Anomali
print(df[['age', 'bmi', 'bp_systolic']].describe())

# Simpan Grafik Anomali Usia
plt.figure(figsize=(6,4))
sns.boxplot(x=df['age'], color='skyblue')
plt.title('Distribusi Usia (Deteksi Anomali)')
plt.tight_layout()
plt.savefig('age_outliers.png')

# 4. Distribusi Target Awal
df_valid_bp = df.dropna(subset=['bp_systolic', 'bp_diastolic'])
df_valid_bp['label'] = ((df_valid_bp['bp_systolic'] >= 140) | (df_valid_bp['bp_diastolic'] >= 90)).astype(int)
print(df_valid_bp['label'].value_counts(normalize=True) * 100)