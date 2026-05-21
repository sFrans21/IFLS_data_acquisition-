import pandas as pd
import numpy as np

df = pd.read_csv('dataset_hipertensi_clean.csv')

print("=" * 55)
print("  PROFIL DATASET: dataset_hipertensi_clean.csv")
print("=" * 55)

print(f"\n[UKURAN]")
print(f"  Baris   : {len(df):,}")
print(f"  Kolom   : {len(df.columns)}")

print(f"\n[SEMUA KOLOM & TIPE DATA]")
print(df.dtypes.to_string())

print(f"\n[DISTRIBUSI TARGET]")
print(df['label_hypertension'].value_counts())
print(f"  Proporsi positif: {df['label_hypertension'].mean():.2%}")

print(f"\n[STATISTIK DESKRIPTIF]")
print(df.describe().round(2).to_string())

print(f"\n[MISSING VALUES]")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("  Tidak ada missing value ✅")
else:
    print(missing)

print(f"\n[SAMPLE 3 BARIS PERTAMA]")
print(df.head(3).to_string())