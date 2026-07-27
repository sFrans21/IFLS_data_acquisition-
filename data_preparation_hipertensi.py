

"""
============================================================================
SCRIPT DATA PREPARATION — PREDIKSI HIPERTENSI
Mengikuti urutan diagram alur yang dikoreksi berdasarkan hasil EDA:

  0. Pemeriksaan duplikat kombinasi hhid14 dan pid14
  1. Pembersihan kode tersamar (998, 8, 9 -> NaN)
  2. Filter responden usia >= 18 tahun
  3. Filter nilai mustahil tekanan darah dengan batas medis klinis, dan nilai yang kosong
  4. Pembentukan label target hipertensi (sistolik>=140 ATAU diastolik>=90)
  5. Penghapusan tekanan darah dari himpunan fitur (cegah kebocoran data)
  6. Filter outlier berat/tinggi badan & hitung BMI
  7. Feature Engineering active_status dari variabel aktivitas fisik
  7. Encoding biner & penanganan Structural Missing Values (frekuensi = 0)
  8. Split data latih-uji lalu imputasi (median & modus)
============================================================================
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

PATH_DATA = "master_dataset_raw_final.csv"

def garis(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

print("--- MEMULAI DATA PREPARATION ---")
df = pd.read_csv(PATH_DATA, low_memory=False)
print(f"Populasi awal: {len(df)} responden, {df.shape[1]} kolom")

# ===========================================================================
# LANGKAH 0 — PEMERIKSAAN DUPLIKAT kombinasi hhid14 dan pid14
# ===========================================================================
garis("LANGKAH 0: Pemeriksaan duplikat pada kombinasi 'hhid14' dan 'pid14'")
dup = df[df.duplicated(subset=['hhid14', 'pid14'], keep=False)]
print(f"  Ditemukan {len(dup)} baris duplikat dari {dup[['hhid14', 'pid14']].drop_duplicates().shape[0]} hhid14 dan pid14 unik.")

# ===========================================================================
# LANGKAH 1 — PEMBERSIHAN KODE TERSAMAR (SENTINEL) -> NaN
# ===========================================================================
garis("LANGKAH 1: Pembersihan kode tersamar")

df.loc[df["age"] > 150, "age"] = np.nan


feature_cols = [
    'has_tobacco', 'has_diabetes', 'has_high_cholesterol', 'has_kidney_disease', 'has_stroke', 'has_fried_food', 'has_noodles', 'has_fast_food', 'hard_act_last7d', 'moderate_act_last7d', 'walking_last7d'
]
for col in feature_cols:
    if col in df.columns:
        df[col] = df[col].replace({8: np.nan, 9: np.nan})

print("  Kode sentinel (998, 8, 9) pada umur dan kondisi kronis -> NaN")

# ===========================================================================
# FILTER  — HANYA RESPONDEN USIA ≥ 18 TAHUN
# ===========================================================================
garis("FILTER: Responden usia >= 18 tahun")
n_sebelum = len(df)
df = df[df["age"] >= 18].copy()
print(f"  Responden usia < 18 tahun yang dihapus : {n_sebelum - len(df)}")

# ===========================================================================
# LANGKAH 2 — FILTER NILAI KOSONG & NILAI MUSTAHIL PADA TEKANAN DARAH 
# ===========================================================================
garis("LANGKAH 2: Koreksi nilai mustahil tekanan darah")
bad_bp = df["bp_diastolic"] >= df["bp_systolic"].fillna(False)

print(f"  Tekanan darah mustahil disisihkan : {int(bad_bp.sum())} baris -> NaN")
df.loc[bad_bp, ["bp_systolic", "bp_diastolic"]] = np.nan

df = df.dropna(subset=["bp_systolic", "bp_diastolic"]).copy() #drop baris td yang kosong

# Buang outlier Antropometri sebelum hitung BMI
df.loc[(df["height_cm"] < 100) | (df["height_cm"] > 200), "height_cm"] = np.nan
df.loc[(df["weight_kg"] < 25) | (df["weight_kg"] > 200), "weight_kg"] = np.nan
df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2

# ===========================================================================
# LANGKAH 3 — PEMBENTUKAN LABEL TARGET HIPERTENSI
# ===========================================================================
garis("LANGKAH 3: Pembentukan label target hipertensi")

df["label_hypertension"] = (
    (df["bp_systolic"] >= 140) | (df["bp_diastolic"] >= 90)
).astype(int)
prop = df["label_hypertension"].mean()
print(f"  Proporsi hipertensi : {prop:.1%} (tidak hipertensi {1-prop:.1%})")

# ===========================================================================
# LANGKAH 4 — PENGHAPUSAN TEKANAN DARAH DARI HIMPUNAN FITUR
# ===========================================================================
LEAKAGE_COLS = ["bp_systolic", "bp_diastolic"]

# ===========================================================================
# LANGKAH 5 — HITUNG BMI, ENCODING & PENANGANAN STRUCTURAL MISSING
# ===========================================================================
garis("LANGKAH 5: BMI, encoding & perbaikan Structural Missing Values")


# B. Encoding Biner
def to_binary(s, satu=1):
    return s.map({satu: 1, 3: 0})

df["is_female"] = (df["sex"] == 3).astype(float)
df.loc[df["sex"].isna(), "is_female"] = np.nan

biner_map = [
    "has_tobacco", "is_diabetes", "is_high_cholesterol", "is_kidney_disease", "is_stroke", "has_fried_food", "has_noodles", "has_fast_food", "hard_act_last7d", 
    "moderate_act_last7d", "walking_last7d"
]
for col in biner_map:
    if col in df.columns:
        df[col] = to_binary(df[col])

# Penyesuaian nama untuk konsistensi
df = df.rename(columns={
    "is_diabetes": "has_diabetes",
    "is_high_cholesterol": "has_high_cholesterol",
    "is_kidney_disease": "has_kidney_disease",
    "is_stroke": "has_stroke"
})



# TAMBAHAN: Feature Engineering active_status (Berdasarkan Riskesdas 2013)
print("  Membuat fitur 'active_status' berdasarkan hard_act dan moderate_act...")

# Inisialisasi daata NaN agar data yang kosong tetap kosong
df["active_status"] = np.nan

# Kondisi Kurang Aktif (0): Tidak melakukan aktivitas berat DAN sedang
kondisi_tidak_aktif = (df["hard_act_last7d"] == 0) & (df["moderate_act_last7d"] == 0)
df.loc[kondisi_tidak_aktif, "active_status"] = 0

# Kondisi Aktif (1): Melakukan aktivitas berat ATAU sedang
kondisi_aktif = (df["hard_act_last7d"] == 1) | (df["moderate_act_last7d"] == 1)
df.loc[kondisi_aktif, "active_status"] = 1

# Hapus fitur lama agar tidak terjadi redundansi/multikolinearitas
df = df.drop(columns=["hard_act_last7d", "moderate_act_last7d", "walking_last7d"], errors="ignore")


# # C. Structural Missing Values -> Jika 'has_' = 0, maka 'freq_' = 0
# diet_pairs = [
#     ("has_noodles", "freq_noodles"),
#     ("has_fast_food", "freq_fast_food"),
#     ("has_fried_food", "freq_fried_food")
# ]

# print("  Menerapkan nilai 0 pada frekuensi diet jika kuesioner dijawab 'Tidak' (0)...")
# for has_col, freq_col in diet_pairs:
#     if has_col in df.columns and freq_col in df.columns:
#         df.loc[df[has_col] == 0, freq_col] = 0


# ===========================================================================
# AUDIT PASCA-STRUCTURAL MISSING & SELEKSI FITUR OTOMATIS
# ===========================================================================
garis("AUDIT PERSENTASE SISA MISSING VALUE (MCAR)")

# Daftar seluruh fitur kandidat yang ingin dievaluasi
fitur_kandidat = [
    "age", "is_female", "bmi",
    "has_tobacco", "has_diabetes", "has_high_cholesterol", "has_kidney_disease", "has_stroke", "has_fried_food", "has_noodles", "has_fast_food", "active_status"
]

# Hitung ulang persentase missing value setelah pembersihan structural zero
mcar_report = (df[fitur_kandidat].isna().mean() * 100).round(2)
print("Persentase sisa nilai hilang (MCAR) per fitur kandidat:")
print(mcar_report.to_string())

# D. Seleksi Fitur Otomatis Berdasarkan Threshold 40%
THRESHOLD = 40.0
FITUR = [col for col in fitur_kandidat if mcar_report[col] <= THRESHOLD]
FITUR_DROPPED = [col for col in fitur_kandidat if mcar_report[col] > THRESHOLD]

garis("HASIL SELESAI FITUR (THRESHOLD 40%)")
print(f"  [LOLOS] Fitur dengan sisa NaN <= {THRESHOLD}%:\n  {FITUR}\n")
print(f"  [DROP]  Fitur dengan sisa NaN > {THRESHOLD}%:\n  {FITUR_DROPPED}\n")

TARGET   = "label_hypertension"
# KOL_SUKU = "ethnicity"   # nama kolom suku bangsa (dari ar15d) di master dataset

# (LANGKAH 5) FITUR sudah terpilih di atas via threshold missing 40%.
# Sertakan ethnicity MENTAH agar ikut ter-split; belum di-encode di sini.
# punya_suku   = KOL_SUKU in df.columns
# kolom_ekstra = [KOL_SUKU] if punya_suku else []
# if not punya_suku:
#     print(f"  [PERINGATAN] Kolom '{KOL_SUKU}' tidak ditemukan — suku dilewati.")

df_model = df[FITUR + [TARGET]].copy()

# Missing suku diisi kategori eksplisit (fillna konstanta -> tidak ada kebocoran).
# if punya_suku:
#     df_model[KOL_SUKU] = df_model[KOL_SUKU].astype("object").fillna("Tidak Diketahui")

print(f"  Jumlah fitur numerik/biner terpilih : {len(FITUR)}")


# ===========================================================================
# LANGKAH 6 — SPLIT & IMPUTASI
# ===========================================================================
garis("LANGKAH 6: Split data latih-uji & Imputasi")
X, y = df_model[FITUR], df_model[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Klasifikasi variabel untuk imputasi
num_cols = ["age", "bmi"]  
cat_cols = [
    "is_female", "has_tobacco", "has_diabetes", "has_high_cholesterol", "has_kidney_disease", "has_stroke",
    "has_fast_food", "has_fried_food", "has_noodles", 
    "active_status"
]

imp_num = SimpleImputer(strategy="median")
imp_cat = SimpleImputer(strategy="most_frequent")

Xtr = X_train.copy(); Xte = X_test.copy()
Xtr[num_cols] = imp_num.fit_transform(X_train[num_cols])
Xte[num_cols] = imp_num.transform(X_test[num_cols])
Xtr[cat_cols] = imp_cat.fit_transform(X_train[cat_cols])
Xte[cat_cols] = imp_cat.transform(X_test[cat_cols])


# # ===========================================================================
# # LANGKAH 6b — GROUPING SUKU (murni frekuensi 1%) + ONE-HOT
# # ===========================================================================
# if punya_suku:
#     garis("LANGKAH 6b: Grouping suku (1%) + one-hot")
#     AMBANG = 0.01

#     # -- LANGKAH 1: BIKIN KAMUS dari TRAIN saja --
#     batas = AMBANG * len(Xtr)
#     freq  = Xtr[KOL_SUKU].value_counts()
#     dipertahankan = freq[freq >= batas].index.tolist()
#     mapping = {s: (s if s in dipertahankan else "Lainnya") for s in freq.index}

#     print(f"  n_train={len(Xtr)} | batas 1% = {batas:.1f} responden")
#     print(f"  Dipertahankan ({len(dipertahankan)}): {dipertahankan}")
#     print(f"  Digabung -> Lainnya: {[s for s in freq.index if s not in dipertahankan]}")

#     # -- LANGKAH 2: PAKAI KAMUS ke TRAIN & TEST (dua-duanya) --
#     Xtr[KOL_SUKU] = Xtr[KOL_SUKU].map(mapping).fillna("Lainnya")
#     Xte[KOL_SUKU] = Xte[KOL_SUKU].map(mapping).fillna("Lainnya")  # kategori tak dikenal -> Lainnya

#     # one-hot: fit di train, samakan kolom test (jaminan jumlah kolom identik)
#     dtr = pd.get_dummies(Xtr[KOL_SUKU], prefix="suku")
#     dte = pd.get_dummies(Xte[KOL_SUKU], prefix="suku").reindex(columns=dtr.columns, fill_value=0)

#     Xtr = pd.concat([Xtr.drop(columns=[KOL_SUKU]), dtr], axis=1)
#     Xte = pd.concat([Xte.drop(columns=[KOL_SUKU]), dte], axis=1)
#     Xte = Xte[Xtr.columns]   # samakan urutan kolom test = train

#     print(f"  Kolom suku: train={dtr.shape[1]} | test={dte.shape[1]}  (harus sama)")


# ===========================================================================
# FINALISASI
# ===========================================================================
garis("FINALISASI")
assert list(Xtr.columns) == list(Xte.columns), "Kolom train & test tidak identik!"
print(f"  Total fitur akhir: {Xtr.shape[1]} (train) / {Xte.shape[1]} (test)")


Xtr.assign(label_hypertension=y_train.values).to_csv("train_hipertensi.csv", index=False)
Xte.assign(label_hypertension=y_test.values).to_csv("test_hipertensi.csv", index=False)

print("\nMissing value setelah imputasi:")
print(f"Train : {Xtr.isna().sum().sum()} sel")
print(f"Test  : {Xte.isna().sum().sum()} sel")
print("--- DATA PREPARATION SELESAI. ---")
